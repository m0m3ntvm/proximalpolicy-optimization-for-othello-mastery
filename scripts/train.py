"""SHA-ZA PPO training loop, extracted from othello.ipynb (cells 5/8/10/12/13/18/20).

Same algorithm and default hyperparameters as the published notebook run.
Run with:

  uv run python scripts/train.py                  # from scratch
  uv run python scripts/train.py --load-checkpoint  # resume from save_dir

NOTE: self-play must run under a spawn context on Windows; see main().
"""
import argparse
import copy
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.multiprocessing as mp
from tabulate import tabulate
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm

from othello import OthelloBoard, OthelloRL, PPO_Loss, miniMax, parallelSelfPlay


def calcEpsilon(step, epsilon_max, epsilon_min, scale_fac, steps):
    return (epsilon_max - epsilon_min) * np.exp(-1 * (step * scale_fac) / steps) + epsilon_min


def Test(agent_A, agent_B=None, test_num=250, board_base_size=8, num_boards=1):
    # model to evaluation mode
    agent_A.cpu().eval()
    agent_A.return_val = False
    if agent_B:
        agent_B.cpu().eval()
        agent_B.return_val = False

    score = np.array([0, 0])

    for i in tqdm(range(test_num), desc=f"model {agent_A.name} vs {agent_B.name if agent_B else 'random'}"):
        # change the size of the board from 8 to 10 to 12 and so on ...
        board_size = board_base_size + 2 * (i % num_boards)  # dynamic board size
        Board = OthelloBoard()
        #########################################################
        if agent_B:
            score += Board.modelVsModel(agent_A, agent_B)
        else:
            score += Board.modelVsModel(agent_A, agent_A, Epsilon_2=1.0)

    return score


def RunTests(Q, ref, engine, win_trace, test_num):
    # testing the model against random and engine
    #####################################################################
    score_random = Test(Q, test_num=test_num)
    win_trace['Agent_vs_Random'].append(round((score_random[0] / (score_random[0] + score_random[1])) * 100, 4))
    # test model against pretrained reference model
    ######################################################################
    if ref is not None:
        score_ref = Test(Q, ref, test_num=test_num)
        win_trace[f"Agent_vs_{ref.name}"].append(round((score_ref[0] / (score_ref[0] + score_ref[1])) * 100, 4))
    # test model against engine
    #####################################################################
    score_engine = Test(Q, engine, test_num=test_num // 2)
    win_trace["Agent_vs_Engine"].append(round((score_engine[0] / (score_engine[0] + score_engine[1])) * 100, 4))

    results = [
        (f"{Q.name} vs random", f"{win_trace['Agent_vs_Random'][-1]}%"),
        (f"{Q.name} vs {engine.name}", f"{win_trace['Agent_vs_Engine'][-1]}%"),
    ]
    if ref is not None:
        results.insert(1, (f"{Q.name} vs {ref.name}", f"{win_trace[f'Agent_vs_{ref.name}'][-1]}%"))

    table = tabulate(results, headers=["Comparison", f"Win Rate for {Q.name}"])
    print(table)


def fit_ppo(data_loader, agents, loss_dict, optim, scheduler, device, epochs_per_step, batch_size, echo=True):
    agents[0].to(device).train()  # model to training mode
    agents[1].to(device).eval()  # model to evaluation mode
    ##############################
    model_clone = copy.deepcopy(agents[0])  # later the old model
    ##############################
    agents[0].return_val = True
    agents[1].return_val = False
    ##############################
    avg_total_loss = 0.0
    avg_policy_loss = 0.0
    avg_value_loss = 0.0
    avg_entropy_loss = 0.0

    for ep in range(epochs_per_step):
        optim.zero_grad()  # gradients accumulation
        for batch_index, ((x, mask), act, ret) in enumerate(tqdm(data_loader)):
            X = x.to(device)
            Mask = mask.to(device)
            #####################
            policy_new, values = agents[0](X, Mask)  # output from the model under training
            policy_old = agents[1](X, Mask)
            loss, policy_loss, value_loss, entropy_loss = criterion(policy_old, policy_new,
                                                                    act.to(device), values, ret.to(device))  # calculating ppo loss
            loss.backward()
            ############################
            avg_total_loss += loss.item()
            avg_policy_loss += policy_loss.item()
            avg_value_loss += value_loss.item()
            avg_entropy_loss += entropy_loss.item()

        optim.step()

    num_batch = len(data_loader)
    total_ep = epochs_per_step * num_batch  # total number of epochs
    #############################
    loss_dict['total_loss'].append(avg_total_loss / total_ep)
    loss_dict['policy_loss'].append(avg_policy_loss / total_ep)
    loss_dict['value_loss'].append(avg_value_loss / total_ep)
    loss_dict['entropy_loss'].append(avg_entropy_loss / total_ep)
    if echo:
        print(f"data size : {num_batch * batch_size} states.")
        print(f"Avg total loss : {loss_dict['total_loss'][-1]}, Avg policy loss : {loss_dict['policy_loss'][-1]}.")
        print(f"Avg value loss : {loss_dict['value_loss'][-1]}, Avg entropy loss : {loss_dict['entropy_loss'][-1]}.")

    ########################################################
    scheduler.step()
    model_clone.name = "old"
    agents = [agents[0], model_clone]
    #######################################################
    agents[0].return_val = False
    agents[1].return_val = False  # only policy head works

    return agents, loss_dict, optim, scheduler


def main():
    parser = argparse.ArgumentParser(description='SHA-ZA PPO training (published configuration).')
    parser.add_argument('--learning-rate', type=float, default=0.5e-4)
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--test-num', type=int, default=250, help='games played in each test point')
    parser.add_argument('--test-freq', type=int, default=25, help='test every N steps')
    parser.add_argument('--game-per-worker', type=int, default=512)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--epochs-per-step', type=int, default=1)
    parser.add_argument('--load-checkpoint', action='store_true', help='resume from save-dir')
    parser.add_argument('--save-dir', type=str, default='training_checkpoints',
                        help='directory for models/logs (created if missing)')
    parser.add_argument('--ref-path', type=str, default=None,
                        help='optional reference model (notebook used /workspace/shaza_old.pth)')
    parser.add_argument('--epsilon-max', type=float, default=0.05)
    parser.add_argument('--epsilon-min', type=float, default=0.0)
    parser.add_argument('--scale-fac', type=float, default=4, help='controls epsilon decay rate')
    parser.add_argument('--engine-depth', type=int, default=7, help='minimax engine depth for evaluation')
    parser.add_argument('--internal-channels', type=int, default=64)
    parser.add_argument('--policy-net-depth', type=int, default=8)
    parser.add_argument('--value-coefficient', type=float, default=0.5)
    parser.add_argument('--entropy-coefficient', type=float, default=0.09)
    parser.add_argument('--clip-param', type=float, default=0.2)
    args = parser.parse_args()

    global criterion  # used inside fit_ppo, mirroring the notebook's module-level variable

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, '')

    # Hyperparameters
    learning_rate = args.learning_rate
    steps = args.steps
    batch_size = args.batch_size
    test_num = args.test_num
    test_frq = args.test_freq
    game_per_worker = args.game_per_worker
    num_workers = args.num_workers
    epochs_per_step = args.epochs_per_step
    load_check_point = args.load_checkpoint
    current_step = 1
    epsilon_max = args.epsilon_max
    epsilon_min = args.epsilon_min
    scale_fac = args.scale_fac
    engine_depth = args.engine_depth
    internal_channels = args.internal_channels
    policy_net_depth = args.policy_net_depth
    value_coefficient = args.value_coefficient
    entropy_coefficient = args.entropy_coefficient
    clip_param = args.clip_param

    gamma_lr = np.exp(np.log(1 / 8) / (steps))  # reduction factor for learning rate
    print(f"**discount factor is : {gamma_lr: .010f}, LR reaches 50% in {int(np.log(0.8) / np.log(gamma_lr))} steps.")
    step_lr = 1  # step size for the learning rate
    print(f"number of games per step : {game_per_worker * num_workers}")
    print(f"random seed : {torch.initial_seed()}, numpy : {np.random.get_state()[1][0]}.")

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Your current device is: {device}.")

    # Create 2 instances of OthelloRL (new = training, old = behavior policy)
    agents = [OthelloRL(internal_channels=internal_channels, res_depth=policy_net_depth, name="new", return_val=True),
              OthelloRL(internal_channels=internal_channels, res_depth=policy_net_depth, name="old", return_val=True)]

    if load_check_point:
        try:
            agents[0].load(os.path.join(save_path, f"shaza_{agents[0].name}.pth"))
            agents[1].load(os.path.join(save_path, f"shaza_{agents[1].name}.pth"))
        except Exception as e:
            print(e)

    # load the reference model (optional)
    ref = None
    if args.ref_path:
        ref = OthelloRL().to(device)
        if ref.load(path=args.ref_path):
            ref.name = "refernce"
        else:
            print(f"WARNING: reference model {args.ref_path} failed to load; skipping vs-ref tests.")
            ref = None

    # best checkpoint
    best_model = copy.deepcopy(agents[0])
    best_model.name = "best_model"

    agents[0] = agents[0].to(device)
    agents[1] = agents[1].to(device)

    agents[0].train()
    agents[1].eval()
    # Define the loss function and optimizer
    criterion = PPO_Loss(clip_param=clip_param, value_coefficient=value_coefficient,
                         entropy_coefficient=entropy_coefficient, echo=True)
    optim = torch.optim.AdamW(agents[0].parameters(), lr=learning_rate)

    if load_check_point:
        try:
            optim.load_state_dict(torch.load(os.path.join(save_path, "optimizer.pth"), map_location=device)["optimizer_state_dict"])
        except Exception as e:
            print(e)

    scheduler = StepLR(optim, step_size=step_lr, gamma=gamma_lr)

    if load_check_point:
        try:
            scheduler.load_state_dict(torch.load(os.path.join(save_path, "scheduler.pth"), map_location=device)["scheduler_state_dict"])
        except Exception as e:
            print(e)

    # Lists to store training metrics
    win_trace = {"Agent_vs_Random": [],
                 "Agent_vs_Engine": [],
                 f"{agents[0].name}_vs_{best_model.name}": []}
    if ref is not None:
        win_trace[f"Agent_vs_{ref.name}"] = []
    loss = {"total_loss": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy_loss": []}
    time_step = []  # store the time in sec for each epoch

    if load_check_point:
        try:
            data = np.load(os.path.join(save_path, "log.npy"), allow_pickle=True).item()
            win_trace = data["win trace"]
            loss = data["total loss"]
            time_step = data["avg time ber epoch"]
            current_step = data["step"] + 1
            print("check point has been loaded")
        except Exception as e:
            print(e)

    print(f"summary of model : {agents[0].name}")
    summary(agents[0], [(2, 8, 8), (1, 8, 8)])

    agents[0].return_val = False
    agents[1].return_val = False

    engine = miniMax(name="E", depth=engine_depth)

    try:
        avg_time = 0.0
        avg_loss = 0.0
        # 1 => steps
        for step in range(current_step, steps + 1):
            start = time.time()  # start counting time for the loop
            agents[0].to(device).eval()  # simulation in cpu
            agents[1].to(device).eval()
            ################
            agents[0].return_val = False
            agents[1].return_val = False  # only policy head works
            ################
            # generating data from self playing
            print("-------------------------------------------------------------")
            print(f"Step Num : {step}/{steps}.")
            print("-------------------------------------------------------------")
            data = parallelSelfPlay(model=agents[0], device=device, workers=num_workers,
                                    Epsilon=calcEpsilon(step, epsilon_max, epsilon_min, scale_fac, steps),
                                    boltzmann=True, game_per_worker=game_per_worker)
            # training loader
            train_loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
            # train the model
            print("")
            agents, loss, optim, scheduler = fit_ppo(data_loader=train_loader, agents=agents, loss_dict=loss,
                                                     optim=optim, device=device, scheduler=scheduler,
                                                     epochs_per_step=epochs_per_step, batch_size=batch_size)
            agents[0].cpu().eval()
            agents[1].cpu().eval()

            avg_time += (time.time() - start) / test_frq  # counting the average time for an epoch

            print(f"time stamp : {datetime.now()}")

            if step % test_frq == 0:
                print("\ntesting...\n")
                RunTests(agents[0], ref, engine, win_trace, test_num)
                score = Test(agents[0], best_model)  # new vs old
                print(f"model {agents[0].name} vs {best_model.name}:  {(score[0] / (score[0] + score[1]) * 100):.04f} %.")
                win_trace[f"{agents[0].name}_vs_{best_model.name}"].append((score[0] / (score[0] + score[1]) * 100))
                ##########################################
                time_step.append(avg_time)
                ###########################################
                data_log = {
                    "win trace": win_trace,
                    "total loss": loss,
                    "step": step,
                    "time stamp": str(datetime.now()),
                    "avg time ber epoch": time_step,
                }
                if win_trace[f"{agents[0].name}_vs_{best_model.name}"][-1] > 50.0:
                    best_model = copy.deepcopy(agents[0])
                    best_model.name = "best_model"
                    # save model
                    for agent in agents:
                        agent.save(os.path.join(save_path, f"shaza_{agent.name}.pth"))
                        torch.save({"optimizer_state_dict": optim.state_dict()},
                                   os.path.join(save_path, "optimizer.pth"))
                        torch.save({"scheduler_state_dict": scheduler.state_dict()},
                                   os.path.join(save_path, "scheduler.pth"))

                    # save log
                    np.save(os.path.join(save_path, "log.npy"), data_log)

                # reset vars
                avg_time = 0.0
                avg_loss = 0.0

    except Exception as e:
        log = {
            "message": f"{e}",
            "time stamp": str(datetime.now()),
        }
        print(log)
        np.save(os.path.join(save_path, "errorlog.npy"), log)


if __name__ == "__main__":
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    mp.set_start_method('spawn', force=True)
    main()
