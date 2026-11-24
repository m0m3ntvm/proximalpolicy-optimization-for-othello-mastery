"""Parallel self-play drivers (multiprocessing and vectorized lockstep)."""
import multiprocessing as mp
import random

import numpy as np

from othello.game.board import OthelloBoard


def asyncSelfPlay(model, device, Epsilon: float = 0.0, boltzmann: bool = False, game_per_worker: int = 10, board_size: int = 8) -> list:
    """Play a number of games for a single worker."""
    data = []
    for game in range(game_per_worker):
        # Define the linear model
        Board = OthelloBoard(size=board_size)  # generate a random board with dynamic size
        ##################################################
        _, _data = Board.selfPlay(model, epsilon=Epsilon, device=device, boltzmann=boltzmann)  # self-playing
        ##################################################
        data = data + _data if _data else data

    return data


def process_func(model, device, results_queue, Epsilon, boltzmann, game_per_worker, board_size, seed=None) -> None:
    # each worker owns its CPU cores: keep torch single-threaded so N workers
    # don't oversubscribe the machine with N x num_cores threads
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    if seed is not None:
        # reproducible worker streams: every game draws from these two RNGs
        np.random.seed(seed)
        random.seed(seed)
    result = asyncSelfPlay(model, device, Epsilon=Epsilon, boltzmann=boltzmann, game_per_worker=game_per_worker, board_size=board_size)
    results_queue.append(result)


def parallelSelfPlay(model, device, workers: int = 4, Epsilon: float = 0.0, boltzmann: bool = False,
                     game_per_worker: int = 10, board_size: int = 8, batch_games: int = 0,
                     seed: int = None) -> list:
    """Play games in multiple workers.

    batch_games > 0 switches to the vectorized environment: a single process
    plays all workers * game_per_worker games in lockstep, with batch_games
    games per model forward. This amortizes the per-ply torch/GPU overhead
    and is much faster on GPUs (see othello.rl.vecenv).

    seed: when set, runs are reproducible — batched mode derives per-game
    streams from it (independent of workers and batch); classic mode seeds
    each worker with seed + rank.
    """
    if batch_games > 0:
        from othello.rl.vecenv import batchedSelfPlay
        model.to(device)
        model.eval()
        model.return_val = False
        return batchedSelfPlay(model, device, workers * game_per_worker, Epsilon=Epsilon,
                               boltzmann=boltzmann, board_size=board_size, batch=batch_games,
                               seed=seed)

    results_queue = mp.Manager().list()

    model.to(device)
    model.eval()
    model.share_memory()

    processes = []

    for rank in range(workers):
        p = mp.Process(target=process_func, args=(model, device, results_queue, Epsilon, boltzmann, game_per_worker, board_size,
                                                  None if seed is None else seed + rank), name=f'self playing worker -{rank}')
        p.start()
        processes.append(p)
        print(f'Started : {p.name}')

    for p in processes:
        p.join()
        print(f'Finished : {p.name}')

    result = []
    for r_q in results_queue:
        result = result + r_q

    return result
