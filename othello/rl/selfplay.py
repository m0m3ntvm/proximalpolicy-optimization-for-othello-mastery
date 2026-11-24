"""Parallel self-play drivers (multiprocessing)."""
import multiprocessing as mp

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


def process_func(model, device, results_queue, Epsilon, boltzmann, game_per_worker, board_size) -> None:
    result = asyncSelfPlay(model, device, Epsilon=Epsilon, boltzmann=boltzmann, game_per_worker=game_per_worker, board_size=board_size)
    results_queue.append(result)


def parallelSelfPlay(model, device, workers: int = 4, Epsilon: float = 0.0, boltzmann: bool = False, game_per_worker: int = 10, board_size: int = 8) -> list:
    """Play games in multiple workers."""
    results_queue = mp.Manager().list()

    model.to(device)
    model.eval()
    model.share_memory()

    processes = []

    for rank in range(workers):
        p = mp.Process(target=process_func, args=(model, device, results_queue, Epsilon, boltzmann, game_per_worker, board_size), name=f'self playing worker -{rank}')
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
