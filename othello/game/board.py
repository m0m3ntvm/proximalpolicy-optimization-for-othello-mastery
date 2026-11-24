"""Othello game state, canonicalization, and match/self-play drivers."""
import random

import numpy as np
import torch

from othello.game.rules import decide, evaluate, get_valid_moves


class OthelloBoard:
    """An Othello board, always oriented so +1 pieces belong to the player to move."""

    def __init__(self, size: int = 8) -> None:
        self.size = size
        self.gamma = np.exp(np.log(0.01) / (size ** 2 - 4))  # dynamic gamma according to the board size
        self.board = np.zeros((self.size, self.size), dtype=np.short)
        ##########################################################
        self.board[self.size // 2 - 1][self.size // 2 - 1] = 1.0
        self.board[self.size // 2][self.size // 2] = 1.0
        self.board[self.size // 2 - 1][self.size // 2] = -1.0
        self.board[self.size // 2][self.size // 2 - 1] = -1.0
        ########################################################
        self.player = True  # True for white and False for black
        ########################################################
        self.white_history = []
        self.black_history = []
        ########################################################
        self.white_mask = []
        self.black_mask = []
        ########################################################
        self.white_actions = []
        self.black_actions = []
        ########################################################
        self.white_return = []
        self.black_return = []
        ########################################################
        self.white_return_scale = []
        self.black_return_scale = []
        #######################################################
        self.last_player_moved = True  # save if the last player was able to move

    def resetBoard(self) -> None:
        # reset the board to the initial state
        self.board = np.zeros(size=(self.size, self.size), dtype=np.short)
        ##########################################################
        self.board[self.size // 2 - 1][self.size // 2 - 1] = 1.0
        self.board[self.size // 2][self.size // 2] = 1.0
        self.board[self.size // 2 - 1][self.size // 2] = -1.0
        self.board[self.size // 2][self.size // 2 - 1] = -1.0
        ########################################################
        self.player = True  # True for white and False for black
        #######################################################
        self.last_player_moved = True  # save if the last player was able to move

    def resetMem(self) -> None:
        # remove the state of the system except the board and the player
        ########################################################
        self.white_history = []
        self.black_history = []
        ########################################################
        self.white_mask = []
        self.black_mask = []
        ########################################################
        self.white_actions = []
        self.black_actions = []
        ########################################################
        self.white_return = []
        self.black_return = []
        ########################################################
        self.white_return_scale = []
        self.black_return_scale = []

    def getBoard(self):
        # return the board as numpy array
        # return all possible moves
        # return opponents pieces under attack
        # return the mask
        mask, under_attack, moves = get_valid_moves(self.board)
        element = (
            np.expand_dims(np.concatenate((np.expand_dims(self.board, axis=0), np.expand_dims(under_attack, axis=0)), axis=0), axis=0).astype(np.float32),
            np.expand_dims(np.expand_dims(mask, axis=0), axis=0).astype(np.float32)
        )
        return element, moves

    def switchPlayer(self) -> bool:
        self.player = not self.player
        self.board = np.rot90(self.board, k=2) * -1
        return self.player

    def drawBoard(self, background_color=np.array([238, 238, 210]), boarder_color=np.array([92, 64, 51]), quality_factor=128, boarder_thickness=2, checker_size="m"):
        # every block of the board will be quality_factor x quality_factor pixels
        board_shape = self.board.shape  # getting the shape of the board
        image_len = self.size * quality_factor  # the w & h of the image
        image = np.ones(shape=(image_len, image_len, 1)) * background_color  # creating a plank image with the background color
        mask, _, _ = get_valid_moves(self.board)
        # draw boxes
        for i in range(0, image_len):
            if i % quality_factor < boarder_thickness or quality_factor - i % quality_factor < boarder_thickness:
                # drawing horizontal boarders
                image[i, :] = boarder_color
                # drawing vertical boarders
                image[:, i, :] = boarder_color
        ####################################################
        image = image / 255.0  # normalize the image 0 -> 1
        checker = np.ones(shape=(quality_factor, quality_factor, 3))  # generating a checker
        # draw the actual checker
        x = (7 - (1 if checker_size == "m" else 2 if checker_size == "l" else 0)) * (quality_factor // 4) ** 2
        for i in range(0, quality_factor):
            for j in range(0, quality_factor):
                ############################################################
                if i ** 2 + j ** 2 - quality_factor * i - quality_factor * j + x > 0:
                    checker[i, j] = np.array([0, 0, 0])

        map_rows, map_cols, _ = image.shape
        # putting checkers on the board
        for raw_index, raw in enumerate(self.board):
            for cell_index, cell in enumerate(raw):
                raw_start, raw_end = raw_index * quality_factor, raw_index * quality_factor + quality_factor
                cell_start, cell_end = cell_index * quality_factor, cell_index * quality_factor + quality_factor

                if int(cell) == 1:
                    # Update the image for white pieces
                    image[raw_start:raw_end, cell_start:cell_end] = (
                        image[raw_start:raw_end, cell_start:cell_end] * (1 - checker) + checker)

                if int(cell) == -1:
                    # Update the image for black pieces
                    image[raw_start:raw_end, cell_start:cell_end] = (
                        image[raw_start:raw_end, cell_start:cell_end] * (1 - checker))

                if mask[raw_index][cell_index] == 1:
                    # Update the image for possible moves (make them darker)
                    image[raw_start:raw_end, cell_start:cell_end] = (
                        image[raw_start:raw_end, cell_start:cell_end] * 0.85)

        return image

    def modelMove(self, OthelloRL, switching: bool = True, epsilon: float = 0.0, boltzmann: bool = True, device=None) -> bool:
        # let the model make a move
        with torch.no_grad():
            (x, mask), all_moves = self.getBoard()

            if np.random.rand() >= epsilon:
                if device:
                    output_tensor = OthelloRL(torch.from_numpy(x).to(device),
                                              torch.from_numpy(mask).to(device)).cpu().numpy()
                else:
                    output_tensor = OthelloRL(torch.from_numpy(x),
                                              torch.from_numpy(mask)).numpy()

                if len(all_moves) > 0:  # check if any move is available
                    if boltzmann:
                        # boltzmann exploration
                        I, J = decide(probabilities=np.squeeze(output_tensor, axis=0))
                    else:
                        # greedy exploration
                        max_indices = np.argmax(output_tensor.reshape(-1, output_tensor.shape[-2] * output_tensor.shape[-1]), axis=1)
                        I, J = np.divmod(max_indices, output_tensor.shape[-1])
                        I, J = I[0], J[0]

                    board_change = all_moves[(I, J)]
                else:
                    I, J = None, None
                    board_change = None  # no move
                    # game ends or player has no move
            else:
                # random exploration
                if len(all_moves) > 0:
                    I, J = random.choice(list(all_moves.keys()))
                    board_change = all_moves[(I, J)]
                else:
                    I, J = None, None
                    board_change = None  # no move
                    # game ends or player has no move

        game_state = self.makeMove(helpers=(x, mask), x=I, y=J, board_change=board_change,
                                   switching=switching)
        # return if the game ended
        return game_state

    def humanMove(self, x: int, y: int, switching: bool = True) -> bool:
        # let the model make a move
        (X, mask), all_moves = self.getBoard()
        if len(all_moves) == 0:
            I, J = None, None
            board_change = None
        elif (x, y) not in list(all_moves):
            raise Exception(f"Invalid move : {x,y}")
        else:
            board_change = all_moves[(x, y)]
            I, J = x, y

        game_state = self.makeMove(helpers=(X, mask), x=I, y=J, board_change=board_change,
                                   switching=switching)

        return game_state

    def makeMove(self, helpers, x: int = 0, y: int = 0, board_change=None, switching: bool = True) -> bool:
        # given x & y this function is responsible of making moves on the board and switch player for the other turn

        if board_change is not None:
            if (self.board[x, y] != 0.0):
                print(f"{x,y} is already taken.")

            self.last_player_moved = True
            infrence_input, mask = helpers
            # check if the position is inside the board
            self.saveBoard(x, y, infrence_input=infrence_input, mask=mask)
            # take the position
            self.board[x, y] = 1.0

            self.board[board_change] = 1.0
        else:
            if not self.last_player_moved:
                # if last player also wasn't able to move
                self.switchPlayer()
                return False
            else:
                self.last_player_moved = False

        # switching player for the next turn
        if switching:
            self.switchPlayer()

        return False if (self.board != 0).all() else True

    def saveBoard(self, x, y, infrence_input, mask):
        # Save the board for training later
        board = np.zeros((1, 1, self.size, self.size))
        board[0, 0, x, y] = 1

        scalar_array = self.gamma ** np.sum(self.board == 0)

        ############################
        if self.player:
            self.white_history.append(np.squeeze(infrence_input))
            self.white_mask.append(np.squeeze(mask, axis=0))
            self.white_actions.append(np.squeeze(board, axis=0))
            self.white_return_scale.append(scalar_array)
        else:
            self.black_history.append(np.squeeze(infrence_input))
            self.black_mask.append(np.squeeze(mask, axis=0))
            self.black_actions.append(np.squeeze(board, axis=0))
            self.black_return_scale.append(scalar_array)

    def selfPlay(self, OthelloRL, start_explore=(0, 32), switching=True, epsilon=0.0, boltzmann=True, device=None):
        # self play one game
        start, finish = start_explore  # play a random number of moves
        start_explore_depth = random.randint(start, min(finish, 32))  # upper bound of 32
        ##################################################

        for _ in range(0, start_explore_depth):
            self.modelMove(OthelloRL, switching=True, epsilon=1.0, boltzmann=False, device=device)

        self.resetMem()

        while self.modelMove(OthelloRL, switching=switching, epsilon=epsilon, boltzmann=boltzmann, device=device):
            pass  # play to the end of the game

        delta = evaluate(board=self.board)

        winner = self.player if delta > 0 else not self.player if delta < 0 else -1
        # no actions
        if len(self.white_actions) == 0 and len(self.black_actions) == 0:
            return winner, None

        data = []
        for i in range(len(self.white_actions)):
            white_return = 0.0 if winner == -1 else self.white_return_scale[i].astype(np.float32) if winner else -1 * self.white_return_scale[i].astype(np.float32)

            data.append(((self.white_history[i].astype(np.float32), self.white_mask[i].astype(np.float32)), self.white_actions[i].astype(np.float32), white_return))

        for i in range(len(self.black_actions)):
            black_return = 0.0 if winner == -1 else -1 ** self.black_return_scale[i].astype(np.float32) if winner else self.black_return_scale[i].astype(np.float32)

            data.append(((self.black_history[i].astype(np.float32), self.black_mask[i].astype(np.float32)), self.black_actions[i].astype(np.float32), black_return))

        return winner, data

    def modelVsModel(self, OthelloRL_1, OthelloRL_2, start_explore=(0, 32), Epsilon_1=0.0, Epsilon_2=0.0, boltzmann=False):
        # this function returns true,false if the first model won
        # draw => false,false
        # lost => false,true
        # true,true is invalid
        start, finish = start_explore  # play a random number of moves
        start_explore_depth = random.randint(start, min(finish, 62))  # upper bound of 32
        ##################################################

        for _ in range(0, start_explore_depth):
            self.modelMove(OthelloRL_1, switching=True, epsilon=1.0, boltzmann=False)

        singal = True  # true when the game ends

        while True:
            singal = self.modelMove(OthelloRL=OthelloRL_1, epsilon=Epsilon_1, boltzmann=boltzmann)
            if not singal:
                delta = evaluate(board=self.board)
                return np.array([delta < 0, delta > 0])

            singal = self.modelMove(OthelloRL=OthelloRL_2, epsilon=Epsilon_2, boltzmann=boltzmann)
            if not singal:
                delta = evaluate(board=self.board)
                return np.array([delta > 0, delta < 0])
