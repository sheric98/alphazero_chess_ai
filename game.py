import chess
import encode
import copy
import util
import numpy as np


# wrapper for chess board
class Game:
    def __init__(self, board=None):
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board
        self.encoded = None
        self.valid_moves = None
        self.state = None
        self.state_map = {'*': 2, '1-0': 1, '0-1': -1, '1/2-1/2': 0}

    def get_encoded(self):
        if self.encoded is None:
            self.encoded = encode.encode_board(self.board)
        return self.encoded

    def get_key(self):
        return self.get_encoded().tobytes()

    def get_valid_moves(self):
        if self.valid_moves is None:
            self.valid_moves = set(map(lambda x: x.uci(), self.board.legal_moves))
        return self.valid_moves

    def get_mask(self):
        return encode.get_prob_mask(self.get_valid_moves())

    def make_move(self, uci):
        self.encoded = None
        self.state = None
        self.valid_moves = None
        self.board.push_uci(uci)

    def coord_to_move(self, ind):
        tup = np.unravel_index(ind, (8, 8, 73))
        starting = (tup[0], tup[1])
        starting_alg = encode.get_alg(starting)
        plane = tup[2]
        diff = encode.decode_dir(plane)
        if util.represents_int(diff[1]):
            final_char = ''
            diff1 = diff[0]
            diff2 = diff[1]
        else:
            final_char = diff[1]
            diff1 = 1 if self.board.turn else -1
            diff2 = diff[0]
        ending = (tup[0] + diff1, tup[1] + diff2)
        ending_alg = encode.get_alg(ending)
        return starting_alg + ending_alg + final_char

    def get_game_state(self):
        if self.state is None:
            self.state = self.state_map[self.board.result(claim_draw=True)]
        return self.state

    def copy_game(self):
        return copy.deepcopy(self)

    def normalize(self):
        if not self.board.turn:
            return Game(self.board.mirror())
        return self

    def move_and_normalize(self, uci):
        game = self.copy_game()
        game.make_move(uci)
        game.normalize()
        return game

    def copy_and_normalize(self):
        return self.copy_game().normalize()

if __name__ == '__main__':
    game = Game()
    game.make_move('d2d4')
    game.make_move('g8f6')
    game.make_move('d4d5')
    game.make_move('f6g8')
    game.make_move('d5d6')
    game.make_move('g8f6')
    game.make_move('d6c7')
    game.make_move('f6g8')
    print(game.get_valid_moves())
    uci = 'c7b8b'
    row, col = encode.get_coord(uci)
    plane = encode.get_plane_ind(uci)
    print(row, col, plane)
    print(game.coord_to_move((row, col, plane)))