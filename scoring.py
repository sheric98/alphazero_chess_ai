import numpy as np

piece_values = {'K': 3, 'Q': 14, 'R': 5, 'B': 3.25, 'N': 3, 'P': 1,
                'k': -3, 'q': -14, 'r': -5, 'b': -3.25, 'n': -3, 'p': -1}


def eval_board(board):
    player = 1 if board.turn else -1
    fen = board.board_fen()

    val = 0.0
    tot = 0
    for char in fen:
        if char in piece_values:
            val += (player * piece_values[char])
            tot += abs(piece_values[char])

    v = val / tot
    return np.tanh(3*v)