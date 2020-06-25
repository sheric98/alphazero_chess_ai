import chess.pgn
from game import Game
import encode
import numpy as np
import util
import network


datadir = 'pgn_files/'


def get_pgn_training(names, path):
    training = []
    for name in names:
        path = datadir + name
        pgn = open(path)

        pgn_game = chess.pgn.read_game(pgn)
        while pgn_game is not None:
            res = pgn_game.headers['Result']
            placeholder = []
            board = pgn_game.board()
            game = Game(board)

            for move in pgn_game.mainline_moves():
                player = 1 if game.board.turn else -1
                uci_move = move.uci()
                probs = encode.get_prob_mask([uci_move])
                assert np.sum(probs) == 1
                normalized_encoded = game.copy_and_normalize().get_encoded()
                placeholder.append((normalized_encoded, probs, player))
                game.make_move(uci_move)

            # check if not resigned
            state = game.get_game_state()
            if (res == '1/2-1/2' and state == 0) or abs(state) == 1:
                for tup in placeholder:
                    train = (tup[0], state * tup[2], tup[1])
                    training.append(train)

            pgn_game = chess.pgn.read_game(pgn)

        pgn.close()

    util.save_prev_training(training, path)
    return training


def sl_train_net(net, training, nepochs=5):
    network.train_mult_epochs(net, training, nepochs, print_all=True)