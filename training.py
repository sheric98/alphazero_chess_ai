from game import Game
from mcst import MCST
import numpy as np
from collections import deque as deq
import time
import util
import network
import encode
import self_play as sp


def play_through_examples(net, thresh=15, max_moves=500, preds=None,
                          use_self_val=False):
    i = 0
    game = Game()
    mcst = MCST(net, preds=preds, early_term=True)
    end = 2
    training_tups = []
    player = 1

    while end == 2:
        normalized_board = game.copy_and_normalize()
        if use_self_val:
            v, p = mcst.get_pred(normalized_board)
            v *= player
        else:
            v = None
        if game.get_full_moves() >= max_moves:
            end = game.early_rollout(value=v)
            break
        resign = game.check_resign(value=v)
        if resign is not None:
            end = resign
            break

        temp = 1 if i < thresh else 0
        probs = mcst.get_probs(normalized_board, temp)
        board = normalized_board.get_encoded()
        training_tups.append((board, probs, player))
        move = np.random.choice(len(probs), p=probs)
        uci_move = normalized_board.coord_to_move(move)
        if player == -1:
            uci_move = encode.flip_move(uci_move)
        game.make_move(uci_move)
        player *= -1
        i += 1

        end = game.get_game_state()

    ret = []
    for tup in training_tups:
        train = (tup[0], end*tup[2], tup[1])
        ret.append(train)

    return ret


def train_net(net, path, training_path, niters=100, neps=100,
              queue_cap=100000, prev_training_cap=20, start=0):
    if net is None:
        curr_net = util.load_net(path)
        prev_training = util.load_prev_training(training_path)
        print(len(prev_training))
    else:
        curr_net = net
        util.save_net(curr_net, path)
        prev_training = []

    for i in range(start, niters):
        print('Starting iter %d' % (i + 1))
        start = time.time()
        iter_training = deq([], maxlen=queue_cap)

        pred_dict = {}
        for j in range(neps):
            nepstart = time.time()
            train = play_through_examples(curr_net, preds=pred_dict)
            iter_training.extend(train)
            nepend = time.time()
            print('nep iteration %d: %f' % (j, nepend - nepstart))

        prev_training.append(iter_training)

        if len(prev_training) > prev_training_cap:
            prev_training.pop(0)

        end1 = time.time()
        print('Generating Training: %f seconds' % (end1 - start))

        util.save_prev_training(prev_training, training_path)

        training = []
        for x in prev_training:
            training.extend(x)

        prev_net = util.load_net(path)

        network.train_mult_epochs(curr_net, training)

        end2 = time.time()
        print('Training Model: %f seconds' % (end2 - end1))

        new_pred_dict = {}

        comp = sp.compare_nets(curr_net, prev_net, prev_pred_dict=pred_dict,
                               new_pred_dict=new_pred_dict)

        end3 = time.time()
        print('Final Comparison: %f seconds' % (end3 - end2))
        if not comp:
            print('reject model')
            curr_net = prev_net
        else:
            print('update model')
            util.save_net(curr_net, path)

        print('Total Iteration Time: %f seconds\n' % (end3 - start))

    util.save_net(curr_net, path)
    return curr_net
