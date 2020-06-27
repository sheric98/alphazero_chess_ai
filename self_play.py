from game import Game
from mcst import MCST
import encode


# tree 1 goes first
def play_single(tree1, tree2, best_tree=1, use_best_val=False,
                max_moves=500):
    game = Game()
    end = 2
    player = 1

    while end == 2:
        normalized_board = game.copy_and_normalize()
        if use_best_val:
            if best_tree == 1:
                v, p = tree1.get_pred(normalized_board)
                v *= player
            else:
                v, p = tree2.get_pred(normalized_board)
                v *= player
        else:
            v = None

        # check for early termination
        if game.get_full_moves() >= max_moves:
            end = game.early_rollout(value=v)
            break
        resign = game.check_resign(value=v)
        if resign is not None:
            end = resign
            break


        tree = tree1 if game.board.turn else tree2
        move = tree.get_move(normalized_board)
        uci_move = normalized_board.coord_to_move(move)
        if player == -1:
            uci_move = encode.flip_move(uci_move)
        game.make_move(uci_move)
        player *= -1
        end = game.get_game_state()

    return end


def play_mult_games(tree1, tree2, num_games=40):
    per_round = num_games // 2
    wins_1 = 0
    wins_2 = 0
    draws = 0

    # play with tree1 going first
    for i in range(per_round):
        res = play_single(tree1, tree2)
        if res == 1:
            wins_1 += 1
        elif res == -1:
            wins_2 += 1
        else:
            draws += 1

    # play with tree2 going first
    for i in range(per_round):
        res = play_single(tree2, tree1)
        if res == 1:
            wins_2 += 1
        elif res == -1:
            wins_1 += 1
        else:
            draws += 1

    return wins_1, wins_2, draws


# returns True if new net should replace prev net
def compare_nets(new_net, prev_net, thresh=0.55, prev_pred_dict=None,
                 new_pred_dict=None):
    new_mcst = MCST(new_net, preds=new_pred_dict, early_term=True)
    prev_mcst = MCST(prev_net, preds=prev_pred_dict, early_term=True)

    new_wins, prev_wins, draws = play_mult_games(new_mcst, prev_mcst)
    print(new_wins, prev_wins, draws)

    if new_wins + prev_wins == 0:
        return False
    new_win_rate = new_wins / (new_wins + prev_wins)
    if new_win_rate > thresh:
        return True
    return False