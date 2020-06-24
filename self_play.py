from game import Game
from mcst import MCST


# tree 1 goes first
def play_single(tree1, tree2):
    game = Game()
    end = 2

    while end == 2:
        normalized_board = game.copy_and_normalize()
        tree = tree1 if game.board.turn else tree2
        move = tree.get_move(normalized_board)
        uci_move = game.coord_to_move(move)
        game.make_move(uci_move)
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
def compare_nets(new_net, prev_net, thresh=0.55):
    new_mcst = MCST(new_net)
    prev_mcst = MCST(prev_net)

    new_wins, prev_wins, draws = play_mult_games(new_mcst, prev_mcst)
    print(new_wins, prev_wins, draws)

    if new_wins + prev_wins == 0:
        return False
    new_win_rate = new_wins / (new_wins + prev_wins)
    if new_win_rate > thresh:
        return True
    return False