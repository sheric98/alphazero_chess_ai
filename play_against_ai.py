from mcst import MCST
from game import Game
import encode


def get_player_move(game):
    while True:
        uci_move = str(input('input uci form of move:\n')).lower().strip()
        if uci_move in game.get_valid_moves():
            return uci_move


def play_net(net, depth=50, player=1):
    mcst = MCST(net, num_sims=depth)
    game = Game()

    while game.get_game_state() == 2:
        player_to_move = 1 if game.board.turn else -1
        if player_to_move == player:
            print(game.board)
            move = get_player_move(game)
        else:
            normalized_board = game.copy_and_normalize()
            coord = mcst.get_move(normalized_board)
            move = normalized_board.coord_to_move(coord)
            if player == 1:  # this means the computer is -1
                move = encode.flip_move(move)
            print('CPU makes move ' + move)
        game.make_move(move)

    print(game.board)
    end = game.get_game_state()
    if end == player:
        print('You win!')
    elif end == -player:
        print('You lose!')
    else:
        print('Draw!')
