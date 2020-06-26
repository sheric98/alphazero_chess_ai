import training as tr
import util
from network import NNet
import play_against_ai as play
import supervised as sl



if __name__ == '__main__':
    path = 'models/chess_ai'
    tr_path = 'prev_training/chess_training'
    sl_path = 'prev_training/sl_training'

    sl_names = ['ficsgamesdb_202004_standard2000_nomovetimes_139538.pgn',
                'ficsgamesdb_202001_standard2000_nomovetimes_139652.pgn',
                'ficsgamesdb_202002_standard2000_nomovetimes_139651.pgn',
                'ficsgamesdb_202003_standard2000_nomovetimes_139650.pgn']

    print(util.device)

    res = util.ask_question('Supervised Learning?')
    if res:
        next_res = util.ask_question('Use previous network?')
        if next_res:
            nnet = util.load_net(path)
        else:
            nnet = NNet()
            nnet.to(util.device)

        sl_train = sl.get_pgn_training(sl_names, sl_path)
        print('Finished obtaining training examples.')
        sl.sl_train_net(nnet, sl_train)
        util.save_net(nnet, path)
        exit(0)

    res = util.ask_question('Reinforcement Learning?')
    if res:
        next_res = util.ask_question('Use previous network?')
        if next_res:
            nnet = None
            start_iter = util.ask_for_int('Start at what iteration?')
        else:
            start_iter = 0
            nnet = NNet()
            nnet.to(util.device)
        tr.train_net(nnet, path, tr_path, start=start_iter)
        exit(0)

    res = util.ask_question('Play against net?')
    if res:
        nnet = util.load_net(path)
        next_res = util.ask_question('Would you like to go first?')
        player = 1 if next_res else -1
        play.play_net(nnet, player=player)
        exit(0)
