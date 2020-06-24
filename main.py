import training as tr
import util
from network import NNet
import chess



if __name__ == '__main__':
    path = 'models/chess_ai'
    tr_path = 'prev_training/chess_training'

    print(util.device)

    nnet = NNet()
    nnet.to(util.device)

    tr.train_net(nnet, path, tr_path)