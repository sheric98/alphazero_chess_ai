import torch
import _pickle as pickle
from network import NNet


def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


# save network
def save_net(net, path):
    # Save model
    torch.save(net.state_dict(), path)


# load network
def load_net(path):
    net = NNet()
    net.to(device)
    state_dict = torch.load(path, map_location=device)
    net.load_state_dict(state_dict)
    return net


def save_and_load(net, path):
    save_net(net, path)
    copied = load_net(path)
    return copied


def save_prev_training(prev_training, path):
    with open(path,'wb') as output:
        pickle.dump(prev_training,output, -1)


def load_prev_training(path):
    with open(path,'rb') as file:
        ret = pickle.load(file)
    return ret


SAMP_SPACE = 8*8*73


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'