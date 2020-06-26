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
    with open(path, 'wb') as output:
        pickle.dump(prev_training, output, -1)


def load_prev_training(path):
    with open(path,'rb') as file:
        ret = pickle.load(file)
    return ret


def ask_question(question):
    res = ''
    while res != 'y' and res != 'n':
        res = str(input(question + ' (y/n):')).lower().strip()
        if res[:1] == 'y':
            return True
        if res[:1] == 'n':
            return False


def ask_for_int(question):
    while True:
        res = str(input(question)).strip()
        if not represents_int(res):
            continue
        inp = int(res)
        return inp

SAMP_SPACE = 8*8*76


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'