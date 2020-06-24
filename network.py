import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import util


class ResBlock(nn.Module):
    def __init__(self, channels=256):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        res = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += res
        x = F.relu(x)
        return x


class FinBlock(nn.Module):
    def __init__(self, channels=256):
        super(FinBlock, self).__init__()
        self.convP = nn.Conv2d(channels, 2, 1, bias=False)
        self.bnP = nn.BatchNorm2d(2)
        self.linearP = nn.Linear(8*8*2, 8*8*76)
        self.convV = nn.Conv2d(channels, 4, 1, bias=False)
        self.bnV = nn.BatchNorm2d(4)
        self.linear1V = nn.Linear(8*8*4, 8*8*channels)
        self.linear2V = nn.Linear(8*8*channels, 1)

    def forward(self, x):
        p = F.relu(self.bnP(self.convP(x)))
        p = p.view(-1, 8*8*2)
        p = F.log_softmax(self.linearP(p), dim=1)

        v = F.relu(self.bnV(self.convV(x)))
        v = v.view(-1, 8*8*4)
        v = F.relu(self.linear1V(v))
        v = torch.tanh(self.linear2V(v))

        return v, p


class NNet(nn.Module):
    def __init__(self, nres=7, step_size=0.001, channels=256):
        super(NNet, self).__init__()
        self.step_size = step_size
        self.nres = 7
        self.conv = nn.Conv2d(18, channels, 5, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        for res in range(nres):
            setattr(self, "res_%d" % res, ResBlock(channels))
        self.finBlock = FinBlock(channels)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.step_size)

    def forward(self, x):
        x = x.view(-1, 18, 8, 8)
        x = F.relu(self.bn(self.conv(x)))
        for res in range(self.nres):
            x = getattr(self, 'res_%d' % res)(x)
        v, p = self.finBlock.forward(x)
        return v, p

    def predict(self, game):
        self.eval()
        enc = game.get_encoded()
        inp = torch.from_numpy(enc).float().to(util.device)
        with torch.no_grad():
            v, p = self(inp)

        return v.data.cpu().numpy()[0][0], p.exp().data.cpu().numpy()[0]

    def calc_loss(self, v, v_targs, p, p_targs):
        v_loss = torch.sum((v.view(-1) - v_targs) ** 2) / v_targs.size()[0]
        p_loss = -torch.sum(p * p_targs) / p_targs.size()[0]
        return v_loss + p_loss

    def get_loss(self, games, v_targs, p_targs):
        v, p = self.forward(games)
        return self.calc_loss(v, v_targs, p, p_targs)

    def run_grad(self, games, v_targs, p_targs):
        # Compute loss
        loss = self.get_loss(games, v_targs, p_targs)
        # Zero out gradients
        self.optimizer.zero_grad()
        # Compute gradients
        loss.backward()
        # Update parameters based on gradients
        self.optimizer.step()

        return loss


def train_one_epoch(net, training, batch_size=256, thresh=10):
    net.train()
    device = util.device
    np.random.shuffle(training)
    N = len(training)
    tot_loss = 0
    for j in range(0, N, batch_size):
        batch = training[j:j+batch_size]
        B = len(batch)
        if B < thresh:
            N -= B
            break
        boards, v_targs, p_targs = list(zip(*batch))
        boards = np.stack(boards)
        p_targs = np.stack(p_targs)
        boards = torch.FloatTensor(boards.astype(np.float64)).to(device)
        v_targs = torch.FloatTensor(np.array(v_targs).astype(np.float64)).to(device)
        p_targs = torch.FloatTensor(p_targs).to(device)
        loss = net.run_grad(boards, v_targs, p_targs)
        tot_loss += (loss * B)
    return tot_loss / N


def train_mult_epochs(net, training, num_epochs=20):
    loss = 0
    for i in range(num_epochs):
        loss = train_one_epoch(net, training)
    print('Loss is %f' % loss)
