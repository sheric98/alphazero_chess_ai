import numpy as np


class MCST:
    def __init__(self, net, cpuct=1, num_sims=25):
        self.net = net
        self.cpuct = cpuct
        self.num_sims = num_sims
        self.Qs = {}
        self.Ns = {}
        self.N_pairs = {}
        self.masks = {}
        self.board_vals = {}
        self.policies = {}

    def is_expanded(self, game):
        key = game.get_key()
        return key in self.policies

    def expand(self, game):
        key = game.get_key()
        v, p = self.net.predict(game)
        self.masks[key] = game.get_mask()
        p *= self.masks[key]
        div = np.sum(p)
        if div == 0:
            print('Mask is all 0 in Expand')
            print('Backup to uniform distribution')
            p = self.masks[key]
            div = np.sum(self.masks[key])
        p /= div
        self.policies[key] = p
        self.Ns[key] = 0
        return v

    def get_value(self, game):
        key = game.get_key()
        if key not in self.board_vals:
            self.board_vals[key] = game.get_game_state()
        return self.board_vals[key]

    def calc_U(self, game, move):
        key = game.get_key()
        pair = (key, move)
        Q = 0
        N_pair = 0
        prob = self.policies[key][move]
        mask = self.masks[key]
        if mask[move] == 0:
            return -1
        N = self.Ns[key] if self.Ns[key] != 0 else 1e-8  # small default value
        if pair in self.Qs:
            Q = self.Qs[pair]
            N_pair = self.N_pairs[pair]
        return Q + self.cpuct * prob * np.sqrt(N) / (1 + N_pair)

    def get_best_move(self, game):
        scores = []
        key = game.get_key()
        sample_space = len(self.policies[key])
        for move in range(sample_space):
            scores.append(self.calc_U(game, move))
        best = np.argmax(np.array(scores))
        return best

    def update_pair(self, game, move, v):
        key = game.get_key()
        pair = (key, move)
        Q = 0
        N_pair = 0
        if pair in self.Qs:
            Q = self.Qs[pair]
            N_pair = self.N_pairs[pair]
        self.Qs[pair] = (Q * N_pair + v) / (1 + N_pair)
        self.N_pairs[pair] = N_pair + 1

    def search(self, game):
        end_val = self.get_value(game)
        if end_val != 2:
            return -end_val

        if not self.is_expanded(game):
            return -self.expand(game)

        best_move = self.get_best_move(game)
        uci_move = game.coord_to_move(best_move)
        next_game = game.move_and_normalize(uci_move)
        v = self.search(next_game)
        self.update_pair(game, best_move, v)

        key = game.get_key()
        self.Ns[key] += 1

        return -v

    def get_probs(self, game, temp=1):
        key = game.get_key()
        sample_space = len(self.policies[key])

        for i in range(self.num_sims):
            self.search(game)

        counts = []
        for move in range(sample_space):
            pair = (key, move)
            count = 0
            if pair in self.N_pairs:
                count = self.N_pairs[pair]
            counts.append(count)
        counts = np.asarray(counts).astype(float)

        if temp == 0:
            best_moves = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best = np.random.choice(best_moves)
            probs = np.zeros(sample_space)
            probs[best] = 1
            return probs

        div = np.sum(counts)
        if div == 0:
            print('All Moves 0 in Get Prob')

        probs = counts / div
        return probs

    def get_move(self, game):
        return np.argmax(self.get_probs(game, temp=0))