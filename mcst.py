import numpy as np
import util

class MCST:
    def __init__(self, net, cpuct=1, num_sims=25, preds=None,
                 early_term=False, use_self_val=False, max_moves=500):
        self.net = net
        self.cpuct = cpuct
        self.num_sims = num_sims
        self.Qs = {}
        self.Ns = {}
        self.N_pairs = {}
        self.masks = {}
        self.board_vals = {}
        self.policies = {}
        self.early_term = early_term
        self.use_self_val = use_self_val
        self.max_moves = max_moves
        if preds is None:
            self.preds = {}
        else:
            self.preds = preds

    def is_expanded(self, game):
        key = game.get_key()
        return key in self.policies

    def get_pred(self, game):
        key = game.get_key()
        if key not in self.preds:
            self.preds[key] = self.net.predict(game)
        return self.preds[key]

    def expand(self, game):
        key = game.get_key()
        v, p = self.get_pred(game)
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
            end = game.get_game_state()
            # check for early rollout
            if end == 2 and self.early_term:
                if self.use_self_val:
                    v, p = self.get_pred(game)
                else:
                    v = None
                if game.get_full_moves() >= self.max_moves:
                    end = game.early_rollout(value=v)
                else:
                    resign = game.check_resign(value=v)
                    if resign is not None:
                        end = resign
            self.board_vals[key] = end
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
        sample_space = util.SAMP_SPACE
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
        sample_space = util.SAMP_SPACE

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