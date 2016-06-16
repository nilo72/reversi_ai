import numpy as np
import random
from util import to_offset, numpify, max_q_move, double_expand
import math

# MAX_MEM_LEN = 1000  # no matter what, do not allow memory past this amount


class ExperienceReplay:
    def __init__(self, size=100000):
        self.memory = []
        self.MEM_LEN = size

    def remember(self, experience):
        self.memory.append(experience)
        if len(self.memory) > self.MEM_LEN:
            self.memory.pop(0)

    def set_replay_len(self, val):
        if val == -1:
            self.MEM_LEN = float('inf')
        else:
            assert val > 0
            self.MEM_LEN = val

    def get_replay(self, model, minibatch_size):
        # (s, a, r, sp, legal, win) tuple
        s, a, r, sp, l, win = range(6)  # indices

        if minibatch_size > len(self.memory):
            minibatch_size = len(self.memory)
        replays = random.sample(self.memory, minibatch_size)

        # now format for training
        board_size = model.input_shape[-1]
        inputs = np.empty((minibatch_size, 1, board_size, board_size))
        targets = np.empty((minibatch_size, board_size**2))
        for index, replay in enumerate(replays):
            experience = replay.get_all()
            if experience[win] is False and not experience[l]:
                continue  # no legal moves, and not a win
            move = int(to_offset(experience[a], board_size))
            state = numpify(experience[s])

            state_prime = double_expand(numpify(experience[sp]))
            prev_qvals = model.predict(state_prime)

            q_prime = None
            if experience[win] is False:
                next_qvals = model.predict(state_prime)
                _, best_q = max_q_move(next_qvals, experience[l], board_size)
                # q_prime = (1 - ALPHA) * \
                #  prev_qvals[0][move] + ALPHA * (experience[r] + best_q)
                q_prime = experience[r] + best_q
            else:
                # q_prime = (1 - ALPHA) * prev_qvals[0][move] + ALPHA * experience[r]
                q_prime = experience[r]
                assert q_prime != 0  # reward for win should be nonzero
            prev_qvals[0][move] = q_prime
            inputs[index] = state
            targets[index] = prev_qvals  # was updated with new target

        return inputs, targets

    def __len__(self):
        return len(self.memory)


class Experience:
    def __init__(self, s, a, sp, r, l, win):
        self.s = s
        self.a = a
        self.sp = sp
        self.r = r
        self.l = l
        self.win = win

    def get_all(self):
        """Return the (s, a, r, sp, l, win) tuple of this experience."""
        return self.s, self.a, self.r, self.sp, self.l, self.win
