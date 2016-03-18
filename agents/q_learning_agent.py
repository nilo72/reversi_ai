import random
import pickle
import gzip
from agents.agent import Agent
from game.reversi import Reversi
from util import *
import pdb


class QLearningAgent(Agent):

    def __init__(self, reversi=None, color=None, **kwargs):
        self.q_vals = dmap()
        # self.unpkl()
        # these defaults prohibit exploring,
        # are only enabled during learning
        episodes = kwargs.get('episodes', 100)
        self.learn(episodes)

        self.alpha = 0.0
        self.discount = 1
        self.explore = 0.0
        self.reversi = reversi
        self.color = color

    def get_q_val(self, state, action):
        return self.q_vals[(state, action)]

    def get_policy(self, state, legal_actions):
        """The policy decides the best action to take
        in each state (or None if no actions exist)"""
        if not legal_actions:
            return None

        # if we are black, we want maximum q-value.
        # if we are white, we want minimum q-value
        best_actions = []
        if self.color == BLACK:
            best_q_val = -float('inf')
            for action in legal_actions:
                q_val = self.get_q_val(state, action)
                if q_val > best_q_val:
                    best_actions = [action]
                    best_q_val = q_val
                elif q_val == best_q_val:
                    best_actions.append(action)
        elif self.color == WHITE:
            best_q_val = float('inf')
            for action in legal_actions:
                q_val = self.get_q_val(state, action)
                if q_val < best_q_val:
                    best_actions = [action]
                    best_q_val = q_val
                elif q_val == best_q_val:
                    best_actions.append(action)

        return random.choice(best_actions)

    def get_action(self, state, legal_actions):
        """Return the best action to take in the given state."""
        if not legal_actions:
            return None
        if self.coin_flip(self.explore):
            return random.choice(legal_actions)
        else:
            best_action = self.get_policy(state, legal_actions)
            return best_action

    @staticmethod
    def coin_flip(p):
        """True with prop p, else false"""
        val = random.random()
        return val < p

    def update(self, state, action, next_state, reward):
        """Use the state/action/state/reward information to
        update our q-value estimation for this state.
        
        See here: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.109.4577&rep=rep1&type=pdf"""
        legal_actions = self.reversi.get_legal_moves(next_state)
        value = reward
        if legal_actions:
            max_q = max(self.get_q_val(next_state, action)
                        for action in legal_actions)
            value += (self.discount * max_q)
        self.q_vals[(state, action)] = (1 - self.alpha) * \
            self.get_q_val(state, action) + (self.alpha * value)

    def pkl(self):
        f = gzip.open('qlearned.pkl', 'wb')
        pickle.dump(self.q_vals, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    def unpkl(self):
        try:
            # f = gzip.open('qlearned.pkl', 'rb')
            print('opening file...')
            f = open('qlearned.pklr', 'rb')
            print('file opened. unloading...')
            self.q_vals = pickle.load(f)
            f.close()
            print('unloading complete.')
        except FileNotFoundError:
            self.q_vals = dmap()
            print("couldn't open qlearn database! using empty qvalue map.")

    def learn(self, num_games):
        dimensions = (8, 8)
        self.alpha = 0.1
        self.discount = 1.0
        self.color = BLACK
        # probability to take a random action
        self.explore = 0.05
        for t in range(1, num_games + 1):
            print('starting episode {}'.format(t))
            self.reversi = Reversi(dimensions)
            reversi = self.reversi
            state = reversi.get_state()
            self.q_vals[state] = 0

            # run a simulation "episode".  A win for black gives reward 1,
            # a win for white gives reward -1, all else is 0.
            while True:
                legal_actions = self.reversi.get_legal_moves(state)
                best_action = self.get_action(state, legal_actions)
                if best_action is None:
                    break
                next_state = reversi.next_state(state, *best_action)
                reward = 0
                winner = reversi.winner(next_state)
                if winner == BLACK:
                    reward = 1
                elif winner == WHITE:
                    reward = -1
                self.update(state, best_action, next_state, reward)

                state = next_state

                if winner is not False:
                    break
            print('episode {} complete'.format(t))
        # self.pkl()


class dmap(dict):
    """This class acts as a map with a default value."""

    def __getitem__(self, arg):
        self.setdefault(arg, 0)
        return dict.__getitem__(self, arg)
