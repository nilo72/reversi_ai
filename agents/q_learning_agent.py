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
            print('best val: {}'.format(best_q_val))
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
        if reward != 0:
            pass
        legal_actions = self.reversi.get_legal_moves(next_state)
        value = reward
        if legal_actions:
            max_q = max(self.get_q_val(next_state, next_action)
                        for next_action in legal_actions)
            value += (self.discount * max_q)

        old_q_val = self.get_q_val(state, action)
        new_q_val = (1 - self.alpha) * old_q_val + (self.alpha * value)
        self.q_vals[(state, action)] = new_q_val

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
        dimensions = (4, 4)
        self.alpha = 0.1
        self.discount = 1.0
        self.color = BLACK
        # probability to take a random action
        self.explore = 0.05
        for t in range(1, num_games + 1):
            print('starting episode {}/{}'.format(t, num_games))
            self.reversi = Reversi(dimensions)
            reversi = self.reversi
            state = reversi.get_state()
            self.q_vals[state] = 0

            # run a simulation "episode".  A win for black gives reward 1,
            # a win for white gives reward -1, all else is 0.
            while not reversi.winner(state):
                state = self.black_move(reversi, state)
                if not state:
                    break
            print('episode {} complete'.format(t))

    @staticmethod
    def winner_reward(winner):
        if winner == BLACK:
            return 10
        elif winner == WHITE:
            return -10
        return 0

    def white_move(self, reversi, state):
        """Makes a move for white, and returns the resulting
        state.  If there are no actions it can take, it returns the
        initial state, but passes its turn to black."""
        assert state[1] == WHITE
        legal_actions = reversi.get_legal_moves(state)
        if not legal_actions:
            # turn passes to black
            state = (state[0], BLACK)
            return state
        picked = random.choice(legal_actions)
        return reversi.next_state(state, *picked)

    def black_move(self, reversi, state):
        """Make black's move, then make white's move,
        then returns the resulting state."""
        assert state[1] == BLACK

        # Make Black's move.
        legal_actions = reversi.get_legal_moves(state)
        if not legal_actions:
            # turn passes to white if we have no moves
            state = (state[0], WHITE)
            return self.white_move(reversi, state)
        best_action = self.get_action(state, legal_actions)
        next_state = reversi.next_state(state, *best_action)
        winner = reversi.winner(next_state)
        reward = self.winner_reward(winner)
        if winner:
            self.update(state, best_action, next_state, reward)
            return None
        opponent_move = self.white_move(reversi, next_state)
        winner = reversi.winner(opponent_move)
        reward = self.winner_reward(winner)
        self.update(state, best_action, opponent_move, reward)
        if winner:
            return None

        return opponent_move
        # self.pkl()


class dmap(dict):
    """This class acts as a map with a default value."""

    def __getitem__(self, arg):
        self.setdefault(arg, 0)
        return dict.__getitem__(self, arg)
