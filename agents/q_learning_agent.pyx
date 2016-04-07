import pdb
import random
import os.path
from collections import deque
from copy import deepcopy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.models import model_from_json
from keras.optimizers import RMSprop
from keras.optimizers import SGD

from agents.agent import Agent
from agents.random_agent import RandomAgent

from util import *
from numpy import array

# change these to change where models/weights are stored/loaded
MODEL_FILENAME = 'net_weights/q_model.json'
WEIGHTS_FILENAME = 'net_weights/q_weights'

# Amount of nodes in the hidden layer
HIDDEN_SIZE = 44
# weight for neural network refitting -- NOT the alpha value
LEARNING_RATE = 0.1
# according to the research paper, momentum of 0 is effective
MOMENTUM = 0.0

WIN_REWARD = 1
LOSE_REWARD = -1

ALPHA = 0.01

# pick the optimizer
# optimizer = SGD(lr=LEARNING_RATE, momentum=MOMENTUM)
optimizer = RMSprop()  # don't change RMSprop values, they should be defaults

class QLearningAgent(Agent):
    def __init__(self, reversi, color, **kwargs):
        self.reversi = reversi
        self.color = color
        self.board_size = self.reversi.board.get_size()
        self.kwargs = kwargs

        self.learning_mode = self.kwargs.get('learning_mode', False)

        # initialize the model
        self.model = self.get_model(self.kwargs.get('model_file', False), self.board_size)
        self.MAX_REPLAY = self.kwargs.get('max_replay', 64)
        self.replay_buffer = []

        if not self.kwargs.get('weights_file', False):
            print('weights_file parameter needed for q_learning_agent. quitting.')
            quit()
        else:
            self.weights_file = self.kwargs['weights_file'] + '_' + color_name[self.color]
            if self.kwargs.get('weights_num', False):
                self.weights_file += '_' + str(float(self.kwargs['weights_num']))
            if not os.path.exists(self.weights_file):
                print("couldn't find file {}, will create it when necessary.".format(self.weights_file))
            else:
                print('loading weights file {}'.format(self.weights_file))
                self.load_weights(self.weights_file, self.model)

        self.prev_move = None
        self.prev_state = None
        self.prev_qs = None
        self.alpha = ALPHA
        if self.learning_mode:
            self.reset_learning()  # redundant but useful
            self.epsilon = 1  # decrease it to 0 gradually
        else:
            info('learning mode NOT enabled.')

        self.save_model(self.model)

    def save_weights(self, suffix):
        save_loc = WEIGHTS_FILENAME + '_' + color_name[self.color] + suffix
        info('saving weights to {}'.format(save_loc))
        self.model.save_weights(save_loc, overwrite=True)

    def reset(self):
        """Called when a Reversi game resets itself, giving the agent
            the chance to prepare itself for the next game."""
        self.reset_learning()

    def reset_learning(self):
        """Reset this agent to its beginning state for learning."""
        info('reset learning called [{}]'.format(color_name[self.color]))
        self.replay_buffer = deque([])
        self.prev_state = None
        self.prev_qs = None
        self.prev_move = None
        self.alpha = ALPHA  # should not change

    def set_replay_len(self, length):
        self.MAX_REPLAY = min(length, 64)

    @staticmethod
    def load_weights(weights_file, model):
        """Given the name of a weights file and the model,
            populate the model with the weights."""
        info('looking for weights_file {}'.format(weights_file))
        if os.path.exists(weights_file):
            model.load_weights(weights_file)
            info('loaded existing weights file {}'.format(weights_file))
        else:
            print("couldn't find weights file {}. quitting.".format(weights_file))
            quit()

    def observe_win(self, state):
        """Called by Reversi when the game is over.
           The state is the final board state of the game."""

        # call get_action to train our agent on the winning move
        self.train_network(state, [])

    def get_action(self, game_state, legal):
        # pdb.set_trace()
        assert game_state[1] == self.color
        if self.learning_mode:
            # learning mode enabled, so reinforce
            return self.train_network(game_state, legal)
        else:
            # just follow the policy
            if not legal:
                return None
            else:
                best_move, _ = self.policy(game_state, legal)
                return best_move

    def train_network(self, game_state, legal):
        if not self.prev_state:
            # on the first move, just set this up for future
            self.prev_state = game_state

            q_vals = self.model.predict(self.numpify(game_state), batch_size=1)
            self.prev_qs = q_vals

            move, _ = self.best_move_val(legal, q_vals)
            self.prev_move = move
            return move
        else:
            # reinforce the previous move with our new info
            reward = 0
            winner = self.reversi.winner(game_state)
            if winner == self.color:
                reward = WIN_REWARD
            elif winner == opponent[self.color]:
                reward = LOSE_REWARD
            elif winner is not False:
                raise ValueError

            # if there are no moves but the game isn't over yet, don't reinforce yet
            if not legal and winner is False:
                return None  # pass turn
            else:
                # store transition in replay buffer
                self.replay_buffer.append((deepcopy(self.prev_state), deepcopy(self.prev_move), deepcopy(game_state), reward, deepcopy(self.prev_qs), deepcopy(legal), winner)) # TODO: does this make deep copy? we hope so
                if len(self.replay_buffer) > self.MAX_REPLAY:
                    self.replay_buffer.popleft()

                # decide our best move
                next_qs = self.model.predict(self.numpify(game_state), batch_size=1)
                best_move, best_q = self.best_move_val(legal, next_qs)

                # train on random transition from the buffer
                s, a, sprime, r, pq, a_primes, wn = random.choice(self.replay_buffer)
                self.update_model(self.model, s, a, pq, sprime, a_primes, r, wn)

                # epsilon greedy exploration
                if self.epsilon > random.random():
                    if legal:
                        best_move = random.choice(legal)
                    else:
                        best_move = None

                self.prev_move = best_move
                self.prev_qs = next_qs
                self.prev_state = game_state

                return best_move

    def set_epsilon(self, val):
        self.epsilon = val

    def policy(self, state, legal):
        if not legal:
            return None, None
        else:
            q_vals = self.model.predict(self.numpify(state), batch_size=1)
            best_move, best_q = self.best_move_val(legal, q_vals)
            return best_move, best_q

    def update_model(self, model, prev_state, prev_action, prev_qs, new_state, new_moves, reward, winner):
        # Q(s,a) = (1-alpha) * Q(s,a) + alpha * (r + maxQ(s', a'))
        prev_qs = prev_qs
        offset = self.to_offset(prev_action)
        if winner is False:
            if not new_moves:
                return None, None
            # update previous estimate based on our guess for future
            next_qs = model.predict(self.numpify(new_state), batch_size=1)
            best_move, max_q = self.best_move_val(new_moves, next_qs)

            info('[{}] updating {} from {} to'.format(color_name[self.color], prev_action, prev_qs[0][offset]))
            prev_qs[0][offset] = (1 - self.alpha) * prev_qs[0][offset] + \
                    self.alpha * (reward + max_q)
            info('{}'.format(prev_qs[0][offset]))

            model.fit(self.numpify(prev_state), prev_qs, batch_size=1, nb_epoch=1, verbose=0)

            return best_move, next_qs
        else:
            # update previous estimate based on entirely on how we won or lost
            info('[{}] updating {} from {} to'.format(color_name[self.color], prev_action, prev_qs[0][offset]))
            prev_qs[0][offset] = (1 - self.alpha) * prev_qs[0][offset] + \
                    self.alpha * reward # no max_q because no future moves, game over
            info(prev_qs[0][offset])
            model.fit(self.numpify(prev_state), prev_qs, batch_size=1, nb_epoch=1, verbose=0)
            return None, None # no future moves, nothing to return

    @staticmethod
    def numpify(state):
        """Given a state (board, color) tuple, return the numpy
        version of the board's array."""
        board = state[0].board
        assert len(board) > 0
        size = len(board) * len(board[0])
        return array(board).reshape(1, size)

    def best_move_val(self, moves, q_vals):
        """Given a list of moves and a q_val array,
        return the move with the highest q_val and the q_val."""
        if not moves:
            # no moves available
                return None, None
        else:
            best_q = None
            best_move = None

            for move in moves:
                offset = self.to_offset(move)
                val = q_vals[0][offset]
                info('{}: {}'.format(move, val))
                if best_q is None or val > best_q:
                    best_q = val
                    best_move = [move]
                elif best_q == val:
                    best_move.append(move)

            return random.choice(best_move), best_q

    def to_offset(self, move):
        """Given x,y coords, convert to an offset
        of an equal flattened array."""
        x, y = move
        return (y * self.board_size) + x

    @staticmethod
    def get_model(model_file, board_size):
        """Load the model from disk, or create a new one
        if none is found on disk."""
        model = None
        if model_file:
            try:
                print('opening file {}'.format(model_file))
                model = model_from_json(open(model_file).read())
            except FileNotFoundError:
                print("couldn't find model file {}. quitting.".format(model_file))
                quit()
            print('loading existing model file {}'.format(MODEL_FILENAME))
        else:
            info('generating new model')
            size = board_size ** 2
            model = Sequential()
            model.add(Dense(HIDDEN_SIZE, init='zero', input_shape=(size,)))
            model.add(Activation('relu'))
            # model.add(Dropout(0.2))

            # model.add(Dense(256, init='zero'))
            # model.add(Activation('relu'))

            model.add(Dense(size, init='zero'))
            model.add(Activation('linear'))  # tanh or linear

        model.compile(loss='mse', optimizer=optimizer)
        return model

    @staticmethod
    def save_model(model):
        """Save the model layout to disk."""
        as_json = model.to_json()
        with open(MODEL_FILENAME, 'w') as f:
            f.write(as_json)
            info('model saved to {}'.format(MODEL_FILENAME))
