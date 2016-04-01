import random
import os.path
from copy import deepcopy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.models import model_from_json
from keras.optimizers import RMSprop

from agents.agent import Agent
from agents.random_agent import RandomAgent

from util import *
from numpy import array

# change these to change where models/weights are stored/loaded
MODEL_FILENAME = '8x8_duel_network/q_model.json'
WEIGHTS_FILENAME = '8x8_duel_network/q_weights'

# after this many epochs, save to a new weight file
# so that if overfitting occurs and the model starts to get
# worse, we have saved versions of earlier weights.
WEIGHT_GEN_LENGTH = 10000

# Amount of nodes in the hidden layer
HIDDEN_SIZE = 44

WIN_REWARD = 1
LOSE_REWARD = -1

class QLearningAgent(Agent):

    def __init__(self, reversi, color, **kwargs):
        self.reversi = reversi
        self.color = color
        self.board_size = self.reversi.board.get_size()
        self.kwargs = kwargs

        self.learning_mode = self.kwargs.get('learning_mode', False)

        # initialize the model
        self.model = self.get_model(self.kwargs.get('model_file', False), self.board_size)
        self.model.compile(loss='mse', optimizer=RMSprop())

        weights_file = self.kwargs.get('weights_file', False)
        if weights_file is not False:
            weights_file += '_' + color_name[self.color]
            self.load_weights(weights_file, self.model)
        else:
            info('no weights file given; using default weights.')

        if self.learning_mode:
            info('learning mode enabled.')
            self.prev_state = None
            self.prev_q = None
            self.prev_move = None

            self.alpha = 0.8
        else:
            info('learning mode NOT enabled.')




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

    def observe_win(self, state, winner):
        """Called by Reversi when the game is over.
           The state is the final board state of the game."""

        # call get_action to train our agent on the winning move
        self.get_action(state)


    def get_action(self, game_state):
        legal = self.reversi.legal_moves(game_state)

        if self.learning_mode:
            # learning mode enabled, so reinforce

            if not self.prev_state:
                # on the first move, just set this up for future
                self.prev_state = state

                q_vals = self.model.predict(self.numpify(game_state), batch_size = 1)[0]
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

                # if there are no moves but the game isn't over yet, don't reinforce yet
                if not legal and winner is False:
                    return None # pass turn

                best_move, new_qs = self.update_model(self.model, self.prev_state, \
                        self.prev_move, self.prev_qs, game_state, reward, winner)
                self.prev_move = best_move
                self.prev_qs = new_qs
                self.prev_state = game_state

                return best_move
        else:
            # just follow the policy
            if not legal:
                return None
            else:
                return self.policy(game_state, legal)

    @staticmethod
    def update_model(self, model, prev_state, prev_action, q_vals, new_moves, reward, winner):
        """Given the necessary parameters, reinforce the model. For convenience, returns the
            best move its q_val that it found while reinforcing. Do NOT call this method
            in a game state where you have no moves, UNLESS it is a winning game state."""
        # Q(s,a) = (1-alpha) * Q(s,a) + alpha * (r + maxQ(s', a'))
        move_offset = self.to_offset(prev_action)
        max_q = None
        next_q_vals = None
        best_move = None
        if winner is False:
            next_q_vals = model.predict(self.numpify(new_state), batch_size = 1)
            best_move, max_q = self.best_move_val(new_moves, next_q_vals)
        else:
            # we have a winner! Don't need max_q, it's all in the reward
            max_q = 0

        q_vals[move_offset] = (1 - self.alpha) * q_vals[move_offset] + \
                self.alpha * (reward + max_q)
        model.fit(self.numpify(prev_state), q_vals, batch_size = 1, nb_epoch=1)

        if winner is False:
            # for convenience, return best move and q_vals
            return best_move, next_q_vals
        else:
            # the game is won, no values to return
            return None, None


    def policy(self, game_state, legal):
        if not legal:
            return None

        # ask neural network for best valued action
        q_vals = self.model.predict(self.numpify(game_state), batch_size = 1)
        best_move, best_val = self.best_move_val(legal, q_vals)
        return best_move


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

            best_q = None
            best_move = None

            for move in moves:
                offset = QLearningAgent.to_offset(move[0], move[1], self.board_size)
                val = q_vals[0][offset]
                info('{}: {}'.format(move, val))
                if best_q is None or val > best_q:
                    best_q = val
                    best_move = move

            return best_move, best_q


    @staticmethod
    def to_offset(x, y, width):
        """Given x,y coords, convert to an offset
        of an equal flattened array."""
        offset = 0
        offset += (y * width)
        offset += x
        return offset

    @staticmethod
    def get_model(model_file, board_size):
        """Load the model from disk, or create a new one
        if none is found on disk."""
        if model_file is not False:
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
            model.add(Activation('tanh'))
            # model.add(Dropout(0.2))

            # model.add(Dense(256, init='zero'))
            # model.add(Activation('relu'))

            model.add(Dense(size, init='zero'))
            model.add(Activation('tanh')) # tanh or linear

        return model

    @staticmethod
    def save_model(model):
        """Save the model layout to disk."""
        as_json = model.to_json()
        with open(MODEL_FILENAME, 'w') as f:
            f.write(as_json)
            info('model saved to {}'.format(MODEL_FILENAME))

