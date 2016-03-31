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

MODEL_FILENAME = 'q_model.json'
WEIGHTS_FILENAME = 'q_weights'

# after this many epochs, save to a new weight file
# so that if overfitting occurs and the model starts to get
# worse, we have saved versions of earlier weights.
WEIGHT_GEN_LENGTH = 10000 

class QLearningAgent(Agent):

    def __init__(self, reversi, color, **kwargs):
        self.reversi = reversi
        self.color = color
        self.board_size = self.reversi.board.get_size()
        self.kwargs = kwargs

        # initialize the model
        self.model = self.get_model()
        self.model.compile(loss='mse', optimizer=RMSprop())

        weights_file = self.kwargs.get('weights_file', False)
        info('looking for weights_file {}'.format(weights_file))
        if weights_file:
            if os.path.exists(weights_file):
                info('loading existing weights file {}'.format(weights_file))
                self.model.load_weights(weights_file)
            else:
                print("couldn't find weights file {}. quitting.".format(weights_file))
                quit()
        else:
            info('Not loading a weights file because it was not found or chosen.')

    def get_action(self, game_state):
        return self.policy(game_state)

    def policy(self, game_state):
        legal = self.reversi.legal_moves(game_state)
        if not legal:
            return None

        # ask neural network for best valued action
        q_vals = self.model.predict(self.numpify(game_state), batch_size = 1)
        best_move, best_val = self.best_move_val(legal, q_vals)
        return best_move

    def update_model(self, model, q_vals, state, action, new_state, reward):
        """
        model: the model
        q_vals: the prediction given initially for all actions in state
        state: the earlier state
        action: the action taken from state
        new_state: the resulting state from taking action in state
        reward: the reward for transitioning to new_state
        """
        # Q(s,a) = (1-alpha) * Q(s,a) + alpha * (r + Q(s', a'))
        # find the max next Q of the new state (max Q(s', a'))
        if q_vals is None:
            q_vals = model.predict(self.numpify(state), batch_size = 1)

        legal = self.reversi.legal_moves(new_state)
        next_qs = model.predict(self.numpify(new_state), batch_size = 1)
        best_move, best_q = self.best_move_val(legal, next_qs)

        if not legal:
            # if no possible moves, best_q is just 0
            best_q = 0

        move_offset = self.to_offset(action[0], action[1], self.board_size)
        q_vals[0][move_offset] = (1 - self.alpha) * q_vals[0][move_offset] + \
                self.alpha * (reward + best_q)

        model.fit(self.numpify(state), q_vals, batch_size=1, nb_epoch=1,verbose=0)

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
            offset = self.to_offset(move[0], move[1], self.board_size)
            val = q_vals[0][offset]
            info('{}: {}'.format(move, val))
            if best_q is None or val > best_q:
                best_q = val
                best_move = move

        return best_move, best_q


    def train(self, epochs):
        # new one from non-working version
        model = self.model
        self.board_size = self.reversi.board.get_size()

        self.alpha = 0.8
        self.epsilon = 0.1

        wins = []
        WIN_REWARD = 1
        LOSE_REWARD = -1

        for i in range(1, epochs + 1):
            print('starting epoch {} ({:5.2f}%)'.format(i, (i / epochs) * 100))
            self.reversi.reset()
            state = deepcopy(self.reversi.get_state())

            prev_action = None
            old_qs = None
            prev_state = None

            # in this epoch, play a game and refit as we go
            white_agent = RandomAgent(self.reversi, WHITE)

            while True:
                # white and black take turns until game is over
                # we break out when game is over
                if state[1] == WHITE:
                    # white's turn
                    legal = self.reversi.legal_moves(state)
                    move = None
                    if legal:
                        move = random.choice(legal)
                    state = self.reversi.next_state(state, move)
                    assert(state[1] == BLACK)

                elif state[1] == BLACK:
                    # it's black's turn

                    # first let's see where our last move landed us after white's play
                    if prev_state is not None:
                        # train model on where we ended up
                        reward = 0
                        winner = self.reversi.winner(state)
                        if winner == BLACK:
                            reward = WIN_REWARD
                        elif winner == WHITE:
                            reward = LOSE_REWARD
                        elif winner is not False:
                            raise ValueError

                        self.update_model(model, old_qs, prev_state, prev_action, state, reward)

                        if winner is not False:
                            wins.append(winner)
                            break # the game is over, move on to next epoch

                    # now make our move based on highest legal q_val
                    q_vals = model.predict(self.numpify(state), batch_size = 1)
                    legal = self.reversi.legal_moves(state)

                    move = None
                    if random.random() < self.epsilon:
                        # epsilon greedy, pick random move
                        if legal:
                            move = random.choice(legal)
                    else:
                        move, val = self.best_move_val(legal, q_vals)

                    # make our move, saving old values for next network refit
                    if legal:
                        # if we had a move and took it
                        prev_state = state
                        old_qs = q_vals
                        prev_action = move
                    state = self.reversi.next_state(state, move)
                    assert(state[1] == WHITE)

            if i % WEIGHT_GEN_LENGTH == 0:
                info('saving model at epoch {}'.format(i))
                model.save_weights(WEIGHTS_FILENAME + '_' + str(i / WEIGHT_GEN_LENGTH))

        info('training complete.')
        info('summary:')
        black_wins = len([win for win in wins if win == BLACK])
        white_wins = len([win for win in wins if win == WHITE])
        info(' black: {} ({})'.format(black_wins, black_wins / (black_wins + white_wins)))
        info(' white: {} ({})'.format(white_wins, white_wins / (black_wins + white_wins)))
        model.save_weights(WEIGHTS_FILENAME, overwrite=True)
        self.save_model(model)

    @staticmethod
    def to_offset(x, y, width):
        """Given x,y coords, convert to an offset
        of an equal flattened array."""
        offset = 0
        offset += (y * width)
        offset += x
        return offset

    def get_model(self):
        """Load the model from disk, or create a new one
        if none is found on disk."""
        model = None
        model_file = self.kwargs.get('model_file', False)
        if model_file is not False:
            try:
                print('trying to open file {}'.format(model_file))
                model = model_from_json(open(model_file).read())
            except FileNotFoundError:
                print("couldn't find model file {}. quitting.".format(self.kwargs.get('model_file')))
                quit()
            info('loading existing model file {}'.format(MODEL_FILENAME))
        else:
            info('generating new model')
            size = self.reversi.board.get_size() ** 2
            model = Sequential()
            model.add(Dense(42, init='lecun_uniform', input_shape=(size,)))
            model.add(Activation('relu'))
            # model.add(Dropout(0.2))

            # model.add(Dense(256, init='zero'))
            # model.add(Activation('relu'))

            model.add(Dense(size, init='lecun_uniform'))
            model.add(Activation('linear'))

        return model

    @staticmethod
    def save_model(model):
        """Save the model layout to disk."""
        as_json = model.to_json()
        with open(MODEL_FILENAME, 'w') as f:
            f.write(as_json)
            info('model saved to {}'.format(MODEL_FILENAME))

