import random
from copy import deepcopy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.models import model_from_json
from keras.optimizers import RMSprop

from agents.agent import Agent
from agents.random_agent import RandomAgent

from util import *
from numpy import array
import pdb

MODEL_FILENAME = 'q-model.json'
WEIGHTS_FILENAME = 'q_weights.h5'

class QLearningAgent(Agent):

    def __init__(self, reversi, color):
        self.reversi = reversi
        self.color = color

    def get_action(self, game_state, legal_moves):
        if not legal_moves:
            return None

        return self.policy(game_state, legal_moves)

    def policy(self, game_state, legal_moves):
        # run through neural network, return best move
        pass

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

        best_q = None
        if legal:
            next_qs = model.predict(self.numpify(new_state), batch_size = 1)
            for each in legal:
                offset = self.to_offset(each[0], each[1], self.board_size)
                val = next_qs[0][offset]
                if best_q is None:
                    best_q = val
                elif val > best_q:
                    best_q = val
        else:
            best_q = 0

        move_offset = self.to_offset(action[0], action[1], self.board_size)
        q_vals[0][move_offset] = (1 - self.alpha) * q_vals[0][move_offset] + \
                self.alpha * (reward + best_q)

        model.fit(self.numpify(state), q_vals, batch_size=1, nb_epoch=1,verbose=1)

    @staticmethod
    def numpify(state):
        """Given a state (board, color) tuple, return the numpy
        version of the board's array."""
        board = state[0].board
        assert len(board) > 0
        size = len(board) * len(board[0])
        if size != 16:
            print('wtf, size was {}'.format(size))
        return array(board).reshape(1, size)



    def train(self, epochs):
        model = self.get_model()
        model.compile(loss='mse', optimizer=RMSprop())
        model.load_weights(WEIGHTS_FILENAME)

        self.board_size = self.reversi.board.get_size()

        self.alpha = 0.8
        self.epsilon = 0.05

        wins = []

        for i in range(1, epochs + 1):
            print('starting epoch {}'.format(i))
            self.reversi.reset()
            state = deepcopy(self.reversi.get_state())
            print('starting state:')
            self.reversi.print_board(state)
            print()

            prev_action = None
            old_qs = None
            prev_state = None

            # in this epoch, play a game and refit as we go
            white_agent = RandomAgent(self.reversi, WHITE)

            while True:
                # white and black take turns until game is over
                # we break out when game is over
                if state[1] == WHITE:
                    # what's turn
                    legal = self.reversi.legal_moves(state)
                    move = None
                    if legal:
                        move = random.choice(legal)
                    state = self.reversi.next_state(state, move)

                elif state[1] == BLACK:
                    # it's black's turn

                    # first let's see where our last move landed us after white's play
                    if prev_state is not None:
                        # train model on where we ended up
                        reward = 0
                        winner = self.reversi.winner(state)
                        if winner == BLACK:
                            reward = 1
                        elif winner == WHITE:
                            reward = 0 # todo: try playing with this
                        elif winner is not False:
                            raise ValueError

                        self.update_model(model, old_qs, prev_state, prev_action, state, reward)

                        if winner is not False:
                            wins.append(winner)
                            break # the game is over, move on to next epoch

                    # now make our move based on highest legal q_val
                    q_vals = model.predict(self.numpify(state), batch_size = 1)
                    legal = self.reversi.legal_moves(state)
                    best_move = None
                    best_q = -float('inf')
                    for each in legal:
                        offset = self.to_offset(each[0], each[1], self.board_size)
                        if q_vals[0][offset] > best_q:
                            best_move = each
                            best_q = q_vals[0][offset]

                    # make our move, saving old values for next network refit
                    if legal:
                        # if we had a move and took it
                        prev_state = state
                        old_qs = q_vals
                        prev_action = best_move
                    state = self.reversi.next_state(state, best_move)



        print('training complete.')
        print('summary:')
        black_wins = len([win for win in wins if win == BLACK])
        white_wins = len([win for win in wins if win == WHITE])
        print(' black: {}'.format(black_wins))
        print(' white: {}'.format(white_wins))
        model.save_weights(WEIGHTS_FILENAME)

    @staticmethod
    def to_offset(x, y, width):
        """Given x,y coords, convert to an offset
        of an equal flattened array."""
        offset = 0
        offset += (y * width)
        offset += x
        return offset

    @staticmethod
    def get_model():
        """Load the model from disk, or create a new one
        if none is found on disk."""
        model = None
        try:
            model = model_from_json(open(MODEL_FILENAME).read())
        except FileNotFoundError:
            model = Sequential()
            model.add(Dense(164, init='lecun_uniform', input_shape=(16,)))
            model.add(Activation('relu'))
            # model.add(Dropout(0.2))

            model.add(Dense(150, init='lecun_uniform'))
            model.add(Activation('relu'))
            # model.add(Dropout(0.2))

            model.add(Dense(16, init='lecun_uniform'))
            model.add(Activation('linear'))

        return model

    @staticmethod
    def save_model(model):
        """Save the model layout to disk."""
        as_json = model.to_json()
        with open(MODEL_FILENAME, 'w') as f:
            f.write(as_json)
            print('model saved to {}'.format(filename))

