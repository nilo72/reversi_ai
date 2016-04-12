import numpy as np
import random
from agents import Agent
from keras.layers import Dense, Activation
from keras.models import Sequential, model_from_json
from keras.optimizers import RMSprop, SGD
from util import info, opponent, color_name

MODEL_FILENAME = 'neural/q_model'
WEIGHTS_FILENAME = 'neural/q_weights'
HIDDEN_SIZE = 44
optimizer = None
ALPHA = 0.1
BATCH_SIZE = 25

WIN_REWARD = 1
LOSE_REWARD = -1
optimizer = RMSprop()


class QLearningAgent(Agent):

    def __init__(self, reversi, color, **kwargs):
        self.color = color
        self.reversi = reversi
        self.learning_enabled = kwargs.get('learning_enabled', False)
        self.model = self.get_model(kwargs.get('model_file', None))

        weights_num = kwargs.get('weights_num', '')
        if weights_num != '':
            self.load_weights(str(float(weights_num)))

        # training values
        self.prev_move = None
        self.prev_state = None
        self.epsilon = 0.0
        self.memory = []
        self.MEM_LEN = 1

        self.save_model(self.model)

    def set_epsilon(self, val):
        self.epsilon = val

    def set_replay_len(self, val):
        self.MEM_LEN = val

    def get_action(self, state, legal_moves=None):
        """Agent method, called by the game to pick a move."""
        if legal_moves is None:
            legal_moves = self.reversi.legal_moves(state)

        if not legal_moves:
            # no actions available
            return None
        else:
            if self.learning_enabled:
                self.train(state, legal_moves)
            if self.epsilon > random.random():
                return random.choice(legal_moves)
            else:
                return self.policy(state, legal_moves)

    def remember(self, S, a, r, Sp, l, win):
        self.memory.append((S, a, r, Sp, l, win))
        if len(self.memory) > self.MEM_LEN:
            self.memory.pop(0)

    def get_replay(self, batch_size):
        # (S, a, r, Sp, legal, win) tuple
        S, a, r, Sp, l, win = range(6)  # indices
        model = self.model

        if batch_size > len(self.memory):
            batch_size = len(self.memory)
        replays = random.sample(self.memory, batch_size)

        # now format for training
        board_size = self.reversi.size
        inputs = np.empty((batch_size, board_size ** 2))
        targets = np.empty((batch_size, board_size ** 2))
        for index, replay in enumerate(replays):
            if replay[win] is False and not replay[l]:
                continue  # no legal moves, and not a win
            move = self.to_offset(replay[a])
            state = self.numpify(replay[S])
            state_prime = self.numpify(replay[Sp])
            prev_qvals = model.predict(state)

            q_prime = None
            if win is False:
                next_qvals = model.predict(state_prime)
                _, best_q = self.best_move_val(next_qvals, replay[l])
                q_prime = (1 - ALPHA) * \
                    prev_qvals[0][move] + ALPHA * (replay[r] + best_q)
            else:
                q_prime = (1 - ALPHA) * prev_qvals[0][move] + ALPHA * replay[r]
            prev_qvals[0][move] = q_prime
            inputs[index] = state
            targets[index] = prev_qvals

        return inputs, targets

    @staticmethod
    def numpify(state):
        """Given a state (board, color) tuple, return the flattened numpy
        version of the board's array."""
        board = state[0].board
        assert len(board) > 0
        size = len(board) * len(board[0])
        return np.array(board).reshape(1, size)
        # return np.reshape(board, (1, size))

    def observe_win(self, state):
        """Called by the game at end of game to present the agent with the final board state."""
        if self.learning_enabled:
            winner = self.reversi.winner(state)
            self.train(state, [], winner)

    def reset(self):
        """Resets the agent to prepare it to play another game."""
        self.reset_learning()

    def reset_learning(self):
        self.prev_move = None
        self.prev_state = None
        self.memory = []

    def policy(self, state, legal_moves):
        """The policy of picking an action based on their weights."""
        if not legal_moves:
            return None

        best_move, _ = self.best_move_val(
            self.model.predict(self.numpify(state)),
            legal_moves
        )
        return best_move

    def train(self, state, legal_moves, winner=False):
        model = self.model
        if self.prev_state is None:
            # on first move, no training to do yet
            q_vals = model.predict(self.numpify(state))
            best_move, _ = self.best_move_val(q_vals, legal_moves)

            self.prev_move = best_move
            self.prev_state = state
        else:
            # add new info to replay memory
            reward = 0
            if winner == self.color:
                reward = WIN_REWARD
            elif winner == opponent[self.color]:
                reward = LOSE_REWARD
            elif winner is not False:
                raise ValueError

            self.remember(self.prev_state, self.prev_move,
                          reward, state, legal_moves, winner)

            # get an experience from memory and train on it
            states, targets = self.get_replay(BATCH_SIZE)
            model.train_on_batch(states, targets)

            q_vals = model.predict(self.numpify(state))
            best_move, _ = self.best_move_val(q_vals, legal_moves)

            self.prev_move = best_move
            self.prev_state = state

    def update_model(self, prev_state, prev_qvals, prev_move, state, legal, reward, winner):
        # Q(s,a) = (1-alpha) * Q(s,a) + alpha * (r + maxQ(s', a'))
        pass

    def best_move_val(self, q_vals, legal_moves):
        """Given a list of moves and a q_val array, return the move with the highest q_val and the q_val."""
        if not legal_moves:
            return None, None
        else:
            best_q = None
            best_move = None

            for move in legal_moves:
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
        x, y = move
        return y * self.reversi.size + x

    def get_model(self, filename=None):
        """Given a filename, load that model file; otherwise, generate a new model."""
        model = None
        if filename:
            info('attempting to load model {}'.format(filename))
            try:
                model = model_from_json(open(filename).read())
            except FileNotFoundError:
                print('could not load file {}'.format(filename))
                quit()
            print('loaded model file {}'.format(filename))
        else:
            info('no model file loaded, generating new model.')
            size = self.reversi.size ** 2
            model = Sequential()
            model.add(Dense(HIDDEN_SIZE, init='zero', input_shape=(size,)))
            model.add(Activation('relu'))
            model.add(Dense(size, init='zero'))
            model.add(Activation('linear'))

        model.compile(loss='mse', optimizer=optimizer)
        return model

    @staticmethod
    def save_model(model):
        """Given a model, save it to disk."""
        as_json = model.to_json()
        with open(MODEL_FILENAME, 'w') as f:
            f.write(as_json)
            info('model saved to {}'.format(MODEL_FILENAME))

    def save_weights(self, suffix):
        filename = '{}{}{}{}'.format(WEIGHTS_FILENAME, color_name[self.color], suffix, '.h5')
        info('saving weights to {}'.format(filename))
        self.model.save_weights(filename)

    def load_weights(self, suffix):
        filename = '{}{}_{}{}'.format(WEIGHTS_FILENAME, color_name[self.color], suffix, '.h5')
        info('loading weights from {}'.format(filename))
        self.model.load_weights(filename)
