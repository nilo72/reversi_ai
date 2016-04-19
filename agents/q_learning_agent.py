import random
from agents import Agent
from keras.layers import Dense, Activation
from keras.models import Sequential, model_from_json
from keras.optimizers import RMSprop, SGD
from util import info, opponent, color_name, to_offset, numpify, best_move_val
from agents.experience_replay import ExperienceReplay

MODEL_FILENAME = 'neural/q_model'
WEIGHTS_FILENAME = 'neural/q_weights'
HIDDEN_SIZE = 128
ALPHA = 0.01
BATCH_SIZE = 20

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
        self.load_weights(weights_num)

        # training values
        self.epsilon = 0.0
        if self.learning_enabled:
            self.memory = None
            self.prev_move = None
            self.prev_state = None
            self.MEM_LEN = 1

        if kwargs.get('model_file', None) is None:
            # if user didn't specify a model file, save the one we generated
            self.save_model(self.model)

    def set_epsilon(self, val):
        self.epsilon = val

    def set_memory(self, memory):
        self.memory = memory

    def get_action(self, state, legal_moves=None):
        """Agent method, called by the game to pick a move."""
        if legal_moves is None:
            legal_moves = self.reversi.legal_moves(state)

        if not legal_moves:
            # no actions available
            return None
        else:
            move = None
            if self.epsilon > random.random():
                move = random.choice(legal_moves)
            else:
                move = self.policy(state, legal_moves)
            if self.learning_enabled:
                self.train(state, legal_moves)
                self.prev_move = move
            return move

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

    def policy(self, state, legal_moves):
        """The policy of picking an action based on their weights."""
        if not legal_moves:
            return None

        best_move, _ = best_move_val(
            self.model.predict(numpify(state)),
            legal_moves
        )
        return best_move

    def train(self, state, legal_moves, winner=False):
        assert self.memory is not None, "can't train without setting memory first"
        model = self.model
        if self.prev_state is None:
            # on first move, no training to do yet
            q_vals = model.predict(numpify(state))
            best_move, _ = best_move_val(q_vals, legal_moves)

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

            self.memory.remember(self.prev_state, self.prev_move,
                                 reward, state, legal_moves, winner)

            # get an experience from memory and train on it
            states, targets = self.memory.get_replay(model, BATCH_SIZE, ALPHA)
            model.train_on_batch(states, targets)

            q_vals = model.predict(numpify(state))
            best_move, _ = best_move_val(q_vals, legal_moves)

            self.prev_state = state

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
            print('no model file loaded, generating new model.')
            size = self.reversi.size ** 2
            model = Sequential()
            model.add(Dense(HIDDEN_SIZE, activation='relu', input_dim=size))
            # model.add(Dense(HIDDEN_SIZE, activation='relu'))
            model.add(Dense(size))

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
        filename = '{}{}{}{}'.format(WEIGHTS_FILENAME, color_name[
                                     self.color], suffix, '.h5')
        print('saving weights to {}'.format(filename))
        self.model.save_weights(filename, overwrite=True)

    def load_weights(self, suffix):
        filename = '{}{}{}{}'.format(WEIGHTS_FILENAME, color_name[
                                     self.color], suffix, '.h5')
        print('loading weights from {}'.format(filename))
        try:
            self.model.load_weights(filename)
        except:
            print('Couldn\'t load weights file {}! Will generate it.'.format(filename))
