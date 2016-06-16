from agents import Agent, ExperienceReplay, Experience
import random
from util import numpify, max_q_move, double_expand
from keras.layers import Dense, Convolution2D, Flatten
from keras.models import Sequential, model_from_json
from filenames import MODEL, weights_filename

WIN_REWARD = 1
LOSS_REWARD = -1

MINIBATCH_SIZE = 32
MIN_EPSILON = 0.01

OPTIMIZER = 'rmsprop'


class QLearningAgent(Agent):
    def __init__(self, reversi, color, **kwargs):
        self.reversi = reversi
        self.color = color

        # training values
        self._epsilon = 0.0
        self._replay_memory = None
        self._prev_state = None
        self._prev_action = None
        self._turn_count = 1

        self._training_enabled = kwargs.get('training_enabled', False)
        self._use_existing_weights = kwargs.get('use_existing_weights', False)
        if self._training_enabled:
            self._init_training()
            self._replay_memory = ExperienceReplay()
            self._epsilon = 1.0  # if training, start epsilon at 1.0

        try:
            self._model = self._load_model()
        except FileNotFoundError:
            self._model = self._gen_model()
            self._save_model(self._model)

        if self._use_existing_weights:
            self._load_weights(self._model)

    def get_action(self, game_state, legal_moves):
        """Given the game state and the legal moves, return this agent's
        choice of move (None if no legal moves)."""
        if not legal_moves:
            return None

        picked_move = None
        if self._epsilon > random.random():
            # take random action
            picked_move = random.choice(legal_moves)
        else:
            picked_move, val = self._best_move_val(game_state, legal_moves)

        if self._training_enabled:
            self._train(game_state, legal_moves)
            self._prev_state = game_state
            self._prev_action = picked_move

        return picked_move

    def observe_win(self, state, winner):
        """Called by the game runner when the game is over to present the
        players with the final game state."""
        self._train(state, legal_moves=[], winner=winner)

    def reset(self):
        """Reset the agent so that it is in the proper state to play
        a game from scratch."""
        self._init_training()

    def _init_training(self):
        """Initialize the training values to their defaults."""
        self._prev_state = None
        self._prev_action = None

    def _best_move_val(self, game_state, legal_moves):
        """Given a game state and list of legal moves, return a tuple of
        (best_move, that_moves_qval).
        If no legal moves, returns (None, None)"""
        numpified = double_expand(numpify(game_state))
        q_vals = self._model.predict(numpified, batch_size=1)
        return max_q_move(q_vals, legal_moves, self.reversi.get_board_size())

    def _train(self, state, legal_moves, winner=False):
        """Use q-learning to train the neural network."""
        if self._prev_state is None:
            # first move of the game, nothing to do
            return
        else:
            reward = 0
            if winner == self.color:
                reward = WIN_REWARD
            elif winner is not False:
                reward = LOSS_REWARD

            # store experience in memory
            self._replay_memory.remember(
                Experience(self._prev_state, self._prev_action, state, reward,
                           legal_moves, winner))

            if self._turn_count % MINIBATCH_SIZE == 0:
                # sample experiences from memory
                prev_vals, targets = self._replay_memory.get_replay(
                    self._model, MINIBATCH_SIZE)
                self._model.train_on_batch(prev_vals, targets)

            self._turn_count += 1

    def decrement_epsilon(self, amnt):
        self._epsilon = max(self._epsilon - amnt, MIN_EPSILON)

    def get_epsilon(self):
        return self._epsilon

    def _load_model(self):
        json = open(MODEL, 'r').read()
        model = model_from_json(json)
        model.compile(optimizer=OPTIMIZER, loss='mse')
        print('Loaded model: {}'.format(MODEL))
        return model

    def _save_model(self, model):
        json = model.to_json()
        with open(MODEL, 'w') as f:
            f.write(json)
            print('Model saved as {}'.format(MODEL))

    def _gen_model(self):
        board_size = self.reversi.get_board_size()
        model = Sequential()
        model.add(Convolution2D(nb_filter=16,
                                nb_row=3,
                                nb_col=3,
                                border_mode='same',
                                input_shape=(1, board_size, board_size)))
        model.add(Convolution2D(nb_filter=8,
                                nb_row=3,
                                nb_col=3,
                                border_mode='same',
                                activation='relu'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(board_size**2, activation='linear'))
        model.compile(optimizer=OPTIMIZER, loss='mse')

        print('Generated model with input shape: {}'.format(model.input_shape))
        return model

    def load_weights(self):
        model = self._model
        filename = weights_filename(self.color)
        model.load_weights(filename)
        print('Weights loaded from {}'.format(filename))

    def save_weights(self, suffix=''):
        model = self._model
        filename = weights_filename(self.color, suffix)
        model.save_weights(filename, overwrite=True)
        print('Weights saved to {}'.format(filename))
