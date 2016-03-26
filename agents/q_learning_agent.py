from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.models import model_from_json
from keras.optimizers import RMSprop

from agents.agent import Agent

MODEL_FILENAME = 'q-model.json'

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

    def train(self, epochs):
        model = self.get_model()
        model.compile(loss='mse', optimizer=rms)

    @staticmethod
    def get_model():
        model = None
        try:
            model = model_from_json(open(MODEL_FILENAME).read())
        except FileNotFoundError:
            model = Sequential()
            model.add(Dense(164, init='lecun_uniform', input_shape=(64,)))
            model.add(Activation('relu'))
            # model.add(Dropout(0.2))

            model.add(Dense(150, init='lecun_uniform'))
            model.add(Activation('relu'))
            # model.add(Dropout(0.2))

            model.add(Dense(4, init='lecun_uniform'))
            model.add(Activation('linear'))

            rms = RMSprop()
        
        model.compile(loss='mse', optimizer=rms)
        return model

    @staticmethod
    def save_model(model, filename):
        as_json = model.to_json()
        with open(filename, 'w') as f:
            f.write(as_json)
            print('model saved to {}'.format(filename))

