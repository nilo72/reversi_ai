import random
from agents.agent import Agent
from util import *

class RandomAgent(Agent):
    """An agent that simply chooses
    totally random legal moves."""

    def __init__(self, reversi, color, **kwargs):
        self.reversi = reversi
        self.color = color

    def get_action(self, game_state):
        legal_moves = self.reversi.legal_moves(game_state)
        print('legal moves for {}: {}'.format(color_name[self.color], legal_moves))
        if not legal_moves:
            return None
        return random.choice(legal_moves)
