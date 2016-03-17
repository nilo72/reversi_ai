import random
from agents.agent import Agent

class RandomAgent(Agent):

    def __init__(self, reversi, color, **kwargs):
        pass

    def get_action(self, game_state, legal_moves):
        if len(legal_moves) == 0:
            return None
        return random.choice(legal_moves)
