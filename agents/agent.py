import sys


class Agent:
    """An abstract class defining the interface for a Reversi agent."""

    def __init__(self, reversi, color):
        print('must define agent inteface!')
        sys.exit(1)

    def get_action(self, game_state, legal_moves):
        raise NotImplementedError
