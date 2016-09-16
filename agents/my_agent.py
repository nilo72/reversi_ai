from agents.agent import Agent

class MyAgent(Agent):
    """This agent is controlled by a human, who inputs moves via stdin."""

    def __init__(self, reversi, color, **kwargs):
        self.reversi = reversi
        self.color = color

    def reset(self):
        pass

    def observe_win(self, winner):
        pass

    def get_action(self, game_state, legal):
        if not legal:
            return None
        choice = legal[:1][0]
        return choice
