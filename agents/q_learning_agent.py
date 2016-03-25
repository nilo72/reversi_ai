from agents.agent import Agent

class QLearningAgent(Agent):

    def __init__(self, reversi, color):
        self.reversi = reversi
        self.color = color

    def get_action(self, game_state, legal_moves):
        if not legal_moves:
            return None

        return self.policy(game_state, legal_moves)

    def policy(self, game_state, legal_moves):
        pass


    
