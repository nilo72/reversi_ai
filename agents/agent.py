class Agent:
    """An abstract class defining the interface for a Reversi agent."""

    def get_action(self, game_state, legal_moves):
        raise NotImplementedError
