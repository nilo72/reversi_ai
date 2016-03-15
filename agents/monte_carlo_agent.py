from agents.agent import Agent


class MonteCarloAgent(Agent):
    """An agent utilizing Monte Carlo Tree Search.  I based much of
    this implementation off Jeff Bradberry's informative blog:
    https://jeffbradberry.com/posts/2015/09/intro-to-monte-carlo-tree-search/"""

    def get_action(self, game_state, legal_moves):
        self.current_state = game_state
