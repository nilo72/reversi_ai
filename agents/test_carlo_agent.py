import random
import time
import math
from agents.agent import Agent

SIM_TIME = 3


class MonteCarloAgent(Agent):
    """An agent utilizing Monte Carlo Tree Search.  I based much of
    this implementation off Jeff Bradberry's informative blog:
    https://jeffbradberry.com/posts/2015/09/intro-to-monte-carlo-tree-search/"""

    def __init__(self, reversi, color, **kwargs):
        self.color = color
        self.reversi = reversi

        self.sim_time = kwargs.get('time', SIM_TIME)

        # map each visited state to a list containing the amount of plays it
        # has seen and the amount of times it has won. state -> [wins, plays]
        self.wins_and_plays = {}

    def get_action(self, game_state, legal_moves):
        sims_played = self.simulate(game_state)

        # once simulation is done, pick the most promising next move
        best_val = -float('inf')
        best_ratio = 0
        best_moves = []
        for move in legal_moves:
            next_state = self.reversi.next_state(game_state, move[0], move[1])
            if next_state not in self.wins_and_plays:
                continue
            wins_plays = self.wins_and_plays[next_state]
            val = wins_plays[1]
            if val > best_val:
                best_val = val
                best_ratio = wins_plays[0] / wins_plays[1]
                best_moves = [move]
            elif val == best_val:
                ratio = wins_plays[0] / wins_plays[1]
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_moves = [move]
                elif ratio == best_ratio:
                    best_moves.append(move)

            print('{}: ({}/{})'.format(move, wins_plays[0], wins_plays[1]))
        print('({} sims played)'.format(sims_played))
        assert len(best_moves) > 0
        return random.choice(best_moves)

    def simulate(self, game_state):
        start = time.time()
        sims_played = 0
        while time.time() - start < self.sim_time:
            cur_state = game_state
            if cur_state not in self.wins_and_plays:
                self.wins_and_plays[cur_state] = [0, 0]
            traversed = set()
            traversed.add(cur_state)

            # selection
            while True:
                next_states = [self.reversi.next_state(
                    cur_state, move[0], move[1]) for move in self.reversi.get_legal_moves(cur_state)]
                unvisited = [
                    state for state in next_states if state not in self.wins_and_plays]

                # we're done if we encounter unvisited children, or reach a
                # leaf
                if len(unvisited) != 0 or len(next_states) == 0:
                    break

                cur_plays = self.wins_and_plays[cur_state][1]
                values = {}
                for state in next_states:
                    wins_plays = self.wins_and_plays[state]
                    mean = wins_plays[0] / wins_plays[1]
                    assert cur_plays != 0
                    assert wins_plays[1] != 0
                    C = 1
                    values[state] = mean + \
                        math.sqrt(C * math.log(cur_plays) / wins_plays[1])
                cur_state = max(values, key=values.get)
                traversed.add(cur_state)

            # expansion
            if len(unvisited) != 0:
                cur_state = random.choice(unvisited)

            #simulate
            if cur_state in traversed:
                # don't include cur_state in traversed because
                # run_simulation adds cur_state to its own internal set.
                traversed.remove(cur_state)
            self.run_simulation(cur_state, traversed)
            sims_played += 1

        return sims_played

    def run_simulation(self, game_state, traversed=None):
        visited_states = set()
        state = game_state
        visited_states.add(state)
        if state not in self.wins_and_plays:
            self.wins_and_plays[state] = [0, 0]

        while self.reversi.winner(state) is False:
            legal_moves = self.reversi.get_legal_moves(state)
            if len(legal_moves) == 0:
                # player can't make a move so their turn passes
                state = (state[0], state[1] * -1)
                continue
            picked = random.choice(legal_moves)
            state = self.reversi.next_state(state, picked[0], picked[1])
            visited_states.add(state)
            if state not in self.wins_and_plays:
                self.wins_and_plays[state] = [0, 0]

        won = 1 if self.reversi.winner(state) == self.color else 0
        for visited in visited_states:
            self.wins_and_plays[visited][0] += won
            self.wins_and_plays[visited][1] += 1

        if traversed is not None:
            for each in traversed:
                self.wins_and_plays[each][0] += won
                self.wins_and_plays[each][1] += 1
