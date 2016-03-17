import random
import time
import math
import copy
import pdb
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
        self.states_to_nodes = {}

    def get_action(self, game_state, legal_moves):
        # always make sure you are getting a deep copy
        game_state = copy.deepcopy(game_state)

        move = self.monte_carlo_search(game_state)

        # print(
        #    '({},{}) wins/plays: {}/{}'.format(move[0], move[1], wins_plays[0], wins_plays[1]))
        # print('({} sims played)'.format(sims_played))
        return move

    def monte_carlo_search(self, game_state):
        root = None
        if isinstance(game_state, Node):
            pdb.set_trace()
        if game_state in self.states_to_nodes:
            root = self.states_to_nodes[game_state]
        else:
            root = Node(game_state)
            self.states_to_nodes[game_state] = root

        # even if this is a "recycled" node we've already used,
        # remove its parent as it is now considered our root level node
        root.remove_parent()

        sim_count = 0
        now = time.time()
        while time.time() - now < self.sim_time:
            visited = set()
            picked_node = self.tree_policy(root, visited)
            result = self.simulate(picked_node.get_game_state())
            self.back_prop(picked_node, result)
            sim_count += 1

        for child in root.get_children():
            wins, plays = child.get_wins_plays()
            position = child.get_action()
            print('{}: ({}/{})'.format(position, wins, plays))
        print('{} simulations performed.'.format(sim_count))
        return self.best_action(root)

    def best_action(self, node):
        # pick the action with the most plays, breaking ties.
        most_plays = -float('inf')
        best_wins = -float('inf')
        best_actions = []
        for child in node.get_children():
            wins, plays = child.get_wins_plays()
            if plays > most_plays:
                most_plays = plays
                best_actions = [child.get_action()]
                best_wins = wins
            elif plays == most_plays:
                # break ties with wins
                if wins > best_wins:
                    best_wins = wins
                    best_actions = [child.get_action()]
                elif wins == best_wins:
                    best_actions.append(child.get_action())

        return random.choice(best_actions)

    def back_prop(self, node, delta):
        while node.get_parent() is not None:
            node.add_play()
            node.add_win_delta(delta)
            node = node.get_parent()

        # update root node of entire tree
        node.add_play()
        node.add_win_delta(delta)

    def tree_policy(self, root, visited):
        cur_node = root
        while True:
            # if this is a terminal node, break
            legal_moves = self.reversi.get_legal_moves(
                cur_node.get_game_state())
            if len(legal_moves) == 0:
                break

            # expand unexplored children states, add them to the tree
            if len(cur_node.get_children()) < len(legal_moves):
                next_states_moves = [
                    (self.reversi.next_state(
                        cur_node.get_game_state(), *move), move) for move in legal_moves
                ]
                unseen = [
                    state_move for state_move in next_states_moves
                    if state_move[0] not in visited
                ]

                if len(unseen) == 0:
                    pdb.set_trace()

                to_expand = random.choice(unseen)
                state, move = to_expand
                n = Node(state, move)
                cur_node.add_child(n)
                visited.add(state)
                self.states_to_nodes[n] = state

                return n
            else:
                cur_node = self.best_child(cur_node)
                assert cur_node is not None

        return cur_node

    def best_child(self, node):
        C = 2
        values = {}
        for child in node.get_children():
            wins, plays = child.get_wins_plays()
            parent_plays = node.get_wins_plays()[1]
            values[child] = (wins / plays) \
                + math.sqrt(C * math.log(parent_plays) / plays)

        best_choice = max(values, key=values.get)
        return best_choice

    def simulate(self, game_state):
        state = game_state
        while True:
            legal_moves = self.reversi.get_legal_moves(state)
            if len(legal_moves) == 0:
                break

            picked_move = random.choice(legal_moves)
            state = self.reversi.next_state(state, *picked_move)

        result = self.reversi.winner(state)
        if result == self.color:
            return 1
        else:
            return 0


class Node:

    def __init__(self, game_state, action=None):
        self.game_state = game_state
        self.plays = 0
        self.wins = 0
        self.children = []
        self.parent = None

        # the action that led to this child state
        self.action = action

    def get_game_state(self):
        return self.game_state

    def add_child(self, node):
        self.children.append(node)
        node.parent = self

    def get_action(self):
        return self.action

    def get_children(self):
        return self.children

    def get_parent(self):
        return self.parent

    def hash_children(self):
        return len(self.children) > 0

    def get_wins_plays(self):
        return self.wins, self.plays

    def add_win_delta(self, delta):
        self.wins += delta

    def add_play(self):
        self.plays += 1

    def __hash__(self):
        return hash(self.game_state)

    def remove_parent(self):
        self.parent = None

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.game_state == other.game_state
