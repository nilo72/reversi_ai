import random
import time
import math
import copy
import pdb
from agents.agent import Agent


class MonteCarloAgent(Agent):
    """An agent utilizing Monte Carlo Tree Search.  I based much of
    this implementation off Jeff Bradberry's informative blog:
    https://jeffbradberry.com/posts/2015/09/intro-to-monte-carlo-tree-search/"""

    def __init__(self, reversi, color, **kwargs):
        self.color = color
        self.reversi = reversi
        self.print_info = kwargs.get('print', False)
        self.sim_time = kwargs.get('time', 5)

    def get_action(self, game_state, legal_moves):
        # always make sure you are getting a deep copy
        game_state = copy.deepcopy(game_state)
        move = self.monte_carlo_search(game_state)
        return move

    def info(self, s):
        if self.print_info:
            print(s)

    def monte_carlo_search(self, game_state):
        root = Node(game_state)

        # even if this is a "recycled" node we've already used,
        # remove its parent as it is now considered our root level node
        root.remove_parent()

        sim_count = 0
        now = time.time()
        while time.time() - now < self.sim_time:
            picked_node = self.tree_policy(root)
            result = self.simulate(picked_node.get_game_state())
            self.back_prop(picked_node, result)
            sim_count += 1

        for child in root.get_children():
            wins, plays = child.get_wins_plays()
            position = child.get_action()
            self.info('{}: ({}/{})'.format(position, wins, plays))
        self.info('{} simulations performed.'.format(sim_count))
        return self.best_action(root)

    @staticmethod
    def best_action(node):
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

    @staticmethod
    def back_prop(node, delta):
        while node.get_parent() is not None:
            node.add_play()
            node.add_win_delta(delta)
            node = node.get_parent()

        # update root node of entire tree
        node.add_play()
        node.add_win_delta(delta)

    def tree_policy(self, root):
        cur_node = root
        while True:
            # if this is a terminal node, break
            legal_moves = self.reversi.get_legal_moves(
                cur_node.get_game_state())
            if len(legal_moves) == 0:
                break

            # if children are not fully expanded, expand one or more
            if len(cur_node.get_children()) < len(legal_moves):
                next_states_moves = [
                    (self.reversi.next_state(
                        cur_node.get_game_state(), *move), move) for move in legal_moves
                ]
                unexpanded = [
                    state_move for state_move in next_states_moves
                    if not cur_node.has_child_state(state_move[0])
                ]

                assert len(unexpanded) > 0
                state, move = random.choice(unexpanded)
                n = Node(state, move)
                cur_node.add_child(n)

                return n
            else:
                cur_node = self.best_child(cur_node)

        return cur_node

    @staticmethod
    def best_child(node):
        C = 1
        values = {}
        for child in node.get_children():
            wins, plays = child.get_wins_plays()
            _, parent_plays = node.get_wins_plays()
            values[child] = (wins / plays) \
                + C * math.sqrt(2 * math.log(parent_plays) / plays)

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

        self.child_states = set()

        # the action that led to this child state
        self.action = action

    def get_game_state(self):
        return self.game_state

    def add_child(self, node):
        self.children.append(node)
        self.child_states.add(node.get_game_state())
        node.parent = self

    def has_child_state(self, state):
        return state in self.child_states

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

    def __repr__(self):
        return 'move: {} wins: {} plays: {}'.format(self.action, self.wins, self.plays)

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.game_state == other.game_state
