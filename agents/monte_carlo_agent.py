import random
import time
import math
import copy
import pdb
from agents.agent import Agent
from util import *


class MonteCarloAgent(Agent):
    """An agent utilizing Monte Carlo Tree Search.  I based much of
    this implementation off Jeff Bradberry's informative blog:
    https://jeffbradberry.com/posts/2015/09/intro-to-monte-carlo-tree-search/"""

    def __init__(self, reversi, color, **kwargs):
        self.color = color
        self.reversi = reversi
        self.print_info = kwargs.get('print', False)
        self.sim_time = kwargs.get('time', 5)
        self.state_node = {}

    def get_action(self, game_state, legal_moves):
        # always make sure you are getting a deep copy
        game_state = copy.deepcopy(game_state)
        move = self.monte_carlo_search(game_state)
        return move

    def info(self, s):
        if self.print_info:
            print(s)

    def monte_carlo_search(self, game_state):
        root = None
        if game_state in self.state_node:
            root = self.state_node[game_state]
        else:
            root = Node(game_state)

        # even if this is a "recycled" node we've already used,
        # remove its parent as it is now considered our root level node
        root.parent = None

        sim_count = 0
        now = time.time()
        while time.time() - now < self.sim_time:
            picked_node = self.tree_policy(root)
            result = self.simulate(picked_node.game_state)
            self.back_prop(picked_node, result)
            sim_count += 1

        for child in root.children:
            wins, plays = child.get_wins_plays()
            position = child.move
            self.info('{}: ({}/{})'.format(position, wins, plays))
        self.info('{} simulations performed.'.format(sim_count))
        return self.best_action(root)

    @staticmethod
    def best_action(node):
        # pick the action with the most plays, breaking ties.
        most_plays = -float('inf')
        best_wins = -float('inf')
        best_actions = []
        for child in node.children:
            wins, plays = child.get_wins_plays()
            if plays > most_plays:
                most_plays = plays
                best_actions = [child.move]
                best_wins = wins
            elif plays == most_plays:
                # break ties with wins
                if wins > best_wins:
                    best_wins = wins
                    best_actions = [child.move]
                elif wins == best_wins:
                    best_actions.append(child.move)

        return random.choice(best_actions)

    @staticmethod
    def back_prop(node, delta):
        while node.parent is not None:
            node.plays += 1
            node.wins += delta
            node = node.parent

        # update root node of entire tree
        node.plays += 1
        node.wins += delta

    def tree_policy(self, root):
        cur_node = root
        while True:
            # if this is a terminal node, break
            legal_moves = self.reversi.get_legal_moves(
                cur_node.game_state)
            if len(legal_moves) == 0:
                # if the game is won, break
                if self.reversi.winner(cur_node.game_state):
                    break
                else:
                    # player passes their turn
                    next_state = (cur_node.game_state[0], opponent[
                                  cur_node.game_state[1]])
                    pass_node = Node(next_state)
                    cur_node.add_child(pass_node)
                    self.state_node[next_state] = pass_node
                    cur_node = pass_node
                    continue

            # if children are not fully expanded, expand one or more
            if len(cur_node.children) < len(legal_moves):
                next_states_moves = [
                    (self.reversi.next_state(
                        cur_node.game_state, *move), move) for move in legal_moves
                ]
                unexpanded = [
                    state_move for state_move in next_states_moves
                    if not cur_node.has_child_state(state_move[0])
                ]

                assert len(unexpanded) > 0
                state, move = random.choice(unexpanded)
                n = Node(state, move)
                cur_node.add_child(n)
                self.state_node[state] = n

                return n
            else:
                cur_node = self.best_child(cur_node)

        return cur_node

    @staticmethod
    def best_child(node):
        C = 1
        values = {}
        for child in node.children:
            wins, plays = child.get_wins_plays()
            _, parent_plays = node.get_wins_plays()
            values[child] = (wins / plays) \
                + C * math.sqrt(2 * math.log(parent_plays) / plays)

        best_choice = max(values, key=values.get)
        return best_choice

    def simulate(self, game_state):
        state = copy.deepcopy(game_state)
        while True:
            winner = self.reversi.winner(state)
            if winner is not False:
                if winner == self.color:
                    return 1
                elif winner == opponent[self.color]:
                    return 0
                else:
                    raise ValueError

            moves = self.reversi.get_legal_moves(state)
            if not moves:
                # if no moves, turn passes to opponent
                state = (state[0], opponent[state[1]])
                moves = self.reversi.get_legal_moves(state)

            picked = random.choice(moves)
            state = self.reversi.apply_move(state, *picked)


class Node:

    def __init__(self, game_state, move=None):
        self.game_state = game_state
        self.plays = 0
        self.wins = 0
        self.children = []
        self.parent = None
        self.child_states = set()

        # the move that led to this child state
        self.move = move

    def add_child(self, node):
        self.children.append(node)
        self.child_states.add(node.game_state)
        node.parent = self

    def has_child_state(self, state):
        return state in self.child_states

    def has_children(self):
        return len(self.children) > 0

    def get_wins_plays(self):
        return self.wins, self.plays

    def __hash__(self):
        return hash(self.game_state)

    def __repr__(self):
        return 'move: {} wins: {} plays: {}'.format(self.move, self.wins, self.plays)

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.game_state == other.game_state
