import random
import copy
import pdb
from game.board import Board, BLACK, WHITE, EMPTY
from agents.human_agent import HumanAgent
from agents.random_agent import RandomAgent
from util import *
from cache_dict import CacheDict


class Reversi:
    """This class enforces the rules of the game of Reversi."""

    def __init__(self, dimens, BlackAgent=RandomAgent, WhiteAgent=RandomAgent, **kwargs):
        board = Board(dimens)
        board.init_starting_position()
        self.dimensions = board.get_dimensions()

        self.game_state = (board, BLACK)

        self.legal_cache = CacheDict()
        # self.valid_cache = CacheDict()
        # self.winner_cache = CacheDict()

        black_time = kwargs.get('black_time', 5)
        white_time = kwargs.get('white_time', 5)
        self.white_agent = WhiteAgent(self, WHITE, time=white_time, **kwargs)
        self.black_agent = BlackAgent(self, BLACK, time=black_time, **kwargs)

        # storing legal moves allows us to avoid needlessly recalculating them
        self.legal_white_moves = []
        self.legal_black_moves = []

    def play_game(self):
        self.update_legal_moves(self.get_state())
        while not self.is_won():
            self.print_board()
            game_state = self.get_state()
            turn_color = game_state[1]
            self.update_legal_moves(game_state)
            legal_moves = None
            if turn_color == BLACK:
                legal_moves = self.legal_black_moves
            elif turn_color == WHITE:
                legal_moves = self.legal_white_moves
            else:
                raise ValueError

            if len(legal_moves) == 0:
                print('{} had no moves, and passed their turn.'.format(
                    color_name[turn_color]))
                self.game_state = (game_state[0], opponent[turn_color])
                continue
            else:
                picked = self.agent_pick_move(game_state, legal_moves)
                print('{} plays at {}'.format(
                    color_name[turn_color], str(picked)))
            updated_game_state = self.apply_move(
                game_state, picked[0], picked[1])
            self.game_state = updated_game_state
        self.print_board()

        # figure out who won
        white_count = 0
        black_count = 0
        for y in range(self.dimensions):
            for x in range(self.dimensions):
                piece = self.get_board().piece_at(x, y)
                if piece == WHITE:
                    white_count += 1
                elif piece == BLACK:
                    black_count += 1

        winner = WHITE if white_count > black_count else BLACK
        return winner, white_count, black_count

    def print_board(self):
        print(str(self.get_board()))

    def agent_pick_move(self, game_state, legal_moves):
        picked = None
        color = game_state[1]
        while picked not in legal_moves:
            if color == WHITE:
                picked = self.white_agent.get_action(
                    self.get_state(), legal_moves)
            elif color == BLACK:
                picked = self.black_agent.get_action(
                    self.get_state(), legal_moves)
            else:
                raise ValueError

            if picked is None:
                return None
            elif picked not in legal_moves:
                print(str(picked) + ' is not a legal move!')

        return picked

    def get_legal_moves(self, game_state):
        # cached = self.legal_cache.get(game_state)
        # if cached is not None:
        #    return cached

        board = game_state[0]
        board_size = board.get_dimensions()
        moves = []  # list of x,y positions valid for color

        for y in range(board_size):
            for x in range(board_size):
                if self.is_valid_move(game_state, x, y):
                    moves.append((x, y))

        # self.legal_cache.update(game_state, moves)
        return moves

    def is_valid_move(self, game_state, x, y):
        board = game_state[0]
        color = game_state[1]
        piece = board.piece_at(x, y)
        if piece != EMPTY:
            return False

        opponent = BLACK
        if color == BLACK:
            opponent = WHITE

        # now check in all directions, including diagonal
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue

                # there needs to be >= 1 opponent piece
                # in this given direction, followed by 1 of player's piece
                distance = 1
                yp = (distance * dy) + y
                xp = (distance * dx) + x

                while board.is_in_bounds(xp, yp) and board.piece_at(xp, yp) == opponent:
                    distance += 1
                    yp = (distance * dy) + y
                    xp = (distance * dx) + x

                if distance > 1 and board.is_in_bounds(xp, yp) and board.piece_at(xp, yp) == color:
                    return True
        return False

    def next_state(self, game_state, x, y):
        """Given a game_state and a position for a new piece, return a new game_state
        reflecting the change.  Does not modify the input game_state."""
        game_state_copy = copy.deepcopy(game_state)
        result = self.apply_move(game_state_copy, x, y)
        return result

    def apply_move(self, game_state, x, y):
        """Given a game_state (which includes info about whose turn it is) and an x,y
        position to place a piece, transform it into the game_state that follows this play."""
        color = game_state[1]
        board = game_state[0]
        board.place_stone_at(color, x, y)

        # now flip all the stones in every direction
        opponent = BLACK
        if color == BLACK:
            opponent = WHITE

        # now check in all directions, including diagonal
        to_flip = []
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue

                # there needs to be >= 1 opponent piece
                # in this given direction, followed by 1 of player's piece
                distance = 1
                yp = (distance * dy) + y
                xp = (distance * dx) + x

                flip_candidates = []
                while board.is_in_bounds(xp, yp) and board.piece_at(xp, yp) == opponent:
                    flip_candidates.append((xp, yp))
                    distance += 1
                    yp = (distance * dy) + y
                    xp = (distance * dx) + x

                if distance > 1 and board.is_in_bounds(xp, yp) and board.piece_at(xp, yp) == color:
                    to_flip.extend(flip_candidates)

        for each in to_flip:
            board.flip_stone(each[0], each[1])
            # board.place_stone_at(color, each[0], each[1])

        if game_state[1] == WHITE:
            game_state = (game_state[0], BLACK)
        elif game_state[1] == BLACK:
            game_state = (game_state[0], WHITE)

        return game_state

    def is_won(self):
        """The game is over when neither player can make a move."""
        return self.get_board().is_full() or (len(self.legal_black_moves) == 0 and len(self.legal_white_moves) == 0)

    def winner(self, game_state):
        """Given a game_state, is the game over, and who won?"""
        board = game_state[0]
        if board.is_full():
            black_count, white_count = board.get_stone_counts()
            if black_count > white_count:
                return BLACK
            else:
                # tie goes to white (for now)
                return WHITE
        black_legal = self.get_legal_moves((game_state[0], BLACK))
        white_legal = self.get_legal_moves((game_state[0], WHITE))

        if len(black_legal) == 0 and len(white_legal) == 0:
            black_count, white_count = board.get_stone_counts()
            if black_count > white_count:
                return BLACK
            else:
                # tie goes to white (for now)
                return WHITE
        else:
            return False

    def update_legal_moves(self, game_state):
        legal_moves = self.get_legal_moves(game_state)
        color = game_state[1]
        if color == WHITE:
            self.legal_white_moves = legal_moves
        elif color == BLACK:
            self.legal_black_moves = legal_moves
        else:
            raise ValueError

    def get_board(self):
        """Return the board from the current game_state."""
        return self.game_state[0]

    def get_state(self):
        """Returns a tuple representing the board state."""
        return self.game_state

    def __str__(self):
        return str(self.board)
