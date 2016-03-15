import random
import pdb
from game.board import Board, BLACK, WHITE, EMPTY
from agents.human_agent import HumanAgent
from agents.random_agent import RandomAgent


color_name = {BLACK: 'Black', WHITE: 'white'}


class Reversi:
    """This class enforces the rules of the game of Reversi."""

    def __init__(self, dimens, BlackAgent=RandomAgent, WhiteAgent=RandomAgent):
        self.board = Board(dimens)
        self.board.init_starting_position()
        self.dimensions = self.board.get_dimensions()

        self.white_agent = WhiteAgent()
        self.black_agent = BlackAgent()

        # black goes first
        self.turn = BLACK
        self.play_game()

    def play_game(self):
        while not self.isWon():
            print(str(self.board))
            self.make_move(self.turn)
            if self.turn == WHITE:
                self.turn = BLACK
            elif self.turn == BLACK:
                self.turn = WHITE
        print(str(self.board))

        # figure out who won
        white_count = 0
        black_count = 0
        for y in range(self.dimensions):
            for x in range(self.dimensions):
                piece = self.board.piece_at(x, y)
                if piece == WHITE:
                    white_count += 1
                elif piece == BLACK:
                    black_count += 1

        print('Black: {}. White: {}'.format(black_count, white_count))

        if white_count > black_count:
            print('White wins!!')
        elif black_count > white_count:
            print('Black wins!!')
        else:
            print('Tie game!!')

    def make_move(self, color):
        picked = None
        legal_moves = self.get_valid_moves(color)
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
                print('{} had no moves, and passed their turn.'.format(
                    color_name[color]))
                return
            elif picked not in legal_moves:
                print(str(picked) + ' is not a legal move!')

        self.apply_move(color, picked[0], picked[1])
        print('{} placed stone at {}'.format(color_name[color], str(picked)))

    def get_valid_moves(self, color):
        # pdb.set_trace()
        board_size = self.board.get_dimensions()
        moves = []  # list of x,y positions valid for color

        for y in range(board_size):
            for x in range(board_size):
                if self.is_valid_move(color, x, y):
                    moves.append((x, y))

        return moves

    def is_valid_move(self, color, x, y):
        piece = self.board.piece_at(x, y)
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

                while self.board.is_in_bounds(xp, yp) and self.board.piece_at(xp, yp) == opponent:
                    distance += 1
                    yp = (distance * dy) + y
                    xp = (distance * dx) + x

                if distance > 1 and self.board.is_in_bounds(xp, yp) and self.board.piece_at(xp, yp) == color:
                    return True
        return False

    def apply_move(self, color, x, y):
        """Apply the input move.  Caller is responsible for making sure
        it is a valid move!"""

        self.board.place_stone_at(color, x, y)

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
                while self.board.is_in_bounds(xp, yp) and self.board.piece_at(xp, yp) == opponent:
                    flip_candidates.append((xp, yp))
                    distance += 1
                    yp = (distance * dy) + y
                    xp = (distance * dx) + x

                if distance > 1 and self.board.is_in_bounds(xp, yp) and self.board.piece_at(xp, yp) == color:
                    to_flip.extend(flip_candidates)

        for each in to_flip:
            self.board.place_stone_at(color, each[0], each[1])

    def isWon(self):
        """The game is over when neither player can make a move."""
        white_moves = self.get_valid_moves(WHITE)
        black_moves = self.get_valid_moves(BLACK)

        return len(white_moves) == 0 and len(black_moves) == 0

    def get_state(self):
        """Returns a tuple representing the board state."""

        # gamestate is defined by the board and whose turn it is
        return (self.board, self.turn)
