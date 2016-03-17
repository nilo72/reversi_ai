import numpy as np
from util import *

class Board:

    def __init__(self, dimens):
        self.dimens = dimens
        assert len(dimens) == 2 and dimens[0] == dimens[1]
        assert dimens[0] >= 4 and dimens[0] % 2 == 0
        self.board = np.zeros(self.dimens, dtype=np.int8)
        self.black_stones = 0
        self.white_stones = 0

    def init_starting_position(self):
        # place the 4 starting stones in the center.
        midpoint = self.dimens[0] / 2
        lower = midpoint - 1
        higher = midpoint
        self.board = np.zeros(self.dimens, dtype=np.int8)
        self.board[lower, lower] = BLACK
        self.board[higher, lower] = WHITE
        self.board[lower, higher] = WHITE
        self.board[higher, higher] = BLACK

        self.black_stones = 2
        self.white_stones = 2

    def place_stone_at(self, color, x, y):
        self.board[y, x] = color
        if color == WHITE:
            self.white_stones += 1
        elif color == BLACK:
            self.black_stones += 1

    def flip_stone(self, x, y):
        piece = self.piece_at(x, y)
        if piece == WHITE:
            self.white_stones -= 1
            self.place_stone_at(BLACK, x, y)
        elif piece == BLACK:
            self.black_stones -= 1
            self.place_stone_at(WHITE, x, y)
        else:
            raise TypeError

    def get_stone_counts(self):
        return self.black_stones, self.white_stones

    def is_full(self):
        return self.black_stones + self.white_stones == (self.dimens[0] ** 2)

    def get_dimensions(self):
        return self.dimens[0]

    def get_board(self):
        """Return the raw board.  Try to avoid this if you can use the other getters."""
        return self.board

    def is_in_bounds(self, x, y):
        return 0 <= x < self.dimens[0] and 0 <= y < self.dimens[0]

    def piece_at(self, x, y):
        return self.board[y, x]

    def __str__(self):
        result = ''
        height, width = np.shape(self.board)
        for y in range(height - 1, -1, -1):
            result += str(y) + ' '
            for x in range(width):
                if self.board[y, x] == WHITE:
                    result += 'O'
                elif self.board[y, x] == BLACK:
                    result += 'X'
                elif self.board[y, x] == EMPTY:
                    result += '-'
                else:
                    result += '?'
            result += '\n'

        result += '  '
        for x in range(width):
            result += str(x)
        result += '\n'

        return result

    def __hash__(self):
        return hash(str(self.board))

    def __eq__(self, other):
        return self.board.all() == other.board.all()
