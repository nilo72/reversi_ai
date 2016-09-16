from util import *

WHITE_PIECE = 'O'
BLACK_PIECE = 'X'


class TicTacToeBoard(object):

    def __init__(self, size):
        self.size = size
        assert size >= 3
        self.board = [[EMPTY for _ in range(self.size)]
                      for _ in range(self.size)]
        self.black_stones = 0
        self.white_stones = 0

    def __str__(self):
        result = ''
        for y in range(self.size - 1, -1, -1):
            result += str(y) + '| '
            for x in range(self.size):
                if self.board[y][x] == WHITE:
                    result += WHITE_PIECE + ' '
                elif self.board[y][x] == BLACK:
                    result += BLACK_PIECE + ' '
                elif self.board[y][x] == EMPTY:
                    result += '- '
                else:
                    result += '? '
            result += '\n'

        result += '  '
        for x in range(self.size):
            result += '--'
        result += '\n'
        result += '   '
        for x in range(self.size):
            result += str(x) + ' '

        return result

    def is_full(self):
        return self.black_stones + self.white_stones == (self.size ** 2)

    def get_size(self):
        return self.size