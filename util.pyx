import numpy as np
cimport numpy as np
from cpython cimport bool
BLACK = 1
WHITE = -1
EMPTY = 0


color_name = {BLACK: 'Black', WHITE: 'White'}
opponent = {BLACK: WHITE, WHITE: BLACK}

silent = False
def make_silent(val):
    assert val is True or val is False
    global silent
    silent = val

def info(message):
    if not silent:
        if not message:
            print()
        else:
            print(message)

def info_newline():
    if not silent:
        print()
        
def to_offset(move, size):
    x, y = move
    return y * size + x

def numpify(state):
    """Given a state (board, color) tuple, return the flattened numpy
    version of the board's array."""
    board = state[0].board
    assert len(board) > 0
    size = len(board) * len(board[0])
    return np.array(board).reshape(1, size)
    # return np.reshape(board, (1, size))

cpdef bool is_in_bounds(int x, int y, int size):
    return 0 <= x < size and 0 <= y < size
