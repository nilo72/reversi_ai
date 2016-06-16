import numpy as np
cimport numpy as np
from cpython cimport bool
import math
import random
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

def info(message=None):
    if not silent:
        if not message:
            print()
        else:
            print(message)

def to_offset(move, size):
    x, y = move
    return y * size + x

def numpify(state):
    """Given a state (board, color) tuple, return the numpy
    version of the board's array."""
    board = state[0].board
    assert len(board) > 0
    size = len(board) * len(board[0])
    result = np.array(board)
    result = np.expand_dims(result, axis=0)
    result = np.expand_dims(result, axis=0)
    return result

cpdef bool is_in_bounds(int x, int y, int size):
    return 0 <= x < size and 0 <= y < size

def max_q_move(q_vals, legal_moves):
    """Given a list of moves and a q_val array, return the move with the highest q_val and the q_val."""
    # TODO: make third param for size, so we dont recompute each time
    if not legal_moves:
        return None, None
    else:
        best_q = -float('inf')
        best_move = []
        size = int(math.sqrt(len(q_vals[0])))
        assert size ** 2 == len(q_vals[0])
        for move in legal_moves:
            offset = to_offset(move, size)
            val = q_vals[0][offset]
            # info('{}: {}'.format(move, val))
            if val > best_q:
                best_q = val
                best_move = [move]
            elif best_q == val:
                best_move.append(move)

        return random.choice(best_move), best_q
