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

cpdef bool is_in_bounds(int x, int y, int size):
    return 0 <= x < size and 0 <= y < size
