BLACK = 1
WHITE = -1
EMPTY = 0


color_name = {BLACK: 'Black', WHITE: 'White'}
opponent = {BLACK: WHITE, WHITE: BLACK}

silent = False
def set_output(val):
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
