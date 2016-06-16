from util import color_name
DEFAULT_DIR = 'neural/'
MODEL = DEFAULT_DIR + 'model_file.json'
WEIGHTS = DEFAULT_DIR + 'weights_file'
DATA = DEFAULT_DIR + 'results.txt'


def weights_filename(color):
    return '{}_{}.hd5'.format(WEIGHTS, color_name[color])
