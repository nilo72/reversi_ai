#!/usr/bin/env python3
from sys import argv
import run_game
from prop_parse import prop_parse

AMOUNT = 50

def main():

    input_args = prop_parse(argv)
    input_args['weights_file'] = 'net_weights/q_weights'

    summary = []
    for i in range(1, AMOUNT + 1):
        input_args['weights_num'] = i
        print('testing weights {}'.format(i))
        summary.append(run_game.main(**input_args))

    for index, result in enumerate(summary):
        print('weight {} won {:.2f}%'.format(index, result))


if __name__ == '__main__':
    main()
