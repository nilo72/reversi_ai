#!/usr/bin/env python3
from sys import argv
import run_game
from prop_parse import prop_parse

AMOUNT = 50


def main():

    input_args = prop_parse(argv)
    input_args['weights_file'] = 'neural/q_weights'

    summary = []
    for i in range(1, AMOUNT + 1):
        input_args['weights_num'] = i
        print('testing weights {}'.format(i))
        result = run_game.main(**input_args)
        print('weight {} black won {:.2f}'.format(i, result['Black']))
        summary.append(result)

    for index, result in enumerate(summary):
        print('weight {} black won {:.2f}%'.format(index, result['Black']))


if __name__ == '__main__':
    main()
