#!/usr/bin/env python3
from sys import argv
import time
from game.reversi import Reversi
from agents import random_agent, monte_carlo_agent, human_agent, test_carlo_agent
from util import *


def main():
    amount = 1
    if len(argv) > 1 and argv[1].isdigit():
        amount = int(argv[1])

    board_size = (8, 8)
    bot_time = 10

    agent_args = {
        'BlackAgent': monte_carlo_agent.MonteCarloAgent,
        'WhiteAgent': random_agent.RandomAgent,
        'print': False,
        'white_time': bot_time,
        'black_time': bot_time
    }

    if 'print' in argv:
        agent_args['print'] = True
    for it in argv:
        if it.isdigit():
            agent_args['white_time'] = int(it)
            agent_args['black_time'] = int(it)

    summary = []
    white_wins = 0
    black_wins = 0
    start = time.time()
    for t in range(1, amount + 1):
        print('starting game {} of {}'.format(t, amount))
        reversi = Reversi(board_size, **agent_args)
        winner, white_score, black_score = reversi.play_game()
        if winner == WHITE:
            white_wins += 1
        elif winner == BLACK:
            black_wins += 1
        print('game {} complete.'.format(t))
        message = '{} wins! {}-{}'.format(
            color_name[winner], white_score, black_score)
        print(message)
        summary.append(message)

    print('time: {} minutes'.format((time.time() - start) / 60))
    print('summary: {} games played'.format(len(summary)))
    for each in summary:
        print(each)
    print('Black won {}%'.format(black_wins / (black_wins + white_wins) * 100))

if __name__ == '__main__':
    main()
