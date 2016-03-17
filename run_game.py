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

    # black_agent = monte_carlo_agent.MonteCarloAgent
    black_agent = monte_carlo_agent.MonteCarloAgent
    white_agent = human_agent.HumanAgent
    # white_agent = test_carlo_agent.MonteCarloAgent
    # black_agent = human_agent.HumanAgent
    # white_agent = test_carlo_agent.MonteCarloAgent
    board_size = (8, 8)

    summary = []
    white_wins = 0
    black_wins = 0
    now = time.time()
    for t in range(amount):
        print('starting game {} of {}'.format(t, amount))
        reversi = Reversi(board_size, black_agent, white_agent, black_time=10, white_time=10)
        winner, white_score, black_score = reversi.play_game()
        if winner == WHITE:
            white_wins += 1
        elif winner == BLACK:
            black_wins += 1
        print('game {} complete.'.format(t))
        message = '{} wins! {}-{}'.format(color_name[winner], white_score, black_score)
        summary.append(message)

    print('time: {} minutes'.format((time.time() - now) / 60))


    print('summary: {} games played'.format(len(summary)))
    for each in summary:
        print(each)
    print('Black won {}%'.format(black_wins / (black_wins + white_wins) * 100))

if __name__ == '__main__':
    main()
