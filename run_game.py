#!/usr/bin/env python3
from sys import argv
import time
from game.reversi import Reversi
from agents import random_agent, monte_carlo_agent, human_agent, q_learning_agent
from util import *


def main():

    board_size = 4
    bot_time = 0.5  
    agent_args = {
        'BlackAgent': q_learning_agent.QLearningAgent,
        'WhiteAgent': random_agent.RandomAgent,
        'print': False,
        'white_time': bot_time,
        'black_time': bot_time,
        'episodes': 200
    }

    amount = 1
    if len(argv) > 1 and argv[1].isdigit():
        amount = int(argv[1])
    if len(argv) > 2 and argv[2].isdigit():
        agent_args['white_time'] = int(argv[2])
        agent_args['black_time'] = int(argv[2])
    if 'print' in argv:
        agent_args['print'] = True
    if 'white' in argv:
        agent_args['WhiteAgent'] = monte_carlo_agent.MonteCarloAgent
        agent_args['BlackAgent'] = human_agent.HumanAgent
    elif 'black' in argv:
        agent_args['BlackAgent'] = monte_carlo_agent.MonteCarloAgent
        agent_args['WhiteAgent'] = human_agent.HumanAgent

    for each in argv:
        if each.startswith('weights_file='):
            weights_file = each.split('weights_file=')[1]
            print('sending weights_file: {}'.format(weights_file))
            agent_args['weights_file'] = weights_file

    summary = []
    white_wins = 0
    black_wins = 0
    reversi = Reversi(board_size, **agent_args)
    start = time.time()
    for t in range(1, amount + 1):
        print('starting game {} of {}'.format(t, amount))
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
        reversi.reset()

    print('time: {} minutes'.format((time.time() - start) / 60))
    print('summary: {} games played'.format(len(summary)))
    for each in summary:
        print(each)
    print('Black won {}%'.format(black_wins / (black_wins + white_wins) * 100))

if __name__ == '__main__':
    main()
