#!/usr/bin/env python3
from agents.q_learning_agent import QLearningAgent
from game.reversi import Reversi
from math import floor
from util import *

import sys

SNAPSHOT_AMNT = 5  # this frequently, save a snapshot of the states

def main():
    amount = 5000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
            amount = int(sys.argv[1])

    reversi = Reversi(size=8, WhiteAgent=QLearningAgent, BlackAgent=QLearningAgent, silent=True, learning_mode=True, weights_file='8x8_duel_network/q_weights')
    epsilon = 1.0
    end_exploration = floor(amount * 0.80)
    for i in range(1, amount + 1):
        print('playing game {}/{} ({:2.2f}%)'.format(i, amount, i * 100 / amount))
        reversi.white_agent.set_epsilon(epsilon)
        reversi.black_agent.set_epsilon(epsilon) 
        reversi.reset()
        reversi.play_game()

        if i % SNAPSHOT_AMNT == 0:
            amnt = i / SNAPSHOT_AMNT 
            reversi.white_agent.save_weights('_' + str(amnt))
            reversi.black_agent.save_weights('_' + str(amnt))

        # epsilon begins at 1, reaches 0 at 80% game completion
        epsilon -= (1 / end_exploration)
        epsilon = max(epsilon, 0)

    reversi.white_agent.save_weights('')
    reversi.black_agent.save_weights('')


if __name__ == "__main__":
    main()
