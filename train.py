#!/usr/bin/env python3
import time
from agents.q_learning_agent import QLearningAgent
from game.reversi import Reversi
from math import floor
from util import *

import sys

SNAPSHOT_AMNT = 100  # this frequently, save a snapshot of the states
STOP_EXPLORING = 0.60  # after how many games do we set epsilon to 0?


def main():
    amount = 4
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        amount = int(sys.argv[1])

    reversi = Reversi(size=8, WhiteAgent=QLearningAgent, BlackAgent=QLearningAgent, silent=True, learning_enabled=True)
    epsilon = 1.0
    end_exploration = max(1, floor(amount * STOP_EXPLORING))
    print('exploration will halt at {} games.'.format(end_exploration))

    start = time.time()
    try:
        for i in range(1, amount + 1):
            print('playing game {}/{} ({:2.2f}%)'.format(i, amount, i * 100 / amount))
            reversi.white_agent.set_epsilon(epsilon)
            reversi.black_agent.set_epsilon(epsilon)
            reversi.black_agent.set_replay_len(i)
            reversi.white_agent.set_replay_len(i)
            reversi.play_game()

            if i % SNAPSHOT_AMNT == 0:
                amnt = i / SNAPSHOT_AMNT
                reversi.white_agent.save_weights('_' + str(amnt))
                reversi.black_agent.save_weights('_' + str(amnt))

            # epsilon begins at 1, reaches 0 at 80% game completion
            epsilon -= (1 / end_exploration)
            epsilon = max(epsilon, 0)
    except KeyboardInterrupt:
        print('Stopping.  Will save weights before quitting.')

    seconds = time.time() - start
    print('time: {:.2f} minutes. per game: {:.2f}ms.'.format(seconds / 60.0, (seconds / float(i)) * 1000.0))
    reversi.white_agent.save_weights('')
    reversi.black_agent.save_weights('')


if __name__ == "__main__":
    main()
