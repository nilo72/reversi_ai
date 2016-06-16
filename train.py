#!/usr/bin/env python3
import time
from math import floor
from agents import QLearningAgent
from agents import RandomAgent
from game.reversi import Reversi
from util import *
from filenames import DATA, MODEL, weights_filename
import sys

SNAPSHOT_AMNT = 2000  # this frequently, save a snapshot of the states
STOP_EXPLORING = 0.5  # after how many games do we set epsilon to 0?
TEST_GAMES = 1000

BOARD_SIZE = 8


def main():
    # reset_output_file()
    amount = 40000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        amount = int(sys.argv[1])

    reversi = Reversi(size=BOARD_SIZE,
                      WhiteAgent=QLearningAgent,
                      BlackAgent=QLearningAgent,
                      silent=True,
                      training_enabled=True)

    end_exploration = max(1, floor(amount * STOP_EXPLORING))
    print('exploration will halt at {} games.'.format(end_exploration))

    start = time.time()
    try:
        for i in range(1, amount + 1):
            print('playing game {}/{} ({:3.2f}%) epsilon: {:.2f}'.format(
                i, amount, i * 100 / amount, reversi.black_agent.get_epsilon(
                )))
            reversi.play_game()
            reversi.reset()

            reversi.white_agent.decrement_epsilon(1.0 / end_exploration)
            reversi.black_agent.decrement_epsilon(1.0 / end_exploration)

            if i % SNAPSHOT_AMNT == 0:
                amnt = int(i / SNAPSHOT_AMNT)
                reversi.white_agent.save_weights(str(amnt))
                reversi.black_agent.save_weights(str(amnt))
                play_test_games()

    except KeyboardInterrupt:
        print('Stopping.  Will save weights before quitting.')

    seconds = time.time() - start
    print('time: {:.2f} minutes. per game: {:.2f}ms.'.format(seconds / 60.0, (
        seconds / float(i)) * 1000.0))
    reversi.white_agent.save_weights()
    reversi.black_agent.save_weights()


def reset_output_file():
    with open(DATA, 'w') as f:
        f.write('')


def play_test_games():
    print('playing test games...')
    wincount = 0
    testgame = Reversi(size=BOARD_SIZE,
                       WhiteAgent=RandomAgent,
                       BlackAgent=QLearningAgent,
                       minimax=False,
                       silent=True,
                       model_file=MODEL,
                       model_weights=weights_filename(BLACK), )
    for i in range(TEST_GAMES):
        print('playing test game {}/{}'.format(i, TEST_GAMES))
        winner, _, _ = testgame.play_game()
        print('winner: {}'.format(color_name[winner]))
        if winner == BLACK:
            wincount += 1

    winrate = 100.0 * wincount / TEST_GAMES
    result = '{:.2f}%\n'.format(winrate)
    print('result: {}'.format(result))
    with open(DATA, 'a') as f:
        f.write(str(winrate) + '\n')
    print('done with test games.')


if __name__ == "__main__":
    main()
