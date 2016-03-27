#!/usr/bin/env python3
from agents.q_learning_agent import QLearningAgent
from game.reversi import Reversi
from util import *

def main():
    reversi = Reversi(4)
    test = QLearningAgent(reversi, BLACK)
    test.train(2000)

if __name__ == "__main__":
    main()
