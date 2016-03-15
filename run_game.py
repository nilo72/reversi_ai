#!/usr/bin/env python3
from sys import argv
from game.reversi import Reversi
from agents import *

def main():
    amount = 1
    if len(argv) > 1 and argv[1].isdigit():
        amount = int(argv[1])
    for _ in range(amount):
        reversi = Reversi((8,8), random_agent.RandomAgent, random_agent.RandomAgent)

if __name__ == '__main__':
    main()
