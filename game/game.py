from abc import ABCMeta
import abc


class Game(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def play_game(self):
        pass

    @abc.abstractmethod
    def agent_pick_move(self, state):
        pass