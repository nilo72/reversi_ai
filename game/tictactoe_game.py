import game
from game import Game
from copy import deepcopy
from tictactoe_board import TicTacToeBoard
from util import *
from agents.random_agent import RandomAgent
from cache_dict import CacheDict


class TicTacToeGame(Game):

    def __init__(self, **kwargs):
        super(TicTacToeGame, self).__init__()
        self.size = kwargs.get('size', 3)
        self.board = TicTacToeBoard(self.size)

        WhiteAgent = kwargs.get('WhiteAgent', RandomAgent)
        BlackAgent = kwargs.get('BlackAgent', RandomAgent)
        self.white_agent = WhiteAgent(self, WHITE, **kwargs)
        self.black_agent = BlackAgent(self, BLACK, **kwargs)
        self.game_state = None
        self.legal_cache = None
        make_silent(kwargs.get('silent', False))
        self.reset()

    def reset(self):
        """Reset the game to initial positions."""
        self.game_state = (self.board, BLACK)
        self.legal_cache = CacheDict()
        self.white_agent.reset()
        self.black_agent.reset()

    @staticmethod
    def apply_move(game_state, move):
        """Given a game_state (which includes info about whose turn it is) and an x,y
        position to place a piece, transform it into the game_state that follows this play."""

        # if move is None, then the player simply passed their turn
        if not move:
            game_state = (game_state[0], opponent[game_state[1]])
            return game_state

        x, y = move
        color = game_state[1]
        board = game_state[0]
        board.place_stone_at(color, x, y)
        game_state = (game_state[0], opponent[game_state[1]])
        return game_state

    def play_game(self):
        state = self.get_state()
        self.print_board(state)
        info_newline()

        while self.winner(state) is False:
            color = state[1]
            picked = self.agent_pick_move(state)
            state = self.next_state(state, picked)
            self.print_board(state)
            if not picked:
                info('{} had no moves and passed their turn.'.format(
                    color_name[color]))
            else:
                info('{} plays at {}'.format(color_name[color], str(picked)))
            info_newline()

        black_count = 1
        white_count = 2
        winner = BLACK if black_count > white_count else WHITE

        return winner, white_count, black_count

    def winner(self, game_state):
        t3_board,p = game_state
        board = t3_board.board
        n_rows = len(board)
        lft = [[0] * i for i in range(n_rows)]  # [[], [0], [0, 0], [0, 0, 0]]
        rgt = list(reversed(lft))

        transpositions = {
            'horizontal': board,
            'vertical': zip(*board),
            'diag_forw': zip(*[lft[i] + board[i] + rgt[i] for i in range(n_rows)]),
            'diag_back': zip(*[rgt[i] + board[i] + lft[i] for i in range(n_rows)]),
        }

        for direction, transp in transpositions.iteritems():
            for row in transp:
                s = ''.join(map(str, row))
                for player in range(1, 3):
                    if s.find(str(player) * 3) >= 0:
                        print 'player={0} direction={1}'.format(player, direction)
                        return True
        return False

    def legal_moves(self, game_state, force_cache=False):
        # Note: this is a very naive and inefficient way to find
        # all available moves by brute force.  I am sure there is a
        # more clever way to do this.  If you want better performance
        # from agents, this would probably be the first area to improve.
        if force_cache:
            return self.legal_cache.get(game_state)

        board = game_state[0]
        if board.is_full():
            return []

        cached = self.legal_cache.get(game_state)
        if cached is not None:
            return cached

        board_size = board.get_size()
        moves = []  # list of x,y positions valid for color

        for y in range(board_size):
            for x in range(board_size):
                if self.is_valid_move(game_state, x, y):
                    moves.append((x, y))

        self.legal_cache.update(game_state, moves)
        return moves

    def is_valid_move(self, state, x, y):
        board, color = self.game_state
        piece = board.board[y][x]
        if piece != EMPTY:
            return False
        return True

    @staticmethod
    def print_board(state):
        board = state[0]
        info(board)

    def get_state(self):
        """Returns a tuple representing the board state."""
        return self.game_state

    def get_board(self):
        """Return the board from the current game_state."""
        return self.game_state[0]

    def agent_pick_move(self, state):
        color = state[1]
        legal_moves = self.legal_moves(state)
        picked = None
        if color == WHITE:
            picked = self.white_agent.get_action(state, legal_moves)
        elif color == BLACK:
            picked = self.black_agent.get_action(state, legal_moves)
        else:
            raise ValueError

        if picked is None:
            return None
        elif picked not in legal_moves:
            info(str(picked) + ' is not a legal move! Game over.')
            quit()

        return picked

    def next_state(self, state, move):
        """Given a game_state and a position for a new piece, return a new game_state
                reflecting the change.  Does not modify the input game_state."""
        return self.apply_move(deepcopy(state), move)
