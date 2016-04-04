from agents.agent import Agent


class HumanAgent(Agent):
    """This agent is controlled by a human, who inputs moves via stdin."""

    def __init__(self, reversi, color, **kwargs):
        self.reversi = reversi
        self.color = color

    def reset(self):
        pass

    def observe_win(self, winner):
        pass

    def get_action(self, game_state, legal):
        if not legal:
            return None
        choice = None
        while True:
            raw_choice = input('Enter a move x,y: ')
            if raw_choice == 'pass':
                return None
            if raw_choice == 'exit' or raw_choice == 'quit':
                quit()
            if len(raw_choice) != 3:
                print('input must be 3 long, formatted x,y')
                continue
            if raw_choice[1] != ',':
                print('comma separator not found.')
                continue
            if not raw_choice[0].isdigit() or not raw_choice[2].isdigit():
                print('couldn\'t determine x,y from your input.')
                continue
            choice = (int(raw_choice[0]), int(raw_choice[2]))
            if choice not in legal:
                print('not a legal move. try again.')
                continue
            else:
                break

        return choice
