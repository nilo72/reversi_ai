# reversi-ai
An attempt to build multiple AI agents to play the game of reversi.

The ultimate goal is to create an agent that combines reinforcement learning with Monte Carlo tree search.
So far, the MCTS agent is implemented and working well.  A Q-learning agent is partly implemented and in progress.

To run, use run_game.py, or create your own top-level game runner.
You just have to create a Reversi object and pass it two agents of your choosing.

This is still very much an "in progress" project, so certain design aspects are still in flux.

Example of running using a HumanAgent (controlled by me) and the MonteCarloAgent:

    7 - - - - - - - - 
    6 - - - - - - - - 
    5 - - - - - - - - 
    4 - - - O X - - - 
    3 - - - X O - - - 
    2 - - - - - - - - 
    1 - - - - - - - - 
    0 - - - - - - - - 
      0 1 2 3 4 5 6 7 ```

    Enter a move x,y (or pass to pass): 5,3
    Black plays at (5, 3)
    7 - - - - - - - - 
    6 - - - - - - - - 
    5 - - - - - - - - 
    4 - - - O X - - - 
    3 - - - X X X - - 
    2 - - - - - - - - 
    1 - - - - - - - - 
    0 - - - - - - - - 
      0 1 2 3 4 5 6 7 

    (3, 2): (30/48)
    (5, 4): (15/31)
    (5, 2): (6/19)
    98 simulations performed.
    White plays at (3, 2)
    7 - - - - - - - - 
    6 - - - - - - - - 
    5 - - - - - - - - 
    4 - - - O X - - - 
    3 - - - O X X - - 
    2 - - - O - - - - 
    1 - - - - - - - - 
    0 - - - - - - - - 
      0 1 2 3 4 5 6 7 

    Enter a move x,y (or pass to pass): 2,1
    Black plays at (2, 1)
    7 - - - - - - - - 
    6 - - - - - - - - 
    5 - - - - - - - - 
    4 - - - O X - - - 
    3 - - - O X X - - 
    2 - - - X - - - - 
    1 - - X - - - - - 
    0 - - - - - - - - 
      0 1 2 3 4 5 6 7 

    (5, 4): (9/19)
    (5, 5): (17/28)
    (5, 2): (14/25)
    (6, 3): (13/23)
    (3, 1): (6/14)
    101 simulations performed.
    White plays at (5, 5)
    7 - - - - - - - - 
    6 - - - - - - - - 
    5 - - - - - O - - 
    4 - - - O O - - - 
    3 - - - O X X - - 
    2 - - - X - - - - 
    1 - - X - - - - - 
    0 - - - - - - - - 
      0 1 2 3 4 5 6 7
