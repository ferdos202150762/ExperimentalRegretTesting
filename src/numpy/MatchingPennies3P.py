# Francisco
# Matching Pennies 3 Players Game Example
from NormalFormGame import *
import numpy as np

if __name__ == "__main__":
    # Create a 3-player game with 2, 3, and 2 strategies respectively
    # H is 0 and T is 1 for each player. 
    game = create_normal_form_game(2, 2, 2)

    # Set some payoffs
    game = set_payoff((0, 0, 0) ,game , [2, 2, -2])
    game = set_payoff((1, 0, 0),game, [-1, -1, 1])
    game = set_payoff((0, 1, 0),game, [-1, -1, 1])
    game = set_payoff((1, 1, 0),game,  [2, 2, -2])
    game = set_payoff((0, 0, 1), game, [-2, -2, 2])
    game = set_payoff((1, 0, 1),game,  [0,0,0])
    game = set_payoff((0, 1, 1),game,  [0,0,0])
    game = set_payoff((1, 1, 1),game,  [-2, -2, 2])

    # Mixed strategies (probability distributions) H:0 and T:1 for each player
    mixed_strategies = [
        np.array([1.0, 0.0]),   # Player 1's mixed strategy
        np.array([0.2, 0.8]),   # Player 2's mixed strategy
        np.array([0.4, 0.6])    # Player 3's mixed strategy
    ]

    # Calculate expected payoffs for the mixed strategies
    expected_payoffs = expected_payoff(mixed_strategies, game)
    print("Expected Payoffs:", expected_payoffs)
    # Calculate expected payoffs for each strategy given play from other players
    fixed_player = 0  # Player 1 is fixed
    mixed_strategies_others = mixed_strategies[:fixed_player] + mixed_strategies[fixed_player+1:]
    expected_payoffs_options = expected_payoff_options(fixed_player, mixed_strategies_others, game)
    print("Expected Payoffs Options for Player 1:", expected_payoffs_options)
    # Calculate expected payoffs for player 2 for each strategy given play from other players
    fixed_player = 1  # Player 2 is fixed
    mixed_strategies_others = mixed_strategies[:fixed_player] + mixed_strategies[fixed_player+1:]
    expected_payoffs_options = expected_payoff_options(fixed_player, mixed_strategies_others, game)
    print("Expected Payoffs Options for Player 2:", expected_payoffs_options)
    # Calculate expected payoffs for player 3 for each strategy given play from other players
    fixed_player = 2  # Player 3 is fixed
    mixed_strategies_others = mixed_strategies[:fixed_player] + mixed_strategies[fixed_player+1:]
    expected_payoffs_options = expected_payoff_options(fixed_player, mixed_strategies_others, game)
    print("Expected Payoffs Options for Player 3:", expected_payoffs_options)

    # Check if the mixed strategies form a Nash equilibrium
    is_nash = is_nash_equilibrium(mixed_strategies, game)
    print("Is Nash Equilibrium:", is_nash)