# Author: Francisco Aristi
# License: MIT License
# This code is part of the "ExperimentRegretTesting" project.

import numpy as np
# randoms seed
#np.random.seed(42)
 
from NormalFormGame import *
from Agents import *
import tqdm


def play_game(sigma, normal_game):
    """
    Simulate a game with given mixed strategies and calculate regret."""
    average_regret = np.zeros((N,2))  # Initialize average regret for each player
    for _ in tqdm.tqdm(range(M)):

        #evaluate regret
        for t in range (T):
            choices = [np.random.choice(2, p=agents_sigma[n]) for n in range(N)]
            strategy_profile = [np.eye(2)[choice] for choice in choices]
            for n in range (N):
                # sample strategies for the players from sigma
                strategies_others = strategy_profile[:n] + strategy_profile[n+1:]

                for k in range(2):
                    average_regret[n][k] = (t/(t+1))*(expected_payoff_options(n, strategies_others, normal_game)[k] - expected_payoff(strategy_profile, normal_game)[n])
            
        # End period T:
        # update policy sigma
        # max average regret
        max_regret_agent = np.max(average_regret, axis=1)  #
        for n in range (N):
            #print("first sigma", sigma)
            if max_regret_agent[n] >= rho:
                #print("Randomly select")
                sample = sample_mix_strategies(2)
                sigma[n] = np.array(sample)
            elif max_regret_agent[n] < rho:
                if np.random.rand() <= 1-_lambda:
                    #print("Select best response")
                    None
                else:
                    #print("Randomly select")
                    sample = sample_mix_strategies(2)
                    sigma[n] = np.array(sample)

    print(expected_payoff(sigma, normal_game))
    print(is_nash_equilibrium(sigma, normal_game, epsilon=.1))
    return sigma

    
    




if __name__ == "__main__":
    # Example usage

    game = create_normal_form_game(2, 2, 2)
    N = game.shape[-1]  # Number of players
        # Set some payoffs
    game = set_payoff((0, 0, 0) ,game , [2, 2, -2])
    game = set_payoff((1, 0, 0),game, [-1, -1, 1])
    game = set_payoff((0, 1, 0),game, [-1, -1, 1])
    game = set_payoff((1, 1, 0),game,  [2, 2, -2])
    game = set_payoff((0, 0, 1), game, [-2, -2, 2])
    game = set_payoff((1, 0, 1),game,  [0,0,0])
    game = set_payoff((0, 1, 1),game,  [0,0,0])
    game = set_payoff((1, 1, 1),game,  [-2, -2, 2])
    rho = 0.1
    T = 1000
    _lambda = 0.01
    trials = 10
    M = 100

    # Init Agents
    agents_sigma = [sample_mix_strategies(2) for _ in range(N)]

    print("Agents' mixed strategies:", agents_sigma)
    expected_payoffs = expected_payoff(agents_sigma, game)
    print("Expected Payoffs:", expected_payoffs)
    print("expected values by action", each_player_expected_payoff_options(agents_sigma, game))

    sigma = play_game(agents_sigma, game)
    print("sigma:", sigma)








