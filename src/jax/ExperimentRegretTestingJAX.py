# Author: Francisco Aristi
# License: MIT License
# This code is part of the "ExperimentRegretTesting" project.

import jax.numpy as jnp
# randoms seed
#jnp.random.seed(42)
import numpy as np
from NormalFormGameJAX import *
from AgentsJAX import *
import tqdm
from jax import jit


def play_game(sigma, normal_game):
    """
    Simulate a game with given mixed strategies and calculate regret."""
    average_regret = jnp.zeros((N,2))  # Initialize average regret for each player
    for _ in tqdm.tqdm(range(110)):
        for n in range (N):
            for t in range (T):
                mixed_strategies_others = jnp.concatenate([sigma[:n], sigma[n+1:]], axis=0)

                for k in range(2):
                    average_regret = average_regret.at[n, k].set(
                        (t / (t + 1)) * (
                            expected_payoff_options(n, mixed_strategies_others, normal_game)[k]
                            - expected_payoff(sigma, normal_game)[n]
                        )
                    )
        for n in range (N):
            max_regret_agent = jnp.max(average_regret[n])  # Ensure regret is non-negative
            #print("first sigma", sigma)
            if max_regret_agent >= rho:
                #print("Randomly select")
                sample = sample_mix_strategies(2, 2)
                sigma[n] = jnp.array(2,sample)
                None
            elif max_regret_agent < rho:
                if np.random.rand() <= 1-_lambda:
                    #print("Select best response")
                    None
                else:
                    #print("Randomly select")
                    sample = sample_mix_strategies(2,2)
                    sigma[n] = jnp.array(sample)
                    None
    print(sigma)
    print(expected_payoff(sigma, normal_game))
    print(is_nash_equilibrium(sigma, normal_game, epsilon=.1))

    
    




if __name__ == "__main__":
    # Example usage

    game = create_normal_form_game((2, 2, 2))
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

    # Init Agents
    # Initialize parameters
    N = 3  # Number of agents
    action_space = 2
    
    # Create initial random key
    key = random.PRNGKey(42)
    
    # Method 1: Simple initialization
    key, subkey = random.split(key)
    agents_sigma = initialize_strategies_simple(subkey, N, action_space)

    print("Agents' mixed strategies:", agents_sigma)
    #expected_payoffs = expected_payoff(agents_sigma, game)
    #print("Expected Payoffs:", expected_payoffs)

    regret = play_game(agents_sigma, game)
    print("Regret:", regret)








