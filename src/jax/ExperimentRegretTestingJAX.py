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
from jax import vmap



def sample_regrets(strategy_profile, normal_game, N, K):
    # sample from sigma
    # sigma is a matrix of shape (N, K)

    return  jnp.array([[ expected_payoff_options(n, jnp.concatenate([strategy_profile[:n], strategy_profile[n+1:]], axis=0), normal_game)[k]
                         - expected_payoff(strategy_profile, normal_game)[n] for k in range(K)] for n in range(N)])
                
def sample_strategy(key, probs):

    choice = random.choice(key, 2, p=probs)
    return jnp.eye(2)[choice]



def play_game(sigma, normal_game,M, key):
    """
    Simulate a game with given mixed strategies and calculate regret."""
    parallel_avg_regret_fn = vmap(sample_regrets, in_axes=(0, None, None, None))


    for _ in tqdm.tqdm(range(M)):
        key, subkey = random.split(key)
        subkeys = random.split(subkey, T * N).reshape(T, N, 2)
        strategy_profile = jax.vmap(lambda t_keys: jax.vmap(sample_strategy)(t_keys, sigma))(subkeys.reshape(T, N, 2))
        result = parallel_avg_regret_fn( strategy_profile, game, N, action_space)
        average_regret = result.mean(axis=0)
        for n in range (N):
            max_regret_agent = jnp.max(average_regret[n])  # Ensure regret is non-negative
            #print("first sigma", sigma)
            if max_regret_agent >= rho:
                #print("Randomly select")
                key, subkey = random.split(key)
                sample_sigma = initialize_strategies_simple(subkey, N, action_space)
                sigma.at[n].set(sample_sigma[n])
                
            elif max_regret_agent < rho:
                if np.random.rand() <= 1-_lambda:
                    #print("Select best response")
                    None
                else:
                    #print("Randomly select")
                    key, subkey = random.split(key)
                    sample_sigma = initialize_strategies_simple(subkey, N, action_space)
                    sigma.at[n].set(sample_sigma[n])
                    
    print(sigma)
    print(expected_payoff(sigma, normal_game))
    #print(is_nash_equilibrium(sigma, normal_game, epsilon=.1))
    return result

    
    




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
    T = 10000
    _lambda = 0.01
    M = 1000

    # Init Agents
    # Initialize parameters
    action_space = 2
    
    # Create initial random key
    key = random.PRNGKey(10)
    
    agents_sigma = initialize_strategies_simple(key, N, action_space)



    print("Agents' mixed strategies:", agents_sigma)
    #expected_payoffs = expected_payoff(agents_sigma, game)
    #print("Expected Payoffs:", expected_payoffs)

    regret = play_game(agents_sigma, game,M, key)
    #print("Regret:", regret)








