# Author: Francisco Aristi
# License: MIT License
# This code is part of the "ExperimentRegretTesting" project.

import numpy as np
# randoms seed
#np.random.seed(42)
 
from NormalFormGame import *
from Agents import *
import tqdm


def play_game(sigma, average_regret, normal_game):
    """
    Simulate a game with given mixed strategies and calculate regret."""


    for _ in tqdm.tqdm(range(M)):
        # evaluate regret

        for t in range (T):
            choices = [np.random.choice(2, p=agents_sigma[n]) for n in range(N)]
            strategy_profile = [np.eye(2)[choice] for choice in choices]
            for n in range (N):
                strategies_others = strategy_profile[:n] + strategy_profile[n+1:]
                for k in range(2):
                    average_regret[n][k] = (t/(t+1))*(expected_payoff_options(n, strategies_others, normal_game)[k] - expected_payoff(strategy_profile, normal_game)[n])
                    #print("average regret", average_regret[n][k])

        # update policy sigma  
        # max average regret 
        max_regret_agent = np.max(average_regret, axis=1)  # Update first column with max regret


        for n in range (N):  # Ensure regret is non-negative
            #print("first sigma", sigma)
            if max_regret_agent[n] >= epsilon**(2/3) :
                #print("hard random")
                sample = sample_mix_strategies(2)
                sigma[n] = np.array(sample)
            elif max_regret_agent[n] < epsilon**(2/3) and max_regret_agent[n] >= rho:
                #print("soft random")
                sample = uniform_distribution_L_ball(sigma[n])
                sigma[n] = sample
            elif max_regret_agent[n] < rho:
                if np.random.rand() <= 1-_lambda:
                    #print("maintain")
                    None
                else:
                    #print("soft random")
                    sample = uniform_distribution_L_ball(sigma[n])
                    sigma[n] = sample
    print("Strategies",sigma)
    print("Expected payoffs",expected_payoff(sigma, normal_game))
    result = is_nash_equilibrium(sigma, normal_game, epsilon=.1)
    print(is_nash_equilibrium(sigma, normal_game, epsilon=.1))
    return result

def epsilon_l(l):
    return 1/(2**l)  

def rho_l(l):
    return epsilon_l(l) + epsilon_l(l)**l

def lambda_l(l):
    return epsilon_l(l)**l

def T_l(l):
    ## ceil function
    return min(np.ceil(-(1/(2*epsilon_l(l)**(2*l)) * np.log(epsilon_l(l)**l))),1000)
    
def M_l(l):
    ## ceil function
    return min(2*np.ceil( np.log(2/epsilon_l(l))/ np.log(1/(1-lambda_l(l)) )),100)

def project_onto_simplex(v):
    """
    Projects a vector v onto the probability simplex.
    Ensures the result is non-negative and sums to 1.
    """
    v = np.asarray(v)
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rh = np.where(u + (1 - cssv) / (np.arange(n) + 1) > 0)[0][-1]
    theta = (cssv[rh] - 1) / (rh + 1)
    return np.maximum(v - theta, 0)

def uniform_distribution_L_ball(vector):
    """
    Sample approximately uniformly from the intersection of:
    - An L∞ ball of radius epsilon around `vector`, and
    - The probability simplex (non-negative, sums to 1).
    
    Parameters:
    -----------
    vector : np.array
        Center of the L∞ ball. Must lie in the probability simplex.
    epsilon : float
        Radius of the L∞ ball.
    n_samples : int
        Number of samples to generate.
    
    Returns:
    --------
    np.array
        Shape (n_samples, len(vector)) if n_samples > 1,
        Shape (len(vector),) if n_samples == 1
    """
    vector = np.asarray(vector)
    dim = len(vector)


    # Sample uniform noise in L∞ ball
    noise = np.random.uniform(-epsilon, epsilon, size=dim)
    perturbed = vector + noise
    # Project back to the simplex
    projected = project_onto_simplex(perturbed)
    


    return  np.array(projected)


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

    # Init Agents
    agents_sigma = [sample_mix_strategies(2) for _ in range(N)]
    average_regret = np.zeros((N,2))  # Initialize average regret for each player
    expected_payoffs = expected_payoff(agents_sigma, game)

    for l in range(1,10):
        rho = rho_l(l)
        epsilon = epsilon_l(l)
        _lambda = lambda_l(l)  
        M = int(M_l(l))

        T = int(T_l(l))

        print("rho epsilon, lambda, T,M",(rho, epsilon, _lambda, T, M))
        is_ne = play_game(agents_sigma, average_regret, game)

        # print all expected payoffs given options
        print("expected values by action",each_player_expected_payoff_options(agents_sigma, game))

        if is_ne:
            break






