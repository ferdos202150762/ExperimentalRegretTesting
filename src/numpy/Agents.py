import numpy as np
from scipy.stats import dirichlet
import numba


def sample_mix_strategies(action_space):
    # sample from a dirichlet distribution on the number of strategies s
    sigma = dirichlet.rvs(np.ones(action_space))
    return sigma.squeeze(0)