import jax
import jax.numpy as jnp
from jax import random
from jax import jit
from functools import partial

# Solution 1: Use static_argnums to tell JAX that action_space is static

def sample_mix_strategies(key, action_space):
    """
    Sample mixed strategies from a Dirichlet distribution.
    
    Args:
        key: JAX random key
        action_space: Number of actions/strategies (static argument)
        
    Returns:
        Array of probabilities that sum to 1
    """
    alpha = jnp.ones((action_space,))
    return random.dirichlet(key, alpha)



def initialize_strategies_simple(key, N, action_space=2):
    """Initialize strategies for N agents."""
    agents_sigma = []
    
    for i in range(N):
        key, subkey = random.split(key)
        strategy = sample_mix_strategies(subkey, action_space)  # Note: key first, then action_space
        agents_sigma.append(strategy)
    
    return jnp.stack(agents_sigma)


