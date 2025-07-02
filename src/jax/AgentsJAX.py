import jax
import jax.numpy as jnp
from jax import random
from jax import jit
from functools import partial

# Solution 1: Use static_argnums to tell JAX that action_space is static
@partial(jax.jit, static_argnums=1)
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


# Alternative version with configurable concentration parameter

def sample_mix_strategies_alpha(key, action_space, alpha=1.0):
    """
    Sample mixed strategies with configurable concentration.
    
    Args:
        key: JAX random key
        action_space: Number of actions/strategies
        alpha: Concentration parameter (default=1.0 for uniform)
               - alpha < 1: favor sparse strategies
               - alpha = 1: uniform over simplex
               - alpha > 1: favor balanced strategies
    """
    alpha_array = jnp.ones(action_space) * alpha
    return random.dirichlet(key, alpha_array)


# Version that returns multiple samples at once (vectorized)

def sample_mix_strategies_batch(key, action_space, num_samples):
    """
    Sample multiple mixed strategies at once.
    
    Args:
        key: JAX random key
        action_space: Number of actions/strategies
        num_samples: Number of strategy samples to generate
        
    Returns:
        Array of shape (num_samples, action_space)
    """
    alpha = jnp.ones(action_space)
    
    # Generate multiple samples efficiently
    keys = random.split(key, num_samples)
    samples = jax.vmap(lambda k: random.dirichlet(k, alpha))(keys)
    
    return samples


# Helper function for use in your game loop

def update_strategy_with_sample(sigma, player_idx, key, action_space):
    """
    Update a specific player's strategy with a new sample.
    
    Args:
        sigma: Current strategies array (num_players, action_space)
        player_idx: Index of player to update
        key: JAX random key
        action_space: Number of actions
        
    Returns:
        Updated strategies array
    """
    new_strategy = sample_mix_strategies(key, action_space)
    return sigma.at[player_idx].set(new_strategy)



# Option 1: Simple fix - use proper keys
def initialize_strategies_simple(key, N, action_space=2):
    """Initialize strategies for N agents."""
    agents_sigma = []
    
    for i in range(N):
        key, subkey = random.split(key)
        strategy = sample_mix_strategies(subkey, action_space)  # Note: key first, then action_space
        agents_sigma.append(strategy)
    
    return jnp.stack(agents_sigma)


# Option 2: Vectorized JAX approach (faster)
def initialize_strategies_vectorized(key, N, action_space=2):
    """Initialize strategies for N agents using JAX vectorization."""
    # Split key into N subkeys
    keys = random.split(key, N)
    
    # Vectorize sample_mix_strategies over the keys
    sample_fn = jax.vmap(lambda k: sample_mix_strategies(k, action_space))
    
    return sample_fn(keys)


# Option 3: Direct batch sampling
@jax.jit
def initialize_strategies_batch(key, N, action_space=2):
    """Initialize all strategies at once."""
    alpha = jnp.ones(action_space)
    
    # Generate N samples from Dirichlet distribution
    keys = random.split(key, N)
    strategies = jax.vmap(lambda k: random.dirichlet(k, alpha))(keys)
    
    return strategies


# Your main code should look like this:
if __name__ == "__main__":
    # Initialize parameters
    N = 3  # Number of agents
    action_space = 2
    
    # Create initial random key
    key = random.PRNGKey(42)
    
    # Method 1: Simple initialization
    key, subkey = random.split(key)
    agents_sigma = initialize_strategies_simple(subkey, N, action_space)
    print("Method 1 - Simple:", agents_sigma)
    
    # Method 2: Vectorized initialization (recommended)
    key, subkey = random.split(key)
    agents_sigma = initialize_strategies_vectorized(subkey, N, action_space)
    print("\nMethod 2 - Vectorized:", agents_sigma)
    
    # Method 3: Batch initialization (fastest)
    key, subkey = random.split(key)
    agents_sigma = initialize_strategies_batch(subkey, N, action_space)
    print("\nMethod 3 - Batch:", agents_sigma)
    
    # Verify they sum to 1
    print("\nSums:", jnp.sum(agents_sigma, axis=1))





# Complete example of how to use in your code:
def main():
    # Parameters
    N = 3  # Number of agents
    T = 100
    rho = 0.1
    _lambda = 0.5
    
    # Initialize random state
    key = random.PRNGKey(42)
    
    # Initialize agent strategies
    key, subkey = random.split(key)
    agents_sigma = initialize_strategies_vectorized(subkey, N, action_space=2)
    
    print("Agents' mixed strategies:", agents_sigma)
    
    # Continue with your game...
    # play_game(agents_sigma, game, N, T, rho, _lambda, key)
    

# Error-free version of what you were trying to do:
def your_original_intent_fixed():
    N = 3
    key = random.PRNGKey(42)
    
    # Original attempt: 
    # agents_sigma = jnp.stack([sample_mix_strategies(2, (2)) for _ in range(N)])
    
    # Fixed version:
    agents_sigma = []
    for _ in range(N):
        key, subkey = random.split(key)
        strategy = sample_mix_strategies(subkey, 2)  # key first, then action_space
        agents_sigma.append(strategy)
    
    agents_sigma = jnp.stack(agents_sigma)
    return agents_sigma