# Author: Francisco Aristi
# License: MIT License
# Normal form game mehtod implementation

import jax.numpy as jnp
from jax import jit
from functools import partial


def create_normal_form_game( *strategy_counts):
    """
    Initialize N-player game.
    Args:
        *strategy_counts: Number of strategies for each player
    """
    n_players = len(strategy_counts)
    strategy_counts = strategy_counts
    # Shape: (*strategy_counts, n_players)
    return jnp.zeros(strategy_counts + (n_players,))

@jit
def set_payoff(strategy_profile, payoffs, payoff):
    """Set payoffs for a given strategy profile."""
    payoffs = payoffs.at[strategy_profile].set(jnp.array(payoff))
    return payoffs
@jit
def get_payoff( player, strategy_profile, payoffs):
    """Get payoff for specific player given strategy profile."""
    return payoffs[strategy_profile][player]
@jit
def get_all_payoffs( strategy_profile, payoffs):
    """Get all players' payoffs for strategy profile."""
    return payoffs[strategy_profile]

@jit
def expected_payoff( mixed_strategies, payoffs):
    """
    Calculate expected payoff for a player given mixed strategy profile.
    Args:
        player: Player index (0 to n_players-1)
        mixed_strategies: List of probability distributions, one per player
    """
    # Compute expected payoff using tensor operations
    payoff_tensor = payoffs
    
    # Multiply by each player's mixed strategy

    for mixed_strategy in mixed_strategies:

        payoff_tensor = jnp.tensordot(payoff_tensor, mixed_strategy, axes=(0, 0))

    
    return payoff_tensor


@partial(jit, static_argnums=(0,))
def expected_payoff_options( fixed_player, mixed_strategies_other, payoffs):
    """
    Calculate expected payoff for fixed player for each strategy.
    """
    payoff_tensor = payoffs[...,fixed_player]
    
    # Move fixed player's axis to position 0
    n_players = len(payoff_tensor.shape)
    axes_order = list(range(n_players))
    payoff_tensor = jnp.moveaxis(payoff_tensor, fixed_player, axes_order[-1])
    
    # Now contract all other dimensions (which are at positions 1, 2, ...)
    for mixed_strategy in mixed_strategies_other:
        # Always contract position 1 (since 0 is our fixed player)
        payoff_tensor = jnp.tensordot(payoff_tensor, mixed_strategy, axes=(0, 0))
    
    return payoff_tensor



def is_nash_equilibrium( mixed_strategies, payoffs , epsilon=1e-8):
    """
    Verify if a mixed strategy profile is a Nash equilibrium.
    """
    n_players = len(mixed_strategies)
    
    for player in range(n_players):
        # Get player's expected payoffs for each pure strategy
        # given other players' mixed strategies
        other_strategies = mixed_strategies[:player] + mixed_strategies[player+1:]
        expected_payoffs = expected_payoff_options(player, other_strategies, payoffs)
    
        # Find strategies player is actually mixing (prob > 0)
        mixed_strat = mixed_strategies[player]
        used_strategies = np.where(mixed_strat > epsilon)[0]
        unused_strategies = np.where(mixed_strat <= epsilon)[0]
        
        if len(used_strategies) > 0:
            # Check 1: All used strategies must yield same payoff
            used_payoffs = expected_payoffs[used_strategies]
            if not np.allclose(used_payoffs, used_payoffs[0], atol=epsilon):
                return False  # Not indifferent between mixed strategies
            
            # Check 2: Unused strategies can't be better
            if len(unused_strategies) > 0:
                max_unused = np.max(expected_payoffs[unused_strategies])
                if max_unused > used_payoffs[0] + epsilon:
                    return False  # Player could improve by deviating

    return True

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
        jnp.array([1.0, 0.0]),   # Player 1's mixed strategy
        jnp.array([0.2, 0.8]),   # Player 2's mixed strategy
        jnp.array([0.4, 0.6])    # Player 3's mixed strategy
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