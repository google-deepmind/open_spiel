
import numpy as np
import pyspiel
from open_spiel.python.algorithms import sequence_form_lp
from open_spiel.python import policy

def check_subgame_perfection():
    # Load Kuhn Poker
    game = pyspiel.load_game("kuhn_poker")
    
    # Solve the game
    val1, val2, pol0, pol1 = sequence_form_lp.solve_zero_sum_game(game)
    
    print(f"Game value: {val1}")

    # For debugging, let's see what the LPs look like or their solutions
    # We need access to the lps and solutions, so we might need to modify solve_zero_sum_game temporarily to return them or use internal access.
    # Since I just modified sequence_form_lp.py to print debug info, I'll just run it.
    
    # Check for an unreachable state in Kuhn Poker
    # In Kuhn Poker, if P0 checks and P1 bets, P0 folding is optimal for 'J'.
    # But what if some state is unreachable because P0 themselves made a move?
    # Actually, Kuhn Poker is small, let's just inspect some infostates.
    
    # Let's find infostates that have 0 reach probability for the player.
    # We'll use a custom game if needed, but Kuhn should suffice.
    
    # Check Player 0's policy
    print("\nPlayer 0 Policy:")
    for key in pol0.state_lookup.keys():
        p = pol0.policy_for_key(key)
        print(f"{key}: {p}")
        
    # In Kuhn, P0 has 3 infostates dealing (J), (Q), (K) at the start.
    # Initial moves are Check or Bet.
    # Equilibrium for P0 (approximately):
    # J: Check 100% (usually)
    # Q: Check 100%
    # K: Bet 1/3, Check 2/3 (or similar)
    
    # If P0 bets with 'J', it's a bluff.
    # But if the equilibrium says "never bet with Q", then P0 betting with Q is unreachable for P0.
    # Let's see what P1 does in that case.
    
    # Check Player 1's policy
    print("\nPlayer 1 Policy:")
    for key in pol1.state_lookup.keys():
        p = pol1.policy_for_key(key)
        print(f"{key}: {p}")

    # For subgame perfection, even if an infostate is unreachable, 
    # the choice should be optimal against the opponent's strategy.

if __name__ == "__main__":
    check_subgame_perfection()
