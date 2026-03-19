#!/usr/bin/env python3
"""OpenSpiel Installation Test Script"""

import sys


def test_openspiel():
    """Test OpenSpiel installation and basic functionality"""
    print(" Testing OpenSpiel Installation")
    print("=" * 50)
    
    try:
        import pyspiel
        print(" pyspiel imported successfully")
        game = pyspiel.load_game("tic_tac_toe")
        state = game.new_initial_state()
        print(f" Game loaded: {game.get_type().short_name}")
        print(f" Players: {game.num_players()}")
        print(f" Legal actions: {len(state.legal_actions())}")
        print("\n OpenSpiel is working correctly!")
        return True
    except Exception as e:  # pragma: no cover - diagnostic script
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    success = test_openspiel()
    sys.exit(0 if success else 1)
