#!/usr/bin/env python3
"""Verification script for RL Environment Error Handling improvements."""
import sys
import unittest
sys.path.insert(0, '/home/abhay/openspiel/open_spiel')

from open_spiel.python import rl_environment


class TestRLErrorHandling(unittest.TestCase):
    def test_custom_exceptions_exist(self):
        """Verify custom exception classes are defined."""
        t = "RLEnvironmentError"
        print(f"Checking {t}...", end="")
        self.assertTrue(hasattr(rl_environment, 'RLEnvironmentError'))
        self.assertTrue(hasattr(rl_environment, 'IllegalActionError'))
        self.assertTrue(hasattr(rl_environment, 'InvalidStateError'))
        self.assertTrue(hasattr(rl_environment, 'InvalidParameterError'))
        print(" OK")

    def test_discount_validation(self):
        """Verify discount parameter is validated."""
        print("Checking discount validation...", end="")
        # Valid discount
        env = rl_environment.Environment("tic_tac_toe", discount=0.5)
        
        # Invalid discount > 1.0
        with self.assertRaisesRegex(rl_environment.InvalidParameterError, "Discount must be in"):
            rl_environment.Environment("tic_tac_toe", discount=1.5)
            
        # Invalid discount < 0.0
        with self.assertRaisesRegex(rl_environment.InvalidParameterError, "Discount must be in"):
            rl_environment.Environment("tic_tac_toe", discount=-0.1)
        print(" OK")

    def test_illegal_action_error(self):
        """Verify IllegalActionError is raised instead of generic RuntimeError."""
        print("Checking IllegalActionError...", end="")
        env = rl_environment.Environment("tic_tac_toe", enable_legality_check=True)
        env.reset()
        # 0-8 are valid. 99 is illegal.
        with self.assertRaises(rl_environment.IllegalActionError):
             env.step([99])
        print(" OK")
        
    def test_invalid_parameter_actions(self):
        """Verify InvalidParameterError for wrong number of actions."""
        print("Checking action count validation...", end="")
        env = rl_environment.Environment("tic_tac_toe")
        env.reset()
        with self.assertRaisesRegex(rl_environment.InvalidParameterError, "Expected 1 action"):
            env.step([0, 1]) # Tic-tac-toe expects 1 action per step (for the current player)
        print(" OK")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
