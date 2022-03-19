"""
Tests for optimal_stopping_game.py.
"""

from absl.testing import absltest
from open_spiel.python.games import optimal_stopping_game
from open_spiel.python.games.optimal_stopping_game_config import OptimalStoppingGameConfig
import pyspiel


class OptimalStoppingGameTest(absltest.TestCase):

  def test_default_param(self) -> None:
    """
    Check the game can be converted to a turn-based game.

    :return: None
    """
    game = pyspiel.load_game("python_optimal_stopping_game")
    self.assertEqual(game.config.p, 0.001)

  def test_non_default_param_from_string(self) -> None:
    """
    Check params can be given through the string loading.

    :return: None
    """
    game = pyspiel.load_game(
        "python_optimal_stopping_game(p=0.5)")
    self.assertEqual(game.config.p, 0.5)

  def test_non_default_param_from_dict(self):
    """Check params can be given through a dictionary."""
    game = pyspiel.load_game("python_optimal_stopping_game",
                             {"p": 0.75})
    self.assertEqual(game.config.p, 0.75)

  def test_game_as_turn_based(self) -> None:
    """
    Check the game can be converted to a turn-based game.

    :return: None
    """
    game = pyspiel.load_game("python_optimal_stopping_game")
    turn_based = pyspiel.convert_to_turn_based(game)
    pyspiel.random_sim_test(
        turn_based, num_sims=10, serialize=False, verbose=True)

  def test_game_as_turn_based_via_string(self) -> None:
    """
    Check the game can be created as a turn-based game from a string.

    :return: None
    """
    game = pyspiel.load_game(
        "turn_based_simultaneous_game(game=python_optimal_stopping_game())"
    )
    pyspiel.random_sim_test(game, num_sims=10, serialize=False, verbose=True)

  def test_game_from_cc(self) -> None:
    """
    Runs our standard game tests, checking API consistency.

    :return: None
    """
    game = pyspiel.load_game("python_optimal_stopping_game")
    pyspiel.random_sim_test(game, num_sims=10, serialize=False, verbose=True)


if __name__ == "__main__":
  absltest.main()
