# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for param_social_dilemma.py."""

from absl.testing import absltest
from absl.testing import parameterized

from open_spiel.python.games import param_social_dilemma
import pyspiel


class ParamSocialDilemmaTest(parameterized.TestCase):

  def test_default_params(self):
    game = pyspiel.load_game("python_param_social_dilemma")
    self.assertEqual(game.num_players(), 2)

  @parameterized.parameters(2, 3, 4, 6, 10)
  def test_variable_num_players(self, num_players):
    """Player count is configurable via the (conventional) `players` param."""
    game = pyspiel.load_game(
        "python_param_social_dilemma", {"players": num_players})
    self.assertEqual(game.num_players(), num_players)
    state = game.new_initial_state()
    self.assertEqual(state.current_player(), pyspiel.PlayerId.SIMULTANEOUS)
    self.assertLen(state.legal_actions(0), 2)

  def test_players_out_of_range_raises(self):
    with self.assertRaises(ValueError):
      pyspiel.load_game("python_param_social_dilemma", {"players": 1})
    with self.assertRaises(ValueError):
      pyspiel.load_game("python_param_social_dilemma", {"players": 11})

  def test_two_player_payoffs_match_classic_prisoners_dilemma(self):
    """N=2 case should reduce exactly to the standard 2x2 PD payoff matrix."""
    game = pyspiel.load_game("python_param_social_dilemma", {"players": 2})

    state = game.new_initial_state()
    state.apply_actions([param_social_dilemma.Action.COOPERATE,
                          param_social_dilemma.Action.COOPERATE])
    self.assertSequenceAlmostEqual(state.rewards(), [5, 5])

    state = game.new_initial_state()
    state.apply_actions([param_social_dilemma.Action.DEFECT,
                          param_social_dilemma.Action.DEFECT])
    self.assertSequenceAlmostEqual(state.rewards(), [1, 1])

    state = game.new_initial_state()
    state.apply_actions([param_social_dilemma.Action.DEFECT,
                          param_social_dilemma.Action.COOPERATE])
    self.assertSequenceAlmostEqual(state.rewards(), [10, 0])

  def test_n_player_payoffs_interpolate_linearly(self):
    """A defector's payoff should rise linearly with cooperating peers."""
    game = pyspiel.load_game("python_param_social_dilemma", {"players": 4})
    d = param_social_dilemma.Action.DEFECT
    c = param_social_dilemma.Action.COOPERATE

    # Lone defector among 3 cooperators (all 3 peers cooperate): P + (T-P).
    state = game.new_initial_state()
    state.apply_actions([d, c, c, c])
    self.assertAlmostEqual(state.rewards()[0], 10)  # = T

    # Lone defector among 3 defectors: P + (T-P)*0 = P.
    state = game.new_initial_state()
    state.apply_actions([d, d, d, d])
    self.assertAlmostEqual(state.rewards()[0], 1)  # = P

    # All cooperate: reward R for everyone.
    state = game.new_initial_state()
    state.apply_actions([c, c, c, c])
    self.assertSequenceAlmostEqual(state.rewards(), [5, 5, 5, 5])

  def test_deterministic_rewards_by_default(self):
    game = pyspiel.load_game("python_param_social_dilemma", {"players": 3})
    self.assertEqual(game.get_parameters()["reward_noise_std"], 0.0)
    state = game.new_initial_state()
    state.apply_actions([0, 1, 0])
    while state.current_player() == pyspiel.PlayerId.CHANCE:
      outcomes = state.chance_outcomes()
      # With no noise/dynamic-payoff phases active, only TERMINATION remains.
      self.assertLen(outcomes, 2)
      state.apply_action(outcomes[0][0])

  def test_stochastic_reward_noise(self):
    game = pyspiel.load_game(
        "python_param_social_dilemma",
        {"players": 3, "reward_noise_std": 4.0})
    state = game.new_initial_state()
    state.apply_actions([0, 1, 0])
    self.assertEqual(state.current_player(), pyspiel.PlayerId.CHANCE)
    outcomes = state.chance_outcomes()
    # 5 discretized noise outcomes.
    self.assertLen(outcomes, 5)
    self.assertAlmostEqual(sum(p for _, p in outcomes), 1.0)

  def test_dynamic_payoffs_requires_multiple_regimes(self):
    with self.assertRaises(ValueError):
      pyspiel.load_game(
          "python_param_social_dilemma",
          {"dynamic_payoffs": True, "payoff_regimes": "10 5 1 0"})

  def test_dynamic_payoffs_regime_can_switch(self):
    game = pyspiel.load_game(
        "python_param_social_dilemma", {
            "players": 2,
            "dynamic_payoffs": True,
            "payoff_regimes": "10 5 1 0 20 5 1 0",
            "payoff_change_prob": 1.0,  # Always switch, for determinism.
        })
    state = game.new_initial_state()
    self.assertEqual(state.regime_index(), 0)
    state.apply_actions([0, 0])
    self.assertEqual(state.current_player(), pyspiel.PlayerId.CHANCE)
    # First chance phase is REGIME (no noise configured): SWITCH outcome.
    outcomes = state.chance_outcomes()
    switch_action = max(a for a, _ in outcomes)  # _REGIME_SWITCH > _STAY
    state.apply_action(switch_action)
    self.assertEqual(state.regime_index(), 1)

  def test_game_as_turn_based(self):
    game = pyspiel.load_game("python_param_social_dilemma", {"players": 3})
    turn_based = pyspiel.convert_to_turn_based(game)
    pyspiel.random_sim_test(
        turn_based, num_sims=5, serialize=False, verbose=False)

  @parameterized.parameters(2, 3, 5)
  def test_random_sim(self, num_players):
    """Runs our standard game tests, checking API consistency."""
    game = pyspiel.load_game(
        "python_param_social_dilemma", {"players": num_players})
    pyspiel.random_sim_test(game, num_sims=5, serialize=True, verbose=False)

  def test_random_sim_with_noise_and_dynamic_payoffs(self):
    game = pyspiel.load_game(
        "python_param_social_dilemma", {
            "players": 5,
            "reward_noise_std": 1.5,
            "dynamic_payoffs": True,
            "payoff_regimes": "10 5 1 0 15 6 2 0 8 4 1 0",
            "payoff_change_prob": 0.3,
        })
    pyspiel.random_sim_test(game, num_sims=5, serialize=True, verbose=False)


if __name__ == "__main__":
  absltest.main()
