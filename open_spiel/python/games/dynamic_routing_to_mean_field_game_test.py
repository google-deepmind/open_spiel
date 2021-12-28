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

"""Tests for dynamic_routing_to_mean_field_game."""
from absl.testing import absltest

from open_spiel.python import games  # pylint:disable=unused-import
from open_spiel.python import policy
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.games import dynamic_routing_to_mean_field_game
from open_spiel.python.mfg import games as mfg_games  # pylint:disable=unused-import
from open_spiel.python.mfg.algorithms import mirror_descent
import pyspiel


class DerivedNPlayerPolicyFromMeanFieldPolicyTest(absltest.TestCase):

  def test_state_conversion_method(self):
    """Test N player game state to mean field game state conversion."""
    # TODO(cabannes): test state conversion.

  def test_uniform_mfg_policy_conversion_to_n_player_uniform_policy(self):
    """Test conversion of uniform to uniform policy."""
    mfg_game = pyspiel.load_game("python_mfg_dynamic_routing", {
        "time_step_length": 0.05,
        "max_num_time_step": 100
    })
    n_player_game = pyspiel.load_game("python_dynamic_routing", {
        "time_step_length": 0.05,
        "max_num_time_step": 100
    })
    mfg_derived_policy = (
        dynamic_routing_to_mean_field_game
        .DerivedNPlayerPolicyFromMeanFieldPolicy(
            n_player_game, policy.UniformRandomPolicy(mfg_game)))
    derived_policy_value = expected_game_score.policy_value(
        n_player_game.new_initial_state(), mfg_derived_policy)
    uniform_policy_value = expected_game_score.policy_value(
        n_player_game.new_initial_state(),
        policy.UniformRandomPolicy(n_player_game))
    self.assertSequenceAlmostEqual(derived_policy_value, uniform_policy_value)

  def test_pigou_network_game_outcome_optimal_mfg_policy_in_n_player_game(self):
    """Test MFG Nash equilibrium policy for the Pigou network."""
    # TODO(cabannes): test policy.
    # TODO(cabannes): test game outcome.

  def test_learning_and_applying_mfg_policy_in_n_player_game(self):
    """Test converting learnt MFG policy default game."""
    # learning the Braess MFG Nash equilibrium
    mfg_game = pyspiel.load_game("python_mfg_dynamic_routing")
    omd = mirror_descent.MirrorDescent(mfg_game, lr=1)
    for _ in range(10):
      omd.iteration()
    mfg_policy = omd.get_policy()
    n_player_game = pyspiel.load_game("python_dynamic_routing")
    mfg_derived_policy = (
        dynamic_routing_to_mean_field_game
        .DerivedNPlayerPolicyFromMeanFieldPolicy(n_player_game, mfg_policy))
    expected_game_score.policy_value(n_player_game.new_initial_state(),
                                     mfg_derived_policy)


if __name__ == "__main__":
  absltest.main()
