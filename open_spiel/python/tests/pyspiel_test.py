# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for open_spiel.python.pybind11.pyspiel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl.testing import absltest

from open_spiel.python import policy
import pyspiel

# Specify game names in alphabetical order, to make the test easier to read.
EXPECTED_GAMES = set([
    "backgammon",
    "battleship",
    "blackjack",
    "blotto",
    "breakthrough",
    "bridge",
    "bridge_uncontested_bidding",
    "catch",
    "chess",
    "cliff_walking",
    "clobber",
    "coin_game",
    "connect_four",
    "coop_box_pushing",
    "coop_to_1p",
    "coordinated_mp",
    "cursor_go",
    "dark_chess",
    "dark_hex",
    "dark_hex_ir",
    "deep_sea",
    "efg_game",
    "first_sealed_auction",
    "gin_rummy",
    "go",
    "goofspiel",
    "havannah",
    "hex",
    "hearts",
    "kriegspiel",
    "kuhn_poker",
    "laser_tag",
    "lewis_signaling",
    "leduc_poker",
    "liars_dice",
    "markov_soccer",
    "matching_pennies_3p",
    "matrix_cd",
    "matrix_coordination",
    "matrix_mp",
    "matrix_pd",
    "matrix_rps",
    "matrix_rpsw",
    "matrix_sh",
    "matrix_shapleys_game",
    "misere",
    "negotiation",
    "nfg_game",
    "normal_form_extensive_game",
    "oh_hell",
    "oshi_zumo",
    "othello",
    "oware",
    "pentago",
    "phantom_ttt",
    "phantom_ttt_ir",
    "pig",
    "quoridor",
    "repeated_game",
    "sheriff",
    "skat",
    "start_at",
    "solitaire",
    "stones_and_gems",
    "tarok",
    "tic_tac_toe",
    "tiny_bridge_2p",
    "tiny_bridge_4p",
    "tiny_hanabi",
    "trade_comm",
    "turn_based_simultaneous_game",
    "y",
])


class PyspielTest(absltest.TestCase):

  def test_registered_names(self):
    game_names = pyspiel.registered_names()

    expected = EXPECTED_GAMES
    if os.environ.get("OPEN_SPIEL_BUILD_WITH_HANABI", "OFF") == "ON":
      expected.add("hanabi")
    if os.environ.get("OPEN_SPIEL_BUILD_WITH_ACPC", "OFF") == "ON":
      expected.add("universal_poker")
    expected = sorted(list(expected))
    self.assertCountEqual(game_names, expected)

  def test_no_mandatory_parameters(self):
    # Games with mandatory parameters will be skipped by several standard
    # tests. Mandatory parameters should therefore be avoided if at all
    # possible. We make a list of such games here in order to make implementors
    # think twice about adding mandatory parameters.
    def has_mandatory_params(game):
      return any(param.is_mandatory()
                 for param in game.parameter_specification.values())

    games_with_mandatory_parameters = [
        game.short_name
        for game in pyspiel.registered_games()
        if has_mandatory_params(game)
    ]
    expected = [
        # Mandatory parameters prevent various sorts of automated testing.
        # Only add games here if there is no sensible default for a parameter.
        "misere",
        "turn_based_simultaneous_game",
        "normal_form_extensive_game",
        "repeated_game",
        "start_at",
    ]
    self.assertCountEqual(games_with_mandatory_parameters, expected)

  def test_registered_game_attributes(self):
    games = {game.short_name: game for game in pyspiel.registered_games()}
    self.assertEqual(games["kuhn_poker"].dynamics,
                     pyspiel.GameType.Dynamics.SEQUENTIAL)
    self.assertEqual(games["kuhn_poker"].chance_mode,
                     pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC)
    self.assertEqual(games["kuhn_poker"].information,
                     pyspiel.GameType.Information.IMPERFECT_INFORMATION)
    self.assertEqual(games["kuhn_poker"].utility,
                     pyspiel.GameType.Utility.ZERO_SUM)
    self.assertEqual(games["kuhn_poker"].min_num_players, 2)

  def test_create_game(self):
    game = pyspiel.load_game("kuhn_poker")
    game_info = game.get_type()
    self.assertEqual(game_info.information,
                     pyspiel.GameType.Information.IMPERFECT_INFORMATION)
    self.assertEqual(game.num_players(), 2)

  def test_play_kuhn_poker(self):
    game = pyspiel.load_game("kuhn_poker")
    state = game.new_initial_state()
    self.assertEqual(state.is_chance_node(), True)
    self.assertEqual(state.chance_outcomes(), [(0, 1 / 3), (1, 1 / 3),
                                               (2, 1 / 3)])
    state.apply_action(1)
    self.assertEqual(state.is_chance_node(), True)
    self.assertEqual(state.chance_outcomes(), [(0, 0.5), (2, 0.5)])
    state.apply_action(2)
    self.assertEqual(state.is_chance_node(), False)
    self.assertEqual(state.legal_actions(), [0, 1])
    sampler = pyspiel.UniformProbabilitySampler(0., 1.)
    clone = state.resample_from_infostate(1, sampler)
    self.assertEqual(
        clone.information_state_string(1), state.information_state_string(1))

  def test_othello(self):
    game = pyspiel.load_game("othello")
    state = game.new_initial_state()
    self.assertFalse(state.is_chance_node())
    self.assertFalse(state.is_terminal())
    self.assertEqual(state.legal_actions(), [19, 26, 37, 44])

  def test_tic_tac_toe(self):
    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()
    self.assertFalse(state.is_chance_node())
    self.assertFalse(state.is_terminal())
    self.assertEqual(state.legal_actions(), [0, 1, 2, 3, 4, 5, 6, 7, 8])

  def test_game_parameter_representation(self):
    param = pyspiel.GameParameter(True)
    self.assertEqual(repr(param), "GameParameter(bool_value=True)")
    param = pyspiel.GameParameter(False)
    self.assertEqual(repr(param), "GameParameter(bool_value=False)")
    param = pyspiel.GameParameter("one")
    self.assertEqual(repr(param), "GameParameter(string_value='one')")
    param = pyspiel.GameParameter(1)
    self.assertEqual(repr(param), "GameParameter(int_value=1)")
    param = pyspiel.GameParameter(1.0)
    self.assertEqual(repr(param), "GameParameter(double_value=1)")
    param = pyspiel.GameParameter(1.2)
    self.assertEqual(repr(param), "GameParameter(double_value=1.2)")

  def test_game_parameter_equality(self):
    param1 = pyspiel.GameParameter("one")
    param1_again = pyspiel.GameParameter("one")
    param2 = pyspiel.GameParameter("two")
    self.assertEqual(param1, param1_again)
    self.assertNotEqual(param1, param2)

  def test_game_parameter_can_access_value(self):
    self.assertEqual(pyspiel.GameParameter(True).value(), True)
    self.assertEqual(pyspiel.GameParameter(42).value(), 42)
    self.assertEqual(pyspiel.GameParameter(3.141).value(), 3.141)
    self.assertEqual(pyspiel.GameParameter("spqr").value(), "spqr")
    self.assertEqual(
        pyspiel.GameParameter({
            "a": pyspiel.GameParameter(1.23),
            "b": pyspiel.GameParameter(True)
        }).value(), {
            "a": 1.23,
            "b": True
        })

  def test_game_parameters_from_string_empty(self):
    self.assertEqual(pyspiel.game_parameters_from_string(""), {})

  def test_game_parameters_from_string_simple(self):
    self.assertEqual(pyspiel.game_parameters_from_string("foo"),
                     {"name": pyspiel.GameParameter("foo")})

  def test_game_parameters_from_string_with_options(self):
    self.assertEqual(pyspiel.game_parameters_from_string("foo(x=2,y=true)"),
                     {"name": pyspiel.GameParameter("foo"),
                      "x": pyspiel.GameParameter(2),
                      "y": pyspiel.GameParameter(True)})

  def test_game_type(self):
    game_type = pyspiel.GameType(
        "matrix_mp", "Matching Pennies", pyspiel.GameType.Dynamics.SIMULTANEOUS,
        pyspiel.GameType.ChanceMode.DETERMINISTIC,
        pyspiel.GameType.Information.PERFECT_INFORMATION,
        pyspiel.GameType.Utility.ZERO_SUM,
        pyspiel.GameType.RewardModel.TERMINAL, 2, 2, True, True, False, False,
        dict())
    self.assertEqual(game_type.chance_mode,
                     pyspiel.GameType.ChanceMode.DETERMINISTIC)

  def test_error_handling(self):
    with self.assertRaisesRegex(RuntimeError,
                                "Unknown game 'invalid_game_name'"):
      unused_game = pyspiel.load_game("invalid_game_name")

  def test_can_create_cpp_tabular_policy(self):
    for game_name in ["kuhn_poker", "leduc_poker", "liars_dice"]:
      game = pyspiel.load_game(game_name)

      # We just test that we can create a tabular policy.
      policy.python_policy_to_pyspiel_policy(policy.TabularPolicy(game))

  def test_simultaneous_game_history(self):
    game = pyspiel.load_game("coop_box_pushing")
    state = game.new_initial_state()
    state.apply_action(0)
    state2 = game.new_initial_state()
    state2.apply_actions([0] * game.num_players())
    self.assertEqual(state.history(), state2.history())

  def test_record_batched_trajectories(self):
    for game_name in ["kuhn_poker", "leduc_poker", "liars_dice"]:
      game = pyspiel.load_game(game_name)
      python_policy = policy.TabularPolicy(game)
      tabular_policy = policy.python_policy_to_pyspiel_policy(python_policy)
      policies = [tabular_policy] * 2

      # We test that we can create a batch of trajectories.
      seed = 0
      batch_size = 128
      include_full_observations = False
      pyspiel.record_batched_trajectories(game, policies,
                                          python_policy.state_lookup,
                                          batch_size, include_full_observations,
                                          seed, -1)


if __name__ == "__main__":
  absltest.main()
