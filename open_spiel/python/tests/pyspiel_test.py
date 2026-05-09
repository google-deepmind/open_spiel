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
"""General tests for pyspiel python bindings."""

import json
import os

from absl.testing import absltest
from absl.testing import parameterized

from open_spiel.python import games  # pylint: disable=unused-import
from open_spiel.python import policy
from open_spiel.python.mfg import games as mfgs  # pylint: disable=unused-import
import pyspiel


_FULLY_OPTIONAL_PYTHON_GAMES = frozenset([
    "python_pokerkit_wrapper",
    "python_pokerkit_wrapper_acpc_style",
    "python_repeated_pokerkit"
])
# Specify game names in alphabetical order, to make the test easier to read.
# "Mandatory" = neither optional nor included only if certain flags are set.
EXPECTED_MANDATORY_GAMES = frozenset([
    "2048",
    "add_noise",
    "amazons",
    "antichess",
    "backgammon",
    "banqi",
    "bargaining",
    "battleship",
    "blackjack",
    "blotto",
    "breakthrough",
    "bridge",
    "bridge_uncontested_bidding",
    "cached_tree",
    "catch",
    "chat_game",  # python game locating in python/games/chat_games/
    "checkers",
    "chess",
    "chinese_checkers",
    "cliff_walking",
    "clobber",
    "coin_game",
    "colored_trails",
    "connect_four",
    "coop_box_pushing",
    "coop_to_1p",
    "coordinated_mp",
    "crazy_eights",
    "crazyhouse",
    "cribbage",
    "cursor_go",
    "dark_chess",
    "dark_hex",
    "dark_hex_ir",
    "deep_sea",
    "dots_and_boxes",
    "dou_dizhu",
    "efg_game",
    "einstein_wurfelt_nicht",
    "euchre",
    "first_sealed_auction",
    "gin_rummy",
    "go",
    "gomoku",
    "goofspiel",
    "havannah",
    "hex",
    "hearts",
    "hive",
    "kriegspiel",
    "kuhn_poker",
    "laser_tag",
    "latent_ttt",
    "lewis_signaling",
    "leduc_poker",
    "liars_dice",
    "liars_dice_ir",
    "lines_of_action",
    "maedn",
    "mancala",
    "markov_soccer",
    "matching_pennies_3p",
    "matrix_bos",
    "matrix_brps",
    "matrix_cd",
    "matrix_coordination",
    "matrix_mp",
    "matrix_pd",
    "matrix_rps",
    "matrix_rpsw",
    "matrix_sh",
    "matrix_shapleys_game",
    "mean_field_lin_quad",
    "mfg_crowd_modelling",
    "mfg_crowd_modelling_2d",
    "mfg_dynamic_routing",
    "mfg_garnet",
    "misere",
    "mnk",
    "morpion_solitaire",
    "negotiation",
    "nfg_game",
    "nim",
    "nine_mens_morris",
    "normal_form_extensive_game",
    "oh_hell",
    "oshi_zumo",
    "othello",
    "oware",
    "pentago",
    "pathfinding",
    "phantom_go",
    "phantom_ttt",
    "phantom_ttt_ir",
    "pig",
    "python_ant_foraging",
    "python_block_dominoes",
    "python_dynamic_routing",
    "python_hangman",
    "python_iterated_prisoners_dilemma",
    "python_mfg_crowd_avoidance",
    "python_mfg_crowd_modelling",
    "python_mfg_dynamic_routing",
    "python_mfg_periodic_aversion",
    "python_mfg_predator_prey",
    "python_kuhn_poker",
    "python_team_dominoes",
    "python_tic_tac_toe",
    "python_liars_poker",
    "quoridor",
    "repeated_game",
    "repeated_leduc_poker",
    "rbc",
    "restricted_nash_response",
    "sheriff",
    "shogi",
    "skat",
    "snake",
    "start_at",
    "solitaire",
    "spades",
    "stones_and_gems",
    "tarok",
    "tic_tac_toe",
    "tiny_bridge_2p",
    "tiny_bridge_4p",
    "tiny_hanabi",
    "trade_comm",
    "turn_based_simultaneous_game",
    "twixt",
    "ultimate_tic_tac_toe",
    "xiangqi",
    "y",
    "yacht",
    "zerosum",
])


class PyspielTest(parameterized.TestCase):

  def test_registered_names_is_sorted(self):
    game_names = pyspiel.registered_names()
    self.assertSequenceEqual(game_names, sorted(game_names))

  def test_registered_names_contains_expected_games(self):
    game_names = pyspiel.registered_names()

    expected = list(EXPECTED_MANDATORY_GAMES)
    if (os.environ.get("OPEN_SPIEL_BUILD_WITH_HANABI", "OFF") == "ON" and
        "hanabi" not in expected):
      expected.append("hanabi")
    if (os.environ.get("OPEN_SPIEL_BUILD_WITH_ACPC", "OFF") == "ON" and
        "universal_poker" not in expected):
      expected.append("universal_poker")
      expected.append("repeated_poker")
    # Verify we have registered all mandatory games + games added via flags
    self.assertContainsSubset(expected, game_names)

    # Check that the contents of the remainder are all fully optional games.
    # (If this fails, it likely means that someone added a game but forgot to
    # update either EXPECTED_MANDATORY_GAMES or _FULLY_OPTIONAL_PYTHON_GAMES.)
    remaining_games = set(game_names).difference(set(expected))
    self.assertContainsSubset(expected_subset=remaining_games,
                              actual_set=_FULLY_OPTIONAL_PYTHON_GAMES)

  def test_default_loadable(self):
    # Games which cannmot be loaded with default parameters will be skipped by
    # several standard tests. We make a list of such games here in order to make
    # implementors think twice about making new games non-default-loadable
    non_default_loadable = [
        game.short_name
        for game in pyspiel.registered_games()
        if not game.default_loadable
    ]
    expected = [
        # Being non-default-loadable prevents various automated tests.
        # Only add games here if there is no sensible default for a parameter.
        "add_noise",
        "cached_tree",
        "coop_to_1p",
        "efg_game",
        "nfg_game",
        "misere",
        "turn_based_simultaneous_game",
        "normal_form_extensive_game",
        "repeated_game",
        "restricted_nash_response",
        "start_at",
        "zerosum",
    ]
    if (os.environ.get("OPEN_SPIEL_BUILD_WITH_ACPC", "OFF") == "ON" and
        "repeated_poker" not in expected):
      expected.append("repeated_poker")
    self.assertCountEqual(non_default_loadable, expected)

  def test_registered_game_attributes(self):
    game_list = {game.short_name: game for game in pyspiel.registered_games()}
    self.assertEqual(game_list["kuhn_poker"].dynamics,
                     pyspiel.GameType.Dynamics.SEQUENTIAL)
    self.assertEqual(game_list["kuhn_poker"].chance_mode,
                     pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC)
    self.assertEqual(game_list["kuhn_poker"].information,
                     pyspiel.GameType.Information.IMPERFECT_INFORMATION)
    self.assertEqual(game_list["kuhn_poker"].utility,
                     pyspiel.GameType.Utility.ZERO_SUM)
    self.assertEqual(game_list["kuhn_poker"].min_num_players, 2)

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

  def test_game_string(self):
    game = pyspiel.load_game("tic_tac_toe")
    self.assertEqual(str(game), game.to_string())

  def test_game_parameters_from_string_empty(self):
    self.assertEqual(pyspiel.game_parameters_from_string(""), {})

  def test_game_parameters_from_string_simple(self):
    self.assertEqual(
        pyspiel.game_parameters_from_string("foo"), {"name": "foo"})

  def test_game_parameters_from_string_with_options(self):
    self.assertEqual(
        pyspiel.game_parameters_from_string("foo(x=2,y=true)"), {
            "name": "foo",
            "x": 2,
            "y": True
        })

  def test_game_parameters_from_string_with_subgame(self):
    self.assertEqual(
        pyspiel.game_parameters_from_string(
            "foo(x=2,y=true,subgame=bar(z=False))"), {
                "name": "foo",
                "x": 2,
                "y": True,
                "subgame": {
                    "name": "bar",
                    "z": False
                }
            })

  def test_game_parameters_to_string_empty(self):
    self.assertEqual(pyspiel.game_parameters_to_string({}), "")

  def test_game_parameters_to_string_simple(self):
    self.assertEqual(
        pyspiel.game_parameters_to_string({"name": "foo"}), "foo()")

  def test_game_parameters_to_string_with_options(self):
    self.assertEqual(
        pyspiel.game_parameters_to_string({
            "name": "foo",
            "x": 2,
            "y": True
        }), "foo(x=2,y=True)")

  def test_game_parameters_to_string_with_subgame(self):
    self.assertEqual(
        pyspiel.game_parameters_to_string({
            "name": "foo",
            "x": 2,
            "y": True,
            "subgame": {
                "name": "bar",
                "z": False
            }
        }), "foo(subgame=bar(z=False),x=2,y=True)")

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

  def test_get_game_parameters(self):
    if "universal_poker" in pyspiel.registered_names():
      game = pyspiel.load_game(pyspiel.hunl_game_string("fullgame"))
      print(game)
      params = game.get_parameters()
      print(params)
      param_string = pyspiel.game_parameters_to_string(params)
      print(param_string)
      params2 = pyspiel.game_parameters_from_string(param_string)
      assert params2 is not None
      game_from_params = pyspiel.load_game(
          f"{game.get_type().short_name}{param_string}")
      self.assertGreaterEqual(len(params2), len(params))
      assert game_from_params is not None

  @parameterized.parameters(
      ("universal_poker", pyspiel.hunl_game_string("fullgame")),
      (
          "repeated_poker",
          "repeated_poker(max_num_hands=10,reset_stacks=True,"
          + "rotate_dealer=True,universal_poker_game_string=universal_poker("
          + "betting=nolimit,bettingAbstraction=fullgame,blind=100 50,"
          + "firstPlayer=2 1 1 1,numBoardCards=0 3 1 1,numHoleCards=2,"
          + "numPlayers=2,numRanks=13,numRounds=4,numSuits=4,stack=20000 20000"
          + "))"),
      ("python_pokerkit_wrapper", "python_pokerkit_wrapper()"),
      ("python_pokerkit_wrapper",
       "python_pokerkit_wrapper(variant=FixedLimitSevenCardStud)"),
      ("python_pokerkit_wrapper",
       "python_pokerkit_wrapper(variant=PotLimitOmahaHoldem)"),
      ("python_pokerkit_wrapper_acpc_style",
       "python_pokerkit_wrapper_acpc_style(),"),
      ("kuhn_poker", "kuhn_poker(players=3)"),
      ("tic_tac_toe", "tic_tac_toe"),
      ("breakthrough", "breakthrough(rows=6,columns=6)"))
  def test_game_serialize_deserialize(self, game_name, game_string):
    if game_name in pyspiel.registered_names():
      game = pyspiel.load_game(game_string)
      serialized_game = game.serialize()
      game2 = pyspiel.deserialize_game(serialized_game)
      self.assertEqual(str(game), str(game2))

  def test_game_structs(self):
    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()

    # Test ActionStruct
    action = state.legal_actions()[0]
    action_struct = state.action_to_struct(action)
    self.assertIsNotNone(action_struct)
    self.assertIsInstance(action_struct, pyspiel.SpielStruct)
    self.assertIsInstance(action_struct, pyspiel.ActionStruct)
    json_str = action_struct.to_json()
    self.assertIn('"row":', json_str)
    self.assertIn('"col":', json_str)

    # Test StructToActions
    self.assertEqual([action], state.struct_to_actions(action_struct))

    # Test StateStruct
    state_struct = state.to_struct()
    self.assertIsNotNone(state_struct)
    self.assertIsInstance(state_struct, pyspiel.SpielStruct)
    self.assertIsInstance(state_struct, pyspiel.StateStruct)
    json_str = state_struct.to_json()
    self.assertIn('"board":', json_str)

    # Test ObservationStruct
    observation_struct = state.to_observation_struct()
    self.assertIsNotNone(observation_struct)
    self.assertIsInstance(observation_struct, pyspiel.SpielStruct)
    self.assertIsInstance(observation_struct, pyspiel.ObservationStruct)
    json_str = observation_struct.to_json()
    self.assertIn('"board":', json_str)


class StructApiTest(absltest.TestCase):
  """Tests for struct API."""

  def setUp(self):
    super().setUp()
    self.game = pyspiel.load_game("tic_tac_toe")
    self.ttt = pyspiel.tic_tac_toe
    # Create a reference state with one move applied
    self.ref_state = self.game.new_initial_state()
    self.ref_state.apply_action(0)  # X plays top-left

  # ===== Struct Field Access Tests =====

  def test_state_struct_fields(self):
    """Test that struct fields are accessible and correct."""
    state_struct = self.ref_state.to_struct()
    self.assertEqual(state_struct.current_player, "o")
    self.assertEqual(state_struct.board,
                     ["x", ".", ".", ".", ".", ".", ".", ".", "."])

  def test_observation_struct_fields(self):
    """Test observation struct has correct fields."""
    obs_struct = self.ref_state.to_observation_struct(0)
    self.assertEqual(obs_struct.current_player, "o")
    self.assertEqual(obs_struct.board,
                     ["x", ".", ".", ".", ".", ".", ".", ".", "."])

  def test_action_struct_fields(self):
    """Test action struct has semantic row/col fields via JSON."""
    action = 4  # Center square
    action_struct = self.ref_state.action_to_struct(action)
    # Note: TicTacToeActionStruct fields aren't directly bound in Python yet,
    # but we can verify via JSON
    parsed = json.loads(action_struct.to_json())
    self.assertEqual(parsed["row"], 1)
    self.assertEqual(parsed["col"], 1)

  # ===== JSON Conversion Tests =====

  def test_to_json(self):
    """Test state converts to expected JSON string."""
    state_struct = self.ref_state.to_struct()
    json_str = state_struct.to_json()
    self.assertIn('"current_player":', json_str)
    self.assertIn('"board":', json_str)
    # Verify it's valid JSON
    parsed = json.loads(json_str)
    self.assertEqual(parsed["current_player"], "o")

  def test_struct_from_json(self):
    """Test struct can be constructed from JSON string."""
    json_str = self.ref_state.to_json()
    state_struct = self.ttt.TicTacToeStateStruct(json_str)
    self.assertEqual(state_struct.to_json(), json_str)

  def test_json_round_trip(self):
    """Test JSON -> struct -> JSON preserves data."""
    original_json = self.ref_state.to_json()
    state_struct = self.ttt.TicTacToeStateStruct(original_json)
    round_trip_json = state_struct.to_json()
    self.assertEqual(original_json, round_trip_json)

  # ===== Dict Conversion Tests =====

  def test_to_dict(self):
    """Test state converts to Python dict."""
    state_dict = self.ref_state.to_dict()
    self.assertIsInstance(state_dict, dict)
    self.assertIn("current_player", state_dict)
    self.assertIn("board", state_dict)
    self.assertEqual(state_dict["current_player"], "o")

  def test_state_from_dict(self):
    """Test creating state from dict."""
    state_dict = {
        "board": ["x", "o", ".", ".", ".", ".", ".", ".", "."],
        "current_player": "x",
    }
    state = self.game.new_initial_state(state_dict)
    self.assertEqual(state.to_dict(), state_dict)

  def test_state_to_from_dict_round_trip(self):
    """Test state -> dict -> state round-trip."""
    state = self.game.new_initial_state()
    for action in [0, 1, 4]:
      state.apply_action(action)

    state_dict = state.to_dict()
    new_state = self.game.new_initial_state(state_dict)
    self.assertEqual(new_state.to_dict(), state_dict)

  # ===== Starting State Tests =====

  def test_empty_starting_state(self):
    """Test that normal initial state has no custom starting state."""
    state = self.game.new_initial_state()
    state.apply_action(0)
    self.assertIsNone(state.starting_state())
    self.assertEqual(state.starting_state_str(), "")

  def test_starting_state_from_dict(self):
    """Test creating state from dict sets starting_state correctly."""
    state = self.game.new_initial_state(self.ref_state.to_dict())
    self.assertEmpty(state.history())
    self.assertIsNotNone(state.starting_state())
    self.assertEqual(str(state.starting_state()), str(self.ref_state))

  def test_starting_state_str(self):
    """Test starting_state_str returns JSON."""
    state = self.game.new_initial_state(self.ref_state.to_dict())
    starting_str = state.starting_state_str()
    self.assertNotEqual(starting_str, "")
    # Should be valid JSON
    parsed = json.loads(starting_str)
    self.assertEqual(parsed["current_player"], "o")

  def test_starting_state_unchanged_after_action(self):
    """Test that starting_state doesn't change when actions are applied."""
    state = self.game.new_initial_state(self.ref_state.to_dict())
    original_starting = str(state.starting_state())
    state.apply_action(4)  # Apply an action
    self.assertEqual(str(state.starting_state()), original_starting)

  # ===== Action Struct Tests =====

  def test_action_struct_round_trip(self):
    """Test action -> struct -> actions preserves the action."""
    state = self.game.new_initial_state()
    for action in state.legal_actions():
      action_struct = state.action_to_struct(action)
      recovered_actions = state.struct_to_actions(action_struct)
      self.assertEqual([action], recovered_actions)

  def test_apply_action_struct(self):
    """Test applying action via struct produces same result as direct apply."""
    action = self.ref_state.legal_actions()[0]
    action_struct = self.ref_state.action_to_struct(action)

    state1 = self.game.new_initial_state()
    state1.apply_action(0)
    state1.apply_action(action)

    state2 = self.game.new_initial_state()
    state2.apply_action(0)
    state2.apply_action_struct(action_struct)  # Should succeed (no exception)

    self.assertEqual(str(state1), str(state2))

  def test_validate_action_struct_valid(self):
    """Test validate_action_struct succeeds for valid action."""
    action = self.ref_state.legal_actions()[0]
    action_struct = self.ref_state.action_to_struct(action)
    self.ref_state.validate_action_struct(action_struct)  # Should not raise

  def test_validate_action_struct_invalid(self):
    """Test validate_action_struct raises error for invalid action."""
    state = self.game.new_initial_state()
    center_action = 4
    action_struct = state.action_to_struct(center_action)

    state.apply_action(center_action)

    status = state.validate_action_struct(action_struct)
    self.assertFalse(status.ok())

  def test_apply_action_struct_invalid(self):
    """Test apply_action_struct raises error and doesn't mutate state."""
    state = self.game.new_initial_state()
    center_action = 4
    action_struct = state.action_to_struct(center_action)

    state.apply_action(center_action)
    state_str_before = str(state)

    status = state.apply_action_struct(action_struct)
    self.assertFalse(status.ok())

    self.assertEqual(str(state), state_str_before)

  def test_action_struct_from_json(self):
    """Test action struct can be constructed from JSON."""
    action = 4  # Center
    action_struct = self.ref_state.action_to_struct(action)
    json_str = action_struct.to_json()

    # Reconstruct and verify
    parsed = json.loads(json_str)
    self.assertEqual(parsed["row"], 1)
    self.assertEqual(parsed["col"], 1)

  def test_action_struct_row_col_mapping(self):
    """Test action struct row/col correctly maps to board position."""
    state = self.game.new_initial_state()
    for action in range(9):
      action_struct = state.action_to_struct(action)
      parsed = json.loads(action_struct.to_json())
      expected_row = action // 3
      expected_col = action % 3
      self.assertEqual(parsed["row"], expected_row,
                       f"Action {action} should have row {expected_row}")
      self.assertEqual(parsed["col"], expected_col,
                       f"Action {action} should have col {expected_col}")

  # ===== Vector Action API Tests =====

  def test_struct_to_actions(self):
    """Test struct_to_actions returns list with single action."""
    action = 4
    action_struct = self.ref_state.action_to_struct(action)
    actions = self.ref_state.struct_to_actions(action_struct)
    self.assertIsInstance(actions, list)
    self.assertLen(actions, 1)
    self.assertEqual(actions[0], action)

  def test_actions_to_struct(self):
    """Test actions_to_struct with single action."""
    action = 4
    action_struct = self.ref_state.actions_to_struct([action])
    recovered = self.ref_state.struct_to_actions(action_struct)
    self.assertEqual(recovered, [action])

  def test_actions_to_struct_with_player(self):
    """Test actions_to_struct with player parameter."""
    action = 4
    action_struct = self.ref_state.actions_to_struct(1, [action])
    parsed = json.loads(action_struct.to_json())
    self.assertEqual(parsed["row"], 1)
    self.assertEqual(parsed["col"], 1)

  # ===== Constructor Tests =====

  def test_struct_default_constructor(self):
    """Test struct default constructor."""
    s = self.ttt.TicTacToeStateStruct()
    self.assertEqual(s.current_player, "")
    self.assertEqual(s.board, [])

  def test_struct_json_constructor(self):
    """Test struct JSON string constructor."""
    json_str = (
        '{"current_player": "x", "board": [".", ".", ".", ".", ".", ".", ".",'
        ' ".", "."]}'
    )
    s = self.ttt.TicTacToeStateStruct(json_str)
    self.assertEqual(s.current_player, "x")
    self.assertLen(s.board, 9)

  def test_struct_dict_constructor(self):
    """Test struct dict constructor."""
    d = {
        "current_player": "o",
        "board": ["x", ".", ".", ".", ".", ".", ".", ".", "."]
    }
    s = self.ttt.TicTacToeStateStruct(d)
    self.assertEqual(s.current_player, "o")
    self.assertEqual(s.board[0], "x")


class LoadGameFromJsonTest(absltest.TestCase):
  """Tests for load_game_from_json API."""

  def test_load_game_basic(self):
    """Test basic game loading from JSON."""
    game = pyspiel.load_game_from_json('{"game_name": "tic_tac_toe"}')
    self.assertEqual(game.get_type().short_name, "tic_tac_toe")

  def test_load_game_with_params(self):
    """Test game loading with parameters."""
    game = pyspiel.load_game_from_json(
        '{"game_name": "connect_four", "columns": 8}')
    self.assertEqual(game.get_type().short_name, "connect_four")

  def test_load_game_invalid_name(self):
    """Test that invalid game name raises error."""
    with self.assertRaises(pyspiel.SpielError):
      pyspiel.load_game_from_json('{"game_name": "nonexistent_game"}')

  def test_load_game_missing_name(self):
    """Test that missing game_name raises error."""
    with self.assertRaises(pyspiel.SpielError):
      pyspiel.load_game_from_json('{"some_param": 123}')


if __name__ == "__main__":
  absltest.main()
