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

# Lint as python3
"""Tests for Python Tic-Tac-Toe."""

import difflib
import json
import os
import pickle

from absl.testing import absltest
import numpy as np

from open_spiel.python.algorithms.get_all_states import get_all_states
from open_spiel.python.games import tic_tac_toe
from open_spiel.python.observation import make_observation
import pyspiel

_DATA_DIR = "open_spiel/integration_tests/playthroughs/"


class TicTacToeTest(absltest.TestCase):

  def test_can_create_game_and_state(self):
    """Checks we can create the game and a state."""
    game = tic_tac_toe.TicTacToeGame()
    state = game.new_initial_state()
    self.assertEqual(str(state), "...\n...\n...")

  def test_random_game(self):
    """Tests basic API functions."""
    # This is here mostly to show the API by example.
    # More serious simulation tests are done in python/tests/games_sim_test.py
    # and in test_game_from_cc (below), both of which test the conformance to
    # the API thoroughly.
    game = tic_tac_toe.TicTacToeGame()
    state = game.new_initial_state()
    while not state.is_terminal():
      print(state)
      cur_player = state.current_player()
      legal_actions = state.legal_actions()
      action = np.random.choice(legal_actions)
      print("Player {} chooses action {}".format(cur_player, action))
      state.apply_action(action)
    print(state)
    print("Returns: {}".format(state.returns()))

  def test_game_from_cc(self):
    """Runs our standard game tests, checking API consistency."""
    game = pyspiel.load_game("python_tic_tac_toe")
    pyspiel.random_sim_test(game, num_sims=10, serialize=False, verbose=True)

  def test_playthoughs_consistent(self):
    """Checks the saved C++ and Python playthroughs are the same."""
    test_srcdir = os.environ.get("TEST_SRCDIR", "")
    path = os.path.join(test_srcdir, _DATA_DIR)
    cc_playthrough = os.path.join(path, "tic_tac_toe.txt")
    py_playthrough = os.path.join(path, "python_tic_tac_toe.txt")
    with open(cc_playthrough, encoding="utf-8") as cc:
      with open(py_playthrough, encoding="utf-8") as py:
        diffs = difflib.ndiff(list(cc), list(py))
    diffs = {d for d in diffs if d and d[0] in {"+", "-"}}
    self.assertEqual(
        diffs, {
            "- game: tic_tac_toe\n",
            "+ game: python_tic_tac_toe\n",
            '- GameType.long_name = "Tic Tac Toe"\n',
            '+ GameType.long_name = "Python Tic-Tac-Toe"\n',
            '- GameType.short_name = "tic_tac_toe"\n',
            '+ GameType.short_name = "python_tic_tac_toe"\n',
            '- ToString() = "tic_tac_toe()"\n',
            '+ ToString() = "python_tic_tac_toe()"\n',
            "- Returns() = [0, 0]\n",
            "+ Returns() = [0, -0]\n",
        })

  def test_observation_tensors_same(self):
    """Checks observation tensor is the same from C++ and from Python."""
    game = pyspiel.load_game("python_tic_tac_toe")
    state = game.new_initial_state()
    for a in [4, 5, 2, 3]:
      state.apply_action(a)
    py_obs = make_observation(game)
    py_obs.set_from(state, state.current_player())
    cc_obs = state.observation_tensor()
    np.testing.assert_array_equal(py_obs.tensor, cc_obs)

  def test_pickle(self):
    """Checks pickling and unpickling of game and state."""
    game = pyspiel.load_game("python_tic_tac_toe")
    pickled_game = pickle.dumps(game)
    unpickled_game = pickle.loads(pickled_game)
    self.assertEqual(str(game), str(unpickled_game))
    state = game.new_initial_state()
    for a in [4, 2, 3, 7]:
      state.apply_action(a)
    ser_str = pyspiel.serialize_game_and_state(game, state)
    new_game, new_state = pyspiel.deserialize_game_and_state(ser_str)
    self.assertEqual(str(game), str(new_game))
    self.assertEqual(str(state), str(new_state))
    pickled_state = pickle.dumps(state)
    unpickled_state = pickle.loads(pickled_state)
    self.assertEqual(str(state), str(unpickled_state))

  def test_cloned_state_matches_original_state(self):
    """Check we can clone states successfully."""
    game = tic_tac_toe.TicTacToeGame()
    state = game.new_initial_state()
    state.apply_action(1)
    state.apply_action(2)
    clone = state.clone()

    self.assertEqual(state.history(), clone.history())
    self.assertEqual(state.num_players(), clone.num_players())
    self.assertEqual(state.move_number(), clone.move_number())
    self.assertEqual(state.num_distinct_actions(), clone.num_distinct_actions())

    self.assertEqual(state._cur_player, clone._cur_player)
    self.assertEqual(state._player0_score, clone._player0_score)
    self.assertEqual(state._is_terminal, clone._is_terminal)
    np.testing.assert_array_equal(state.board, clone.board)

  def test_consistent(self):
    """Checks the Python and C++ game implementations are the same."""
    py_game = pyspiel.load_game("python_tic_tac_toe")
    cc_game = pyspiel.load_game("tic_tac_toe")
    py_obs = make_observation(py_game)
    cc_obs = make_observation(cc_game)
    py_states = get_all_states(py_game, to_string=str)
    cc_states = get_all_states(cc_game, to_string=str)
    self.assertCountEqual(list(cc_states), list(py_states))
    for key, cc_state in cc_states.items():
      py_state = py_states[key]
      np.testing.assert_array_equal(py_state.history(), cc_state.history())
      np.testing.assert_array_equal(py_state.returns(), cc_state.returns())
      py_obs.set_from(py_state, 0)
      cc_obs.set_from(cc_state, 0)
      np.testing.assert_array_equal(py_obs.tensor, cc_obs.tensor)

  def test_struct_api(self):
    """Tests the SpielStruct dict-based API on Python games."""
    game = pyspiel.load_game("python_tic_tac_toe")
    state = game.new_initial_state()

    # Test to_struct() returns proper JSON.
    state_struct = state.to_struct()
    state_dict = json.loads(state_struct.to_json())
    self.assertEqual(state_dict["current_player"], "x")
    self.assertEqual(state_dict["board"], ["."] * 9)

    # Test to_json() convenience method (calls to_struct internally).
    state_json_str = state.to_json()
    self.assertEqual(json.loads(state_json_str), state_dict)

    # Test to_dict() convenience method.
    self.assertEqual(state.to_dict(), state_dict)

    # Test action_to_struct().
    action_struct = state.action_to_struct(0, 4)  # center cell
    action_dict = json.loads(action_struct.to_json())
    self.assertEqual(action_dict["row"], 1)
    self.assertEqual(action_dict["col"], 1)

    action_struct2 = state.action_to_struct(0, 0)  # top-left
    action_dict2 = json.loads(action_struct2.to_json())
    self.assertEqual(action_dict2["row"], 0)
    self.assertEqual(action_dict2["col"], 0)

    # Test validate_action_struct() with a legal action.
    status = state.validate_action_struct(action_struct)
    self.assertTrue(status.ok())

    # Test apply_action_struct().
    status = state.apply_action_struct(action_struct)
    self.assertTrue(status.ok())

    # Verify the action was applied.
    self.assertEqual(state.current_player(), 1)  # player o's turn
    state_json_after = json.loads(state.to_struct().to_json())
    self.assertEqual(state_json_after["current_player"], "o")
    self.assertEqual(state_json_after["board"][4], "x")  # center is now x

    # Test validate_action_struct() with an illegal action (cell occupied).
    status = state.validate_action_struct(action_struct)  # center again
    self.assertFalse(status.ok())
    self.assertIn("not legal", status.to_string())

    # Test to_observation_struct() — mirrors state struct for perfect info.
    obs_struct = state.to_observation_struct(0)
    obs_dict = json.loads(obs_struct.to_json())
    self.assertEqual(obs_dict["current_player"], "o")
    self.assertEqual(obs_dict["board"][4], "x")

  def test_struct_matches_cc(self):
    """Checks Python struct JSON matches C++ struct JSON at every state."""
    py_game = pyspiel.load_game("python_tic_tac_toe")
    cc_game = pyspiel.load_game("tic_tac_toe")
    py_state = py_game.new_initial_state()
    cc_state = cc_game.new_initial_state()

    actions = [4, 0, 2, 6, 8, 3, 5, 1, 7]  # a full game
    for action in actions:
      # Compare state structs.
      py_state_dict = json.loads(py_state.to_json())
      cc_state_dict = json.loads(cc_state.to_json())
      self.assertEqual(
          py_state_dict,
          cc_state_dict,
          f"State struct mismatch after actions up to {action}",
      )

      # Compare action structs.
      if not py_state.is_terminal():
        py_action_dict = json.loads(
            py_state.action_to_struct(
                py_state.current_player(), action
            ).to_json()
        )
        cc_action_dict = json.loads(
            cc_state.action_to_struct(
                cc_state.current_player(), action
            ).to_json()
        )
        self.assertEqual(
            py_action_dict,
            cc_action_dict,
            f"Action struct mismatch for action {action}",
        )

      py_state.apply_action(action)
      cc_state.apply_action(action)
      if py_state.is_terminal():
        break

  def test_new_initial_state_from_dict(self):
    """Tests constructing a state from a dict."""
    game = pyspiel.load_game("python_tic_tac_toe")

    # Construct a mid-game state: x in center, o in top-left.
    state_dict = {
        "current_player": "x",
        "board": ["o", ".", ".", ".", "x", ".", ".", ".", "."],
    }
    state = game.new_initial_state(state_dict)
    self.assertEqual(state.current_player(), 0)  # x's turn (2nd move by x)

    # Wait — let's recalculate. board has 1 x and 1 o, so num_x == num_o,
    # meaning it's x's turn (player 0).
    self.assertEqual(state.current_player(), 0)
    self.assertFalse(state.is_terminal())
    self.assertLen(state.legal_actions(), 7)  # 9 - 2 occupied

  def test_new_initial_state_from_struct(self):
    """Tests constructing a state from a StateStruct object."""
    game = pyspiel.load_game("python_tic_tac_toe")
    state = game.new_initial_state()

    # Play a few moves.
    state.apply_action(4)  # x center
    state.apply_action(0)  # o top-left
    state.apply_action(2)  # x top-right

    # Get the struct and reconstruct.
    state_struct = state.to_struct()
    state2 = game.new_initial_state(state_struct)

    # Verify the reconstructed state matches.
    self.assertEqual(str(state), str(state2))
    self.assertEqual(state.current_player(), state2.current_player())
    self.assertEqual(state.legal_actions(), state2.legal_actions())
    self.assertEqual(state.is_terminal(), state2.is_terminal())

  def test_struct_round_trip_full_game(self):
    """Tests struct round-trip at every step of a complete game."""
    game = pyspiel.load_game("python_tic_tac_toe")
    state = game.new_initial_state()

    actions = [4, 0, 2, 6, 8, 3, 5, 1, 7]  # a full game
    for action in actions:
      if state.is_terminal():
        break

      # Round-trip: state → struct → json → new_initial_state → compare.
      state_json = state.to_json()
      state_dict = json.loads(state_json)
      reconstructed = game.new_initial_state(state_dict)

      self.assertEqual(
          str(state), str(reconstructed), f"Mismatch before action {action}"
      )
      self.assertEqual(state.current_player(), reconstructed.current_player())
      self.assertEqual(state.legal_actions(), reconstructed.legal_actions())
      self.assertEqual(state.returns(), reconstructed.returns())
      self.assertEqual(state.is_terminal(), reconstructed.is_terminal())

      state.apply_action(action)

  def test_struct_round_trip_terminal_state(self):
    """Tests round-trip for a terminal state (x wins)."""
    game = pyspiel.load_game("python_tic_tac_toe")
    state = game.new_initial_state()

    # x wins: top row.
    for a in [0, 3, 1, 4, 2]:
      state.apply_action(a)
    self.assertTrue(state.is_terminal())
    self.assertEqual(state.returns(), [1.0, -1.0])

    # Round-trip the terminal state.
    state_struct = state.to_struct()
    reconstructed = game.new_initial_state(state_struct)
    self.assertTrue(reconstructed.is_terminal())
    self.assertEqual(reconstructed.returns(), [1.0, -1.0])
    self.assertEqual(str(state), str(reconstructed))

  def test_struct_round_trip_draw(self):
    """Tests round-trip for a drawn game."""
    game = pyspiel.load_game("python_tic_tac_toe")
    state = game.new_initial_state()

    # A draw game.
    for a in [4, 0, 2, 6, 3, 5, 1, 7, 8]:
      state.apply_action(a)
    self.assertTrue(state.is_terminal())
    self.assertEqual(state.returns(), [0.0, 0.0])

    reconstructed = game.new_initial_state(state.to_struct())
    self.assertTrue(reconstructed.is_terminal())
    self.assertEqual(reconstructed.returns(), [0.0, 0.0])

  def test_struct_round_trip_matches_cc(self):
    """Cross-validates struct round-trip between Python and C++."""
    py_game = pyspiel.load_game("python_tic_tac_toe")
    cc_game = pyspiel.load_game("tic_tac_toe")
    py_state = py_game.new_initial_state()
    cc_state = cc_game.new_initial_state()

    actions = [4, 0, 2, 6, 8, 3, 5]
    for action in actions:
      if py_state.is_terminal():
        break

      # Compare structs.
      py_state_dict = json.loads(py_state.to_json())
      cc_state_dict = json.loads(cc_state.to_json())
      self.assertEqual(py_state_dict, cc_state_dict)

      # Compare action structs.
      py_action_dict = json.loads(
          py_state.action_to_struct(py_state.current_player(), action).to_json()
      )
      cc_action_dict = json.loads(
          cc_state.action_to_struct(cc_state.current_player(), action).to_json()
      )
      self.assertEqual(py_action_dict, cc_action_dict)

      # Reconstruct both from struct and verify they still match.
      py_reconstructed = py_game.new_initial_state(py_state_dict)
      cc_reconstructed = cc_game.new_initial_state(cc_state_dict)
      self.assertEqual(
          json.loads(py_reconstructed.to_json()),
          json.loads(cc_reconstructed.to_json()),
      )

      py_state.apply_action(action)
      cc_state.apply_action(action)

  def test_new_initial_state_invalid_board(self):
    """Tests that invalid board dicts raise errors."""
    game = pyspiel.load_game("python_tic_tac_toe")

    # Wrong board size.
    with self.assertRaises((ValueError, RuntimeError)):
      game.new_initial_state({"board": [".", "."], "current_player": "x"})

    # Invalid piece counts (more o than x).
    with self.assertRaises((ValueError, RuntimeError)):
      game.new_initial_state({
          "board": ["o", "o", ".", ".", "x", ".", ".", ".", "."],
          "current_player": "x",
      })


if __name__ == "__main__":
  absltest.main()
