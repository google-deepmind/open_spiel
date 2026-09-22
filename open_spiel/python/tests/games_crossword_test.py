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

"""Tests for the game-specific functions for crossword."""

from absl import flags
from absl.testing import absltest
import numpy as np

import pyspiel

crossword = pyspiel.crossword
crossword_duel = pyspiel.crossword_duel

flags.DEFINE_string(
    "puzzles_root", "", "The root directory of the crossword puzzles.",
)

flags.DEFINE_string(
    "word_list_file", "", "The word list file to use for the crossword game.",
)

SEED = 238711999


class GamesCrosswordTest(absltest.TestCase):

  def simulate_winning_game(self, state):
    util_return = 0.0
    while not state.is_terminal():
      print(state)
      clue_index = np.random.choice(
          [i for i, solved in enumerate(state.clue_solved()) if solved == 0]
      )
      clue = state.board().clue(clue_index)
      cid = crossword.clue_id(clue)
      answer = state.board().answer(cid)
      action = crossword.CrosswordActionStruct(cid, answer)
      print(f"\ncid = {cid}, applying action: {action}\n")
      status = state.validate_action_struct(action)
      self.assertTrue(status.ok())
      status = state.apply_action_struct(action)
      self.assertTrue(status.ok())
      self.assertGreater(state.rewards()[0], 0.0)
      self.assertGreater(state.returns()[0], util_return)
      util_return = state.returns()[0]
    print(f"Final state: \n{state}")

  def simulate_random_game(self, state):
    action_struct_sampler = state.get_action_struct_sampler(
        int(np.random.randint(0, 2**30))
    )
    while not state.is_terminal():
      print(state)
      action = action_struct_sampler.sample_action_struct()
      status = state.validate_action_struct(action)
      self.assertTrue(status.ok())
      status = state.apply_action_struct(action)
      self.assertTrue(status.ok())
    print(f"Final state: \n{state}")

  def test_default_random_sim(self):
    if not flags.FLAGS.word_list_file:
      print("No word list file provided, skipping random sim.")
      return
    game_string = (
        f"crossword(word_list_file={flags.FLAGS.word_list_file})"
    )
    game = pyspiel.load_game(game_string)
    state = game.new_initial_state()
    self.assertFalse(state.is_chance_node())
    self.simulate_random_game(state)

  def test_default_winning_sim(self):
    game = pyspiel.load_game("crossword")
    state = game.new_initial_state()
    self.assertFalse(state.is_chance_node())
    self.simulate_winning_game(state)

  def test_random_sims_with_chance(self):
    if not flags.FLAGS.word_list_file:
      print("No word list file provided, skipping random sim.")
      return
    if not flags.FLAGS.puzzles_root:
      print("No puzzles root file provided, skipping sim with chance.")
      return
    game_string = (
        f"crossword(puzzles_root={flags.FLAGS.puzzles_root},"
        f"word_list_file={flags.FLAGS.word_list_file})"
    )
    game = pyspiel.load_game(game_string)
    state = game.new_initial_state()
    self.assertTrue(state.is_chance_node())
    outcomes = state.chance_outcomes()
    action_list, prob_list = zip(*outcomes)
    outcome = np.random.choice(action_list, p=prob_list)
    state.apply_action(outcome)
    print(f"Sampled puzzle: {game.crossword_file(outcome)}")
    self.simulate_random_game(state)

  def test_winning_sims_with_chance(self):
    if not flags.FLAGS.word_list_file:
      print("No word list file provided, skipping random sim.")
      return
    if not flags.FLAGS.puzzles_root:
      print("No puzzles root file provided, skipping sim with chance.")
      return
    game_string = (
        f"crossword(puzzles_root={flags.FLAGS.puzzles_root},"
        f"word_list_file={flags.FLAGS.word_list_file})"
    )
    game = pyspiel.load_game(game_string)
    state = game.new_initial_state()
    self.assertTrue(state.is_chance_node())
    outcomes = state.chance_outcomes()
    action_list, prob_list = zip(*outcomes)
    outcome = np.random.choice(action_list, p=prob_list)
    state.apply_action(outcome)
    print(f"Sampled puzzle: {game.crossword_file(outcome)}")
    self.simulate_winning_game(state)

  def test_default_crossword_manually(self):
    action_json_strings = [
        '{"clue_id": "A1", "word": "GO"}',    # valid and correct: +1
        '{"clue_id": "D1", "word": "GAME"}',  # valid and correct: +1
        '{"clue_id": "A4", "word": "ACHE"}',  # valid but incorrect: -1
        '{"clue_id": "A4", "word": "ARGH"}',  # valid but incorrect: -1
        '{"clue_id": "A4", "word": "ARID"}',  # valid and correct: +1
        '{"clue_id": "D3", "word": "ARGH"}',  # invalid (not applied)
        '{"clue_id": "D3", "word": "ABLE"}',  # invalid (not applied)
        '{"clue_id": "D3", "word": "AMEN"}',  # invalid (not applied)
        '{"clue_id": "D3", "word": "ADAM"}',  # valid and correct: +1
        '{"clue_id": "D6", "word": "ITA"}',   # valid and correct: +1
    ]
    game = pyspiel.load_game("crossword")
    state = game.new_initial_state()
    for action_json_string in action_json_strings:
      print("Current state:")
      print(state)
      action_struct = game.ActionStruct(action_json_string)
      print(f"Applying action: {action_struct}")
      status = state.validate_action_struct(action_struct)
      if not status.ok():
        print(f"Invalid action: {status.message}")
      else:
        status = state.apply_action_struct(action_struct)
        self.assertTrue(status.ok())
    print("Final state:")
    print(state)
    self.assertEqual(state.returns()[0], 3)

  def test_create_game_from_crossword_string(self):
    puzzle_contents = """
Title: Default crossword puzzle.
Author: OpenSpiel authors
Copyright: © 2026 OpenSpiel authors
Date: 2025-06-05


GO#A
ARID
M#TA
EDAM

A1. Ancient board game with black and white stones ~ GO
A4. Dry, like a desert ~ ARID
A7. Teaching assistant ~ TA
A8. Dutch cheese ~ EDAM

D1. Chess is an example of a board ____ ~ GAME
D2. Disjunction (logical operator) ~ OR
D3. Popular optimizer ~ ADAM
D6. Invitation to Apply, abbreviated ~ ITA
"""
    game = crossword.create_game_from_crossword_string(puzzle_contents)
    state = game.new_initial_state()
    self.assertFalse(state.is_chance_node())
    action_struct = game.ActionStruct("A1", "GO")
    status = state.apply_action_struct(action_struct)
    self.assertTrue(status.ok())
    self.assertEqual(state.returns()[0], 1)

  def test_bad_json_action(self):
    game = pyspiel.load_game("crossword")
    state = game.new_initial_state()
    with self.assertRaises(RuntimeError):
      action_struct = game.ActionStruct('{"foo": "bar"}')
      state.apply_action_struct(action_struct)
    # See what the error message is.. is it informative?
    try:
      _ = game.ActionStruct('{"foo": "bar"}')
    except RuntimeError as e:
      # Prints "Error creating action struct:
      # json.exception.out_of_range.403] key 'clue_id' not found"
      print(f"Error creating action struct: {e}")


class GamesCrosswordDuelTest(absltest.TestCase):

  def apply_forced_chance_letter(self, state, letter):
    """Applies the chance action drawing `letter`, which must be available."""
    outcomes = state.chance_outcomes()
    action = ord(letter) - ord("A")
    self.assertTrue(any(outcome_action == action
                        for outcome_action, _ in outcomes))
    state.apply_action(action)

  def apply_any_chance_letter(self, state):
    """Applies an arbitrary (don't-care) chance outcome."""
    outcomes = state.chance_outcomes()
    state.apply_action(outcomes[0][0])

  def test_default_word_list(self):
    game = pyspiel.load_game("crossword_duel")
    self.assertEqual(game.num_words(), 10)
    self.assertEqual(game.num_players(), 2)

  def test_create_crossword_duel_game(self):
    game = crossword_duel.create_crossword_duel_game(
        crossword_duel.DUEL_DEFAULT_BOARD, ""
    )
    self.assertEqual(game.num_words(), 10)
    self.assertEqual(game.num_players(), 2)
    state = game.new_initial_state()
    self.assertTrue(state.is_chance_node())

  def test_deal_and_first_move(self):
    game = pyspiel.load_game("crossword_duel")
    state = game.new_initial_state()
    while state.is_chance_node():
      self.apply_any_chance_letter(state)
    self.assertLen(state.rack(0), 7)
    self.assertLen(state.rack(1), 7)
    self.assertEqual(state.current_player(), 0)

    # A word not covering the start cell should be rejected as first move.
    status = state.validate_action_struct(
        crossword_duel.CrosswordDuelActionStruct(
            0, 0, crossword_duel.Direction.ACROSS, "AB"
        )
    )
    self.assertFalse(status.ok())

    # A pass is always legal (direction/word are irrelevant when pass=true).
    pass_action = crossword_duel.CrosswordDuelActionStruct(
        0, 0, crossword_duel.Direction.ACROSS, "X"
    )
    pass_action.pass_ = True
    status = state.validate_action_struct(pass_action)
    self.assertTrue(status.ok())

  def test_legal_and_illegal_placements(self):
    # Forces both racks so that player 0 is guaranteed to hold C/A/T and
    # player 1 is guaranteed to hold B/E/A/R, then checks: (1) a known-legal
    # placement of CAT at the start cell validates as legal; (2) extending
    # the board with a letter the mover's rack doesn't have is rejected for
    # insufficient tiles; and (3) an otherwise-legal word placed away from
    # the board's only tile is rejected for not connecting to an existing
    # tile.
    game = pyspiel.load_game("crossword_duel")
    state = game.new_initial_state()

    forced_p0 = list("CAT")
    forced_p1 = list("BEAR")
    i0, i1 = 0, 0
    while state.is_chance_node():
      dealing_to_p0 = len(state.rack(0)) < 7
      if dealing_to_p0:
        if i0 < len(forced_p0):
          self.apply_forced_chance_letter(state, forced_p0[i0])
          i0 += 1
        else:
          self.apply_any_chance_letter(state)
      else:
        if i1 < len(forced_p1):
          self.apply_forced_chance_letter(state, forced_p1[i1])
          i1 += 1
        else:
          self.apply_any_chance_letter(state)
    self.assertLen(state.rack(0), 7)
    self.assertLen(state.rack(1), 7)
    self.assertEqual(state.current_player(), 0)

    # (1) CAT across the start cell (7, 7) is a legal opening move.
    legal_action = crossword_duel.CrosswordDuelActionStruct(
        7, 7, crossword_duel.Direction.ACROSS, "CAT"
    )
    status = state.validate_action_struct(legal_action)
    self.assertTrue(status.ok())
    status = state.apply_action_struct(legal_action)
    self.assertTrue(status.ok())
    self.assertEqual(state.char_at(7, 7), "C")
    self.assertEqual(state.char_at(7, 8), "A")
    self.assertEqual(state.char_at(7, 9), "T")

    # Drain the refill chance node(s) triggered by player 0's move.
    while state.is_chance_node():
      self.apply_any_chance_letter(state)
    self.assertEqual(state.current_player(), 1)

    # (2) Extending "CAT" into "CATX" needs an 'X' tile that player 1's rack
    # (B/E/A/R + filler) does not have.
    not_enough_tiles = crossword_duel.CrosswordDuelActionStruct(
        7, 7, crossword_duel.Direction.ACROSS, "CATX"
    )
    status = state.validate_action_struct(not_enough_tiles)
    self.assertFalse(status.ok())
    self.assertIn("Not enough", status.message())

    # (3) "BEAR" is spellable from player 1's rack, but placed far from the
    # only tiles on the board.
    not_connected = crossword_duel.CrosswordDuelActionStruct(
        0, 0, crossword_duel.Direction.ACROSS, "BEAR"
    )
    status = state.validate_action_struct(not_connected)
    self.assertFalse(status.ok())
    self.assertIn("connect", status.message())

  def test_action_struct_from_json(self):
    action = crossword_duel.CrosswordDuelActionStruct(
        '{"row": 7, "col": 7, "direction": "across", "word": "CAT", '
        '"pass": false}'
    )
    self.assertEqual(action.row, 7)
    self.assertEqual(action.col, 7)
    self.assertEqual(action.direction, crossword_duel.Direction.ACROSS)
    self.assertEqual(action.word, "CAT")
    self.assertFalse(action.pass_)

  def test_default_random_sim(self):
    # Only 10 words in the default word list, so most random games end
    # quickly via consecutive passes.
    game = pyspiel.load_game("crossword_duel")
    state = game.new_initial_state()
    rng = np.random.RandomState(SEED)
    while not state.is_terminal():
      if state.is_chance_node():
        outcomes = state.chance_outcomes()
        action_list, prob_list = zip(*outcomes)
        action = rng.choice(action_list, p=prob_list)
        state.apply_action(action)
      else:
        sampler = state.get_action_struct_sampler(
            int(rng.randint(0, 2**30))
        )
        action = sampler.sample_action_struct()
        status = state.apply_action_struct(action)
        self.assertTrue(status.ok())
    print(f"Final state: \n{state}")
    self.assertEqual(sum(state.returns()), 0.0)


if __name__ == "__main__":
  np.random.seed(SEED)
  absltest.main()
