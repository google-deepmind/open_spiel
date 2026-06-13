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


if __name__ == "__main__":
  np.random.seed(SEED)
  absltest.main()
