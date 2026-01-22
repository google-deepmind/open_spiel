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
"""Tests for Python Hangman game."""

from absl import flags
from absl.testing import absltest
import numpy as np

from open_spiel.python.games import hangman
import pyspiel

_WORD_LIST_FILE = flags.DEFINE_string("word_list_file", "", "Word list file")

_SEED = 329827811
_NUM_SIMULATIONS = 100


class HangmanTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(_SEED)
    params = {"word_list_file": _WORD_LIST_FILE.value}
    self.game = pyspiel.load_game("python_hangman", params)

  def run_random_simulation(self, word: str | None = None):
    if word is not None:
      # Start with a specific word.
      state = self.game.new_initial_state(word)
    else:
      # Randomly sample a word via an initial chance node.
      state = self.game.new_initial_state()
      self.assertEqual(state.current_player(), pyspiel.PlayerId.CHANCE)
      outcomes = state.chance_outcomes()
      action_list, prob_list = zip(*outcomes)
      action = np.random.choice(action_list, p=prob_list)
      print("Sampled outcome: ",
            state.action_to_string(state.current_player(), action))
      state.apply_action(action)
    print(state)
    self.assertEqual(state.current_player(), 0)
    self.assertLen(state.legal_actions(), hangman.NUM_DISTINCT_ACTIONS)
    while not state.is_terminal():
      action = np.random.choice(state.legal_actions())
      action_string = state.action_to_string(state.current_player(), action)
      print(f"Player {state.current_player()} " +
            f"randomly sampled action: {action_string}")
      state.apply_action(action)
      print(f"Rewards: {state.rewards()}")
      print(state)
    print("Terminal state:")
    print(state)
    print(f"Rewards: {state.rewards()}")
    print(f"Returns: {state.returns()}")

  def test_random_simulations_chance_node(self):
    """Tests basic API functions."""
    for g in range(_NUM_SIMULATIONS):
      print("")
      print(f"Starting game: {g}")
      self.run_random_simulation()

  def test_random_simulations_specific_words(self):
    for word in ["pineapple", "does", "not", "belong", "on", "pizza"]:
      print("")
      print(f"Starting game with fixed word: {word}")
      self.run_random_simulation(word)

  def test_word_with_spaces(self):
    state = self.game.new_initial_state("hello world")
    self.assertEqual(state.current_player(), 0)
    action_sequence = [ord("h") - 97, ord("e") - 97, ord("l") - 97,
                       ord("o") - 97, ord("w") - 97, ord("r") - 97,
                       ord("d") - 97]
    for action in action_sequence:
      print("")
      print(state)
      print(f"Guessing letter: {chr(action + 97)}")
      state.apply_action(action)
      self.assertGreater(state.rewards()[0], 0.0)
    print("Final state:")
    print(state)
    self.assertTrue(state.is_terminal())
    self.assertAlmostEqual(state.returns()[0],
                           7*hangman.CORRECT_GUESS_REWARD + hangman.WIN_REWARD)

  def test_all_incorrect_guesses(self):
    state = self.game.new_initial_state("banana")
    self.assertEqual(state.current_player(), 0)
    action_sequence = [ord("h") - 97, ord("e") - 97, ord("l") - 97,
                       ord("o") - 97, ord("w") - 97, ord("r") - 97]
    for action in action_sequence:
      self.assertFalse(state.is_terminal())
      print("")
      print(state)
      print(f"Guessing letter: {chr(action + 97)}")
      state.apply_action(action)
      self.assertLessEqual(state.rewards()[0], 0.0)
    self.assertTrue(state.is_terminal())
    print("Final state:")
    print(state)
    self.assertAlmostEqual(state.returns()[0], self.game.min_utility())


if __name__ == "__main__":
  absltest.main()
