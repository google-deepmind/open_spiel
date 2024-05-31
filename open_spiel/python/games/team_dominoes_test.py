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
"""Tests for Latin American Python Dominoes."""


from absl.testing import absltest
from open_spiel.python.games import team_dominoes
import pyspiel


class DominoesTest(absltest.TestCase):

  def test_game_from_cc(self):
    """Runs our standard game tests, checking API consistency."""
    game = pyspiel.load_game("python_team_dominoes")
    pyspiel.random_sim_test(game, num_sims=100, serialize=False, verbose=True)

  def test_single_deterministic_game_1(self):
    """Runs a single game where tiles and actions chose deterministically."""
    game = pyspiel.load_game("python_team_dominoes")
    state = game.new_initial_state()
    hand0 = [
        (1.0, 3.0),
        (0.0, 5.0),
        (1.0, 1.0),
        (2.0, 3.0),
        (4.0, 5.0),
        (3.0, 5.0),
        (0.0, 1.0),
    ]
    hand1 = [
        (2.0, 5.0),
        (3.0, 4.0),
        (2.0, 2.0),
        (0.0, 4.0),
        (3.0, 3.0),
        (2.0, 6.0),
        (1.0, 6.0),
    ]
    hand2 = [
        (5.0, 6.0),
        (6.0, 6.0),
        (1.0, 4.0),
        (2.0, 4.0),
        (4.0, 4.0),
        (0.0, 0.0),
        (1.0, 5.0),
    ]
    hand3 = [
        (4.0, 6.0),
        (0.0, 2.0),
        (0.0, 3.0),
        (3.0, 6.0),
        (5.0, 5.0),
        (1.0, 2.0),
        (0.0, 6.0),
    ]

    self.deal_hands(state, [hand0, hand1, hand2, hand3])

    self.apply_action(state, team_dominoes.Action(0, (3.0, 4.0), None))
    self.apply_action(state, team_dominoes.Action(1, (2.0, 4.0), 4.0))
    self.apply_action(state, team_dominoes.Action(2, (1.0, 2.0), 2.0))
    self.apply_action(state, team_dominoes.Action(3, (0.0, 3.0), 3.0))

    self.apply_action(state, team_dominoes.Action(0, (1.0, 3.0), 1.0))
    self.apply_action(state, team_dominoes.Action(1, (3.0, 5.0), 3.0))
    self.apply_action(state, team_dominoes.Action(2, (0.0, 2.0), 0.0))
    self.apply_action(state, team_dominoes.Action(3, (2.0, 5.0), 2.0))

    self.apply_action(state, team_dominoes.Action(0, (1.0, 5.0), 5.0))
    self.apply_action(state, team_dominoes.Action(1, (0.0, 5.0), 5.0))
    self.apply_action(state, team_dominoes.Action(2, (1.0, 1.0), 1.0))
    self.apply_action(state, team_dominoes.Action(3, (0.0, 6.0), 0.0))

    self.apply_action(state, team_dominoes.Action(0, (3.0, 6.0), 6.0))
    self.apply_action(state, team_dominoes.Action(1, (1.0, 6.0), 1.0))
    self.apply_action(state, team_dominoes.Action(2, (5.0, 6.0), 6.0))
    self.apply_action(state, team_dominoes.Action(3, (3.0, 3.0), 3.0))

    self.apply_action(state, team_dominoes.Action(0, (4.0, 5.0), 5.0))
    self.apply_action(state, team_dominoes.Action(1, (4.0, 6.0), 4.0))
    self.apply_action(state, team_dominoes.Action(3, (6.0, 6.0), 6.0))

    self.apply_action(state, team_dominoes.Action(0, (2.0, 6.0), 6.0))
    self.apply_action(state, team_dominoes.Action(1, (2.0, 2.0), 2.0))
    self.apply_action(state, team_dominoes.Action(3, (2.0, 3.0), 3.0))
    # Game is stuck! No player can play any tile as all 2.0s are played

    self.assertTrue(state.is_terminal())
    self.assertEqual(state.returns()[0], -18)
    self.assertEqual(state.returns()[1], 18)
    self.assertEqual(state.returns()[2], -18)
    self.assertEqual(state.returns()[3], 18)

  def test_single_deterministic_game_2(self):
    """Runs a single game where tiles and actions chose deterministically."""
    game = pyspiel.load_game("python_team_dominoes")
    state = game.new_initial_state()
    hand0 = [
        (0.0, 6.0),
        (3.0, 6.0),
        (1.0, 3.0),
        (1.0, 4.0),
        (5.0, 5.0),
        (0.0, 0.0),
        (2.0, 6.0),
    ]
    hand1 = [
        (1.0, 5.0),
        (2.0, 2.0),
        (0.0, 2.0),
        (0.0, 3.0),
        (4.0, 5.0),
        (6.0, 6.0),
        (5.0, 6.0),
    ]
    hand2 = [
        (2.0, 4.0),
        (3.0, 4.0),
        (3.0, 3.0),
        (0.0, 4.0),
        (1.0, 1.0),
        (1.0, 6.0),
        (3.0, 5.0),
    ]
    hand3 = [
        (0.0, 5.0),
        (0.0, 1.0),
        (4.0, 4.0),
        (2.0, 3.0),
        (1.0, 2.0),
        (2.0, 5.0),
        (4.0, 6.0),
    ]

    self.deal_hands(state, [hand0, hand1, hand2, hand3])

    self.apply_action(state, team_dominoes.Action(0, (0.0, 6.0), None))
    self.apply_action(state, team_dominoes.Action(1, (0.0, 5.0), 0.0))
    self.apply_action(state, team_dominoes.Action(2, (2.0, 6.0), 6.0))
    self.apply_action(state, team_dominoes.Action(3, (1.0, 5.0), 5.0))

    self.apply_action(state, team_dominoes.Action(0, (2.0, 3.0), 2.0))
    self.apply_action(state, team_dominoes.Action(1, (3.0, 6.0), 3.0))
    self.apply_action(state, team_dominoes.Action(2, (1.0, 3.0), 1.0))
    self.apply_action(state, team_dominoes.Action(3, (1.0, 6.0), 6.0))

    self.apply_action(state, team_dominoes.Action(0, (3.0, 5.0), 3.0))
    self.apply_action(state, team_dominoes.Action(1, (5.0, 6.0), 5.0))
    self.apply_action(state, team_dominoes.Action(2, (1.0, 1.0), 1.0))
    self.apply_action(state, team_dominoes.Action(3, (4.0, 6.0), 6.0))

    # skipped player 0 (has no 4.0 or 1.0 to play)
    self.apply_action(state, team_dominoes.Action(1, (0.0, 4.0), 4.0))
    self.apply_action(state, team_dominoes.Action(2, (0.0, 1.0), 1.0))
    # skipped player 3 (has no 0.0s to play)

    # skipped over player 0 (has no 0.0s to play)
    self.apply_action(state, team_dominoes.Action(1, (0.0, 0.0), 0.0))
    self.apply_action(state, team_dominoes.Action(2, (0.0, 3.0), 0.0))
    self.apply_action(state, team_dominoes.Action(3, (3.0, 4.0), 3.0))

    # skipped over player 0 (has no 0.0s nor 4.0s to play)
    self.apply_action(state, team_dominoes.Action(1, (0.0, 2.0), 0.0))
    self.apply_action(state, team_dominoes.Action(2, (2.0, 4.0), 2.0))
    self.apply_action(state, team_dominoes.Action(3, (1.0, 4.0), 4.0))

    # skipped over player 0 (has no 1.0s nor 4.0s to play)
    self.apply_action(state, team_dominoes.Action(1, (1.0, 2.0), 1.0))
    # player 1 won (no more tiles to play)

    self.assertTrue(state.is_terminal())
    self.assertEqual(state.returns()[0], -39)
    self.assertEqual(state.returns()[1], 39)
    self.assertEqual(state.returns()[2], -39)
    self.assertEqual(state.returns()[3], 39)

  def apply_action(self, state, action):
    actions_str = team_dominoes._ACTIONS_STR
    state.apply_action(actions_str.index(str(action)))

  def deal_hands(self, state, hands):
    deck = team_dominoes._DECK
    for hand in hands:
      for t in hand:
        state.apply_action(deck.index(t))


if __name__ == "__main__":
  absltest.main()
