# Copyright 2026 DeepMind Technologies Limited
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

"""Hearts known answers.

The move-for-move check against pyspiel is in openspiel_equivalence (four
configs, every parameter).
"""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np

from open_spiel.python.jax.games import hearts
from open_spiel.python.jax.games import registry

try:
  import pyspiel  # pytype: disable=import-error

  _HAS_PYSPIEL = True
except ImportError:
  _HAS_PYSPIEL = False


def _moon_deal() -> list[int]:
  """Deal order (the i-th card goes to seat i % 4) giving North all 13 clubs.

  Returns:
    The 52 cards in deal order.
  """
  clubs = [c for c in range(52) if c % 4 == hearts.CLUBS]
  rest = [c for c in range(52) if c % 4 != hearts.CLUBS]
  order = []
  for i in range(13):
    order += [clubs[i]] + rest[3 * i : 3 * i + 3]
  return order


def _play_lowest(game, actions):
  """Plays `actions` (chance), then the lowest legal card for every seat.

  Args:
    game: The Hearts game.
    actions: The chance outcomes to apply first.

  Returns:
    The terminal state and the actions played.
  """
  apply = jax.jit(game.apply_action)
  mask = jax.jit(game.legal_actions_mask)
  s = game.new_initial_state()
  history = []
  for a in actions:
    s = apply(s, a)
    history.append(a)
  while not bool(game.is_terminal(s)):
    a = int(np.flatnonzero(np.asarray(mask(s)))[0])
    s = apply(s, a)
    history.append(a)
  return s, history


class HeartsTest(parameterized.TestCase):
  """Hearts known answers."""

  def test_registered_with_pyspiel_sizes(self):
    """Registered with pyspiel sizes."""
    game = registry.load_game("hearts")
    self.assertIsInstance(game, hearts.Hearts)
    self.assertEqual(
        (game.num_players, game.num_actions, game.max_chance_outcomes),
        (4, 52, 52),
    )
    self.assertEqual(game.info_state_shape, (5088,))

  def test_rejects_non_bool_parameters(self):
    """Rejects non bool parameters."""
    with self.assertRaises(TypeError):
      hearts.Hearts(jd_bonus=1)

  @parameterized.parameters(
      # North takes every trick: the moon. 26 - (26 - 26) for North, 26 - 26
      # others.
      ("hearts", [26.0, 0.0, 0.0, 0.0]),
      # The jack of diamonds alone is not a point trick, so North still shot
      # the moon at 16 points: 26 - (16 - 26) = 36; a total-is-26 test misses
      # it.
      ("hearts(jd_bonus=true)", [36.0, 0.0, 0.0, 0.0]),
      # The avoid-all-tricks bonus is not paid after a moon.
      ("hearts(avoid_all_tricks_bonus=true)", [26.0, 0.0, 0.0, 0.0]),
  )
  def test_known_answer_moon(self, game_str, expected):
    """Known answer moon.

    Args:
      game_str: the game string.
      expected: the returns.
    """
    game = registry.load_game(game_str)
    s, history = _play_lowest(game, [hearts.PASS_NONE] + _moon_deal())
    np.testing.assert_array_equal(np.asarray(game.returns(s)), expected)
    self.assertTrue(np.all(np.asarray(s.trick_winner) == 0))
    if _HAS_PYSPIEL:
      os_state = pyspiel.load_game(game_str).new_initial_state()
      for a in history:
        os_state.apply_action(a)
      self.assertTrue(os_state.is_terminal())
      np.testing.assert_array_equal(os_state.returns(), expected)

  def test_returns_total_78_or_26_after_a_moon(self):
    """Returns total 78 or 26 after a moon."""
    game = registry.load_game("hearts")
    rng = np.random.default_rng(0)
    for _ in range(8):
      deal = [int(c) for c in rng.permutation(52)]
      s, _ = _play_lowest(game, [int(rng.integers(4))] + deal)
      r = np.asarray(game.returns(s))
      winners = np.asarray(s.trick_winner)
      pts = np.asarray(s.trick_points)
      moon = len(set(winners[pts != 0].tolist())) == 1
      self.assertEqual(r.sum(), 26.0 if moon else 78.0)

  def test_pass_cards_false_keeps_pyspiel_chance_quirk(self):
    """Pass cards false keeps pyspiel chance quirk."""
    game = registry.load_game("hearts(pass_cards=false)")
    s = game.new_initial_state()
    np.testing.assert_array_equal(
        np.flatnonzero(np.asarray(game.chance_outcomes(s))), [0, 1, 2, 3]
    )
    np.testing.assert_array_equal(
        np.flatnonzero(np.asarray(game.legal_actions_mask(s))), [0]
    )


if __name__ == "__main__":
  absltest.main()
