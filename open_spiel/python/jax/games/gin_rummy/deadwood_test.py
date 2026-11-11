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

"""Gin rummy deadwood on hand-verified hands, and zero-sum random games.

1. Hands with known deadwood (pure runs, a set of four, mixed melds, no
   melds) pin `calculate_deadwood_lut` and `hand_total_points`.
2. Seeded random games through the state machine end, and their returns sum
   to zero.

Card encoding, as in OpenSpiel: card = suit * 13 + rank, suits {s=0, c=1,
d=2, h=3}, ranks {A=0 .. K=12}.
"""

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import gin_rummy
from open_spiel.python.jax.games.gin_rummy.deadwood import (
    calculate_deadwood_lut,
)
from open_spiel.python.jax.games.gin_rummy.deadwood import hand_total_points


def _hand(*cards: int) -> jnp.ndarray:
  """Build a 52-d int8 hand mask from card indices.

  Args:
    *cards: The card indices.

  Returns:
    int8[52], the hand mask.
  """
  h = np.zeros(52, dtype=np.int8)
  for c in cards:
    h[c] = 1
  return jnp.asarray(h)


# Helpers for human-readable card construction.
def _spade(rank: int) -> int:
  """The card index of the spade of `rank`.

  Args:
    rank: The rank.

  Returns:
    The card index.
  """
  return 0 * 13 + rank  # s


def _club(rank: int) -> int:
  """The card index of the club of `rank`.

  Args:
    rank: The rank.

  Returns:
    The card index.
  """
  return 1 * 13 + rank  # c


def _diamond(rank: int) -> int:
  """The card index of the diamond of `rank`.

  Args:
    rank: The rank.

  Returns:
    The card index.
  """
  return 2 * 13 + rank  # d


def _heart(rank: int) -> int:
  """The card index of the heart of `rank`.

  Args:
    rank: The rank.

  Returns:
    The card index.
  """
  return 3 * 13 + rank  # h


# Rank constants.
A, T, J, Q, K = 0, 9, 10, 11, 12  # noqa: N816


class DeadwoodTest(absltest.TestCase):
  """Golden-hand deadwood values.

  Each case is hand-verifiable: deadwood = sum of unmelded card point
  values (A=1, 2-9=face, 10/J/Q/K=10).
  """

  def test_pure_10_card_run_is_gin(self):
    """A 10-card run has no deadwood."""
    # As 2s 3s 4s 5s 6s 7s 8s 9s 10s — one run of 10. All melded.
    hand = _hand(*[_spade(r) for r in range(10)])
    self.assertEqual(int(calculate_deadwood_lut(hand)), 0)

  def test_two_sets_of_four_plus_two_face_cards(self):
    """Two sets of four leave the two face cards as deadwood."""
    # A-A-A-A (set of 4) + 2-2-2-2 (set of 4) + Ks (10) + Qs (10).
    # Sets meld; Ks + Qs are deadwood = 20.
    hand = _hand(
        _spade(A),
        _club(A),
        _diamond(A),
        _heart(A),
        _spade(1),
        _club(1),
        _diamond(1),
        _heart(1),
        _spade(K),
        _spade(Q),
    )
    self.assertEqual(int(calculate_deadwood_lut(hand)), 20)

  def test_run_of_three_plus_set_of_three_plus_four_deadwood(self):
    """A run and a set of three leave four deadwood cards."""
    # 3s 4s 5s (run of 3, all spades) + Q-Q-Q (set of 3, suits s/c/d)
    # + 4 deadwood: 8h 9h Jc Kd. Deadwood = 8 + 9 + 10 + 10 = 37.
    hand = _hand(
        _spade(2),
        _spade(3),
        _spade(4),  # 3s 4s 5s
        _spade(Q),
        _club(Q),
        _diamond(Q),  # Q set
        _heart(7),
        _heart(8),  # 8h, 9h (deadwood)
        _club(J),
        _diamond(K),  # Jc, Kd (deadwood)
    )
    self.assertEqual(int(calculate_deadwood_lut(hand)), 8 + 9 + 10 + 10)

  def test_no_melds_all_deadwood(self):
    """A hand with no meld is all deadwood."""
    # A 10-card hand with no possible meld: one card per rank from
    # disjoint ranks, no two consecutive in same suit.
    # Picks: As, 3c, 5d, 7h, 9s, Jc, Kd — only 7 cards. Pad with non-
    # melding extras: 2h, 4s, 6c. That gives:
    # As(1) + 3c(3) + 5d(5) + 7h(7) + 9s(9) + Jc(10) + Kd(10)
    #     + 2h(2) + 4s(4) + 6c(6) = 57 deadwood.
    # Verify no meld possible: ranks are {0,2,4,6,8,10,12,1,3,5} —
    # consecutive pairs exist (As 2h, 3c 4s, 5d 6c, 7h, 9s, Jc, Kd)
    # but never 3 consecutive in the same suit, and no rank repeats.
    hand = _hand(
        _spade(A),
        _club(2),
        _diamond(4),
        _heart(6),
        _spade(8),
        _club(J),
        _diamond(K),
        _heart(1),
        _spade(3),
        _club(5),
    )
    # Confirm raw point total matches; with no melds, deadwood == raw.
    raw = int(hand_total_points(hand))
    self.assertEqual(raw, 57)
    self.assertEqual(int(calculate_deadwood_lut(hand)), 57)


class RandomGameTest(absltest.TestCase):
  """Multi-seed random rollouts — zero-sum + terminal reachability.

  Attributes:
    game: The game.
  """

  def setUp(self):
    """Builds the default game."""
    super().setUp()
    self.game = gin_rummy.GinRummy()

  def _random_rollout(self, seed: int) -> tuple[int, np.ndarray]:
    """Plays one random game to the end.

    Args:
      seed: The rollout seed.

    Returns:
      (steps taken, the final returns).
    """
    rng = np.random.default_rng(seed)
    state = self.game.new_initial_state()
    steps = 0
    while not bool(state["done"]):
      if bool(self.game.is_chance_node(state)):
        p = np.asarray(self.game.chance_outcomes(state))
        action = int(rng.choice(p.size, p=p))
      else:
        mask = np.asarray(self.game.legal_actions_mask(state))
        legal = np.flatnonzero(mask)
        self.assertNotEmpty(
            legal,
            f"seed={seed} step={steps}: no legal actions",
        )
        action = int(rng.choice(legal))
      state = self.game.apply_action(state, jnp.int32(action))
      steps += 1
      self.assertLess(steps, 1500, f"seed={seed}: rollout exceeded cap")
    returns = np.asarray(self.game.returns(state))
    return steps, returns

  def test_multi_seed_rollouts_are_zero_sum_and_terminate(self):
    """Random rollouts terminate and are zero-sum."""
    seeds = [0, 1, 7, 42, 137, 2026]
    for seed in seeds:
      with self.subTest(seed=seed):
        steps, returns = self._random_rollout(seed)
        self.assertEqual(returns.shape, (2,))
        self.assertAlmostEqual(
            float(returns.sum()),
            0.0,
            places=4,
            msg=f"non-zero-sum at seed={seed}: returns={returns}",
        )
        self.assertGreater(
            steps, 21, f"seed={seed}: terminated before deal complete"
        )


if __name__ == "__main__":
  absltest.main()
