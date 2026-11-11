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

"""play_checkpoints with any number of players: seats, keys and payoffs.

Fast: nothing is trained, compiled or played. At two players the seating,
keys and payoffs are exactly the seat-swapped pairs the script played before
it took more players (the Evaluator's keys); at P players deal i is played P
times, a in seat r of its r-th game and b in every other seat.
"""

import sys

import jax
import numpy as np
import pytest

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import arena_engine
from open_spiel.python.scale.core import types
from open_spiel.python.scale.eval import payoff
from open_spiel.python.scale.examples import play_checkpoints


def _key_bytes(keys):
  """The raw data of typed jax keys, as numpy.

  Args:
    keys: typed keys.

  Returns:
    Their key data.
  """
  return np.asarray(jax.random.key_data(keys))


def test_lineup_puts_a_in_one_seat_and_b_in_the_others():
  """Lineup puts a in one seat and b in the others."""
  assert play_checkpoints.lineup_slots(2) == [0, 1]
  assert play_checkpoints.lineup_slots(4) == [0, 1, 1, 1]
  table = arena_lib.rotated_table(3, play_checkpoints.lineup_slots(4), [0] * 4)
  slot = np.asarray(table.seat_slot)
  assert slot.shape == (12, 4)
  for row in range(12):
    r = row % 4
    # a (slot 0) holds seat r of the r-th game of each deal, b the rest
    assert np.flatnonzero(slot[row] == 0).tolist() == [r], (row, slot[row])


@pytest.mark.parametrize("seed,n", [(0, 5), (11, 3)])
def test_two_player_keys_are_the_evaluators(seed, n):
  """Two-player keys are the Evaluator's (arena_engine.deal_keys)."""
  d0, p0 = arena_engine.deal_keys(jax.random.key(seed), n, True)
  d1, p1 = play_checkpoints.match_keys(jax.random.key(seed), n, 2)
  np.testing.assert_array_equal(_key_bytes(d0), _key_bytes(d1))
  np.testing.assert_array_equal(_key_bytes(p0), _key_bytes(p1))


def test_four_player_keys_share_the_deal_and_not_the_policy_keys():
  """Four-player keys share the deal key within a deal, nothing else."""
  dk, pk = play_checkpoints.match_keys(jax.random.key(3), 5, 4)
  dk, pk = _key_bytes(dk), _key_bytes(pk)
  assert dk.shape[0] == 20 and pk.shape[:2] == (20, 4)
  dk = dk.reshape(5, 4, -1)
  assert np.all(dk == dk[:, :1])
  assert len({dk[i, 0].tobytes() for i in range(5)}) == 5
  policy_keys = set()
  for r in range(20):
    policy_keys.update(pk[r, q].tobytes() for q in range(4))
  assert len(policy_keys) == 80
  # A deal's keys do not depend on how many deals are drawn.
  dk2, _ = play_checkpoints.match_keys(jax.random.key(3), 2, 4)
  np.testing.assert_array_equal(_key_bytes(dk2).reshape(2, 4, -1), dk[:2])


def test_two_player_seat_payoffs_are_the_seat_swapped_pairs():
  """Two-player seat payoffs are the seat-swapped pairs."""
  rng = np.random.default_rng(0)
  pay = rng.normal(size=(6, 2))
  a_seat, a_pay, b_pay = play_checkpoints.seat_payoffs(pay, 2)
  np.testing.assert_array_equal(a_seat, [0, 1, 0, 1, 0, 1])
  rows = np.arange(6)
  # the formula the script used before it took more players
  np.testing.assert_array_equal(a_pay, pay[rows, a_seat].reshape(3, 2))
  np.testing.assert_array_equal(b_pay, pay[rows, 1 - a_seat].reshape(3, 2))


def test_four_player_seat_payoffs():
  """Four-player seat payoffs: a's seat, b's mean over its three seats."""
  pay = np.arange(32, dtype=np.float64).reshape(8, 4)
  a_seat, a_pay, b_pay = play_checkpoints.seat_payoffs(pay, 4)
  np.testing.assert_array_equal(a_seat, [0, 1, 2, 3] * 2)
  assert a_pay.shape == b_pay.shape == (2, 4)
  for row in range(8):
    i, r = divmod(row, 4)
    assert a_pay[i, r] == pay[row, r]
    others = [pay[row, q] for q in range(4) if q != r]
    assert b_pay[i, r] == pytest.approx(np.mean(others))
  with pytest.raises(ValueError, match="4 players"):
    play_checkpoints.seat_payoffs(pay[:6], 4)
  with pytest.raises(ValueError, match="4 players"):
    play_checkpoints.seat_payoffs(pay[:, :3], 4)


def test_match_result_takes_any_number_of_seats():
  """match_result: one seat estimate per column; two seats unchanged."""
  rng = np.random.default_rng(1)
  rule = types.ARGMAX
  a2, b2 = rng.normal(size=(50, 2)), rng.normal(size=(50, 2))
  r2 = payoff.match_result("g", "c", 1, True, "a", "b", rule, rule, a2, b2)
  assert len(r2.seat) == 2
  assert r2.seat[1].mean == pytest.approx(a2[:, 1].mean())
  # 2n matches at two seats: the unpaired SE the function always computed
  assert r2.se_unpaired == pytest.approx(
      a2.reshape(-1).std(ddof=1) / np.sqrt(100)
  )
  a4, b4 = rng.normal(size=(50, 4)), rng.normal(size=(50, 4))
  r4 = payoff.match_result("g", "c", 1, True, "a", "b", rule, rule, a4, b4)
  assert len(r4.seat) == 4
  for r in range(4):
    assert r4.seat[r].mean == pytest.approx(a4[:, r].mean())
  assert r4.mean.mean == pytest.approx(a4.mean())
  assert r4.se_unpaired == pytest.approx(
      a4.reshape(-1).std(ddof=1) / np.sqrt(200)
  )
  with pytest.raises(ValueError, match="one shape"):
    payoff.match_result("g", "c", 1, True, "a", "b", rule, rule, a4, b2)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
