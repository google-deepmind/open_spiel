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

"""league.payoff_views on hand-made table records (numpy, no jax).

Known answers: the pair cell is half the paired difference of the pair
scores, exactly antisymmetric and centred under a constant sum c != 0; the
field cell is position 0's mean side return; PFSP win rates count wins of
position 0 over each member where it sits, ties split.
"""

import sys

import numpy as np
import pytest

from open_spiel.python.scale.core import sides as sides_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.eval import designs
from open_spiel.python.scale.eval import payoff
from open_spiel.python.scale.eval import tables
from open_spiel.python.scale.league import payoff_views

TWO = sides_lib.Sides.individual(2, constant_sum=True)
BRIDGE = sides_lib.Sides.partnerships(4, 2, constant_sum=True)
IND3 = sides_lib.Sides.individual(3, constant_sum=True)


def _table(
    sides, design_name, lineup_sides, side_returns, entrants=("a", "b", "c")
):
  """A TableResult whose rows give each table side the given side return.

  Every seat of a side scores its side's return.

  Args:
    sides: the table's Sides.
    design_name: the eval.designs design.
    lineup_sides: i32[n, B, T] the entrant per side position.
    side_returns: f64[n, R, T] each row's side returns.
    entrants: the entrant names.

  Returns:
    The TableResult.
  """
  d = designs.make(design_name, sides)
  side_lineup = np.asarray(lineup_sides, np.int32)  # [n, B, T]
  lineup = (
      designs.seat_lineup(sides, side_lineup)
      if sides.num_players != sides.num_sides
      else side_lineup
  )
  sr = np.asarray(side_returns, np.float64)  # [n, R, T]
  returns = sr[:, :, list(sides.side_of_seat)]
  rules = (types.SAMPLE,) * len(entrants)
  return tables.table_result(
      "g", "c", d, entrants, rules, lineup, returns, 1, True
  )


def test_view_kind_and_designs():
  """View kind and designs."""
  assert (
      payoff_views.view_kind(TWO) == "pair"
      and payoff_views.view_kind(BRIDGE) == "pair"
  )
  assert payoff_views.view_kind(IND3) == "field"
  assert payoff_views.resolve_design("auto", BRIDGE) == "swap2"
  assert payoff_views.resolve_design("auto", IND3) == "cyclic"
  assert (
      payoff_views.resolve_design("cyclic+reversed", IND3) == "cyclic+reversed"
  )
  with pytest.raises(ValueError):
    payoff_views.resolve_design("cyclic", BRIDGE)
  with pytest.raises(ValueError):
    payoff_views.resolve_design("swap2", IND3)
  with pytest.raises(ValueError):
    payoff_views.resolve_design("substitute", IND3)
  assert payoff_views.design_names() == (
      "auto",
      "swap2",
      "cyclic",
      "cyclic+reversed",
      "cyclic+classes",
  )
  # Uneven sides: auto is cyclic+classes at more than two sides; two uneven
  # sides are two roles, a field view (no pair view), and refuse a design that
  # rotates every side.
  three_v_one_v_one = sides_lib.Sides((0, 1, 0, 2, 0), True)
  assert payoff_views.view_kind(three_v_one_v_one) == "field"
  assert (
      payoff_views.resolve_design("auto", three_v_one_v_one) == "cyclic+classes"
  )
  two_v_one = sides_lib.Sides((0, 1, 0), True)
  assert payoff_views.view_kind(two_v_one) == "field"
  assert payoff_views.resolve_design("auto", two_v_one) == "cyclic+classes"
  for name in ("swap2", "cyclic"):
    with pytest.raises(ValueError):
      payoff_views.resolve_design(name, two_v_one)


@pytest.mark.parametrize("c", [0.0, 3.0])
def test_side_cell_is_half_the_paired_difference_antisymmetric_and_centred(c):
  """Side cell is half the paired difference antisymmetric and centred."""
  rng = np.random.default_rng(0)
  n = 64
  x = rng.normal(size=(n, 2))  # a's side return on rows 0, 1
  sr = np.stack([x, c - x], axis=-1)  # [n, 2 rows, 2 sides] by TABLE side
  # swap2: row 0 a at side 0, row 1 a at side 1: a's return is sr[:,0,0],
  # sr[:,1,1].
  sr[:, 1] = sr[:, 1, ::-1]
  res = _table(BRIDGE, "swap2", np.tile([[0, 1]], (n, 1, 1)), sr)
  ma, sa, mb, sb = payoff_views.side_cell(res)
  sa_ = res.deal_scores(0)
  sb_ = res.deal_scores(1)
  want = 0.5 * (sa_ - sb_)
  assert ma == pytest.approx(want.mean(), abs=1e-12)
  assert mb == -ma and sb == sa
  # centred: equals a's score minus c / 2
  assert ma == pytest.approx(sa_.mean() - c / 2, abs=1e-12)
  # a pair scorer (bridge IMPs-like, odd) is applied per position

  def scorer(r0, r1):
    """An odd pair scorer.

    Args:
      r0: Position 0's return.
      r1: Position 1's return.

    Returns:
      tanh(r0 - r1).
    """
    return np.tanh(r0 - r1)

  ma2, _, mb2, _ = payoff_views.side_cell(res, scorer)
  assert ma2 == pytest.approx(
      (
          0.5 * (res.deal_scores(0, 0, scorer) - res.deal_scores(1, 0, scorer))
      ).mean(),
      abs=1e-12,
  )
  assert mb2 == -ma2


def test_side_cell_refuses_more_sides_and_other_designs():
  """Side cell refuses more sides and other designs."""
  n = 4
  sr = np.zeros((n, 3, 3))
  res = _table(IND3, "cyclic", np.tile([[0, 1, 2]], (n, 1, 1)), sr)
  with pytest.raises(ValueError):
    payoff_views.side_cell(res)


def test_field_cell_is_position_zero_side_return():
  """Field cell is position zero side return."""
  n = 16
  rng = np.random.default_rng(1)
  sr = rng.normal(size=(n, 3, 3))
  res = _table(
      IND3,
      "cyclic",
      np.tile([[0, 1, 1]], (n, 1, 1)),
      sr - sr.mean(-1, keepdims=True),
  )
  m, s = payoff_views.field_cell(res)
  e = res.position_score(0)
  assert (m, s) == (e.mean, e.se)
  with pytest.raises(ValueError):
    payoff_views.field_cell(
        _table(TWO, "swap2", np.tile([[0, 1]], (n, 1, 1)), np.zeros((n, 2, 2)))
    )


def test_pfsp_winrates_two_sides_is_the_win_rate_ties_split():
  """Pfsp winrates two sides is the win rate ties split."""
  # a beats b on deal 0 both rows, ties deal 1 row 0, loses deal 1 row 1.
  # a at side 1 gets -1 on row 1: a loses
  sr = np.array([
      [
          [1.0, -1.0],
          [-1.0, 1.0],
      ],  # row 1 swapped sides: a at side 1 (wins: +1)
      [[0.0, 0.0], [1.0, -1.0]],
  ])
  res = _table(TWO, "swap2", np.tile([[0, 1]], (2, 1, 1)), sr)
  w = payoff_views.pfsp_winrates(res)
  assert w == {"b": (pytest.approx((1 + 1 + 0.5 + 0) / 4), 4)}


def test_pfsp_winrates_per_member_where_it_sits():
  """Pfsp winrates per member where it sits."""
  # Three sides, cyclic: entrant 0 at position 0; deal 0 field (b, c), deal 1
  # (c, c).
  lineup = np.array([[[0, 1, 2]], [[0, 2, 2]]])
  sr = np.zeros((2, 3, 3))
  d = designs.make("cyclic", IND3)
  for i in range(2):
    for r in range(3):
      perm = d.side_perms[r]  # base side k at table side perm[k]
      vals = [2.0, 1.0, 3.0] if i == 0 else [2.0, 2.0, 1.0]  # by base position
      for k in range(3):
        sr[i, r, perm[k]] = vals[k]
  res = _table(IND3, "cyclic", lineup, sr)
  w = payoff_views.pfsp_winrates(res)
  # b sits only at position 1 on deal 0 (3 rows): a 2 > 1 -> 3 wins.
  # c: deal 0 position 2 (a 2 < 3: 0), deal 1 positions 1 (tie: 1/2) and 2 (2 >
  # 1: 1).
  assert w["b"] == (1.0, 3)
  assert w["c"] == (pytest.approx((0 * 3 + 0.5 * 3 + 1 * 3) / 9), 9)


def test_check_antisymmetric():
  """Check antisymmetric."""
  pm = payoff.PayoffMatrix(
      ("a", "b"),
      np.array([[0.0, 1.0], [-1.0, 0.0]]),
      np.zeros((2, 2)),
      np.zeros((2, 2)),
      np.zeros((2, 2)),
      0,
      {},
  )
  payoff_views.check_antisymmetric(pm)
  bad = pm._replace(mean=np.array([[0.0, 1.0], [-0.5, 0.0]]))
  with pytest.raises(AssertionError, match="antisymmetric"):
    payoff_views.check_antisymmetric(bad)


def test_league_table_prints_every_entrant():
  """League table prints every entrant."""
  pm = payoff.PayoffMatrix(
      ("a@0", "b@1"),
      np.array([[0.0, 1.0], [-1.0, 0.0]]),
      np.zeros((2, 2)),
      np.zeros((2, 2)),
      np.zeros((2, 2)),
      0,
      {},
  )
  out = payoff_views.league_table(pm, "pair")
  assert "a@0" in out and "b@1" in out and "+1.0000" in out
  with pytest.raises(ValueError):
    payoff_views.league_table(pm, "matrix")


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
