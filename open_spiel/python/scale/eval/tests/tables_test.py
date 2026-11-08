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

"""Seating designs, TableResult estimators and the Evaluator's table methods.

Without jax.

Known answers: hand-made table records; a toy N-seat game whose numpy engine
pays each seat strength(its entrant) - mean strength at the table, plus a
per-deal seat bias that sums to 0 over the seats. Every design that moves
an entrant through every side cancels the bias deal by deal, so position
scores are exact (SE 0) under duplicate deals and noisy under independent
ones.
"""

import itertools
import sys

import numpy as np
import pytest

from open_spiel.python.scale.core import rng
from open_spiel.python.scale.core import sides as sides_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.eval import designs
from open_spiel.python.scale.eval import match
from open_spiel.python.scale.eval import payoff
from open_spiel.python.scale.eval import stats
from open_spiel.python.scale.eval import tables

IND3 = sides_lib.Sides.individual(3, constant_sum=True)
IND2 = sides_lib.Sides.individual(2, constant_sum=True)
TEAM = sides_lib.Sides.partnerships(4, 2, constant_sum=True)


# -- designs -------------------------------------------------------------------
def test_swap2_at_two_seats_is_the_duplicate_pair():
  """Swap2 at two seats is the duplicate pair."""
  d = designs.swap2(IND2)
  assert (
      d.seat_perms == ((0, 1), (1, 0)) and d.num_rows == 2 and d.num_blocks == 1
  )
  rows = designs.table_rows(d, np.array([[[5, 7]], [[1, 2]]]))
  # slots stand in for entrants
  t = payoff.duplicate_table([5, 1], [0, 0], [7, 2], [0, 0])
  np.testing.assert_array_equal(rows.reshape(4, 2), t.seat_slot)
  np.testing.assert_array_equal(
      [perm[0] for perm in d.seat_perms] * 2, t.perspective_seat
  )


def test_swap2_for_partnerships_swaps_whole_sides():
  """Swap2 for partnerships swaps whole sides."""
  d = designs.swap2(TEAM)
  assert d.seat_perms == ((0, 1, 2, 3), (1, 0, 3, 2))
  np.testing.assert_array_equal(d.side_perms, [[0, 1], [1, 0]])
  rows = designs.table_rows(d, designs.seat_lineup(TEAM, [[[4, 9]]]))
  np.testing.assert_array_equal(rows[0], [[4, 9, 4, 9], [9, 4, 9, 4]])


def test_cyclic_and_reversed_cover_the_orders():
  """Cyclic and reversed cover the orders."""
  c = designs.cyclic(IND3)
  np.testing.assert_array_equal(c.side_perms, [[0, 1, 2], [1, 2, 0], [2, 0, 1]])
  cr = designs.cyclic_reversed(IND3)
  assert cr.num_rows == 6
  assert {tuple(x) for x in cr.side_perms.tolist()} == set(
      itertools.permutations(range(3))
  )
  # Every position holds every side once per cyclic replay.
  for d in (c, designs.cyclic(sides_lib.Sides.individual(4, False))):
    for k in range(d.sides.num_sides):
      assert sorted(d.side_perms[:, k].tolist()) == list(
          range(d.sides.num_sides)
      )


def test_swap2_partner_moves_every_entrant_through_every_seat():
  """Swap2 partner moves every entrant through every seat."""
  d = designs.swap2_partner(TEAM)
  assert d.num_rows == 4 and d.seat_perms[:2] == designs.swap2(TEAM).seat_perms
  for q in range(4):
    assert sorted(perm[q] for perm in d.seat_perms) == [0, 1, 2, 3]
  _ = d.side_perms  # whole sides onto whole sides


def test_substitute_blocks_share_policy_key_rows():
  """Substitute blocks share policy key rows."""
  d = designs.substitute(IND3)
  assert (
      d.num_rows == 6
      and d.blocks == (0, 0, 0, 1, 1, 1)
      and d.key_rows == (0, 1, 2) * 2
  )
  assert d.seat_perms[:3] == d.seat_perms[3:] == designs.cyclic(IND3).seat_perms
  np.testing.assert_array_equal(d.rows_of(1), [3, 4, 5])
  plan = designs.key_plan(d, duplicate=True)
  assert plan.deal_index == (0,) * 6 and plan.policy_index == (0, 1, 2, 0, 1, 2)
  assert plan.num_deal_keys == 1 and plan.num_policy_rows == 3
  ind = designs.key_plan(d, duplicate=False)
  assert ind.deal_index == ind.policy_index == tuple(range(6))


def test_make_auto_and_refusals():
  """Make auto and refusals."""
  assert designs.make("auto", IND2).name == "swap2"
  assert designs.make("auto", TEAM).name == "swap2"
  assert designs.make("auto", IND3).name == "cyclic"
  assert designs.make("cyclic+reversed", IND3).num_rows == 6
  with pytest.raises(ValueError, match="unknown design"):
    designs.make("nope", IND3)
  with pytest.raises(ValueError, match="two sides"):
    designs.swap2(IND3)
  with pytest.raises(ValueError, match="at least 2 seats"):
    designs.swap2_partner(IND2)
  uneven = sides_lib.Sides((0, 1, 1), constant_sum=True, rotatable=False)
  with pytest.raises(ValueError, match="rotatable"):
    designs.cyclic(uneven)
  with pytest.raises(ValueError, match="splits side"):
    designs.Design("bad", TEAM, ((0, 1, 2, 3), (0, 1, 3, 2)), (0, 0), (0, 1))
  with pytest.raises(ValueError, match="different seat permutations"):
    designs.Design("bad", IND2, ((0, 1), (1, 0)), (0, 1), (0, 0))
  with pytest.raises(ValueError, match="not a permutation"):
    designs.Design("bad", IND2, ((0, 0),), (0,), (0,))


def test_uneven_sides_rotate_only_equal_size_sides():
  """Uneven sides rotate only equal size sides."""
  two_v_one = sides_lib.Sides((0, 1, 0), constant_sum=True)
  d = designs.make("auto", two_v_one)
  # nothing interchangeable
  assert d.name == "cyclic+classes" and d.seat_perms == ((0, 1, 2),)
  for name in ("swap2", "cyclic", "cyclic+reversed", "swap2+partner"):
    with pytest.raises(ValueError, match="one size|two sides"):
      designs.make(name, two_v_one)
  # 3v1v1: the two lone sides swap, the team of three stays.
  u = sides_lib.Sides((0, 1, 0, 2, 0), constant_sum=True)
  d = designs.make("auto", u)
  assert d.name == "cyclic+classes" and d.num_rows == 2
  assert d.side_perms.tolist() == [[0, 1, 2], [0, 2, 1]]
  assert d.seat_perms == ((0, 1, 2, 3, 4), (0, 3, 2, 1, 4))
  sub = designs.substitute(u)
  assert (
      sub.num_rows == 4
      and sub.blocks == (0, 0, 1, 1)
      and sub.key_rows == (0, 1, 0, 1)
  )
  # 2v2v1v1: both classes shift together, period 2.
  v = sides_lib.Sides((0, 1, 0, 1, 2, 3), constant_sum=False)
  assert designs.make("auto", v).side_perms.tolist() == [
      [0, 1, 2, 3],
      [1, 0, 3, 2],
  ]
  # A design that moves a side onto a side of another size is refused when
  # built.
  with pytest.raises(ValueError, match="splits side|does not permute"):
    designs.Design("bad", u, ((1, 0, 3, 2, 4),), (0,), (0,))
  # Equal rotatable sides: cyclic+classes is cyclic, substitute unchanged.
  for s in (IND3, TEAM, sides_lib.Sides.partnerships(6, 3, True)):
    assert designs.cyclic_classes(s).seat_perms == designs.cyclic(s).seat_perms
  # Not rotatable: the identity only; substitute refuses.
  fixed = sides_lib.Sides((0, 1, 2), constant_sum=True, rotatable=False)
  assert designs.make("auto", fixed).seat_perms == ((0, 1, 2),)
  with pytest.raises(ValueError, match="rotatable"):
    designs.substitute(fixed)


def test_seat_scores_record_a_mixed_side_per_seat():
  """Seat scores record a mixed side per seat."""
  two_v_one = sides_lib.Sides((0, 1, 0), constant_sum=True)
  d = designs.make("auto", two_v_one)
  # a mixed pair: a at seat 0, b at seat 2
  lineup = np.array([[[0, 2, 1]], [[0, 2, 1]]])
  ret = np.array([[[1.0, -1.0, 1.0]], [[-1.0, 1.0, -1.0]]])
  res = _hand(d, lineup, ret)
  assert res.seat_score(0).mean == 0.0 and res.seat_score(1).mean == 0.0
  np.testing.assert_array_equal(res.seat_returns(2)[:, 0], [1.0, -1.0])
  # the side's
  np.testing.assert_array_equal(res.position_returns(0)[:, 0], [1.0, -1.0])
  assert res.check_sum() == 0.0
  np.testing.assert_array_equal(res.seats[:, 0], [[0, 2, 1], [0, 2, 1]])
  with pytest.raises(ValueError, match="seat 3"):
    res.seat_returns(3)


# -- TableResult over hand-made records ----------------------------------------
def _hand(design, lineup, returns, rules=None, names=("a", "b", "c")):
  """A one-hand TableResult of toy returns.

  Args:
    design: The Design.
    lineup: The lineups.
    returns: The per-seat returns.
    rules: Each entrant's rule (default ARGMAX).
    names: The entrants.

  Returns:
    The TableResult.
  """
  rules = rules or [types.ARGMAX] * len(names)
  return tables.table_result(
      "g", "cfg", design, names, rules, lineup, returns, 1, True
  )


def test_as_head_to_head_is_the_two_player_accounting():
  """As head to head is the two player accounting."""
  d = designs.swap2(IND2)
  ret = np.array([
      [[1.0, -1.0], [-1.0, 1.0]],
      [[3.0, -3.0], [-1.0, 1.0]],
      [[0.0, 0.0], [2.0, -2.0]],
  ])
  res = _hand(d, np.tile([[[0, 1]]], (3, 1, 1)), ret, names=("a", "b"))
  h = res.as_head_to_head()
  # column r: a at seat r
  a_pay = np.array([[1.0, 1.0], [3.0, 1.0], [0.0, -2.0]])
  want = payoff.match_result(
      "g", "cfg", 1, True, "a", "b", types.ARGMAX, types.ARGMAX, a_pay, -a_pay
  )
  for f in payoff.MatchResult._fields:
    got, exp = getattr(h, f), getattr(want, f)
    if isinstance(exp, np.ndarray):
      np.testing.assert_array_equal(got, exp)
    else:
      assert got == exp, f
  scored = res.as_head_to_head(scorer=lambda r0, r1: np.sign(r0 + r1))
  assert scored.mean == stats.estimate(np.array([1.0, 1.0, -1.0]))
  assert scored.diff == stats.estimate(np.array([2.0, 2.0, -2.0]))
  assert np.isnan(scored.se_unpaired) and np.isnan(scored.variance_reduction)
  with pytest.raises(ValueError, match="two sides under swap2"):
    _hand(
        designs.cyclic(IND3), np.zeros((2, 1, 3), np.int32), np.zeros((2, 3, 3))
    ).as_head_to_head()


def test_position_estimators_and_placement():
  """Position estimators and placement."""
  d = designs.cyclic(IND3)
  # One deal; position k's entrant is k; rows rotate. returns[r, seat].
  ret = np.array([
      [[3.0, 0.0, -3.0], [-3.0, 3.0, 0.0], [1.0, 1.0, -2.0]],
      [[2.0, -1.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 2.0, -1.0]],
  ])
  res = _hand(d, np.tile([[[0, 1, 2]]], (2, 1, 1)), ret)
  # Position 0 sits at side r on row r.
  np.testing.assert_array_equal(
      res.position_returns(0), [[3.0, 3.0, -2.0], [2.0, 0.0, -1.0]]
  )
  np.testing.assert_array_equal(
      res.position_returns(1), [[0.0, 0.0, 1.0], [-1.0, 0.0, -1.0]]
  )
  assert res.position_score(0) == stats.estimate(
      np.array([4.0 / 3.0, 1.0 / 3.0])
  )
  assert res.diff(0, 1) == stats.estimate(np.array([1.0, 1.0]))
  side = res.by_side(0)
  assert side[0] == stats.estimate(np.array([3.0, 2.0]))
  assert side[2] == stats.estimate(np.array([-2.0, -1.0]))
  # Placement of position 0: deal 0 rows: 1st, 1st, 3rd; deal 1: 1st, tie of
  # three over places 1-3, 2nd-3rd tied.
  pl = res.placement(0)
  want = (
      np.array([
          1 + 1 + 0 + 1 + 1 / 3,
          0 + 0 + 0 + 0 + 1 / 3 + 1 / 2,
          1 + 1 / 3 + 1 / 2,
      ])
      / 6
  )
  np.testing.assert_allclose(pl, want, atol=1e-15)
  assert pl.sum() == pytest.approx(1.0, abs=1e-15)
  assert res.check_sum() == 0.0
  e, p0 = res.entrant_score("a"), res.position_score(0)
  assert e.n == 2 and e.mean == pytest.approx(p0.mean, abs=1e-15)
  assert e.se == pytest.approx(p0.se, abs=1e-15)


def test_entrant_score_over_draws_and_shared_rows():
  """Entrant score over draws and shared rows."""
  d = designs.cyclic(IND3)
  # Deal 0: a in every position (self-play), deal 1: a only at position 0, deal
  # 2: no a.
  lineup = np.array([[[0, 0, 0]], [[0, 1, 2]], [[1, 2, 1]]])
  ret = np.array(
      [[[1.0, 2.0, -3.0]] * 3, [[6.0, 0.0, -6.0]] * 3, [[1.0, 1.0, -2.0]] * 3]
  )
  res = _hand(d, lineup, ret)
  # deal 0: a holds all seats: mean seat return 0; deal 1: 6, 0, -6 over the
  # rotations.
  assert res.entrant_score("a") == stats.estimate(np.array([0.0, 0.0]))
  b = res.entrant_score(1)
  assert b.n == 2
  with pytest.raises(ValueError, match="no entrant"):
    res.entrant_score("zz")


def test_check_sum_raises_on_constant_sum_drift_and_logs_general_sum():
  """Check sum raises on constant sum drift and logs general sum."""
  d = designs.cyclic(IND3)
  ret = np.zeros((2, 3, 3))
  ret[1, 2] = (1.0, 0.0, 0.0)
  with pytest.raises(AssertionError, match="row sums vary"):
    _hand(d, np.zeros((2, 1, 3), np.int32), ret).check_sum()
  gs = designs.cyclic(sides_lib.Sides.individual(3, constant_sum=False))
  assert _hand(gs, np.zeros((2, 1, 3), np.int32), ret).check_sum() is None
  c3 = np.full((2, 3, 3), 1.0)
  assert _hand(d, np.zeros((2, 1, 3), np.int32), c3).check_sum() == 3.0


def test_team_side_returns_and_pair_scores():
  """Team side returns and pair scores."""
  d = designs.swap2(TEAM)
  ret = np.array([
      [[2.0, -2.0, 2.0, -2.0], [5.0, -5.0, 5.0, -5.0]],
      [[1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, -1.0, 1.0]],
  ])
  res = _hand(
      d,
      designs.seat_lineup(TEAM, np.tile([[[0, 1]]], (2, 1, 1))),
      ret,
      names=("ns", "ew"),
  )
  np.testing.assert_array_equal(
      res.side_returns[:, 0], [[2.0, -2.0], [1.0, -1.0]]
  )
  # Position 0 holds NS on row 0 and EW on row 1.
  np.testing.assert_array_equal(
      res.position_returns(0), [[2.0, -5.0], [1.0, 1.0]]
  )
  np.testing.assert_array_equal(
      res.deal_scores(0, scorer=lambda a, b: a + b), [-3.0, 2.0]
  )
  with pytest.raises(ValueError, match="two-row blocks"):
    _hand(
        designs.cyclic(IND3), np.zeros((2, 1, 3), np.int32), np.zeros((2, 3, 3))
    ).deal_scores(0, scorer=lambda a, b: a)


def test_table_result_validates_its_record():
  """Table result validates its record."""
  d = designs.cyclic(IND3)
  with pytest.raises(ValueError, match="shape"):
    _hand(d, np.zeros((2, 1, 3), np.int32), np.zeros((2, 2, 3)))
  with pytest.raises(ValueError, match="outside"):
    _hand(d, np.full((2, 1, 3), 5, np.int32), np.zeros((2, 3, 3)))
  with pytest.raises(ValueError, match="non-finite"):
    _hand(d, np.zeros((2, 1, 3), np.int32), np.full((2, 3, 3), np.nan))


# -- the Evaluator over a numpy N-seat engine ----------------------------------
class _ToyEnv:
  name, num_actions, max_len, zero_sum = "toy_n", 2, 1, True

  def __init__(self, sides):
    self.num_players = sides.num_players
    if not sides.one_seat_per_side:
      self.sides = sides


class _Strength:

  def __init__(self, name, p):
    self.spec = types.PolicySpec(
        "toy_n",
        "none",
        2,
        p,
        False,
        types.Capability.HOST_DRIVEN,
        "jax",
        f"strength_{name}",
    )


def _bias(deal_key, p):
  """A per-deal seat bias, mean zero, from the deal key.

  Args:
    deal_key: The deal's key data.
    p: The players.

  Returns:
    f64[p].
  """
  h = int(np.asarray(deal_key)[1]) % 7
  b = np.array([(h * (q + 1)) % 5 for q in range(p)], np.float64)
  return b - b.mean()


class _ToyEngine:
  """A toy N-seat engine.

  Seat q of a row receives strength(entrant at q) - mean strength +
  bias(deal)[q].

  Attributes:
    calls: every match call, in order.
  """

  def __init__(self):
    self.calls = []

  def row_unit(self) -> int:
    """Rows come in pairs.

    Returns:
      2.
    """
    return 2

  def stack(self, slot, members):
    """The members as a list.

    Args:
      slot: unused.
      members: the members' params.

    Returns:
      list(members).
    """
    del slot  # the toy params are plain numbers
    return list(members)

  def job_keys(self, keys, n, duplicate):
    """Refused: the table path uses table_keys.

    Args:
      keys: unused.
      n: unused.
      duplicate: unused.

    Raises:
      AssertionError: always.
    """
    raise AssertionError("the table path uses table_keys")

  def table_keys(self, keys, n, plan, num_players):
    """The deal and policy keys of every row, as core.arena_engine's.

    Args:
      keys: one key per job.
      n: deals per job.
      plan: the KeyPlan.
      num_players: P.

    Returns:
      (deal keys, policy keys).
    """
    deal, pol = [], []
    for key in keys:
      for i in range(n):
        dk, pk = rng.split(rng.fold_in64(key, i))
        dks = rng.split(dk, plan.num_deal_keys)
        pks = rng.split(pk, plan.num_policy_rows * num_players).reshape(
            plan.num_policy_rows, num_players, -1
        )
        deal += [dks[j] for j in plan.deal_index]
        pol += [pks[j] for j in plan.policy_index]
    return np.stack(deal), np.stack(pol)

  def play(
      self,
      slot_list,
      params,
      table,
      deal_key,
      policy_key,
      num_steps,
      *,
      perspective,
      cache_key,
  ):
    """Pays each seat its entrant's strength minus the table mean, plus bias.

    Args:
      slot_list: unused.
      params: the slots' params (strengths).
      table: the SlotTable.
      deal_key: the deal keys.
      policy_key: the policy keys.
      num_steps: unused.
      perspective: the perspective flag.
      cache_key: unused.

    Returns:
      The payoffs.
    """
    del slot_list, cache_key
    b, p = table.seat_slot.shape
    self.calls.append(
        dict(
            table=table,
            deal=np.asarray(deal_key),
            pol=np.asarray(policy_key),
            perspective=perspective,
        )
    )
    s = np.array(
        [
            [
                params[table.seat_slot[r, q]][table.seat_member[r, q]]
                for q in range(p)
            ]
            for r in range(b)
        ],
        np.float64,
    )
    pay = (
        s
        - s.mean(1, keepdims=True)
        + np.stack([_bias(deal_key[r], p) for r in range(b)])
    )
    rewards = np.zeros((b, num_steps, p))
    done = np.zeros((b, num_steps), bool)
    rewards[:, 0], done[:, 0] = pay, True
    return rewards, done


def _ent(name, strength, p, pol=None):
  """A toy entrant of fixed strength.

  Args:
    name: The entrant.
    strength: Its strength (its params).
    p: The players.
    pol: The policy (default _Strength).

  Returns:
    The match.Entrant.
  """
  return match.Entrant(
      name, pol or _Strength(name, p), float(strength), types.ARGMAX
  )


def _ev(sides=IND3, max_rows=4096, duplicate=True):
  """A toy Evaluator over _ToyEnv and _ToyEngine.

  Args:
    sides: The env's Sides.
    max_rows: The row cap.
    duplicate: Whether deals are duplicated.

  Returns:
    The match.Evaluator.
  """
  return match.Evaluator(
      _ToyEnv(sides),
      match.EvaluatorConfig(name="toy", max_rows=max_rows, duplicate=duplicate),
      engine=_ToyEngine(),
  )


def test_table_position_scores_are_exact_under_duplicate_deals():
  """Table position scores are exact under duplicate deals."""
  ents = [_ent("x", 3.0, 3), _ent("y", 0.0, 3), _ent("z", -3.0, 3)]
  res = _ev().table(ents, 32, rng.key(0))
  assert (
      res.design.name == "cyclic"
      and res.n_deals == 32
      and res.entrants == ("x", "y", "z")
  )
  for k, want in enumerate((3.0, 0.0, -3.0)):
    est = res.position_score(k)
    assert est.mean == pytest.approx(
        want, abs=1e-12
    ) and est.se == pytest.approx(0.0, abs=1e-12)
  # The seat bias is real: by side, the subject's return varies over deals.
  assert max(e.se for e in res.by_side(0)) > 0.0
  assert res.check_sum() == pytest.approx(0.0, abs=1e-12)
  ind = _ev(duplicate=False).table(ents, 32, rng.key(0))
  assert ind.position_score(0).se > 0.0
  assert res.placement(0).sum() == pytest.approx(1.0, abs=1e-12)


def test_vs_field_draws_each_side_from_the_field():
  """Vs field draws each side from the field."""
  pol = _Strength("pool", 3)
  field = [
      _ent("f0", 0.0, 3, pol),
      _ent("f1", 3.0, 3, pol),
      _ent("f2", 9.0, 3, pol),
  ]
  res = _ev().vs_field(
      _ent("me", 6.0, 3), field, [0.5, 0.5, 0.0], 3000, rng.key(1)
  )
  assert np.all(res.lineup[:, 0, 0] == 0)
  drawn = res.lineup[:, 0, 1:] - 1
  assert not np.any(drawn == 2)
  for side in range(2):
    frac = np.mean(drawn[:, side] == 0)
    assert abs(frac - 0.5) < 5 * np.sqrt(0.25 / 3000)
  # Independent sides: the two field seats differ on about half the deals.
  assert abs(np.mean(drawn[:, 0] != drawn[:, 1]) - 0.5) < 5 * np.sqrt(
      0.25 / 3000
  )
  # Exact per deal: 6 - mean(6, f_a, f_b), bias cancelled by the rotation.
  want = 6.0 - (6.0 + np.array([0.0, 3.0, 9.0])[drawn].sum(1)) / 3.0
  np.testing.assert_allclose(res.deal_scores(0), want, atol=1e-12)
  with pytest.raises(ValueError, match="weights"):
    _ev().vs_field(_ent("me", 6.0, 3), field, [0.5, 0.5], 4, rng.key(1))


def test_substitute_is_seat_balanced_and_shares_field_keys():
  """Substitute is seat balanced and shares field keys."""
  ev = _ev()
  pol = _Strength("pool", 3)
  field = [_ent("f0", 1.0, 3, pol), _ent("f1", -2.0, 3, pol)]
  s, i = _ent("subject", 4.0, 3), _ent("incumbent", 1.0, 3)
  res = ev.substitute(s, i, field, [0.3, 0.7], 50, rng.key(2))
  assert res.design.name == "substitute" and res.lineup.shape == (50, 2, 3)
  # one field per deal
  np.testing.assert_array_equal(res.lineup[:, 0, 1:], res.lineup[:, 1, 1:])
  # The subject and the incumbent each hold every side once per deal.
  for b in (0, 1):
    rows = res.design.rows_of(b)
    assert sorted(res.design.side_perms[rows, 0].tolist()) == [0, 1, 2]
  # Exact: (4 - 1) * (1 - 1/3) per deal, whatever the field.
  d = res.block_diff(0)
  assert d.mean == pytest.approx(2.0, abs=1e-12) and d.se == pytest.approx(
      0.0, abs=1e-12
  )
  # The subject's and the incumbent's slots play apart (perspective per slot).
  assert len({c["perspective"] for c in ev.engine.calls}) == 2
  # Keys under the design's plan: one deal key per deal, and row r of block 1
  # gets row r of block 0's policy keys (the engine draws by KeyPlan).
  dk, pk = _ToyEngine().table_keys(
      [rng.key(5)], 3, designs.key_plan(res.design, True), 3
  )
  dk, pk = dk.reshape(3, 6, -1), pk.reshape(3, 6, 3, -1)
  assert np.all(dk == dk[:, :1])
  np.testing.assert_array_equal(pk[:, :3], pk[:, 3:])
  assert not np.array_equal(pk[:, 0], pk[:, 1])


def test_substitute_self_against_self_is_zero_per_deal():
  """Substitute self against self is zero per deal."""
  s = _ent("me", 2.5, 3)
  res = _ev().substitute(s, s, [_ent("f", 0.0, 3)], [1.0], 20, rng.key(3))
  np.testing.assert_array_equal(res.block_diff(0).mean, 0.0)
  np.testing.assert_array_equal(res.deal_scores(0, 0), res.deal_scores(0, 1))


def test_tables_do_not_depend_on_chunking():
  """Tables do not depend on chunking."""
  ents = [_ent("x", 1.0, 3), _ent("y", 2.0, 3), _ent("z", 0.5, 3)]
  big, small = _ev(max_rows=4096), _ev(max_rows=4)
  r1 = big.table(ents, 9, rng.key(4), "cyclic+reversed")
  r2 = small.table(ents, 9, rng.key(4), "cyclic+reversed")
  np.testing.assert_array_equal(r1.returns, r2.returns)
  assert len(small.engine.calls) > len(big.engine.calls)


def test_pairwise_methods_refuse_more_than_two_sides():
  """Pairwise methods refuse more than two sides."""
  ev = _ev()
  a, b = _ent("a", 1.0, 3), _ent("b", 0.0, 3)
  for call in (
      lambda: ev.head_to_head(a, b, 4, rng.key(0)),
      lambda: ev.payoff_matrix([a, b], 4, rng.key(0)),
      lambda: ev.mixture_head_to_head([a], [1.0], [b], [1.0], 4, rng.key(0)),
  ):
    with pytest.raises(ValueError, match="design not valid for these sides"):
      call()
  with pytest.raises(ValueError, match="one entrant per side"):
    ev.table([a, b], 4, rng.key(0))
  with pytest.raises(ValueError, match="two sides"):
    ev.table([a, b, a], 4, rng.key(0), "swap2")


def test_two_teams_play_head_to_head_side_against_side():
  """Two teams play head to head side against side."""
  ev = _ev(TEAM)
  a, b = _ent("a", 2.0, 4), _ent("b", -2.0, 4)
  r = ev.head_to_head(a, b, 16, rng.key(6))
  assert isinstance(r, payoff.MatchResult) and (r.a, r.b) == ("a", "b")
  assert r.mean.mean == pytest.approx(
      2.0, abs=1e-12
  ) and r.mean.se == pytest.approx(0, abs=1e-12)
  pm = ev.payoff_matrix([a, b, _ent("c", 0.0, 4)], 8, rng.key(7))
  np.testing.assert_allclose(
      pm.mean, [[0, 2, 1], [-2, 0, -1], [-1, 1, 0]], atol=1e-12
  )
  pol = _Strength("pop", 4)
  mm = ev.mixture_head_to_head(
      [_ent("p0", 4.0, 4, pol), _ent("p1", 0.0, 4, pol)],
      [0.5, 0.5],
      [b],
      [1.0],
      400,
      rng.key(8),
      a_name="A",
      b_name="B",
  )
  assert (mm.result.a, mm.result.b) == ("A", "B")
  frac = np.mean(mm.a_members == 0)
  assert mm.result.mean.mean == pytest.approx(
      3.0 * frac + 1.0 * (1 - frac), abs=1e-12
  )


def test_a_declared_pair_scorer_scores_two_sides_of_several_seats():
  """A declared pair scorer scores two sides of several seats."""
  env = _ToyEnv(TEAM)
  env.pair_scorer = lambda r0, r1: np.sign(r0 + r1) * 10.0
  ev = match.Evaluator(
      env, match.EvaluatorConfig(name="toy"), engine=_ToyEngine()
  )
  r = ev.head_to_head(_ent("a", 2.0, 4), _ent("b", -2.0, 4), 8, rng.key(9))
  assert r.mean.mean == 10.0 and r.diff.mean == 20.0 and np.isnan(r.se_unpaired)
  # the record stays raw
  np.testing.assert_allclose(r.a_payoff.mean(1), 2.0, atol=1e-12)


def test_an_engine_without_table_keys_is_refused():
  """An engine without table keys is refused."""

  class _Old(_ToyEngine):
    table_keys = None

  ev = match.Evaluator(_ToyEnv(IND3), engine=_Old())
  with pytest.raises(TypeError, match="no table_keys"):
    ev.table([_ent("a", 0.0, 3)] * 3, 4, rng.key(0))


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
