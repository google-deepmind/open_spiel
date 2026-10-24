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

"""Payoff accounting and the Evaluator over a numpy play engine, without jax.

eval.payoff holds the Evaluator's numpy accounting (payoffs per row from a
reward / done log, the MatchResult of a seat-swapped duplicate match, the
PayoffMatrix assembled from pair results, the duplicate slot table) and the
MatchEngine protocol the Evaluator plays through. This file imports no jax
and runs the Evaluator over a numpy engine.

Known answers: hand-made logs and payoffs; a toy game whose engine pays
sign(strength_a - strength_b) plus a per-deal seat-0 bias that duplicate
deals cancel (pair means exact, SE 0), so head_to_head, payoff_matrix and
mixture_head_to_head have closed-form results.
"""

import sys

import numpy as np
import pytest

from open_spiel.python.scale.core import rng
from open_spiel.python.scale.core import slots
from open_spiel.python.scale.core import types
from open_spiel.python.scale.eval import match
from open_spiel.python.scale.eval import payoff
from open_spiel.python.scale.eval import stats


# -- accounting ----------------------------------------------------------------
def test_payoffs_count_the_first_hands_only():
  """`payoffs` sums each row's rewards over its first `hands` hands only."""
  rewards = np.zeros((2, 5, 2), np.float32)
  done = np.zeros((2, 5), bool)
  rewards[0, 1], done[0, 1] = (1.0, -1.0), True
  rewards[0, 3], done[0, 3] = (2.0, -2.0), True
  rewards[1, 4], done[1, 4] = (-0.5, 0.5), True
  out = payoff.payoffs(rewards, done, 1)
  assert out.dtype == np.float64 and out.shape == (2, 2)
  np.testing.assert_array_equal(out, [[1.0, -1.0], [-0.5, 0.5]])
  with pytest.raises(RuntimeError, match="row 1 finished 1 < 2 hands"):
    payoff.payoffs(rewards, done, 2)
  done[1, 2] = True
  np.testing.assert_array_equal(
      payoff.payoffs(rewards, done, 2), [[3.0, -3.0], [-0.5, 0.5]]
  )


def test_match_result_known_answer():
  """match_result has its known mean, SE and variance ratio."""
  a_pay = np.array([[1.0, -1.0], [3.0, 1.0]])
  r = payoff.match_result(
      "g", "cfg", 1, True, "a", "b", types.ARGMAX, types.SAMPLE, a_pay, -a_pay
  )
  assert isinstance(r, payoff.MatchResult)
  assert (
      r.game,
      r.config,
      r.a,
      r.b,
      r.n_deals,
      r.hands_per_match,
      r.duplicate,
  ) == ("g", "cfg", "a", "b", 2, 1, True)
  assert r.mean == stats.estimate(np.array([0.0, 2.0]))
  assert r.seat == (
      stats.estimate(np.array([1.0, 3.0])),
      stats.estimate(np.array([-1.0, 1.0])),
  )
  assert r.diff == stats.estimate(np.array([0.0, 4.0]))
  se_un = float(np.std([1.0, -1.0, 3.0, 1.0], ddof=1) / 2.0)
  assert r.se_unpaired == pytest.approx(se_un, abs=1e-15)
  assert r.variance_reduction == pytest.approx(
      (se_un / r.mean.se) ** 2, abs=1e-12
  )
  flat = payoff.match_result(
      "g",
      "c",
      1,
      True,
      "a",
      "b",
      types.ARGMAX,
      types.ARGMAX,
      np.zeros((3, 2)),
      np.zeros((3, 2)),
  )
  assert np.isnan(flat.variance_reduction)
  pure = payoff.match_result(
      "g",
      "c",
      1,
      True,
      "a",
      "b",
      types.ARGMAX,
      types.ARGMAX,
      np.array([[1.0, -1.0]] * 3),
      np.zeros((3, 2)),
  )
  assert pure.mean.se == 0.0 and pure.variance_reduction == np.inf


def test_matrix_from_results_uses_b_payoffs_below_the_diagonal():
  """matrix_from_results fills cell (j, i) from b's payoffs."""

  def res(a_pay, b_pay):
    """A MatchResult of x vs y from the given payoffs.

    Args:
      a_pay: x's payoffs.
      b_pay: y's payoffs.

    Returns:
      the MatchResult.
    """
    return payoff.match_result(
        "g",
        "c",
        1,
        True,
        "x",
        "y",
        types.ARGMAX,
        types.ARGMAX,
        np.asarray(a_pay, np.float64),
        np.asarray(b_pay, np.float64),
    )

  results = {
      (0, 0): res([[0, 0], [0, 0]], [[0, 0], [0, 0]]),
      (0, 1): res([[1, 1], [1, 1]], [[-2, -2], [-2, -2]]),
      (1, 1): res([[0.5, 0.5], [0.5, 0.5]], [[0, 0], [0, 0]]),
  }
  pm = payoff.matrix_from_results(("p", "q"), results, 2)
  assert pm.names == ("p", "q") and pm.n_deals == 2 and pm.results is results
  np.testing.assert_array_equal(pm.mean, [[0.0, 1.0], [-2.0, 0.5]])
  np.testing.assert_array_equal(pm.se, np.zeros((2, 2)))
  np.testing.assert_array_equal(pm.lo, pm.mean)
  np.testing.assert_array_equal(pm.hi, pm.mean)


def test_duplicate_table_layout():
  """duplicate_table puts a in seat 0 on even rows and seat 1 on odd rows."""
  t = payoff.duplicate_table([0, 0], [0, 1], [1, 1], [2, 0])
  assert isinstance(t, slots.SlotTable)
  np.testing.assert_array_equal(t.seat_slot, [[0, 1], [1, 0], [0, 1], [1, 0]])
  np.testing.assert_array_equal(t.seat_member, [[0, 2], [2, 0], [1, 0], [0, 1]])
  np.testing.assert_array_equal(t.perspective_seat, [0, 1, 0, 1])
  assert (
      t.seat_slot.dtype
      == t.seat_member.dtype
      == t.perspective_seat.dtype
      == np.int32
  )


# -- the Evaluator over a numpy engine -----------------------------------------
class _Env:
  name, num_players, num_actions, max_len = "toy", 2, 2, 1


class _Strength:
  """A policy identity; its params are a strength (float).

  Attributes:
    spec: the policy's spec.
  """

  def __init__(self, name):
    self.spec = types.PolicySpec(
        "toy",
        "none",
        2,
        2,
        False,
        types.Capability.HOST_DRIVEN,
        "jax",
        f"strength_{name}",
    )


class _ToyEngine:
  """A MatchEngine over the toy game.

  MatchEngine over the toy game: one ply per hand, seat 0 receives
  sign(s0 - s1) + bias(deal), seat 1 the negation; numpy keys.

  Attributes:
    plays: How many matches the engine has played.
  """

  def __init__(self):
    self.plays = 0

  def row_unit(self) -> int:
    """Rows in multiples of 2.

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
      the list.
    """
    del slot
    return list(members)

  def job_keys(self, keys, n, duplicate):
    """NumPy deal and policy keys for every job's rows.

    Args:
      keys: one key per job.
      n: deal pairs per job.
      duplicate: whether the two rows of a pair share the deal.

    Returns:
      (deal keys, policy keys).
    """
    deal, pol = [], []
    for key in keys:
      for i in range(n):
        k = rng.fold_in64(key, i)
        d = rng.split(k, 3)
        deal += [d[0], d[0]] if duplicate else [d[0], d[1]]
        pol += [rng.split(d[2], 2), rng.split(d[1], 2)]
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
    """One ply per hand: seat 0 gets sign(s0 - s1) + bias(deal).

    Args:
      slot_list: unused.
      params: each slot's members' strengths.
      table: the slot table of the rows.
      deal_key: per-row deal keys.
      policy_key: unused.
      num_steps: plies.
      perspective: unused.
      cache_key: unused.

    Returns:
      (rewards [B, T, 2], done [B, T]).
    """
    del slot_list, policy_key, perspective, cache_key
    self.plays += 1
    b = table.seat_slot.shape[0]
    s = np.array(
        [
            [
                params[table.seat_slot[r, p]][table.seat_member[r, p]]
                for p in range(2)
            ]
            for r in range(b)
        ],
        np.float64,
    )
    bias = (np.asarray(deal_key)[:, 1] % 3).astype(np.float64) - 1.0
    r0 = np.sign(s[:, 0] - s[:, 1]) + bias
    rewards = np.zeros((b, num_steps, 2))
    done = np.zeros((b, num_steps), bool)
    rewards[:, 0, 0], rewards[:, 0, 1], done[:, 0] = r0, -r0, True
    return rewards, done


def _ent(name, strength, pol=None):
  """A toy entrant of a given strength.

  Args:
    name: The entrant name.
    strength: Its strength (its params).
    pol: Its policy (default a _Strength).

  Returns:
    The Entrant.
  """
  return match.Entrant(
      name, pol or _Strength(name), float(strength), types.ARGMAX
  )


def _ev(max_rows=4096):
  """An evaluator over the toy engine.

  Args:
    max_rows: The evaluator's row cap.

  Returns:
    The Evaluator.
  """
  return match.Evaluator(
      _Env(),
      match.EvaluatorConfig(name="toy", max_rows=max_rows),
      engine=_ToyEngine(),
  )


def test_head_to_head_closed_form_through_the_engine():
  """head_to_head through the engine matches the closed form."""
  ev = _ev()
  r = ev.head_to_head(_ent("strong", 2.0), _ent("weak", 1.0), 64, rng.key(0))
  assert (r.a, r.b, r.game, r.config, r.n_deals) == (
      "strong",
      "weak",
      "toy",
      "toy",
      64,
  )
  # duplicate deals cancel the seat bias
  assert r.mean.mean == 1.0 and r.mean.se == 0.0
  np.testing.assert_array_equal(r.b_payoff, -r.a_payoff)
  assert r.se_unpaired > 0.0 and r.variance_reduction == np.inf
  # Biased seats: a scores 1 + bias(d) from seat 0 and 1 - bias(d) from seat 1,
  # with bias(d) in {-1, 0, 1}.
  bias = r.a_payoff[:, 0] - 1.0
  assert set(np.unique(bias)) <= {-1.0, 0.0, 1.0} and np.std(bias) > 0.0
  np.testing.assert_array_equal(r.a_payoff[:, 1], 1.0 - bias)
  assert r.seat[0].mean == pytest.approx(1.0 + bias.mean(), abs=1e-12)
  assert r.seat[1].mean == pytest.approx(1.0 - bias.mean(), abs=1e-12)
  ind = match.Evaluator(
      _Env(), match.EvaluatorConfig(duplicate=False), engine=_ToyEngine()
  ).head_to_head(_ent("s", 2.0), _ent("w", 1.0), 64, rng.key(0))
  assert ind.mean.se > 0.0  # independent deals keep the bias noise


def test_payoff_matrix_cells_equal_head_to_head_under_pair_keys():
  """Each payoff matrix cell equals head_to_head under its pair key."""
  ents = [_ent("a", 3.0), _ent("b", 1.0), _ent("c", 2.0)]
  key = rng.key(4)
  pm = _ev().payoff_matrix(ents, 16, key)
  assert pm.names == ("a", "b", "c")
  np.testing.assert_array_equal(pm.mean, [[0, 1, 1], [-1, 0, -1], [-1, 1, 0]])
  for i in range(3):
    for j in range(i, 3):
      r = _ev().head_to_head(ents[i], ents[j], 16, match.pair_key(key, i, j))
      np.testing.assert_array_equal(r.a_payoff, pm.results[(i, j)].a_payoff)


def test_results_do_not_depend_on_chunking():
  """Results do not depend on row chunking."""
  a, b = _ent("a", 1.0), _ent("b", 1.0)
  big, small = _ev(4096), _ev(4)
  r1 = big.head_to_head(a, b, 10, rng.key(1))
  r2 = small.head_to_head(a, b, 10, rng.key(1))
  np.testing.assert_array_equal(r1.a_payoff, r2.a_payoff)
  assert small.engine.plays > big.engine.plays


def test_mixture_head_to_head_draws_members_by_weight():
  """mixture_head_to_head draws members by weight."""
  pol_a = _Strength("pop_a")  # a population plays one architecture: one slot
  pa = [_ent("a0", 5.0, pol_a), _ent("a1", -5.0, pol_a), _ent("a2", 0.0, pol_a)]
  pb = [_ent("b0", 1.0)]
  mm = _ev().mixture_head_to_head(
      pa, [0.75, 0.25, 0.0], pb, [1.0], 4000, rng.key(2), a_name="A", b_name="B"
  )
  assert mm.result.a == "A" and mm.result.b == "B"
  assert not np.any(mm.a_members == 2) and np.all(mm.b_members == 0)
  frac = np.mean(mm.a_members == 0)
  assert abs(frac - 0.75) < 5 * np.sqrt(0.75 * 0.25 / 4000)
  # pair payoff is +1 for a0 and -1 for a1: the mean is 2 * frac - 1 exactly.
  assert mm.result.mean.mean == pytest.approx(2 * frac - 1, abs=1e-12)
  with pytest.raises(ValueError, match="weights"):
    _ev().mixture_head_to_head(pa, [0.5, 0.5], pb, [1.0], 4, rng.key(2))


def test_evaluator_refuses_non_two_player_games():
  """The Evaluator refuses games without 2 players."""

  class _Env3(_Env):
    num_players = 3

  with pytest.raises(ValueError, match="2 players"):
    match.Evaluator(_Env3(), engine=_ToyEngine())


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
