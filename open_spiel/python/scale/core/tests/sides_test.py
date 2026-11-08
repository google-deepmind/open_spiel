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

"""core.sides: the seat partition, side returns, pair scores and lineups.

* every constructor gives a partition of the seats into sides numbered in
  order of first appearance; malformed partitions are refused;
* side returns are per-side means of seat returns, the same in numpy and
  JAX (and under jit), and the identity for one seat per side;
* an env without `sides` resolves to one side per seat (every two-player
  env: T = 2), and a declared `sides` must fit the env;
* mean_pair_score is bit for bit the two-player Evaluator's per-deal-pair
  mean;
* Lineup and SideFill refuse malformed fills.
"""

import itertools
import sys
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import sides as sides_lib
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.league import config_common as common


def _env(p, zero_sum=True):
  """A stand-in env with `p` players.

  Args:
    p: The players.
    zero_sum: Whether the game is zero-sum.

  Returns:
    The env stand-in.
  """
  return types.SimpleNamespace(name=f"toy{p}", num_players=p, zero_sum=zero_sum)


@pytest.mark.parametrize("p", [2, 3, 4, 6])
def test_individual_is_one_side_per_seat(p):
  """Individual is one side per seat."""
  s = sides_lib.Sides.individual(p, constant_sum=True)
  assert s.side_of_seat == tuple(range(p))
  assert (s.num_players, s.num_sides, s.one_seat_per_side) == (p, p, True)
  assert s.is_two_player == (p == 2)
  assert all(s.seats(i) == (i,) for i in range(p))


@pytest.mark.parametrize("p,t", [(4, 2), (6, 2), (6, 3), (8, 4)])
def test_partnerships_partition_the_seats(p, t):
  """Partnerships partition the seats."""
  s = sides_lib.Sides.partnerships(p, t, constant_sum=True)
  assert s.num_sides == t and s.side_sizes == (p // t,) * t
  covered = sorted(itertools.chain.from_iterable(s.seats(i) for i in range(t)))
  assert covered == list(range(p))
  assert all(s.side(seat) == seat % t for seat in range(p))
  assert not s.one_seat_per_side and not s.is_two_player


def test_bridge_shape():
  """Bridge shape."""
  s = sides_lib.Sides.partnerships(4, 2, constant_sum=True)
  assert s.seats(0) == (0, 2) and s.seats(1) == (1, 3)


@pytest.mark.parametrize("bad", [(0,), (1, 0), (0, 2, 1), (0, 0), (0, 1, 3)])
def test_malformed_partitions_are_refused(bad):
  """Malformed partitions are refused."""
  with pytest.raises(ValueError):
    sides_lib.Sides(bad, constant_sum=True, rotatable=False)


def test_unequal_sides_rotate_only_within_equal_size_classes():
  """Unequal sides rotate only within equal size classes."""
  # 1 vs 2 (skat-like)
  s = sides_lib.Sides((0, 1, 1), constant_sum=True, rotatable=False)
  assert s.side_sizes == (1, 2) and not s.fully_rotatable
  assert s.rotation_classes == ((0,), (1,)) and s.rotation_period == 1
  # rotatable, but nothing to swap
  r = sides_lib.Sides((0, 1, 1), constant_sum=True, rotatable=True)
  assert r.rotation_classes == ((0,), (1,)) and not r.fully_rotatable
  # 3v1v1: seats 0, 2, 4 a team of three, seats 1 and 3 one each.
  u = sides_lib.Sides((0, 1, 0, 2, 0), constant_sum=True)
  assert u.side_sizes == (3, 1, 1) and u.rotation_classes == ((0,), (1, 2))
  assert u.rotation_period == 2
  assert [u.rotate_side(k, 1) for k in range(3)] == [0, 2, 1]
  assert [u.rotate_side(k, 2) for k in range(3)] == [0, 1, 2]
  # 2v2v1v1: classes {0, 1} and {2, 3}, period 2.
  v = sides_lib.Sides((0, 1, 0, 1, 2, 3), constant_sum=False)
  assert v.rotation_classes == ((0, 1), (2, 3)) and v.rotation_period == 2
  # Fully rotatable sides: one class, rotate_side is (k + r) % T.
  f = sides_lib.Sides.individual(4, True)
  assert f.fully_rotatable and f.rotation_classes == ((0, 1, 2, 3),)
  assert [f.rotate_side(k, 3) for k in range(4)] == [
      3,
      0,
      1,
      2,
  ] and f.rotation_period == 4
  assert sides_lib.Sides(
      (0, 1, 0, 1), True, rotatable=False
  ).rotation_classes == (
      (0,),
      (1,),
  )


def test_type_checks():
  """Type checks."""
  with pytest.raises(TypeError):
    sides_lib.Sides((0, 1), constant_sum=1)
  with pytest.raises(TypeError):
    sides_lib.Sides((0, True), constant_sum=True)
  with pytest.raises(ValueError):
    sides_lib.Sides.partnerships(4, 3, True)
  with pytest.raises(ValueError):
    sides_lib.Sides.individual(1, True)
  with pytest.raises(ValueError):
    sides_lib.Sides.individual(2, True).seats(2)


def test_side_returns_are_side_means_numpy():
  """Side returns are side means numpy."""
  s = sides_lib.Sides((0, 1, 0, 1, 2, 2), constant_sum=False, rotatable=True)
  r = np.arange(12, dtype=np.float64).reshape(2, 6)
  got = s.side_returns(r)
  want = np.stack(
      [
          (r[:, 0] + r[:, 2]) / 2,
          (r[:, 1] + r[:, 3]) / 2,
          (r[:, 4] + r[:, 5]) / 2,
      ],
      -1,
  )
  assert got.shape == (2, 3) and np.array_equal(got, want)


def test_side_returns_identity_for_one_seat_per_side():
  """Side returns identity for one seat per side."""
  r = np.random.default_rng(0).normal(size=(5, 3))
  got = sides_lib.Sides.individual(3, False).side_returns(r)
  assert np.array_equal(got, r) and got.dtype == r.dtype


def test_side_returns_jax_matches_numpy_and_jits():
  """Side returns JAX matches numpy and jits."""
  s = sides_lib.Sides.partnerships(4, 2, constant_sum=True)
  r = np.random.default_rng(1).normal(size=(7, 4))
  want = s.side_returns(r)
  got = s.side_returns(jnp.asarray(r))
  assert isinstance(got, jax.Array)
  np.testing.assert_array_equal(np.asarray(got), want)
  np.testing.assert_array_equal(
      np.asarray(jax.jit(s.side_returns)(jnp.asarray(r))), want
  )


def test_side_returns_refuses_wrong_width():
  """Side returns refuses wrong width."""
  with pytest.raises(ValueError, match=r"\[\.\.\., 4\]"):
    sides_lib.Sides.partnerships(4, 2, True).side_returns(np.zeros((3, 2)))


def test_bridge_partner_seats_give_the_seat_return():
  """Bridge partner seats give the seat return."""
  # Bridge seats of one partnership score the same: the side return is that
  # score.
  ns = np.array([-620.0, 100.0, 1430.0])
  r = np.stack([ns, -ns, ns, -ns], -1)
  got = sides_lib.Sides.partnerships(4, 2, True).side_returns(r)
  assert np.array_equal(got, np.stack([ns, -ns], -1))


def test_sides_of_defaults_to_one_side_per_seat():
  """Sides of defaults to one side per seat."""
  env = kuhn.KuhnEnv()
  assert not hasattr(env, "sides")
  s = sides_lib.sides_of(env)
  assert (
      s == sides_lib.Sides.individual(2, True)
      and s.is_two_player
      and s.constant_sum
  )
  gs = sides_lib.sides_of(_env(3, zero_sum=False))
  assert gs.num_sides == 3 and not gs.constant_sum


def test_sides_of_checks_a_declared_sides():
  """Sides of checks a declared sides."""
  env = _env(4)
  env.sides = sides_lib.Sides.partnerships(
      4, 2, constant_sum=bool(env.zero_sum)
  )
  assert sides_lib.sides_of(env).num_sides == 2
  env.sides = sides_lib.Sides.individual(3, True)
  with pytest.raises(ValueError, match="seats"):
    sides_lib.sides_of(env)
  env.sides = (0, 1, 0, 1)
  with pytest.raises(TypeError):
    sides_lib.sides_of(env)


def test_mean_pair_score_is_the_evaluator_pair_mean_bitwise():
  """Mean pair score is the evaluator pair mean bitwise."""
  a_payoff = np.random.default_rng(2).normal(size=(1000, 2)) * 1e3
  got = sides_lib.mean_pair_score(a_payoff[:, 0], a_payoff[:, 1])
  assert got.dtype == np.float64
  # match.Evaluator's per-deal-pair unit
  assert np.array_equal(got, a_payoff.mean(1))


def test_pair_scorer_of_defaults_to_the_mean():
  """Pair scorer of defaults to the mean."""
  assert sides_lib.pair_scorer_of(kuhn.KuhnEnv()) is sides_lib.mean_pair_score


def test_lineup_validation():
  """Lineup validation."""
  lu = sides_lib.lineup(
      ("learner", "pool", "self", "fixed"), members=["anchor@0"]
  )
  assert [f.kind for f in lu.fills] == ["learner", "pool", "self", "fixed"]
  assert lu.fills[3].member == "anchor@0" and lu.rotate == "cyclic"
  lu.check(sides_lib.Sides.individual(4, False))
  with pytest.raises(ValueError, match="fills for"):
    lu.check(sides_lib.Sides.partnerships(4, 2, True))
  with pytest.raises(ValueError, match="learner"):
    sides_lib.lineup(("pool", "pool"))
  with pytest.raises(ValueError):
    sides_lib.lineup(("learner", "fixed"))
  with pytest.raises(ValueError):
    sides_lib.lineup(("learner", "pool"), members=["x@1"])
  with pytest.raises(ValueError):
    sides_lib.lineup(("learner", "pool"), rotate="random")
  with pytest.raises(ValueError):
    sides_lib.SideFill("pool", learner=1)
  with pytest.raises(ValueError):
    sides_lib.SideFill("dealer")
  with pytest.raises(ValueError, match="rotatable"):
    sides_lib.lineup(("learner", "pool")).check(
        sides_lib.Sides((0, 1, 1), True, rotatable=False)
    )
  sides_lib.lineup(("learner", "pool"), rotate="none").check(
      sides_lib.Sides((0, 1, 1), True, rotatable=False)
  )


def test_per_seat_lineups():
  """Per seat lineups."""
  bridge = sides_lib.Sides.partnerships(4, 2, True)
  lu = sides_lib.lineup((("learner", "fixed"), "pool"), members=("partner@3",))
  assert lu.fills[0] == (
      sides_lib.SideFill("learner"),
      sides_lib.SideFill("fixed", member="partner@3"),
  )
  assert [f.kind for f in lu.units] == ["learner", "fixed", "pool"]
  lu.check(bridge)
  with pytest.raises(ValueError, match="per-seat fill of 3 seats"):
    sides_lib.lineup((("learner", "self", "pool"), "pool")).check(bridge)
  with pytest.raises(ValueError, match="learner"):
    sides_lib.lineup((("pool", "self"), "pool"))  # no learner seat anywhere
  with pytest.raises(ValueError, match="at least one seat"):
    sides_lib.Lineup(((), sides_lib.SideFill("learner")))
  with pytest.raises(TypeError):
    sides_lib.Lineup(
        ((sides_lib.SideFill("learner"), "pool"), sides_lib.SideFill("pool"))
    )
  # A per-seat fill sized for an uneven side: 2v1 with learner + frozen partner.
  sides_lib.lineup(
      (("learner", "fixed"), "pool"), rotate="none", members=("p@0",)
  ).check(sides_lib.Sides((0, 1, 0), True))
  with pytest.raises(
      ValueError, match="per-seat fill of 2 seats for a side of 1"
  ):
    sides_lib.lineup(("pool", ("learner", "fixed")), members=("p@0",)).check(
        sides_lib.Sides((0, 1, 0), True)
    )
  assert hash(lu) == hash(
      sides_lib.lineup((("learner", "fixed"), "pool"), members=("partner@3",))
  )
  assert common.plain(lu)["fills"][0] == [
      {"kind": "learner", "learner": 0, "member": ""},
      {"kind": "fixed", "learner": 0, "member": "partner@3"},
  ]


def test_lineup_is_hashable_and_plain():
  """Lineup is hashable and plain."""
  lu = sides_lib.lineup(("learner", "pool"))
  assert hash(lu) == hash(sides_lib.lineup(("learner", "pool")))
  assert common.plain(lu) == {
      "fills": [
          {"kind": "learner", "learner": 0, "member": ""},
          {"kind": "pool", "learner": 0, "member": ""},
      ],
      "rotate": "cyclic",
  }


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
