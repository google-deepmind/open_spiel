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

"""league.sampler: mix rules and the slot-table draw.

league.sampler: mix rules with known weights and the pure, deterministic
draw of a slot table at fixed K.

  * Mix: self-play mass + mass per member name; validated (finite, >= 0,
    sums to 1, no member named "self").
  * Rules (pure numpy, known answers): uniform; pfsp (pfsp_weights +
    returns_to_pfsp_priorities: linear / squared / variance, floor, all-zero
    -> uniform); pmix ((1 - lambda - eps) self + lambda active + eps audit,
    mass with no member folding to self-play); lagged_audit (self-play, a
    lagged learner snapshot, a uniform audit pool); meta (a meta-solver
    strategy over named members); pipeline (pipeline-PSRO: (1 - share) meta +
    share uniform over the lower workers).
  * sample(state, key): the SlotTable (slot 0 = learner, seat r % P) and the
    pools' stacked params, unchanged in shape (fixed K). Stratified draws give
    exact per-seat counts from the weights (largest remainder); the same key
    gives the same table, another key another permutation with the same
    counts; iid draws match the weights in frequency.

Runs without jax (jax_free_control_test): pools hold a stub policy and numpy
params, keys are core.rng keys.
The draw through the Arena is sampler_arena_test.
"""

import sys

import numpy as np
import pytest

from open_spiel.python.scale.core import rng
from open_spiel.python.scale.core import types
from open_spiel.python.scale.league import sampler


class _StubPolicy:
  """A policy identity only (the sampler never calls a policy).

  Attributes:
    spec: the PolicySpec.
  """

  def __init__(self, arch_name="tabular"):
    self.spec = types.PolicySpec(
        "kuhn_poker",
        "kuhn_poker_info_index",
        2,
        2,
        False,
        types.Capability.NATIVE_JAX,
        "jax",
        arch_name,
    )


def _pol():
  """A stub policy.

  Returns:
    The policy.
  """
  return _StubPolicy()


def _pool(names, k=None):
  """A pool of `names` padded to k slots.

  Args:
    names: The member names.
    k: The slots.

  Returns:
    The Pool.
  """
  pol = _pol()
  k = len(names) if k is None else k
  names = tuple(names) + (None,) * (k - len(names))
  logits = np.stack([np.full((12, 2), float(i), np.float32) for i in range(k)])
  return sampler.Pool(
      policy=pol, rule=types.SAMPLE, names=names, params={"logits": logits}
  )


# -- Mix -----------------------------------------------------------------------
def test_mix_validation():
  """Mix validation."""
  sampler.Mix(0.25, {"a": 0.75})
  sampler.Mix(1.0, {})
  with pytest.raises(ValueError):
    sampler.Mix(0.5, {"a": 0.4})  # sums to 0.9
  with pytest.raises(ValueError):
    sampler.Mix(0.5, {"a": -0.1, "b": 0.6})
  with pytest.raises(ValueError):
    sampler.Mix(0.0, {"self": 1.0})
  with pytest.raises(ValueError):
    sampler.Mix(float("nan"), {"a": 1.0})
  m = sampler.Mix(0.0, {"a": 0.5, "b": 0.5, "c": 0.0})
  assert m.support() == ("a", "b")


# -- rules ---------------------------------------------------------------------
def test_uniform_rule():
  """Uniform rule."""
  m = sampler.uniform(["a", "b", "c", "d"], self_play=0.2)
  assert m.self_play == pytest.approx(0.2)
  for n in "abcd":
    assert m.weights[n] == pytest.approx(0.2)
  # nothing to place: all self-play
  assert sampler.uniform([], self_play=0.3).self_play == 1.0


def test_pfsp_weights_known_values():
  """Pfsp weights known values."""
  w = sampler.pfsp_weights([0.2, 0.5, 0.8], mode="linear")
  np.testing.assert_allclose(w, np.array([0.2, 0.5, 0.8]) / 1.5)
  w = sampler.pfsp_weights([0.2, 0.5, 0.8], mode="squared")
  np.testing.assert_allclose(w, np.array([0.04, 0.25, 0.64]) / 0.93)
  w = sampler.pfsp_weights([0.2, 0.5, 0.8], mode="variance")
  np.testing.assert_allclose(w, np.array([0.16, 0.25, 0.16]) / 0.57)
  np.testing.assert_allclose(
      sampler.pfsp_weights([0.0, 0.0], mode="squared"), [0.5, 0.5]
  )
  np.testing.assert_allclose(
      sampler.pfsp_weights([0.0, 1.0], mode="squared", floor=1.0),
      [1.0 / 3.0, 2.0 / 3.0],
  )
  with pytest.raises(ValueError):
    sampler.pfsp_weights([50.0], mode="linear")  # a percent, not a probability
  with pytest.raises(ValueError):
    sampler.pfsp_weights([0.5], mode="cubic")


def test_returns_to_priorities_logistic():
  """Returns to priorities logistic."""
  p = sampler.returns_to_pfsp_priorities([0.0, 5.0, -5.0], scale=5.0)
  np.testing.assert_allclose(
      p, [0.5, 1.0 / (1.0 + np.exp(-1.0)), 1.0 / (1.0 + np.exp(1.0))]
  )
  with pytest.raises(ValueError):
    sampler.returns_to_pfsp_priorities([np.nan])


def test_pfsp_rule_from_returns_and_winrates():
  """Pfsp rule from returns and winrates."""
  m = sampler.pfsp(
      ["a", "b"], winrates=[0.25, 0.75], mode="linear", self_play=0.5
  )
  assert m.self_play == pytest.approx(0.5)
  assert m.weights["a"] == pytest.approx(0.125) and m.weights[
      "b"
  ] == pytest.approx(0.375)
  m = sampler.pfsp(["a", "b"], returns=[0.0, 0.0], mode="squared")
  assert m.weights["a"] == pytest.approx(0.5)
  with pytest.raises(ValueError):
    sampler.pfsp(["a"], winrates=[0.5], returns=[1.0])


def test_pmix_rule_bands_and_folding():
  """Pmix rule bands and folding."""
  m = sampler.pmix(
      active={"a": 0.75, "b": 0.25},
      lambda_t=0.2,
      epsilon=0.05,
      audit=["c", "d"],
  )
  assert m.self_play == pytest.approx(0.75)
  assert m.weights["a"] == pytest.approx(0.15) and m.weights[
      "b"
  ] == pytest.approx(0.05)
  assert m.weights["c"] == pytest.approx(0.025) and m.weights[
      "d"
  ] == pytest.approx(0.025)
  # No audit member: epsilon folds to self-play. No active member: lambda folds.
  m = sampler.pmix(active={}, lambda_t=0.0, epsilon=0.05, audit=[])
  assert m.self_play == pytest.approx(1.0) and not m.support()
  m = sampler.pmix(active={}, lambda_t=0.2, epsilon=0.05, audit=["c"])
  assert m.self_play == pytest.approx(0.95)
  m = sampler.pmix(
      active={"a": 1.0},
      lambda_t=0.2,
      epsilon=0.05,
      audit=[],
      archived=["z"],
      archive_epsilon=0.02,
  )
  assert m.weights["z"] == pytest.approx(0.02) and m.self_play == pytest.approx(
      0.78
  )
  with pytest.raises(ValueError):
    # no self-play left
    sampler.pmix(active={"a": 1.0}, lambda_t=0.7, epsilon=0.4, audit=["c"])


def test_lagged_audit_rule():
  """Lagged audit rule."""
  m = sampler.lagged_audit(
      lagged="main-lag@500",
      audit=["p1", "p2", "p3"],
      self_play=0.4,
      lagged_mass=0.3,
  )
  assert m.self_play == pytest.approx(0.4)
  assert m.weights["main-lag@500"] == pytest.approx(0.3)
  for n in ("p1", "p2", "p3"):
    assert m.weights[n] == pytest.approx(0.1)
  m = sampler.lagged_audit(
      lagged=None, audit=["p1"], self_play=0.4, lagged_mass=0.3
  )
  assert m.self_play == pytest.approx(0.7) and m.weights["p1"] == pytest.approx(
      0.3
  )
  m = sampler.lagged_audit(
      lagged="l@1", audit=[], self_play=0.0, lagged_mass=0.5
  )
  assert m.weights["l@1"] == pytest.approx(
      0.5
  ) and m.self_play == pytest.approx(0.5)


def test_meta_and_pipeline_rules():
  """Meta and pipeline rules."""
  m = sampler.meta(["a", "b", "c"], np.array([0.5, 0.0, 0.5]))
  assert m.weights == {"a": 0.5, "b": 0.0, "c": 0.5} and m.self_play == 0.0
  with pytest.raises(ValueError):
    sampler.meta(["a", "b"], np.array([0.5, 0.6]))
  m = sampler.pipeline(
      ["a", "b"], np.array([0.25, 0.75]), lower=["w0", "w1"], active_share=0.4
  )
  assert m.weights["a"] == pytest.approx(0.15) and m.weights[
      "b"
  ] == pytest.approx(0.45)
  assert m.weights["w0"] == pytest.approx(0.2) and m.weights[
      "w1"
  ] == pytest.approx(0.2)
  m = sampler.pipeline(["a"], np.array([1.0]), lower=[], active_share=0.4)
  assert m.weights == {
      "a": 1.0
  }  # level 0: no lower workers, the meta-strategy alone


# -- sample --------------------------------------------------------------------
def _state(mix, batch=40, pools=None, stratified=True):
  """LeagueSlots over the given pools.

  Args:
    mix: The Mix.
    batch: The rows.
    pools: The pools (default one of a, b, c).
    stratified: Whether draws are stratified.

  Returns:
    The LeagueSlots.
  """
  pools = (_pool(["a", "b", "c"], k=5),) if pools is None else pools
  return sampler.LeagueSlots(
      num_players=2, batch=batch, pools=pools, mix=mix, stratified=stratified
  )


def _rows_of(draw, name):
  """The rows the draw gave `name`.

  Args:
    draw: The draw.
    name: The member.

  Returns:
    The count.
  """
  return draw.counts.get(name, 0)


def test_sample_exact_counts_and_seats_with_stratified_draws():
  """Sample exact counts and seats with stratified draws."""
  mix = sampler.Mix(0.2, {"a": 0.5, "b": 0.3, "c": 0.0})
  st = _state(mix, batch=1000)
  d = sampler.sample(st, rng.key(0))
  assert d.counts == {"self": 200, "a": 500, "b": 300}
  t = d.table
  assert t.seat_slot.dtype == np.int32 and t.seat_member.dtype == np.int32
  assert t.perspective_seat.dtype == np.int32
  np.testing.assert_array_equal(t.perspective_seat, np.arange(1000) % 2)
  rows = np.arange(1000)
  assert (t.seat_slot[rows, t.perspective_seat] == 0).all()
  assert (t.seat_member[rows, t.perspective_seat] == 0).all()
  opp = 1 - t.perspective_seat
  self_rows = t.seat_slot[rows, opp] == 0
  assert self_rows.sum() == 200
  a_rows = (t.seat_slot[rows, opp] == 1) & (t.seat_member[rows, opp] == 0)
  b_rows = (t.seat_slot[rows, opp] == 1) & (t.seat_member[rows, opp] == 1)
  assert a_rows.sum() == 500 and b_rows.sum() == 300
  # Exact per seat: each opponent faces the learner in each seat equally often.
  for sel, n in ((self_rows, 100), (a_rows, 250), (b_rows, 150)):
    assert (
        sel[t.perspective_seat == 0].sum() == n
        and sel[t.perspective_seat == 1].sum() == n
    )


def test_sample_is_a_pure_function_of_state_and_key():
  """Sample is a pure function of state and key."""
  mix = sampler.Mix(0.0, {"a": 0.25, "b": 0.25, "c": 0.5})
  st = _state(mix, batch=64)
  d1, d2 = sampler.sample(st, rng.key(3)), sampler.sample(st, rng.key(3))
  for x, y in zip(d1.table, d2.table):
    np.testing.assert_array_equal(x, y)
  d3 = sampler.sample(st, rng.key(4))
  assert d3.counts == d1.counts
  assert not np.array_equal(d3.table.seat_member, d1.table.seat_member)
  # The state is not mutated and the params pass through at fixed K.
  assert d1.params[0] is st.pools[0].params
  assert d1.params[0]["logits"].shape == (5, 12, 2)


def test_largest_remainder_rounding_is_exact_and_deterministic():
  """Largest remainder rounding is exact and deterministic."""
  mix = sampler.Mix(0.0, {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})
  d = sampler.sample(_state(mix, batch=10), rng.key(0))
  assert sum(d.counts.values()) == 10
  # Per seat 5 rows each over three equal weights: 2,2,1 per seat; 4,4,2 total
  # in category order ("self" first, then names sorted; ties go to the
  # earlier category).
  assert d.counts == {"a": 4, "b": 4, "c": 2}


def test_iid_draws_match_the_weights_in_frequency():
  """Iid draws match the weights in frequency."""
  mix = sampler.Mix(0.1, {"a": 0.6, "b": 0.3})
  st = _state(mix, batch=20000, stratified=False)
  d = sampler.sample(st, rng.key(7))
  for name, w in (("self", 0.1), ("a", 0.6), ("b", 0.3)):
    se = np.sqrt(w * (1 - w) / 20000)
    assert abs(d.counts.get(name, 0) / 20000 - w) < 4 * se, (name, d.counts)


def test_sample_refuses_unknown_or_empty_members():
  """Sample refuses unknown or empty members."""
  with pytest.raises(KeyError):
    sampler.sample(_state(sampler.Mix(0.0, {"zzz": 1.0})), rng.key(0))
  st = sampler.LeagueSlots(
      num_players=2,
      batch=8,
      pools=(_pool(["a"], k=3),),
      mix=sampler.Mix(0.0, {"a": 1.0}),
  )
  sampler.sample(st, rng.key(0))
  with pytest.raises(ValueError):
    sampler.LeagueSlots(
        num_players=2,
        batch=8,
        pools=(_pool(["a", "a"]),),
        mix=sampler.Mix(1.0, {}),
    )


def test_two_pools_map_names_to_slots():
  """Two pools map names to slots."""
  p1 = _pool(["a", "b"], k=2)
  p2 = sampler.Pool(
      policy=_StubPolicy("uniform"),
      rule=types.SAMPLE,
      names=("uniform",),
      params={},
  )
  st = sampler.LeagueSlots(
      num_players=2,
      batch=8,
      pools=(p1, p2),
      mix=sampler.Mix(0.0, {"b": 0.5, "uniform": 0.5}),
  )
  d = sampler.sample(st, rng.key(0))
  opp = 1 - d.table.perspective_seat
  rows = np.arange(8)
  slots, members = d.table.seat_slot[rows, opp], d.table.seat_member[rows, opp]
  assert sorted(set(zip(slots.tolist(), members.tolist()))) == [(1, 1), (2, 0)]
  assert d.counts == {"b": 4, "uniform": 4}


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
