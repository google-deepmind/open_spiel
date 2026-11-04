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

"""Evaluator head_to_head / payoff_matrix: contract and known-answer tests.

Kuhn (actions pass 0 / bet 1). Hand-computed values (also exact in
exploitability_test through the game tree):
  always-bet vs uniform: +1/2 as p0, +1/4 as p1, so 3/8 per game.
  always-bet vs always-pass: +1 in either seat (the passer folds to a bet).
  a deterministic policy against itself in seat-swapped duplicate deals: the
  two games of a pair are mirror images, so every pair sums to exactly 0.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.ckpt import artifact
from open_spiel.python.scale.core import arena_engine
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.envs import leduc
from open_spiel.python.scale.eval import match
from open_spiel.python.scale.eval import stats
from open_spiel.python.scale.networks import policy as policy_lib
from open_spiel.python.scale.networks.tests import helpers
from open_spiel.python.scale.policies import registry
from open_spiel.python.scale.policies import tabular
from open_spiel.python.scale.policies import uniform
from open_spiel.python.scale.testing import fixtures

PASS, BET = 0, 1
KUHN_INDEX = f"{kuhn.__name__}:info_index"


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """The kuhn Env.

  Returns:
    the Env.
  """
  return kuhn.KuhnEnv()


def _tab(env):
  """A Kuhn tabular policy.

  Args:
    env: The env.

  Returns:
    The policy.
  """
  return tabular.TabularPolicy(
      "kuhn_poker",
      kuhn.NUM_INFO_STATES,
      env.num_actions,
      env.num_players,
      KUHN_INDEX,
  )


def _always(pol, action, name):
  """An entrant that always plays `action`.

  Args:
    pol: The tabular policy.
    action: The action.
    name: The entrant name.

  Returns:
    The Entrant.
  """
  logits = (
      jnp.zeros((kuhn.NUM_INFO_STATES, 2), jnp.float32).at[:, action].set(10.0)
  )
  return match.Entrant(name, pol, pol.params_from_logits(logits), types.ARGMAX)


def _uniform(env, name="uniform"):
  """A uniform entrant.

  Args:
    env: The Kuhn env.
    name: The entrant name.

  Returns:
    The Entrant.
  """
  pol = uniform.UniformPolicy("kuhn_poker", env.num_actions, env.num_players)
  return match.Entrant(name, pol, {}, types.SAMPLE)


def test_always_bet_vs_uniform_known_value(env):
  """Always-bet against uniform has its known value."""
  tab = _tab(env)
  ev = match.Evaluator(env)
  r = ev.head_to_head(
      _always(tab, BET, "bet"), _uniform(env), 4096, jax.random.key(0)
  )
  assert r.n_deals == 4096 and r.a == "bet" and r.b == "uniform" and r.duplicate
  assert r.a_payoff.shape == (4096, 2) and r.a_payoff.dtype == np.float64
  assert r.b_payoff.shape == (4096, 2) and r.b_payoff.dtype == np.float64
  assert abs(r.mean.mean - 0.375) < 4 * r.mean.se, r.mean
  assert abs(r.seat[0].mean - 0.5) < 4 * r.seat[0].se, r.seat
  assert abs(r.seat[1].mean - 0.25) < 4 * r.seat[1].se, r.seat
  assert r.mean.lo == pytest.approx(
      r.mean.mean - stats.Z95 * r.mean.se, abs=1e-12
  )
  assert r.mean.hi == pytest.approx(
      r.mean.mean + stats.Z95 * r.mean.se, abs=1e-12
  )
  # Zero-sum: b's payoff mirrors a's game by game, so the paired difference is
  # 2x.
  np.testing.assert_array_equal(r.b_payoff, -r.a_payoff)
  assert r.diff.mean == pytest.approx(2 * r.mean.mean, abs=1e-12)
  assert r.diff.se == pytest.approx(2 * r.mean.se, abs=1e-12)
  # Kuhn payoffs are integers in {-2, -1, 1, 2}.
  assert set(np.unique(r.a_payoff)) <= {-2.0, -1.0, 1.0, 2.0}


def test_mirror_match_is_exactly_zero_only_with_duplicate_deals(env):
  """A mirror match is exactly zero only with duplicate deals.

  Duplicate deals share the deal across the seat swap: a deterministic
  policy against itself sums to 0 on every pair (SE 0). Independent deals do
  not.

  Args:
    env: the kuhn Env (fixture).
  """
  tab = _tab(env)
  a, b = _always(tab, BET, "bet"), _always(tab, BET, "bet2")
  dup = match.Evaluator(env).head_to_head(a, b, 256, jax.random.key(1))
  np.testing.assert_array_equal(dup.a_payoff.sum(1), 0.0)
  assert dup.mean.mean == 0.0 and dup.mean.se == 0.0
  assert dup.variance_reduction == np.inf
  ind = match.Evaluator(
      env, match.EvaluatorConfig(duplicate=False)
  ).head_to_head(a, b, 256, jax.random.key(1))
  # Independent +-2 games: SE of a pair mean = sqrt(4 / 2) / sqrt(256) ~ 0.088.
  assert not ind.duplicate and np.any(ind.a_payoff.sum(1) != 0)
  assert 0.06 < ind.mean.se < 0.12, ind.mean


def test_duplicate_deals_reduce_variance(env):
  """Duplicate deals give a smaller SE than independent deals."""
  tab = _tab(env)
  a, b = _always(tab, BET, "bet"), _uniform(env)
  n, key = 2048, jax.random.key(2)
  dup = match.Evaluator(env).head_to_head(a, b, n, key)
  ind = match.Evaluator(
      env, match.EvaluatorConfig(duplicate=False)
  ).head_to_head(a, b, n, key)
  for r in (dup, ind):
    assert abs(r.mean.mean - 0.375) < 4 * r.mean.se, r.mean
  assert dup.mean.se < 0.9 * ind.mean.se, (dup.mean.se, ind.mean.se)
  assert dup.variance_reduction > 1.2, dup.variance_reduction
  # The unpaired SE is what 2n independent games would give: close to ind's.
  assert dup.se_unpaired == pytest.approx(ind.mean.se, rel=0.15)


def test_policy_keys_are_separate_from_deal_keys(env):
  """Policy keys are separate from deal keys.

  Stochastic play is not mirrored across the swap (each rotation draws its
  own policy keys) although the deal is; same key -> identical result.

  Args:
    env: the kuhn Env (fixture).
  """
  u1, u2 = _uniform(env, "u1"), _uniform(env, "u2")
  ev = match.Evaluator(env)
  r = ev.head_to_head(u1, u2, 512, jax.random.key(3))
  assert np.mean(r.a_payoff.sum(1) != 0) > 0.3
  r2 = ev.head_to_head(u1, u2, 512, jax.random.key(3))
  np.testing.assert_array_equal(r.a_payoff, r2.a_payoff)
  r3 = ev.head_to_head(u1, u2, 512, jax.random.key(4))
  assert not np.array_equal(r.a_payoff, r3.a_payoff)


def test_chunking_does_not_change_results(env):
  """Row chunking does not change results."""
  tab = _tab(env)
  a, b = _always(tab, BET, "bet"), _uniform(env)
  big = match.Evaluator(env).head_to_head(a, b, 100, jax.random.key(5))
  small = match.Evaluator(env, match.EvaluatorConfig(max_rows=16)).head_to_head(
      a, b, 100, jax.random.key(5)
  )
  np.testing.assert_array_equal(big.a_payoff, small.a_payoff)


def test_leduc_duplicate_shares_the_public_card(env):
  """Leduc duplicate deals share the public card.

  Leduc has a mid-hand chance node (the board card): mirror play of a
  deterministic always-call policy still sums to 0 on every pair.

  Args:
    env: the kuhn Env (fixture).
  """
  del env
  le = leduc.LeducEnv()
  pol = fixtures.HistoryPolicy("leduc_poker", le.num_actions, le.num_players)
  # always call
  params = pol.make_params(c=jnp.zeros(3), b=jnp.array([0.0, 1.0, 0.0]))
  a = match.Entrant("call", pol, params, types.ARGMAX)
  b = match.Entrant("call2", pol, params, types.ARGMAX)
  r = match.Evaluator(le).head_to_head(a, b, 128, jax.random.key(6))
  np.testing.assert_array_equal(r.a_payoff.sum(1), 0.0)
  assert np.any(r.a_payoff != 0)


def test_callback_batched_entrant_equals_native(env):
  """A CALLBACK_BATCHED entrant plays exactly like the NATIVE_JAX one.

  Mixed capabilities through the Arena: a CALLBACK_BATCHED host table
  plays exactly like the NATIVE_JAX table with the same logits.

  Args:
    env: the kuhn Env (fixture).
  """
  logits = np.asarray(jax.random.normal(jax.random.key(7), (12, 2)), np.float32)
  tab = _tab(env)
  native = match.Entrant("t", tab, tab.params_from_logits(logits), types.SAMPLE)
  cb = fixtures.CallbackTabularPolicy("kuhn_poker", 12, 2, 2, KUHN_INDEX)
  cb.set_host_tables(logits[None])
  host = match.Entrant("t", cb, {"member": jnp.int32(0)}, types.SAMPLE)
  ev = match.Evaluator(env)
  key = jax.random.key(8)
  r_native = ev.head_to_head(native, _uniform(env), 256, key)
  r_cb = ev.head_to_head(host, _uniform(env), 256, key)
  np.testing.assert_array_equal(r_native.a_payoff, r_cb.a_payoff)


def test_host_driven_entrant_runs_in_host_arena(env):
  """A HOST_DRIVEN entrant runs in the HostArena.

  A HOST_DRIVEN numpy entrant: HostArena reproduces the in-graph result of
  the same maths in JAX (HistoryPolicy) exactly.

  Args:
    env: the kuhn Env (fixture).
  """
  w = np.array([[0.0, 2.0], [1.5, -1.0]], np.float32)
  jpol = fixtures.HistoryPolicy("kuhn_poker", 2, 2)
  tpol = fixtures.NumpyHistoryPolicy("kuhn_poker", 2, 2)
  ja = match.Entrant("h", jpol, jpol.make_params(w=w), types.SAMPLE)
  ta = match.Entrant("h", tpol, tpol.make_params(w=w), types.SAMPLE)
  ev = match.Evaluator(env)
  key = jax.random.key(9)
  r_j = ev.head_to_head(ja, _uniform(env), 64, key)
  r_t = ev.head_to_head(ta, _uniform(env), 64, key)
  np.testing.assert_array_equal(r_j.a_payoff, r_t.a_payoff)


def test_payoff_matrix_known_answers_and_cells_equal_head_to_head(env):
  """Payoff matrix cells have known values and equal head_to_head."""
  tab = _tab(env)
  bet, pas, uni = (
      _always(tab, BET, "bet"),
      _always(tab, PASS, "pass"),
      _uniform(env),
  )
  ev = match.Evaluator(env)
  key = jax.random.key(10)
  m = ev.payoff_matrix([bet, pas, uni], 512, key)
  assert m.names == ("bet", "pass", "uniform")
  for arr in (m.mean, m.se, m.lo, m.hi):
    assert arr.shape == (3, 3) and arr.dtype == np.float64
  # always-bet beats always-pass by exactly 1 in every game.
  assert m.mean[0, 1] == 1.0 and m.se[0, 1] == 0.0 and m.mean[1, 0] == -1.0
  # Deterministic entrants against themselves: exactly 0.
  assert m.mean[0, 0] == 0.0 and m.mean[1, 1] == 0.0 and m.se[0, 0] == 0.0
  assert abs(m.mean[0, 2] - 0.375) < 4 * m.se[0, 2]
  # Zero-sum: the matrix is antisymmetric off the diagonal, SEs symmetric.
  np.testing.assert_array_equal(
      m.mean, -m.mean.T * (1 - np.eye(3)) + np.diag(np.diag(m.mean))
  )
  np.testing.assert_array_equal(m.se, m.se.T)
  np.testing.assert_allclose(m.lo, m.mean - stats.Z95 * m.se, atol=1e-12)
  np.testing.assert_allclose(m.hi, m.mean + stats.Z95 * m.se, atol=1e-12)
  # Each cell is the head_to_head of that pair under its pair key, exactly.
  for i, j in ((0, 2), (1, 2), (2, 2)):
    r = ev.head_to_head(
        [bet, pas, uni][i], [bet, pas, uni][j], 512, match.pair_key(key, i, j)
    )
    np.testing.assert_array_equal(m.results[(i, j)].a_payoff, r.a_payoff)
    assert m.mean[i, j] == r.mean.mean and m.se[i, j] == r.mean.se


def test_entrant_from_artifact(env, tmp_path):
  """Entrant.from_artifact names the entrant and takes the manifest's rule."""
  tab = _tab(env)
  params = tab.params_from_logits(jnp.zeros((12, 2)).at[:, BET].set(5.0))
  artifact.write_artifact(
      str(tmp_path),
      30,
      tab,
      params,
      rules_sha="r" * 8,
      config_hash="c",
      algo="test",
      env_name="kuhn_poker",
      default_rule=types.ARGMAX,
      run_id="321",
  )
  e = match.Entrant.from_artifact(artifact.load_artifact(str(tmp_path), 30))
  assert e.name == "321@30" and e.rule == types.ARGMAX
  r = match.Evaluator(env).head_to_head(
      e, _uniform(env), 1024, jax.random.key(11)
  )
  assert abs(r.mean.mean - 0.375) < 4 * r.mean.se


def test_rule_override_and_wrap(env):
  """config.rule overrides every entrant's rule.

  config.wrap rewrites every entrant before play.

  Args:
    env: the kuhn Env (fixture).
  """
  tab = _tab(env)
  # Logits favour bet but sampling would still pass sometimes; ARGMAX override
  # makes it always-bet: vs always-pass exactly +1 everywhere.
  soft = match.Entrant(
      "soft",
      tab,
      tab.params_from_logits(jnp.zeros((12, 2)).at[:, BET].set(0.5)),
      types.SAMPLE,
  )
  ev = match.Evaluator(env, match.EvaluatorConfig(rule=types.ARGMAX))
  r = ev.head_to_head(soft, _always(tab, PASS, "pass"), 64, jax.random.key(12))
  np.testing.assert_array_equal(r.a_payoff, 1.0)
  seen = []

  def wrap(e):
    """Records and returns the entrant.

    Args:
      e: the entrant.

    Returns:
      e.
    """
    seen.append(e.name)
    return e

  match.Evaluator(env, match.EvaluatorConfig(wrap=wrap)).head_to_head(
      soft, _uniform(env), 4, jax.random.key(13)
  )
  assert sorted(seen) == ["soft", "uniform"]


def test_fails_loud(env):
  """Bad arguments to the Evaluator fail loud."""
  tab = _tab(env)
  ev = match.Evaluator(env)
  a = _always(tab, BET, "bet")
  with pytest.raises(ValueError, match="2 deal pairs"):
    # SE needs >= 2 deals
    ev.head_to_head(a, _uniform(env), 1, jax.random.key(0))
  with pytest.raises(ValueError, match="no reference"):
    ev.vs_reference(a, 8, jax.random.key(0))  # no reference configured
  wrong = uniform.UniformPolicy("kuhn_poker", 3, 2)
  with pytest.raises(ValueError, match="3 actions"):
    ev.head_to_head(
        a, match.Entrant("w", wrong, {}, types.SAMPLE), 4, jax.random.key(0)
    )
  with pytest.raises(TypeError, match="DecisionRule"):
    match.Entrant("x", tab, {}, "argmax")  # rule must be a DecisionRule


def test_injected_encoders_never_share_a_slot(env):
  """Policies with injected encoders never share a slot.

  Two NetworkPolicies whose injected encoders share a qualname (two lambdas
  from one factory) are different policies and never share a slot; a
  registry-rebuildable policy still groups by spec.

  Args:
    env: the kuhn Env (fixture).
  """
  # Reaches private _slot_key of the code under test.
  # pylint: disable=protected-access
  onehot = registry.resolve_callable(helpers.KUHN_ONEHOT)

  def mk(scale):
    """An encoder lambda; every call makes one with the same qualname.

    Args:
      scale: the encoding's scale.

    Returns:
      the encoder.
    """
    return lambda info: onehot(info) * scale

  def net(fn):
    """A NetworkPolicy with encoder fn.

    Args:
      fn: the injected encoder.

    Returns:
      the policy.
    """
    return policy_lib.NetworkPolicy(
        "kuhn_poker",
        "kuhn_test",
        2,
        2,
        "",
        [12],
        helpers.mlp_config(),
        encoder_fn=fn,
    )

  a, b = net(mk(1.0)), net(mk(-1.0))
  # same qualname
  assert a.spec.arch_kwargs["encoder"] == b.spec.arch_kwargs["encoder"]
  ea = match.Entrant("a", a, a.init_params(jax.random.key(0)), types.ARGMAX)
  eb = match.Entrant("b", b, b.init_params(jax.random.key(0)), types.ARGMAX)
  # Reaches private _slot_key of the code under test.
  assert match._slot_key(ea) != match._slot_key(eb)  # pylint: disable=protected-access
  t1, t2 = _tab(env), _tab(env)
  # Reaches private _slot_key of the code under test.
  assert (
      match._slot_key(_always(t1, BET, "x"))  # pylint: disable=protected-access
      # Reaches private _slot_key of the code under test.
      == match._slot_key(_always(t2, PASS, "y"))
  )  # pylint: disable=protected-access


class _Keyed(uniform.UniformPolicy):
  """A uniform policy that declares which instances may share a slot."""

  def __init__(self, key):
    super().__init__("kuhn_poker", 2, 2)
    self._key = key

  def stack_key(self):
    """The key given at construction.

    Returns:
      the key.
    """
    return self._key


def test_stack_key_decides_slot_sharing(env):
  """The stack key decides slot sharing.

  Policies with equal non-None stack keys share a slot when they play the
  same rule; a policy whose stack key is None shares only with itself.

  Args:
    env: the kuhn Env (fixture).
  """
  # Reaches private _slot_key of the code under test.
  # pylint: disable=protected-access
  a, b, c = _Keyed("k"), _Keyed("k"), _Keyed(None)
  ea, eb = (
      match.Entrant(n, p, {}, types.SAMPLE) for n, p in (("a", a), ("b", b))
  )
  ec1, ec2 = (match.Entrant(n, c, {}, types.SAMPLE) for n in ("c1", "c2"))
  # Reaches private _slot_key of the code under test.
  assert match._slot_key(ea) == match._slot_key(eb)  # pylint: disable=protected-access
  # Reaches private _slot_key of the code under test.
  assert match._slot_key(ea) != match._slot_key(ec1)  # pylint: disable=protected-access
  # Reaches private _slot_key of the code under test.
  assert (
      match._slot_key(ec1)  # pylint: disable=protected-access
      # Reaches private _slot_key of the code under test.
      == match._slot_key(ec2)
  )  # pylint: disable=protected-access
  # Reaches private _slot_key of the code under test.
  assert (
      match._slot_key(ea)  # pylint: disable=protected-access
      # Reaches private _slot_key of the code under test.
      != match._slot_key(match.Entrant("a", a, {}, types.ARGMAX))
  )  # pylint: disable=protected-access
  m = match.Evaluator(env).payoff_matrix([ea, eb], 8, jax.random.key(3))
  assert m.mean.shape == (2, 2)


def test_segment_plies_fits_the_budget():
  """segment_plies fits the log budget and plays at least one ply."""
  # 512 rows x 1001 plies x 8192 actions does not fit one segment.
  seg = arena_engine.segment_plies(512, 1001, 8192)
  assert 1 <= seg < 1001
  per_ply = 512 * (
      arena_engine.BYTES_PER_ACTION * 8192 + arena_engine.BYTES_FIXED
  )
  assert seg * per_ply <= arena_engine.LOG_BUDGET_BYTES
  assert arena_engine.segment_plies(8, 10, 2) == 10  # a small game: one rollout
  assert arena_engine.segment_plies(1, 10, 2, budget=1) == 1


@pytest.mark.parametrize("hands", [1, 3])
def test_ply_segments_do_not_change_results(env, hands):
  """Ply segments do not change results.

  A rollout cut into 1-ply segments (log budget 1 byte) gives the same
  payoffs as one rollout: keys follow row_ply, the state carries over.

  Args:
    env: the kuhn Env (fixture).
    hands: hands per match (parametrized).
  """
  tab = _tab(env)
  logits = np.asarray(
      jax.random.normal(jax.random.key(11), (12, 2)), np.float32
  )
  a = match.Entrant("t", tab, tab.params_from_logits(logits), types.SAMPLE)
  cfg = match.EvaluatorConfig(hands_per_match=hands)
  one = match.Evaluator(env, cfg).head_to_head(
      a, _uniform(env), 64, jax.random.key(12)
  )
  eng = arena_engine.ArenaEngine(env, hands, log_budget_bytes=1)
  seg = match.Evaluator(env, cfg, engine=eng).head_to_head(
      a, _uniform(env), 64, jax.random.key(12)
  )
  np.testing.assert_array_equal(one.a_payoff, seg.a_payoff)
  np.testing.assert_array_equal(one.b_payoff, seg.b_payoff)


def test_ply_segments_over_a_mid_hand_chance_node():
  """Segments over a mid-hand chance node (leduc) give the 1-segment result."""
  le = leduc.LeducEnv()
  pol = fixtures.HistoryPolicy("leduc_poker", le.num_actions, le.num_players)
  params = pol.make_params(c=jnp.zeros(3), b=jnp.array([0.0, 1.0, 0.0]))
  a = match.Entrant("call", pol, params, types.SAMPLE)
  u = match.Entrant(
      "u",
      uniform.UniformPolicy("leduc_poker", le.num_actions, 2),
      {},
      types.SAMPLE,
  )
  one = match.Evaluator(le).head_to_head(a, u, 64, jax.random.key(13))
  seg = match.Evaluator(
      le, engine=arena_engine.ArenaEngine(le, 1, log_budget_bytes=1)
  ).head_to_head(a, u, 64, jax.random.key(13))
  np.testing.assert_array_equal(one.a_payoff, seg.a_payoff)


def test_ply_segments_of_a_host_driven_entrant(env):
  """A HOST_DRIVEN entrant played in segments gives the one-segment result."""
  w = np.array([[0.0, 2.0], [1.5, -1.0]], np.float32)
  tpol = fixtures.NumpyHistoryPolicy("kuhn_poker", 2, 2)
  ta = match.Entrant("h", tpol, tpol.make_params(w=w), types.SAMPLE)
  r1 = match.Evaluator(env).head_to_head(
      ta, _uniform(env), 32, jax.random.key(14)
  )
  r2 = match.Evaluator(
      env, engine=arena_engine.ArenaEngine(env, 1, log_budget_bytes=1)
  ).head_to_head(ta, _uniform(env), 32, jax.random.key(14))
  np.testing.assert_array_equal(r1.a_payoff, r2.a_payoff)


def test_progress_reports_whole_deal_pairs_in_order(env):
  """Progress reports whole deal pairs, in order.

  head_to_head(progress=): after every row chunk, a's payoffs of the deal
  pairs played so far; the last report is the result's a_payoff.

  Args:
    env: the kuhn Env (fixture).
  """
  tab = _tab(env)
  a, b = _always(tab, BET, "bet"), _uniform(env)
  seen = []
  ev = match.Evaluator(env, match.EvaluatorConfig(max_rows=16))
  r = ev.head_to_head(
      a, b, 40, jax.random.key(15), progress=lambda p: seen.append(p.copy())
  )
  assert [s.shape[0] for s in seen] == [8, 16, 24, 32, 40]
  for s in seen:
    np.testing.assert_array_equal(s, r.a_payoff[: s.shape[0]])
  plain = match.Evaluator(env, match.EvaluatorConfig(max_rows=16)).head_to_head(
      a, b, 40, jax.random.key(15)
  )
  np.testing.assert_array_equal(plain.a_payoff, r.a_payoff)
  with pytest.raises(ValueError, match="one job"):
    # Reaches private _play of the code under test.
    ev._play(  # pylint: disable=protected-access
        [a, b],
        [(0, 1), (1, 0)],
        4,
        [jax.random.key(1)] * 2,
        progress=lambda p: None,
    )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
