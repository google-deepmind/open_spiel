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

"""TemperatureDrop numerics and HostArena support.

* arena.tempered divides in f64 exactly as decide divides by a rule's
temperature: a drop to T and DecisionRule(kind, T) give bit-identical
probabilities, log-probs and samples (known answer at T = 0.3, where an f32
quotient differs); * HostArena accepts a TemperatureDrop on a HOST_DRIVEN slot,
and a numpy slot with a drop reproduces the same maths' JAX slot with the same
drop in the Arena; the drop changes the decisions it should and only those.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import decide as decide_lib
from open_spiel.python.scale.core import host_arena
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core.tests import numpy_policies
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.policies import uniform
from open_spiel.python.scale.testing import fixtures

tree_map = jax.tree_util.tree_map


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """A Kuhn env.

  Returns:
    The env.
  """
  return kuhn.KuhnEnv()


def _stack(params, k=1):
  """Stacks `params` k times on a new leading axis.

  Args:
    params: The params.
    k: The copies.

  Returns:
    The stacked params.
  """
  return tree_map(lambda x: jnp.stack([x] * k), params)


def _logits(b=64, a=5, seed=0):
  """Random logits.

  Args:
    b: The rows.
    a: The actions.
    seed: The seed.

  Returns:
    f32[b, a].
  """
  return jax.random.normal(jax.random.key(seed), (b, a), jnp.float32) * 3.0


def test_tempered_is_f64_division_known_answer():
  """Tempered divides f64 logits by an f64 temperature (known answer)."""
  slot = arena_lib.Slot(
      uniform.UniformPolicy("kuhn_poker", 5, 2),
      1,
      types.SAMPLE,
      temperature_drop=arena_lib.TemperatureDrop(2, 0.3),
  )
  lg = _logits()
  hp = jnp.asarray(
      np.arange(64) % 4, jnp.int64
  )  # plies 0..3: T = 1, 1, 0.3, 0.3
  rule, scaled = arena_lib.tempered(slot, lg, hp)
  assert rule == types.DecisionRule("sample", 1.0)
  assert scaled.dtype == jnp.float64
  t = np.where(np.asarray(hp) < 2, 1.0, 0.3)
  want = np.asarray(lg, np.float64) / t[:, None]
  np.testing.assert_array_equal(np.asarray(scaled), want)
  # An f32 quotient would be a different number at T = 0.3.
  f32 = (np.asarray(lg) / np.float32(0.3)).astype(np.float64)
  assert (f32[np.asarray(hp) >= 2] != want[np.asarray(hp) >= 2]).any()


@pytest.mark.parametrize("kind", ["sample", "argmax"])
@pytest.mark.parametrize("temp", [0.3, 0.25, 2.0])
def test_drop_equals_rule_temperature_bit_for_bit(kind, temp):
  """Rows past the drop decide exactly as DecisionRule(kind, temp) does.

  On the raw logits: same actions, same f32 logp, same f64 probs.

  Args:
    kind: the decision rule's kind.
    temp: the drop's temperature.
  """
  slot = arena_lib.Slot(
      uniform.UniformPolicy("kuhn_poker", 5, 2),
      1,
      types.DecisionRule(kind, 1.0),
      temperature_drop=arena_lib.TemperatureDrop(1, temp),
  )
  lg = _logits(seed=3)
  legal = (
      jax.random.bernoulli(jax.random.key(4), 0.7, lg.shape).at[:, 0].set(True)
  )
  keys = jax.random.split(jax.random.key(5), lg.shape[0])
  hp = jnp.full((lg.shape[0],), 3, jnp.int64)
  rule, scaled = arena_lib.tempered(slot, lg, hp)
  got = arena_lib.decide_rows(rule, scaled, legal, keys)
  want = arena_lib.decide_rows(types.DecisionRule(kind, temp), lg, legal, keys)
  for g, w in zip(got, want):
    np.testing.assert_array_equal(np.asarray(g), np.asarray(w))
  pg_ = jax.vmap(
      lambda l, m, k: decide_lib.decide(l[None], m[None], rule, k)[1][0]
  )(scaled, legal, keys)
  pw = jax.vmap(
      lambda l, m, k: decide_lib.decide(
          l[None], m[None], types.DecisionRule(kind, temp), k
      )[1][0]
  )(lg, legal, keys)
  np.testing.assert_array_equal(np.asarray(pg_), np.asarray(pw))


def _decide_scaled(logits, rule):
  """Decide's own scaled logits (f64), read from inside decide.

  the value its softmax and argmax / categorical run on (core.decide: logits /
  T, clipped to the f64 range).

  Args:
    logits: the logits.
    rule: the DecisionRule.

  Returns:
    the f64 logits decide's softmax ran on.
  """
  seen = {}
  real = jax.nn.softmax

  def spy(x, axis=-1):
    """Records softmax's input, then calls the real softmax.

    Args:
      x: the scaled logits.
      axis: the softmax axis.

    Returns:
      softmax(x).
    """
    seen["scaled"] = x
    return real(x, axis=axis)

  legal = jnp.ones(jnp.shape(logits), jnp.bool_)
  orig, decide_lib.jax.nn.softmax = decide_lib.jax.nn.softmax, spy
  try:
    decide_lib.decide(logits, legal, rule, jax.random.key(0))
  finally:
    decide_lib.jax.nn.softmax = orig
  return np.asarray(seen["scaled"])


@pytest.mark.parametrize("temp", [0.3, 0.25, 0.1, 2.0, 2.5e-308, 7.0])
def test_drop_and_decide_compute_the_same_f64_value(temp):
  """A TemperatureDrop hands decide decide's own f64 value, bit for bit.

  The value a TemperatureDrop hands decide, as decide scales it at T = 1, is
  decide's own f64 value at DecisionRule(kind, temp) on the raw logits,
  including where logits / T overflows the f64 range (temp 2.5e-308: decide's
  clip).

  Args:
    temp: the temperature.
  """
  slot = arena_lib.Slot(
      uniform.UniformPolicy("kuhn_poker", 5, 2),
      1,
      types.SAMPLE,
      temperature_drop=arena_lib.TemperatureDrop(1, temp),
  )
  lg = _logits(seed=11)
  hp = jnp.full((lg.shape[0],), 5, jnp.int64)
  rule, scaled = arena_lib.tempered(slot, lg, hp)
  assert rule.temperature == 1.0 and scaled.dtype == jnp.float64
  via_drop = _decide_scaled(scaled, rule)
  direct = _decide_scaled(lg, types.DecisionRule("sample", temp))
  assert via_drop.dtype == direct.dtype == np.float64
  np.testing.assert_array_equal(via_drop, direct)
  if temp == 2.5e-308:  # |logit| > 4.5 overflows: decide clips to the f64 range
    assert (
        np.isfinite(direct).all()
        and (np.abs(direct) == np.finfo(np.float64).max).any()
    )


def test_no_drop_leaves_rule_and_logits_alone():
  """Without a TemperatureDrop tempered returns the rule and logits as is."""
  slot = arena_lib.Slot(
      uniform.UniformPolicy("kuhn_poker", 5, 2), 1, types.SAMPLE
  )
  lg = _logits()
  rule, out = arena_lib.tempered(slot, lg, jnp.zeros((64,), jnp.int64))
  assert rule is types.SAMPLE and out is lg


def _pair(env, drop):
  """The same temperature-drop match in the JAX and the host arena.

  Args:
    env: The env.
    drop: The temperature drop.

  Returns:
    The two arenas' batches and logs.
  """
  w = np.array([[0.3, -0.2], [0.1, 0.4]], np.float32)
  b_ = np.array([0.0, 0.25], np.float32)
  jax_pol = fixtures.HistoryPolicy("kuhn_poker", 2, 2)
  host_pol = numpy_policies.NumpyHistoryPolicy("kuhn_poker", 2, 2)
  unif = uniform.UniformPolicy("kuhn_poker", 2, 2)
  table = arena_lib.rotated_table(16, [0, 1], [0, 0])
  dk, pk = arena_lib.duplicate_keys(jax.random.key(1), 16, 2)
  ref = arena_lib.Arena(
      env,
      [
          arena_lib.Slot(jax_pol, 1, types.SAMPLE, drop),
          arena_lib.Slot(unif, 1, types.SAMPLE),
      ],
  )
  _, ba, la = ref.collect(
      [_stack(jax_pol.make_params(w=w, b=b_)), _stack({})], table, dk, pk, 16
  )
  host = host_arena.HostArena(
      env,
      [
          arena_lib.Slot(host_pol, 1, types.SAMPLE, drop),
          arena_lib.Slot(unif, 1, types.SAMPLE),
      ],
  )
  _, bh, lh = host.collect(
      [[host_pol.make_params(w=w, b=b_)], _stack({})], table, dk, pk, 16
  )
  return (ba, la), (bh, lh)


def _hand_ply(new_hand):
  """Each ply's index in its hand.

  Args:
    new_hand: bool[B, T], the new_hand flags.

  Returns:
    int64[B, T].
  """
  hp = np.zeros(new_hand.shape, np.int64)
  for t in range(1, new_hand.shape[1]):
    hp[:, t] = np.where(new_hand[:, t], 0, hp[:, t - 1] + 1)
  return hp


def test_host_driven_slot_takes_a_temperature_drop(env):
  """A HOST_DRIVEN slot in HostArena takes a TemperatureDrop."""
  drop = arena_lib.TemperatureDrop(1, 0.25)
  (ba, la), (bh, lh) = _pair(env, drop)
  np.testing.assert_array_equal(np.asarray(lh.action), np.asarray(la.action))
  np.testing.assert_allclose(
      np.asarray(bh.behavior_logp), np.asarray(ba.behavior_logp), atol=1e-6
  )
  # behavior_logits stay the policy's own (untempered) logits.
  np.testing.assert_allclose(
      np.asarray(bh.behavior_logits), np.asarray(ba.behavior_logits), atol=1e-6
  )
  # Known answer on the host batch: logp = log softmax(logits / T(h))[action].
  valid = np.asarray(bh.valid)
  hp = _hand_ply(np.asarray(bh.steps.new_hand))
  lg = np.asarray(bh.behavior_logits, np.float64)
  t = np.where(hp >= 1, 0.25, 1.0)[..., None]
  z = lg / t
  logsm = (
      z
      - np.log(np.exp(z - z.max(-1, keepdims=True)).sum(-1, keepdims=True))
      - z.max(-1, keepdims=True)
  )
  act = np.maximum(np.asarray(bh.action), 0)
  want = np.take_along_axis(logsm, act[..., None], -1)[..., 0]
  assert (valid & (hp >= 1)).any() and (valid & (hp == 0)).any()
  np.testing.assert_allclose(
      np.asarray(bh.behavior_logp)[valid], want[valid], atol=1e-6
  )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
