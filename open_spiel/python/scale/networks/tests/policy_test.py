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

"""Tests for NetworkPolicy alone.

The Policy contract (check_policy) for every torso
kind, the registry round trip, x64 dtype contracts, the RNN carry, masks
through decide, chunking and determinism.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import decide as decide_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.encoders import registry as enc_registry
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.networks import actor_critic
from open_spiel.python.scale.networks import policy as policy_lib
from open_spiel.python.scale.networks.tests import helpers
from open_spiel.python.scale.policies import registry
from open_spiel.python.scale.testing import check_f64
from open_spiel.python.scale.testing import check_forward_dtypes
from open_spiel.python.scale.testing import check_policy
from open_spiel.python.scale.testing import steps as steps_lib


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """A Kuhn env.

  Returns:
    The env.
  """
  return kuhn.KuhnEnv()


@pytest.fixture(name="kuhn_steps", scope="module")
def _kuhn_steps_fixture(env):
  """Two-ply Kuhn SeatSteps for seat 1.

  Args:
    env: The Kuhn env.

  Returns:
    The SeatSteps.
  """
  return steps_lib.two_ply_steps(env, batch=8, seat=1)


def _kuhn_policy(network, encoder=helpers.KUHN_ONEHOT, obs_shape=(12,), **kw):
  """A Kuhn network policy.

  Args:
    network: The network config.
    encoder: The encoder.
    obs_shape: The observation shape.
    **kw: More NetworkPolicy options.

  Returns:
    The policy.
  """
  return policy_lib.NetworkPolicy(
      "kuhn_poker", "kuhn_onehot", 2, 2, encoder, list(obs_shape), network, **kw
  )


def _two_params(pol):
  """Two different params of `pol`.

  Args:
    pol: The policy.

  Returns:
    (params a, params b).
  """
  pa = pol.init_params(jax.random.key(1))
  pb = jax.tree.map(lambda x: x + 0.5, pa)
  return pa, pb


KUHN_CONFIGS = {
    "mlp_shared_f32": (helpers.mlp_config(), helpers.KUHN_ONEHOT, (12,)),
    "mlp_separate_bf16": (
        helpers.mlp_config("bfloat16", "separate"),
        helpers.KUHN_ONEHOT,
        (12,),
    ),
    "mlp_hl_gauss": (
        helpers.mlp_config(value="hl_gauss"),
        helpers.KUHN_ONEHOT,
        (12,),
    ),
    "cnn": (helpers.cnn_config(), helpers.KUHN_PLANES, (2, 6, 1)),
    "rnn": (helpers.rnn_config(), helpers.KUHN_ONEHOT, (12,)),
}


@pytest.mark.parametrize("name", sorted(KUHN_CONFIGS))
def test_check_policy(name, kuhn_steps):
  """Check policy."""
  network, enc, shape = KUHN_CONFIGS[name]
  pol = _kuhn_policy(network, enc, shape)
  assert pol.spec.stateful == (name == "rnn")
  check_policy.check_policy(pol, *_two_params(pol), *kuhn_steps)


@pytest.mark.parametrize("name", sorted(KUHN_CONFIGS))
def test_registry_round_trip(name, kuhn_steps):
  """Registry round trip."""
  network, enc, shape = KUHN_CONFIGS[name]
  pol = _kuhn_policy(network, enc, shape)
  again = registry.build(pol.spec.arch_name, pol.spec.arch_kwargs)
  assert isinstance(again, policy_lib.NetworkPolicy) and again.spec == pol.spec
  p = pol.init_params(jax.random.key(3))
  s = kuhn_steps[1]
  c = pol.observe(p, pol.init_carry(p, 8, s.seat, None), s)
  a = pol.evaluate(p, c, s, None)
  b = again.evaluate(
      p, again.observe(p, again.init_carry(p, 8, s.seat, None), s), s, None
  )
  np.testing.assert_array_equal(np.asarray(a.logits), np.asarray(b.logits))


def test_injected_encoder_cannot_be_rebuilt():
  """Injected encoder cannot be rebuilt."""
  pol = policy_lib.NetworkPolicy(
      "kuhn_poker",
      "kuhn_onehot",
      2,
      2,
      "unused",
      [12],
      helpers.mlp_config(),
      encoder_fn=helpers.kuhn_onehot,
  )
  assert pol.spec.arch_kwargs["encoder"].startswith("injected:")
  with pytest.raises(ValueError, match="injected"):
    registry.build(pol.spec.arch_name, pol.spec.arch_kwargs)


def test_bad_configs_fail_loud():
  """Bad configs fail loud."""
  with pytest.raises(ValueError, match="unknown network config keys"):
    _kuhn_policy({**helpers.mlp_config(), "dropout": 0.1})
  with pytest.raises(TypeError):
    _kuhn_policy(
        {**helpers.mlp_config(), "torso": {"kind": "mlp", "width": [3]}}
    )
  with pytest.raises(ValueError, match="torso kind"):
    _kuhn_policy({**helpers.mlp_config(), "torso": {"kind": "lstm"}})
  with pytest.raises(ValueError, match="value_layout"):
    _kuhn_policy(helpers.mlp_config(), value_layout="mine")
  pol = _kuhn_policy(helpers.mlp_config(), obs_shape=(13,))
  p = pol.init_params(jax.random.key(0))
  with pytest.raises(ValueError, match="obs_shape"):
    s = steps_lib.two_ply_steps(kuhn.KuhnEnv(), batch=2, seat=0)[0]
    pol.evaluate(p, (), s, None)


ALL_CONFIGS = dict(KUHN_CONFIGS)


@pytest.mark.parametrize("name", sorted(ALL_CONFIGS))
def test_x64_declared_dtypes(name):
  """X64 declared dtypes.

  Under x64: params f32, outputs f32, and the forward has no f64 op except
  where a component declares one (the HL-Gauss support edges).

  Args:
    name: the network config's name in ALL_CONFIGS.
  """
  assert jax.config.jax_enable_x64
  network, enc, shape = ALL_CONFIGS[name]
  pol = _kuhn_policy(network, enc, shape)
  step = steps_lib.two_ply_steps(kuhn.KuhnEnv(), batch=2, seat=0)[0]
  p = pol.init_params(jax.random.key(0))
  assert {l.dtype for l in jax.tree.leaves(p)} == {jnp.dtype(jnp.float32)}
  c = pol.init_carry(p, 2, step.seat, None)
  assert all(l.dtype == jnp.float32 for l in jax.tree.leaves(c))

  def run(p, c, s):
    """Observe then evaluate.

    Args:
      p: the params.
      c: the carry.
      s: the SeatStep.

    Returns:
      (carry, PolicyOutput).
    """
    c = pol.observe(p, c, s)
    return c, pol.evaluate(p, c, s, None)

  c2, out = run(p, c, step)
  for leaf in jax.tree.leaves((c2, out)):
    assert leaf.dtype == jnp.float32, leaf.dtype
  f64_eqns = [
      l
      for l in str(jax.make_jaxpr(run)(p, c, step)).splitlines()
      if "f64[" in l
  ]
  # Declared f64: the HL-Gauss support edges (hl_gauss.make_support).
  declared = name == "mlp_hl_gauss"
  assert bool(f64_eqns) == declared, f64_eqns[:5]
  # the bf16 configs do compute in bf16
  if network["compute_dtype"] == "bfloat16":
    mm = [
        e
        for e in check_f64.equations(jax.make_jaxpr(run)(p, c, step).jaxpr)
        if e.primitive.name == "dot_general"
    ]
    bf16 = [
        e for e in mm if all(v.aval.dtype == jnp.bfloat16 for v in e.invars)
    ]
    assert mm and len(bf16) >= len(mm) // 2, (len(bf16), len(mm))


def test_f64_observation_is_cast_to_the_compute_dtype():
  """F64 observation is cast to the compute dtype."""
  pol = policy_lib.NetworkPolicy(
      "x", "x", 3, 2, helpers.OBS_FIELD, [5], helpers.mlp_config()
  )
  p = pol.init_params(jax.random.key(0))
  x = np.random.default_rng(0).normal(size=(4, 5))
  a = pol.evaluate(
      p, (), helpers.obs_steps(jnp.asarray(x, jnp.float64), num_actions=3), None
  )
  b = pol.evaluate(
      p, (), helpers.obs_steps(jnp.asarray(x, jnp.float32), num_actions=3), None
  )
  np.testing.assert_array_equal(np.asarray(a.logits), np.asarray(b.logits))


def test_value_layouts_and_aux():
  """Value layouts and aux."""
  x = jnp.asarray(np.random.default_rng(0).normal(size=(4, 5)), jnp.float32)
  for layout in ("seat", "zero_sum"):
    pol = policy_lib.NetworkPolicy(
        "x",
        "x",
        3,
        2,
        helpers.OBS_FIELD,
        [5],
        helpers.mlp_config(value="hl_gauss"),
        value_layout=layout,
    )
    p = pol.init_params(jax.random.key(0))
    raw = pol.forward(p, (), helpers.obs_steps(x, num_actions=3))
    out = pol.evaluate(p, (), helpers.obs_steps(x, seat=1, num_actions=3), None)
    np.testing.assert_array_equal(
        np.asarray(out.value[:, 1]), np.asarray(raw["value"])
    )
    other = np.asarray(out.value[:, 0])
    np.testing.assert_array_equal(
        other, 0.0 if layout == "seat" else -np.asarray(raw["value"])
    )
    assert set(out.aux) == {"deadwood", "value_logits"}
    assert out.aux["value_logits"].shape == (4, 11) and out.aux[
        "deadwood"
    ].shape == (4,)


def _rnn_policy():
  """A recurrent toy network policy.

  Returns:
    The policy.
  """
  return policy_lib.NetworkPolicy(
      "x", "x", 3, 2, helpers.OBS_FIELD, [5], helpers.rnn_config()
  )


def test_rnn_carry_step2_depends_on_step1_input():
  """Rnn carry step2 depends on step1 input.

  The carry is used: step-2 logits change when the step-1 input is perturbed;
  a new_hand at step 2 erases step 1.
  """
  pol = _rnn_policy()
  p = pol.init_params(jax.random.key(0))
  rng = np.random.default_rng(0)
  x1, x2 = rng.normal(size=(2, 4, 5)).astype(np.float32)
  s1 = helpers.obs_steps(x1, num_actions=3)
  s1b = helpers.obs_steps(x1 + 1.0, num_actions=3)
  s2 = helpers.obs_steps(x2, num_actions=3, new_hand=False, new_match=False)

  def logits2(first, second):
    """The logits at `second` after observing `first`.

    Args:
      first: the first SeatStep.
      second: the second SeatStep.

    Returns:
      The logits, numpy.
    """
    c = pol.init_carry(p, 4, first.seat, None)
    c = pol.observe(p, pol.observe(p, c, first), second)
    return np.asarray(pol.evaluate(p, c, second, None).logits)

  a, b = logits2(s1, s2), logits2(s1b, s2)
  assert np.abs(a - b).min() > 0.0
  s2_reset = s2._replace(new_hand=jnp.ones((4,), bool))
  np.testing.assert_array_equal(logits2(s1, s2_reset), logits2(s1b, s2_reset))
  # the last action seen at step 2 also reaches the carry (use_last_action)
  s2_act = s2._replace(
      last_action=jnp.full((4,), 1, jnp.int32),
      last_actor=jnp.zeros((4,), jnp.int32),
  )
  assert not np.array_equal(logits2(s1, s2), logits2(s1, s2_act))


def test_decide_masks_network_logits():
  """Decide masks network logits."""
  pol = policy_lib.NetworkPolicy(
      "x", "x", 6, 2, helpers.OBS_FIELD, [5], helpers.mlp_config()
  )
  p = pol.init_params(jax.random.key(0))
  x = jnp.asarray(np.random.default_rng(1).normal(size=(64, 5)), jnp.float32)
  legal = np.random.default_rng(2).random((64, 6)) < 0.4
  legal[np.arange(64), np.arange(64) % 6] = True
  out = pol.evaluate(p, (), helpers.obs_steps(x, legal=legal), None)
  for rule in (types.ARGMAX, types.SAMPLE):
    a, probs = decide_lib.decide(
        out, jnp.asarray(legal), rule, jax.random.key(0)
    )
    assert legal[np.arange(64), np.asarray(a)].all()
    assert (np.asarray(probs)[~legal] == 0).all()


def test_chunked_forward_equals_unchunked():
  """Chunked forward equals unchunked."""
  cfg = helpers.mlp_config()
  x = jnp.asarray(np.random.default_rng(0).normal(size=(7, 5)), jnp.float32)
  a = policy_lib.NetworkPolicy("x", "x", 3, 2, helpers.OBS_FIELD, [5], cfg)
  b = policy_lib.NetworkPolicy(
      "x", "x", 3, 2, helpers.OBS_FIELD, [5], {**cfg, "chunk_size": 3}
  )
  p = a.init_params(jax.random.key(0))
  assert jax.tree.structure(p) == jax.tree.structure(
      b.init_params(jax.random.key(0))
  )
  oa, ob = a.forward(p, (), helpers.obs_steps(x, num_actions=3)), b.forward(
      p, (), helpers.obs_steps(x, num_actions=3)
  )
  np.testing.assert_allclose(
      np.asarray(oa["logits"]), np.asarray(ob["logits"]), atol=1e-6
  )
  np.testing.assert_allclose(
      np.asarray(oa["value"]), np.asarray(ob["value"]), atol=1e-6
  )


@pytest.mark.parametrize("name", sorted(ALL_CONFIGS))
def test_deterministic(name):
  """Deterministic."""
  network, enc, shape = ALL_CONFIGS[name]
  pol = _kuhn_policy(network, enc, shape)
  step = steps_lib.two_ply_steps(kuhn.KuhnEnv(), batch=3, seat=0)[0]
  p = pol.init_params(jax.random.key(0))
  assert jax.tree.all(
      jax.tree.map(
          lambda a, b: bool((a == b).all()),
          p,
          pol.init_params(jax.random.key(0)),
      )
  )
  f = jax.jit(
      lambda p, s: pol.evaluate(
          p, pol.observe(p, pol.init_carry(p, 3, s.seat, None), s), s, None
      )
  )
  a, b = f(p, step), f(p, step)
  np.testing.assert_array_equal(np.asarray(a.logits), np.asarray(b.logits))
  np.testing.assert_array_equal(np.asarray(a.value), np.asarray(b.value))


def test_obs_format_identifies_the_encoder():
  """Obs format identifies the encoder.

  An artifact's obs_format never names an encoder other than the one used,
  nor a shape other than the declared one.
  """
  name = "kuhn_poker_info_index"
  shape = list(enc_registry.get(name).shape)
  net = helpers.mlp_config()
  ok = policy_lib.NetworkPolicy("kuhn_poker", name, 2, 2, name, shape, net)
  assert ok.spec.obs_format == name == ok.spec.arch_kwargs["encoder"]
  with pytest.raises(ValueError, match="IS the obs_format"):
    policy_lib.NetworkPolicy(
        "kuhn_poker", "kuhn_onehot", 2, 2, name, shape, net
    )
  with pytest.raises(ValueError, match="declared shape"):
    policy_lib.NetworkPolicy(
        "kuhn_poker", name, 2, 2, name, [d + 1 for d in shape], net
    )
  # A 'module:attr' (test) encoder may not borrow a registry name ...
  with pytest.raises(ValueError, match="names a registry encoder"):
    policy_lib.NetworkPolicy(
        "kuhn_poker", name, 2, 2, helpers.KUHN_ONEHOT, [12], net
    )
  # ... nor may an injected one.
  with pytest.raises(ValueError, match="names a registry encoder"):
    policy_lib.NetworkPolicy(
        "kuhn_poker", name, 2, 2, "", [12], net, encoder_fn=lambda info: info
    )
  # A free label is fine for a test encoder.
  policy_lib.NetworkPolicy(
      "kuhn_poker", "kuhn_onehot", 2, 2, helpers.KUHN_ONEHOT, [12], net
  )


# Real networks through the testing contract checker: configs that declare no
# f64 anywhere and one compute dtype are checked strictly (f32 params, f32
# outputs, no f64 value, every matmul in the declared compute dtype).
# Declared-f64 configs (HL-Gauss edges) are covered by test_x64_declared_dtypes.
STRICT_CONFIGS = {
    "mlp_shared_f32": (helpers.mlp_config(), helpers.KUHN_ONEHOT, (12,)),
    "mlp_shared_bf16": (
        helpers.mlp_config("bfloat16"),
        helpers.KUHN_ONEHOT,
        (12,),
    ),
    "mlp_separate_bf16": (
        helpers.mlp_config("bfloat16", "separate"),
        helpers.KUHN_ONEHOT,
        (12,),
    ),
    "cnn_f32": (helpers.cnn_config(), helpers.KUHN_PLANES, (2, 6, 1)),
    # CNNTorso declares its conv dtype separately (conv_dtype, default f32);
    # a uniform-bf16 CNN declares both.
    "cnn_bf16": (
        {
            **helpers.cnn_config("bfloat16"),
            "torso": {
                **helpers.cnn_config()["torso"],
                "conv_dtype": "bfloat16",
            },
        },
        helpers.KUHN_PLANES,
        (2, 6, 1),
    ),
    "rnn_f32": (helpers.rnn_config(), helpers.KUHN_ONEHOT, (12,)),
    "rnn_bf16": (helpers.rnn_config("bfloat16"), helpers.KUHN_ONEHOT, (12,)),
}


@pytest.mark.parametrize("name", sorted(STRICT_CONFIGS))
def test_real_networks_pass_check_forward_dtypes(name):
  """Real networks pass check forward dtypes."""
  network, enc, shape = STRICT_CONFIGS[name]
  pol = _kuhn_policy(network, enc, shape)
  step = steps_lib.two_ply_steps(kuhn.KuhnEnv(), batch=2, seat=0)[0]
  p = pol.init_params(jax.random.key(0))
  c = pol.init_carry(p, 2, step.seat, None)

  def apply(params, carry, s):
    """Observe then evaluate.

    Args:
      params: the params.
      carry: the carry.
      s: the SeatStep.

    Returns:
      (carry, PolicyOutput).
    """
    carry = pol.observe(params, carry, s)
    return carry, pol.evaluate(params, carry, s, None)

  check_forward_dtypes.check_forward_dtypes(
      apply,
      p,
      (c, step),
      param_dtype=jnp.float32,
      out_dtype=jnp.float32,
      compute_dtype=jnp.dtype(network["compute_dtype"]),
  )


def test_stack_key_groups_equal_networks():
  """Stack key groups equal networks."""
  a = _kuhn_policy(helpers.mlp_config())
  b = _kuhn_policy(helpers.mlp_config())
  c = _kuhn_policy(helpers.mlp_config(widths=(8,)))
  assert a.stack_key() == b.stack_key() and a.stack_key() != c.stack_key()
  injected = policy_lib.NetworkPolicy(
      "kuhn_poker",
      "kuhn_onehot",
      2,
      2,
      "unused",
      [12],
      helpers.mlp_config(),
      encoder_fn=helpers.kuhn_onehot,
  )
  assert injected.stack_key() is None


def test_build_network_refuses_reserved_and_repeated_aux_keys():
  """Build network refuses reserved and repeated aux keys."""
  cfg = helpers.mlp_config(aux=False)
  with pytest.raises(ValueError, match="reserved"):
    actor_critic.build_network(
        dict(cfg, aux_heads=[{"key": "value_logits"}]), 3
    )
  with pytest.raises(ValueError, match="repeat"):
    actor_critic.build_network(
        dict(cfg, aux_heads=[{"key": "a"}, {"key": "a"}]), 3
    )
  actor_critic.build_network(dict(cfg, aux_heads=[{"key": "a"}]), 3)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
