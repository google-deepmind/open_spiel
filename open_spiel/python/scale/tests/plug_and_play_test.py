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

"""Components that pass alone compose by name, and each exists once.

every encoder is an encoders.registry.Encoder reachable by name; a NetworkPolicy
built with a bare encoder name encodes a batched InfoState, with a scalar or an
HL-Gauss value head; HL-Gauss has one implementation shared by the value head
and the losses.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.encoders import registry as enc_registry
from open_spiel.python.scale.envs import leduc
from open_spiel.python.scale.learners import losses
from open_spiel.python.scale.networks import heads
from open_spiel.python.scale.networks import hl_gauss
from open_spiel.python.scale.policies import registry as policy_registry
from open_spiel.python.scale.testing import check_policy
from open_spiel.python.scale.testing import steps as steps_lib


def _mlp(value="scalar"):
  """A small MLP network config.

  Args:
    value: The value head kind.

  Returns:
    The config.
  """
  vh = {"kind": value}
  if value == "hl_gauss":
    vh.update(num_bins=11, v_min=-2.0, v_max=2.0)
  return {
      "compute_dtype": "float32",
      "torso": {"kind": "mlp", "widths": [16, 16], "layout": "shared"},
      "policy_head": {"kind": "dense"},
      "value_head": vh,
  }


def _two_params(pol):
  """Two different params of `pol`.

  Args:
    pol: The policy.

  Returns:
    (params a, params b).
  """
  pa = pol.init_params(jax.random.key(1))
  return pa, jax.tree.map(lambda x: x + 0.5, pa)


def test_every_encoder_is_the_one_encoder_type():
  """Every registered encoder is an Encoder, exactly."""
  for name in enc_registry.names():
    assert enc_registry.get(name).__class__ is enc_registry.Encoder, name


def _leduc_policy(value):
  """A Leduc network policy with value head `value`.

  Args:
    value: The value head kind.

  Returns:
    (env, encoder, policy).
  """
  env = leduc.LeducEnv()
  enc = enc_registry.get("leduc_poker_info_state")
  pol = policy_registry.build(
      "network",
      dict(
          game=env.name,
          obs_format=enc.name,
          num_actions=env.num_actions,
          num_players=env.num_players,
          encoder=enc.name,
          obs_shape=list(enc.shape),
          network=_mlp(value),
          value_layout="seat",
      ),
  )
  return env, enc, pol


def test_network_policy_by_registry_encoder_name_leduc():
  """A NetworkPolicy built by encoder name passes check_policy on leduc."""
  env, _, pol = _leduc_policy("scalar")
  check_policy.check_policy(
      pol, *_two_params(pol), *steps_lib.two_ply_steps(env, batch=8, seat=1)
  )


def test_network_policy_with_an_hl_gauss_value_head_encodes_a_batch():
  """A NetworkPolicy with an hl_gauss value head encodes a batch."""
  env, enc, pol = _leduc_policy("hl_gauss")
  s1, s2 = steps_lib.two_ply_steps(env, batch=8, seat=0)
  x = pol.encode(s1.info)
  assert x.shape == (8,) + tuple(enc.shape) and x.dtype == jnp.float32
  check_policy.check_policy(pol, *_two_params(pol), s1, s2)


def test_hl_gauss_is_one_implementation():
  """Losses and heads use the one hl_gauss implementation."""
  assert losses.HLGaussSupport is hl_gauss.Support
  assert losses.hl_gauss_support is hl_gauss.make_support
  assert losses.hl_gauss_logits_to_scalar is hl_gauss.logits_to_scalar
  head = heads.ValueHead(kind="hl_gauss", num_bins=11, v_min=-2.0, v_max=2.0)
  s = head.support()
  assert s.__class__ is hl_gauss.Support
  # A value head's support feeds the loss directly; the loss equals the module's
  # CE.
  logits = jax.random.normal(jax.random.key(0), (5, 11), jnp.float32)
  y = jnp.asarray([-3.0, -1.0, 0.0, 0.7, 2.0], jnp.float32)
  np.testing.assert_array_equal(
      np.asarray(losses.hl_gauss_cross_entropy(logits, y, s)),
      np.asarray(hl_gauss.cross_entropy(logits, y, s)),
  )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
