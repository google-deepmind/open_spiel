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

"""Tests for ActorCritic's forward chunking plan.

Rows split into ceil(n / chunk_size) equal chunks (rounded up to 8 rows, never
above chunk_size), so the padding stays under 8 rows per chunk. Known answers,
and the chunked forward equals the unchunked one at a size that splits
unevenly.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.networks import actor_critic
from open_spiel.python.scale.networks import policy as policy_lib
from open_spiel.python.scale.networks.tests import helpers


def test_chunk_plan_known_answers():
  """Chunk plan known answers."""
  plan = actor_critic.chunk_plan
  assert plan(25344, 16384) == (2, 12672)  # no padding
  assert plan(101376, 16384) == (7, 14488)  # 40 pad rows
  assert plan(16384, 16384) == (1, 16384)
  assert plan(16385, 16384) == (2, 8200)
  assert plan(7, 3) == (3, 3)
  assert plan(10, 3) == (4, 3)
  for n, c in ((25344, 16384), (16385, 16384), (7, 3), (1000, 999), (5, 1)):
    num, size = plan(n, c)
    assert size <= c and num * size >= n and (num - 1) * size < n, (
        n,
        c,
        num,
        size,
    )


def test_uneven_chunks_equal_the_unchunked_forward():
  """Uneven chunks equal the unchunked forward."""
  cfg = helpers.mlp_config()
  x = jnp.asarray(np.random.default_rng(1).normal(size=(10, 5)), jnp.float32)
  a = policy_lib.NetworkPolicy("x", "x", 3, 2, helpers.OBS_FIELD, [5], cfg)
  b = policy_lib.NetworkPolicy(
      "x", "x", 3, 2, helpers.OBS_FIELD, [5], {**cfg, "chunk_size": 3}
  )
  p = a.init_params(jax.random.key(0))
  s = helpers.obs_steps(x, num_actions=3)
  oa, ob = a.forward(p, (), s), b.forward(p, (), s)
  np.testing.assert_allclose(
      np.asarray(oa["logits"]), np.asarray(ob["logits"]), atol=1e-6
  )
  np.testing.assert_allclose(
      np.asarray(oa["value"]), np.asarray(ob["value"]), atol=1e-6
  )
  ga = jax.grad(lambda q: jnp.sum(a.forward(q, (), s)["logits"] ** 2))(p)
  gb = jax.grad(lambda q: jnp.sum(b.forward(q, (), s)["logits"] ** 2))(p)
  for u, v in zip(jax.tree.leaves(ga), jax.tree.leaves(gb)):
    np.testing.assert_allclose(
        np.asarray(u), np.asarray(v), atol=1e-5, rtol=1e-5
    )


# The recording encoder lives in helpers, a module loaded under one name
# wherever this test file is imported from, so the encoder the policy loads
# by name records into the list this test reads.
ENCODED_ROWS = helpers.ENCODED_ROWS
RECORDING_OBS = helpers.RECORDING_OBS


def test_the_encoder_runs_per_chunk():
  """The encoder runs per chunk.

  The encoder is part of the chunked forward, so an encoder's per-row
  intermediates (gin cardtokens' [160, 189] touch gathers) never exist for
  every row at once. Known answer: 10 rows at chunk_size 3 encode as 4 chunks
  of 3 (2 pad rows), forward and gradient; the outputs equal the unchunked
  forward's.
  """
  cfg = helpers.mlp_config()
  x = jnp.asarray(np.random.default_rng(2).normal(size=(10, 5)), jnp.float32)
  a = policy_lib.NetworkPolicy("x", "x", 3, 2, helpers.OBS_FIELD, [5], cfg)
  b = policy_lib.NetworkPolicy(
      "x", "x", 3, 2, RECORDING_OBS, [5], {**cfg, "chunk_size": 3}
  )
  p = a.init_params(jax.random.key(0))
  s = helpers.obs_steps(x, num_actions=3)
  ENCODED_ROWS.clear()
  ob = b.forward(p, (), s)
  # one trace of the scanned chunk body
  assert ENCODED_ROWS == [3], ENCODED_ROWS
  oa = a.forward(p, (), s)
  for k in ("logits", "value"):
    assert ob[k].shape == oa[k].shape and ob[k].dtype == jnp.float32, (
        k,
        ob[k].shape,
    )
    np.testing.assert_allclose(np.asarray(oa[k]), np.asarray(ob[k]), atol=1e-6)
  ENCODED_ROWS.clear()
  ga = jax.grad(lambda q: jnp.sum(a.forward(q, (), s)["logits"] ** 2))(p)
  gb = jax.grad(lambda q: jnp.sum(b.forward(q, (), s)["logits"] ** 2))(p)
  assert ENCODED_ROWS and set(ENCODED_ROWS) == {3}, ENCODED_ROWS
  for u, v in zip(jax.tree.leaves(ga), jax.tree.leaves(gb)):
    np.testing.assert_allclose(
        np.asarray(u), np.asarray(v), atol=1e-5, rtol=1e-5
    )
  ENCODED_ROWS.clear()
  # n <= chunk_size: one call
  b.forward(p, (), helpers.obs_steps(x[:3], num_actions=3))
  assert ENCODED_ROWS == [3], ENCODED_ROWS


def test_remat_is_configurable_default_on_same_math():
  """Remat is configurable default on same math.

  Network "remat" (default true: nn.remat per chunk): off
  stores every chunk's activations instead of recomputing them. Same values and
  gradients; the jaxpr has the remat (checkpoint) primitive only when on.
  """
  cfg = {**helpers.mlp_config(), "chunk_size": 3}
  x = jnp.asarray(np.random.default_rng(2).normal(size=(10, 5)), jnp.float32)
  on = policy_lib.NetworkPolicy("x", "x", 3, 2, helpers.OBS_FIELD, [5], cfg)
  on2 = policy_lib.NetworkPolicy(
      "x", "x", 3, 2, helpers.OBS_FIELD, [5], {**cfg, "remat": True}
  )
  off = policy_lib.NetworkPolicy(
      "x", "x", 3, 2, helpers.OBS_FIELD, [5], {**cfg, "remat": False}
  )
  p = on.init_params(jax.random.key(0))
  s = helpers.obs_steps(x, num_actions=3)

  def loss(pol):
    """The sum of squared logits of `pol`, as a function of params.

    Args:
      pol: The network policy.

    Returns:
      The loss function.
    """
    return lambda q: jnp.sum(pol.forward(q, (), s)["logits"] ** 2)

  jaxprs = {
      n: str(jax.make_jaxpr(jax.grad(loss(pol)))(p))
      for n, pol in (("on", on), ("on2", on2), ("off", off))
  }
  assert "checkpoint" in jaxprs["on"] or "remat" in jaxprs["on"]
  # absent == true: the default program unchanged
  assert jaxprs["on"] == jaxprs["on2"]
  assert "checkpoint" not in jaxprs["off"] and "remat" not in jaxprs["off"]
  np.testing.assert_array_equal(
      np.asarray(on.forward(p, (), s)["logits"]),
      np.asarray(off.forward(p, (), s)["logits"]),
  )
  for u, v in zip(
      jax.tree.leaves(jax.grad(loss(on))(p)),
      jax.tree.leaves(jax.grad(loss(off))(p)),
  ):
    np.testing.assert_allclose(
        np.asarray(u), np.asarray(v), atol=1e-6, rtol=1e-6
    )
  with pytest.raises(ValueError, match="remat"):
    policy_lib.NetworkPolicy(
        "x", "x", 3, 2, helpers.OBS_FIELD, [5], {**cfg, "remat": "no"}
    )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
