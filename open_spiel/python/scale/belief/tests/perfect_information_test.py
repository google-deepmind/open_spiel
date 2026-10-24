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

"""Tests for PerfectInformationSampler.

The sampled world is the observed state itself,
K copies of it, and imperfect-information games are refused.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.belief import perfect_information
from open_spiel.python.scale.envs import connect_four
from open_spiel.python.scale.envs import leduc
from open_spiel.python.scale.envs import tic_tac_toe

tree_map = jax.tree_util.tree_map


def _random_states(env, n, plies, seed):
  """N states after up to `plies` uniform-random moves (terminals kept).

  Args:
    env: The env.
    n: The number of states.
    plies: The most random moves.
    seed: The seed.

  Returns:
    The states, leaves [n, ...].
  """
  key = jax.random.key(seed)
  s = jax.vmap(env.init)(jax.random.split(key, n))
  for t in range(plies):
    k = jax.random.fold_in(key, t)
    legal = jax.vmap(env.legal_mask)(s)
    logits = jnp.where(legal, jnp.float32(0.0), -jnp.inf)
    a = jax.random.categorical(k, logits).astype(jnp.int32)
    term = jax.vmap(env.terminal)(s)
    s2 = jax.vmap(env.step)(s, jnp.maximum(a, 0), jax.random.split(k, n))
    s = tree_map(
        lambda x, y, term=term: jnp.where(
            term.reshape(term.shape + (1,) * (x.ndim - 1)), y, x
        ),
        s2,
        s,
    )
  return s


@pytest.mark.parametrize(
    "env_cls,plies",
    [(connect_four.ConnectFourEnv, 11), (tic_tac_toe.TicTacToeEnv, 5)],
)
def test_round_trip_is_the_true_state(env_cls, plies):
  """Round trip is the true state."""
  env = env_cls()
  sampler = perfect_information.PerfectInformationSampler(env)
  n = 16
  s = _random_states(env, n, plies, seed=3)
  actor = jnp.maximum(jax.vmap(env.actor)(s), 0)
  info = jax.vmap(env.info_state)(s, actor)
  params = sampler.init_params(jax.random.key(0))
  carry = sampler.init_carry(params, n, actor, jax.random.key(1))
  worlds, logw = sampler.sample_worlds(
      params, carry, info, 1, jax.random.key(2)
  )
  np.testing.assert_array_equal(np.asarray(logw), np.zeros((n, 1)))
  for got, want in zip(
      jax.tree_util.tree_leaves(worlds), jax.tree_util.tree_leaves(s)
  ):
    assert got.shape == (n, 1) + want.shape[1:], (got.shape, want.shape)
    assert got.dtype == want.dtype, (got.dtype, want.dtype)
    np.testing.assert_array_equal(np.asarray(got[:, 0]), np.asarray(want))


def test_k_worlds_are_copies():
  """K worlds are copies."""
  env = connect_four.ConnectFourEnv()
  sampler = perfect_information.PerfectInformationSampler(env)
  s = _random_states(env, 4, 6, seed=1)
  info = jax.vmap(env.info_state)(s, jnp.zeros((4,), jnp.int32))
  worlds, _ = sampler.sample_worlds({}, (), info, 3, jax.random.key(0))
  for leaf in jax.tree_util.tree_leaves(worlds):
    assert leaf.shape[:2] == (4, 3)
    for k in range(3):
      np.testing.assert_array_equal(
          np.asarray(leaf[:, k]), np.asarray(leaf[:, 0])
      )


def test_observe_is_identity_and_stateless():
  """Observe is identity and stateless."""
  env = connect_four.ConnectFourEnv()
  sampler = perfect_information.PerfectInformationSampler(env)
  assert not sampler.stateful and sampler.is_perfect_information
  assert sampler.params_source == "none"
  assert not sampler.kwargs
  c = sampler.init_carry({}, 2, jnp.zeros((2,), jnp.int32), jax.random.key(0))
  assert sampler.observe({}, c, None) == c


def test_imperfect_information_env_is_refused():
  """Imperfect information env is refused."""
  with pytest.raises(ValueError, match="hidden information"):
    perfect_information.PerfectInformationSampler(leduc.LeducEnv())


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
