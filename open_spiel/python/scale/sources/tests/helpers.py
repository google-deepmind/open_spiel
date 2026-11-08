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

"""Test helpers for sources/.

Test helpers for sources/: hand-built TrajectoryBatches and kuhn
policies (teacher / student / opponents).
"""

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.policies import tabular

KUHN_INDEX = f"{kuhn.__name__}:info_index"


def kuhn_tabular(env=None):
  """A kuhn TabularPolicy.

  Args:
    env: unused (the kuhn env).

  Returns:
    The TabularPolicy.
  """
  env = env or kuhn.KuhnEnv()
  return tabular.TabularPolicy(
      "kuhn_poker",
      kuhn.NUM_INFO_STATES,
      env.num_actions,
      env.num_players,
      KUHN_INDEX,
  )


def constant_logits(
    action: int, num_states: int = kuhn.NUM_INFO_STATES, num_actions: int = 2
):
  """f32[S, A] logits that play `action` with probability ~1 (1 - 1e-30).

  Args:
    action: the action played.
    num_states: S.
    num_actions: A.

  Returns:
    The logits.
  """
  x = np.full((num_states, num_actions), -80.0, np.float32)
  x[:, action] = 80.0
  return jnp.asarray(x)


def fake_batch(
    reward,
    valid,
    done,
    seat,
    num_actions=2,
    num_players=2,
    to_move=None,
    labels=None,
):
  """A TrajectoryBatch [B, T] with the given reward/valid/done/seat.

  Zeros elsewhere (info is a f32[B, T, 1] array).

  Args:
    reward: f32[B, T].
    valid: bool[B, T].
    done: bool[B, T].
    seat: int[B, T].
    num_actions: A.
    num_players: P.
    to_move: int[B, T] (default seat).
    labels: the batch labels.

  Returns:
    The TrajectoryBatch.
  """
  reward = jnp.asarray(reward, jnp.float32)
  b, t = reward.shape
  valid = jnp.asarray(valid, jnp.bool_)
  done = jnp.asarray(done, jnp.bool_)
  seat = jnp.broadcast_to(jnp.asarray(seat, jnp.int32), (b, t))
  if to_move is None:
    to_move = jnp.where(valid, seat, 1 - seat)
  to_move = jnp.asarray(to_move, jnp.int32)
  steps = types.SeatStep(
      info=jnp.zeros((b, t, 1), jnp.float32),
      legal_mask=jnp.broadcast_to(valid[..., None], (b, t, num_actions)),
      seat=seat,
      to_move=to_move,
      new_hand=jnp.zeros((b, t), jnp.bool_),
      new_match=jnp.zeros((b, t), jnp.bool_),
      last_action=jnp.full((b, t), types.NO_ACTION, jnp.int32),
      last_actor=jnp.full((b, t), types.NO_ACTOR, jnp.int32),
  )
  boot = jax.tree.map(lambda x: x[:, 0], steps)
  return types.TrajectoryBatch(
      steps=steps,
      init_carry=(),
      action=jnp.zeros((b, t), jnp.int32),
      behavior_logits=jnp.zeros((b, t, num_actions), jnp.float32),
      behavior_logp=jnp.zeros((b, t), jnp.float32),
      value_pred=jnp.zeros((b, t, num_players), jnp.float32),
      target_policy=jnp.zeros((b, t, num_actions), jnp.float32),
      target_value=jnp.zeros((b, t, num_players), jnp.float32),
      reward=reward,
      done=done,
      valid=valid,
      labels=dict(labels or {}),
      aux={},
      bootstrap_step=boot,
  )


def assert_tree_equal(a, b):
  """Same tree structure, same dtypes, bitwise-equal values."""
  la, ta = jax.tree_util.tree_flatten(a)
  lb, tb = jax.tree_util.tree_flatten(b)
  assert ta == tb, (ta, tb)
  for x, y in zip(la, lb):
    x, y = np.asarray(x), np.asarray(y)
    assert x.dtype == y.dtype, (x.dtype, y.dtype)
    assert x.shape == y.shape, (x.shape, y.shape)
    np.testing.assert_array_equal(x, y)
