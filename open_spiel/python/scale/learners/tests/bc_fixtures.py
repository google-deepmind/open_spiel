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

"""Fixtures the behaviour-cloning tests share (a module, not a test file).

`cfg` builds a BCConfig with the tests' defaults; `two_ply_batch` builds a
Kuhn batch with one acting ply per row and a fixed target policy.
"""

from typing import Any, Sequence

import jax
import jax.numpy as jnp

from open_spiel.python.scale.core import types
from open_spiel.python.scale.learners import bc
from open_spiel.python.scale.learners import losses
from open_spiel.python.scale.testing import steps as steps_lib

F32 = jnp.float32


def cfg(**kw: Any) -> bc.BCConfig:
  """Returns a BCConfig with the tests' defaults, updated by `kw`.

  Args:
    **kw: BCConfig fields that replace the defaults.

  Returns:
    The config.
  """
  args = dict(
      loss=losses.DistillLossConfig(
          policy="kl_target", value="mse", value_coef=0.5
      ),
      learning_rate=1e-2,
      weight_decay=0.0,
      max_grad_norm=10.0,
  )
  args.update(kw)
  return bc.BCConfig(**args)


def two_ply_batch(
    env: types.Env, target: Sequence[float]
) -> types.TrajectoryBatch:
  """Returns a batch of 4 rows of two plies; seat 0 acts at ply 0 only.

  Args:
    env: A two-player env (the tests use Kuhn poker).
    target: The target policy at the acting ply, one entry per action.

  Returns:
    The batch: only ply 0 is valid, with target policy `target`; every
    other field is zero.
  """
  s1, s2 = steps_lib.two_ply_steps(env, batch=4, seat=0)
  steps = jax.tree.map(lambda a, b: jnp.stack([a, b], axis=1), s1, s2)
  b, t, a = 4, 2, env.num_actions
  tp = jnp.zeros((b, t, a), F32).at[:, 0].set(jnp.asarray(target, F32))
  return types.TrajectoryBatch(
      steps=steps,
      init_carry=(),
      action=jnp.zeros((b, t), jnp.int32),
      behavior_logits=jnp.zeros((b, t, a), F32),
      behavior_logp=jnp.zeros((b, t), F32),
      value_pred=jnp.zeros((b, t, 2), F32),
      target_policy=tp,
      target_value=jnp.zeros((b, t, 2), F32),
      reward=jnp.zeros((b, t), F32),
      done=jnp.zeros((b, t), bool),
      valid=jnp.zeros((b, t), bool).at[:, 0].set(True),
      labels={},
      aux={},
      bootstrap_step=s2,
  )
