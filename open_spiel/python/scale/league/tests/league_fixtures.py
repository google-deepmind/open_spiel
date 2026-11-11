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

"""Shared builders for the League loop tests (kuhn, tiny PG learner)."""

from typing import Any

import jax
import numpy as np

from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.learners import losses
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.networks import catalog
from open_spiel.python.scale.policies import registry


def kuhn_policy_kwargs(width: int = 32) -> dict[str, Any]:
  """Network policy kwargs for kuhn poker.

  Args:
    width: the hidden layer width.

  Returns:
    The kwargs for registry.build("network", ...).
  """
  return common.network_policy_kwargs(
      "kuhn_poker",
      "kuhn_poker_info_index",
      [12],
      2,
      2,
      catalog.mlp([width], "relu", "separate"),
  )


def kuhn_policy(width: int = 32):
  """A kuhn poker network policy.

  Args:
    width: the hidden layer width.

  Returns:
    The policy.
  """
  return registry.build("network", kuhn_policy_kwargs(width))


def br_learner(policy, lr: float = 1e-2, ent: float = 0.01):
  """A PPO best responder for kuhn (constant LR, gamma 1).

  Args:
    policy: the policy to train.
    lr: the constant learning rate.
    ent: the entropy coefficient.

  Returns:
    The PG learner.
  """
  cfg = pg.PGConfig(
      loss="ppo",
      lr=lr,
      lr_end=lr,
      anneal_lr=False,
      num_updates=1,
      max_grad_norm=1.0,
      update_epochs=4,
      num_minibatches=4,
      gamma=1.0,
      gae_lambda=1.0,
      pg=losses.PGLossConfig(clip_eps=0.2, vf_coef=0.5, clip_value=False),
      ent_coef=schedules.Schedule("constant", ent),
      kl_coef=schedules.Schedule("constant", 0.0),
  )
  return pg.PGLearner(policy, cfg)


def init_params(policy, seed: int = 0):
  """Initial params of `policy`.

  Args:
    policy: the policy.
    seed: the jax key seed.

  Returns:
    The params.
  """
  return policy.init_params(jax.random.key(seed))


def env():
  """A kuhn poker env.

  Returns:
    The kuhn.KuhnEnv.
  """
  return kuhn.KuhnEnv()


def leaves_equal(a, b) -> bool:
  """Whether two trees have bitwise-equal leaves.

  Args:
    a: a tree.
    b: another tree.

  Returns:
    True when the leaves match in number and value.
  """
  la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
  return len(la) == len(lb) and all(
      np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb)
  )
