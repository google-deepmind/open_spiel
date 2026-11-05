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

r"""A euchre league smoke: one learner partnership against a pool.

Euchre (envs.euchre, OpenSpiel's corrected euchre) is a partnership game:
North-South against East-West, two sides of two seats, zero-sum. The learner
holds one side (both seats; the side alternates by row) and the other side
draws one member from the learner's opponent mixture per row, which plays
both of its seats (league.sampler.draw_table, the league's default seating
for two sides of several seats, so the config carries no lineup field). The
pool is lagged snapshots harvested every SNAPSHOT_EVERY updates (newest
MAX_POOL), POOL_SLOTS of them seated per round, a quarter of the rows on the
latest snapshot. Returns are the makers' score, -4..4. A tiny network and a
few deals: a smoke of the wiring, not a training run (tracking tier; its
numbers are not findings). A real run scales NUM_ENVS, NUM_UPDATES and
WIDTHS through get_config's arguments and policy_kwargs.

  python -m open_spiel.python.scale.train --config=euchre_smoke \
      --checkpoint_dir=/tmp/euchre
"""

import dataclasses
from typing import Any

from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.encoders import euchre as euchre_enc
from open_spiel.python.scale.envs import euchre
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import kinds
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.learners import losses
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.networks import catalog

NAME = "euchre_smoke"
NUM_UPDATES = 6
NUM_ENVS = 16
NUM_STEPS = 32  # a hand is at most 31 decisions
SNAPSHOT_EVERY = 2
MAX_POOL = 3
POOL_SLOTS = 2
SELF_WEIGHT = 0.25
EVAL_DEALS = 4
WIDTHS = (32,)
VALUE_SCALE = 4.0  # returns are -4..4


def policy_kwargs(widths=WIDTHS) -> dict[str, Any]:
  """A separate-torso MLP over OpenSpiel's euchre information-state tensor.

  f32[959].

  Args:
    widths: the hidden widths.

  Returns:
    The network's arch_kwargs.
  """
  return common.network_policy_kwargs(
      euchre.EuchreEnv.name,
      euchre_enc.INFO_STATE.name,
      euchre_enc.INFO_STATE.shape,
      euchre.NUM_ACTIONS,
      euchre.NUM_PLAYERS,
      catalog.mlp(list(widths), "relu", "separate"),
  )


def learner(num_updates: int = NUM_UPDATES) -> kinds.PG:
  """The PPO learner of the smoke.

  Args:
    num_updates: the run length in updates.

  Returns:
    The kinds.PG spec.
  """
  return kinds.PG(
      pg.PGConfig(
          loss="ppo",
          lr=2.5e-4,
          lr_end=2.5e-4,
          anneal_lr=False,
          num_updates=int(num_updates),
          max_grad_norm=0.5,
          adam_eps=1e-5,
          use_adamw=False,
          update_epochs=1,
          num_minibatches=2,
          shuffle="flat",
          gamma=1.0,
          gae_lambda=0.95,
          pg=losses.PGLossConfig(clip_eps=0.2, vf_coef=0.5, clip_value=False),
          ent_coef=schedules.constant(0.01),
          kl_coef=schedules.constant(0.0),
          value_scale=VALUE_SCALE,
      )
  )


def get_config(
    num_updates: int = NUM_UPDATES, num_envs: int = NUM_ENVS
) -> runner.LeagueRunConfig:
  """The smoke's run config (module doc).

  Args:
    num_updates: the run length in updates.
    num_envs: Arena rows per update.

  Returns:
    The LeagueRunConfig.
  """
  cfg = runner.snapshot_pool(
      NAME,
      euchre.EuchreEnv.name,
      learner(num_updates),
      policy_kwargs(),
      num_envs=int(num_envs),
      num_steps=NUM_STEPS,
      num_updates=int(num_updates),
      snapshot_every=SNAPSHOT_EVERY,
      harvest_every=SNAPSHOT_EVERY,
      max_pool=MAX_POOL,
      self_weight=SELF_WEIGHT,
      pool_slots=POOL_SLOTS,
      keep_every=int(num_updates),
      notes=(
          "euchre league smoke: 2 partnerships, learner pair v pool pair; "
          "tracking tier"
      ),
  )
  league = dataclasses.replace(
      cfg.league, payoff="evaluator", eval_deals=EVAL_DEALS, eval_max_rows=64
  )
  return dataclasses.replace(cfg, league=league)
