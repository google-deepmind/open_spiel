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

r"""A four-player hearts league smoke: one learner against a pool.

Every seat is its own side.

Hearts (envs.hearts, OpenSpiel's hearts) is a free-for-all: four sides of one
seat, general-sum. The learner holds one seat (row r: seat r % 4) and the
other three are pool fills: each draws its own member from the learner's
opponent mixture, independently per row and per seat
(league.sampler.draw_table, the league's default seating at more than two
sides, so the config carries no lineup field). The pool
is lagged snapshots harvested every SNAPSHOT_EVERY updates (newest MAX_POOL),
POOL_SLOTS of them seated per round, a quarter of the rows on the latest
snapshot. The league's payoff matrix is the field view: cell (i, j) is i's
return against a field of j under the cyclic design (league.payoff_views),
in raw hearts returns (26 - points). A tiny network and a few deals: a smoke
of the wiring, not a training run (tracking tier; its numbers are not
findings).

  python -m open_spiel.python.scale.train --config=hearts_league_smoke \
      --checkpoint_dir=/tmp/hearts
"""

import dataclasses
from typing import Any

from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.encoders import hearts as hearts_enc
from open_spiel.python.scale.envs import hearts
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import kinds
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.learners import losses
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.networks import catalog

NAME = "hearts_league_smoke"
NUM_UPDATES = 6
NUM_ENVS = 16
NUM_STEPS = 64
SNAPSHOT_EVERY = 2
MAX_POOL = 3
POOL_SLOTS = 2
SELF_WEIGHT = 0.25
EVAL_DEALS = 4
WIDTHS = (32,)
VALUE_SCALE = 26.0  # returns are 0..36


def policy_kwargs(widths=WIDTHS) -> dict[str, Any]:
  """A separate-torso MLP over pyspiel's hearts information-state tensor.

  f32[5088].

  Args:
    widths: the hidden widths.

  Returns:
    The network's arch_kwargs.
  """
  return common.network_policy_kwargs(
      hearts.HeartsEnv.name,
      hearts_enc.INFO_STATE.name,
      hearts_enc.INFO_STATE.shape,
      hearts.NUM_CARDS,
      hearts.NUM_PLAYERS,
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


def get_config(num_updates: int = NUM_UPDATES) -> runner.LeagueRunConfig:
  """The smoke's run config (module doc).

  Args:
    num_updates: the run length in updates.

  Returns:
    The LeagueRunConfig.
  """
  cfg = runner.snapshot_pool(
      NAME,
      hearts.HeartsEnv.name,
      learner(num_updates),
      policy_kwargs(),
      num_envs=NUM_ENVS,
      num_steps=NUM_STEPS,
      num_updates=int(num_updates),
      snapshot_every=SNAPSHOT_EVERY,
      harvest_every=SNAPSHOT_EVERY,
      max_pool=MAX_POOL,
      self_weight=SELF_WEIGHT,
      pool_slots=POOL_SLOTS,
      keep_every=int(num_updates),
      notes=(
          "hearts league smoke: 4 seats, learner + 3 pool fills, field "
          "payoffs; tracking tier"
      ),
  )
  league = dataclasses.replace(
      cfg.league, payoff="evaluator", eval_deals=EVAL_DEALS, eval_max_rows=64
  )
  return dataclasses.replace(cfg, league=league)
