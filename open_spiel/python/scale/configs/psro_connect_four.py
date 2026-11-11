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

"""PSRO on Connect Four with PPO best responses.

PSRO on Connect Four: 10 PSRO iterations of PPO best responses (an MLP
256x2 torso, 1024 rows x 67 plies, 305 updates each), a uniform meta-solver,
50 seat-swapped game pairs per payoff entry (eval_deals; Connect Four has no
chance, so a pair is one game from each seat). Games run on within a row across
updates, so the segment length is free; 67 plies span one or more Connect
Four games (at most 42 moves) per row per update, and 305 updates make about
21M plies per best response.
"""

from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import kinds
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.learners import losses
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.networks import catalog


def get_config() -> runner.LeagueRunConfig:
  """The run config of this example.

  Returns:
    The runner.LeagueRunConfig.
  """
  br_updates = 305
  learner = pg.PGConfig(
      loss="ppo",
      lr=3e-4,
      lr_end=3e-4,
      anneal_lr=False,
      num_updates=br_updates,
      max_grad_norm=0.5,
      adam_eps=1e-5,
      use_adamw=False,
      weight_decay=0.0,
      update_epochs=4,
      num_minibatches=8,
      shuffle="flat",
      gamma=0.99,
      gae_lambda=0.95,
      pg=losses.PGLossConfig(clip_eps=0.2, vf_coef=0.5, clip_value=False),
      ent_coef=schedules.Schedule("constant", 0.01),
      kl_coef=schedules.Schedule("constant", 0.0),
      value_ema=True,
      value_ema_beta=0.925,
  )
  return runner.psro(
      "psro_connect_four",
      "connect_four",
      kinds.PG(learner),
      common.network_policy_kwargs(
          "connect_four",
          "connect_four_flat_planes",
          [126],
          7,
          2,
          catalog.mlp([256, 256], "relu", "shared"),
      ),
      num_envs=1024,
      num_steps=67,
      psro_iterations=10,
      br_updates=br_updates,
      meta_solver="uniform",
      eval_deals=50,
      warm_start=False,
  )
