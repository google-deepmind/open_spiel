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

r"""A three-player league smoke on Kuhn poker.

Fictitious play over a population.

Three-player Kuhn poker (envs.kuhn_nplayer, OpenSpiel's kuhn_poker(players=3),
no game package needed) under PSRO with the uniform meta-solver: each
iteration trains a best response against the uniform mixture of the
population, admits it, and re-evaluates. At three players every table seats
the learner on one side and draws each other side independently from the
mixture (league.sampler.draw_table), and the league's payoff matrix is the
field view (league.payoff_views): cell (i, j) is i's return against a field
of j under the cyclic design. Numbers it reports are relative to that field,
not equilibrium claims; small enough to run end to end on a CPU in minutes.

  python -m open_spiel.python.scale.train --config=kuhn3_league_smoke \
      --checkpoint_dir=/tmp/kuhn3
"""

from typing import Any

from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.encoders import kuhn_nplayer as kuhn_nplayer_enc
from open_spiel.python.scale.envs import kuhn_nplayer
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import kinds
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.learners import losses
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.networks import catalog

NAME = "kuhn3_league_smoke"
PLAYERS = 3
PSRO_ITERATIONS = 3
BR_UPDATES = 40
NUM_ENVS = 256
NUM_STEPS = 16
EVAL_DEALS = 256
WIDTHS = (64, 64)


def policy_kwargs(widths=WIDTHS) -> dict[str, Any]:
  """A separate-torso MLP over kuhn_nplayer's information-state tensor.

  f32[6N-1].

  Args:
    widths: the hidden widths.

  Returns:
    The network's arch_kwargs.
  """
  return common.network_policy_kwargs(
      kuhn_nplayer.KuhnNPlayerEnv.name,
      kuhn_nplayer_enc.INFO_STATE,
      (6 * PLAYERS - 1,),
      kuhn_nplayer.KuhnNPlayerEnv.num_actions,
      PLAYERS,
      catalog.mlp(list(widths), "relu", "separate"),
  )


def learner(br_updates: int = BR_UPDATES) -> kinds.PG:
  """A PPO best response with a constant step size.

  A multi-learner league's learners train from scratch per slot.

  Args:
    br_updates: updates per best response.

  Returns:
    The kinds.PG spec.
  """
  return kinds.PG(
      pg.PGConfig(
          loss="ppo",
          lr=1e-3,
          lr_end=1e-3,
          anneal_lr=False,
          num_updates=int(br_updates),
          max_grad_norm=0.5,
          adam_eps=1e-5,
          use_adamw=False,
          update_epochs=4,
          num_minibatches=4,
          shuffle="flat",
          gamma=1.0,
          gae_lambda=0.95,
          pg=losses.PGLossConfig(clip_eps=0.2, vf_coef=0.5, clip_value=False),
          ent_coef=schedules.constant(0.01),
          kl_coef=schedules.constant(0.0),
      )
  )


def get_config(
    psro_iterations: int = PSRO_ITERATIONS, br_updates: int = BR_UPDATES
) -> runner.LeagueRunConfig:
  """The smoke's run config (module doc).

  Args:
    psro_iterations: the PSRO iterations.
    br_updates: updates per best response.

  Returns:
    The LeagueRunConfig.
  """
  return runner.psro(
      NAME,
      kuhn_nplayer.KuhnNPlayerEnv.name,
      learner(br_updates),
      policy_kwargs(),
      num_envs=NUM_ENVS,
      num_steps=NUM_STEPS,
      psro_iterations=int(psro_iterations),
      br_updates=int(br_updates),
      meta_solver="uniform",
      eval_deals=EVAL_DEALS,
      notes=(
          "three-player kuhn league smoke: uniform-meta PSRO, field payoffs; "
          "tracking tier"
      ),
  )
