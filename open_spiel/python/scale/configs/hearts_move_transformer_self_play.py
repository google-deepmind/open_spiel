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

r"""Hearts from scratch: one move transformer on all four seats, ppo_magnet.

OpenSpiel's hearts at pyspiel's default options (envs.hearts OPTIONS), four
seats, each for itself. One shared network plays every seat from that
seat's own view, in pure self-play (configs.recipes.ppo_magnet_self_play):
no pool, no snapshot, no fixed opponent.

  python -m open_spiel.python.scale.train \
      --config=hearts_move_transformer_self_play \
      --checkpoint_dir=/tmp/hearts_move_transformer

A smoke run on a CPU:

  python -m open_spiel.python.scale.train \
      --config=hearts_move_transformer_self_play \
      --checkpoint_dir=/tmp/hmt_smoke --num_updates=2 --num_envs=8 \
      --num_steps=16 --set=learner.learner.minibatch_size=64 \
      --set=learner.learner.sgd_steps=2 --set=eval.0.deals=4

The recipe:
  * reward: each seat's return relative to the field (the loss's field
    credit: its 26 - points minus the mean of the other three seats'), so
    self-play is zero-sum and a moon costs every other seat 26 / 3; the
    value predicts it over VALUE_SCALE (13), in [-2, 2].
  * network: catalog "hearts_card_move_transformer_bf16" (depth 8, 8 heads,
    D = 384, FFN 4D, about 14.5M params; bf16 matmuls, f32 params) over the
    card planes encoders.hearts_card_planes [56, 52], one token per card,
    every move a card key dotted with the decision token's query; the
    value head's kernel and bias drawn from U(+-1/sqrt(D)) (VALUE_INIT) and
    each attention head's keys centred over the tokens (CENTER_KEYS).
  * learner: configs.recipes.ppo_magnet_config: PPO-clip 0.2 on raw
    advantages + 0.1 KL to the behaviour policy + an annealed uniform-magnet
    KL (temperature clip(0.05 / (n + 1)**0.3, 0.001, 0.1)) + value MSE, no
    entropy bonus; GAE(0.5) / TD(0.8), gamma 1; one epoch of SGD_STEPS (128)
    minibatches of at most 4,096 plies; Adam eps 1e-8, clip 0.267, step size
    clip(0.5 / (n + 1)**1.1, 5e-6, 1e-4). SCHEDULES names the two clocks:
    "continued" (the default): from update RESCALE_STEP (3,000) on both
    schedules read n = 14 x the update (RESCALE_TIME_SCALE), the clock of a
    3,000-update run continued under a 14x stretched clock: the step size
    drops to its floor and the temperature to about 0.002. "single": n = the
    update throughout (one clock for the whole run).
  * value rows: besides its own decisions, a seat's value is trained where a
    round-robin row clock would have it wait for a trick winner's lead
    (hearts_card_planes.waiting_row), linked into its TD / GAE chain.
  * rollout: NUM_ENVS (4,096) tables x NUM_STEPS plies per update, every
    seat a perspective seat (16,384 batch rows); about 98 decisions per
    table per update, so about 403,000 decisions plus about 114,000 value
    rows per update (517,000 trained plies, measured on one H100), under
    128 x 4,096 = 524,288.
  * length: NUM_UPDATES (6,000) updates, about 3.1 billion trained plies;
    a keep every KEEP_EVERY (40) updates.
  * in-run tracking eval: vs_uniform every UNIFORM_EVERY updates (the
    learner on one seat against three uniform-random seats; a uniform seat
    averages 19.5 of 26). The moon rates of the rollout are logged every
    update (envs.hearts_telemetry).

Every in-run number is TRACKING tier.
"""

import dataclasses

from open_spiel.python.scale.configs import recipes
from open_spiel.python.scale.encoders import hearts_card_planes as planes
from open_spiel.python.scale.envs import hearts
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.networks import card_move_transformer
from open_spiel.python.scale.networks import catalog

NAME = "hearts_move_transformer_self_play"
NETWORK = "hearts_card_move_transformer_bf16"
NUM_ENVS = 4096
#: Plies per update: a hand is 64 decisions (52 without a pass) plus its
#: terminal ply, so 100 plies are about 98 decisions per table.
NUM_STEPS = 100
NUM_UPDATES = 6000
KEEP_EVERY = 40
#: 4 seats x (33 - 1) rows: SGD steps per update, each of at most 4,096 plies.
SGD_STEPS = 128
VALUE_SCALE = 13.0
RESCALE_STEP = 3000
RESCALE_TIME_SCALE = 14.0
#: The schedule clocks: name -> (rescale_step, rescale_time_scale).
SCHEDULES = {
    "continued": (RESCALE_STEP, RESCALE_TIME_SCALE),
    "single": (0, 1.0),
}
SCHEDULE = "continued"
VALUE_ROWS = f"{planes.__name__}:{planes.waiting_row.__name__}"
#: A perspective row decides on about a quarter of its plies (one seat of
#: four acts at each ply) and trains about 7 value rows per 100 plies (0.316
#: of the plies trained in all, measured on one H100): at most
#: 0.35 of the batch's plies are trained (PGLearner.train_rows raises
#: TrainRowsOverflowError above it; nothing is dropped).
MAX_DECISION_SHARE = 0.35
#: The value head's init and attention key centring
#: (card_move_transformer.with_options): a uniform fan-in value-head init
#: (kernel and bias U(+-1/sqrt(D))) and per-head keys centred over the tokens
#: before the attention product.
VALUE_INIT = "torch_uniform"
CENTER_KEYS = True
UNIFORM_EVAL = "vs_uniform"
UNIFORM_EVERY, UNIFORM_DEALS, UNIFORM_SEED = 200, 512, 65535
SEED = 1

NOTES = (
    "hearts from scratch: one card move transformer on all four seats, "
    "ppo_magnet, field-relative reward; in-run evals are TRACKING tier"
)


def policy_kwargs(
    network: str = NETWORK,
    *,
    value_init: str = VALUE_INIT,
    center_keys: bool = CENTER_KEYS,
) -> dict[str, object]:
  """The network policy's arch_kwargs.

  Args:
    network: the catalog network.
    value_init: card_move_transformer.with_options's value_init.
    center_keys: card_move_transformer.with_options's center_keys.

  Returns:
    The arch_kwargs.
  """
  return common.network_policy_kwargs(
      hearts.HeartsEnv.name,
      planes.NAME,
      (planes.N_CHANNEL, planes.N_CARD),
      hearts.NUM_CARDS,
      hearts.NUM_PLAYERS,
      card_move_transformer.with_options(
          catalog.get(network), value_init=value_init, center_keys=center_keys
      ),
  )


def learner(num_updates: int = NUM_UPDATES, schedule: str = SCHEDULE):
  """The learner's PGConfig (module doc).

  Args:
    num_updates: the run length in updates.
    schedule: a SCHEDULES name.

  Returns:
    The pg.PGConfig.

  Raises:
    ValueError: an unknown schedule.
  """
  if schedule not in SCHEDULES:
    raise ValueError(f"schedule {schedule!r} is not one of {sorted(SCHEDULES)}")
  rescale_step, rescale_time_scale = SCHEDULES[schedule]
  return recipes.ppo_magnet_config(
      num_updates=num_updates,
      sgd_steps=SGD_STEPS,
      value_scale=VALUE_SCALE,
      rescale_step=rescale_step,
      rescale_time_scale=rescale_time_scale,
      value_rows=VALUE_ROWS,
      field_players=hearts.NUM_PLAYERS,
      max_decision_share=MAX_DECISION_SHARE,
  )


def evals(
    every: int = UNIFORM_EVERY, deals: int = UNIFORM_DEALS
) -> tuple[runner.InLoopEval, ...]:
  """The in-run tracking evals.

  Args:
    every: updates between evals.
    deals: duplicate deals per eval.

  Returns:
    The evals.
  """
  return (
      runner.InLoopEval(
          UNIFORM_EVAL, every=every, deals=deals, seed=UNIFORM_SEED
      ),
  )


def build(
    name: str = NAME,
    *,
    num_updates: int = NUM_UPDATES,
    schedule: str = SCHEDULE,
    eval: object = None,  # pylint: disable=redefined-builtin
    notes: str = NOTES,
    metadata: object = None,
    value_init: str = VALUE_INIT,
    center_keys: bool = CENTER_KEYS,
) -> runner.LeagueRunConfig:
  """The run config (module doc).

  Args:
    name: the run name.
    num_updates: the run length in updates.
    schedule: a SCHEDULES name.
    eval: the in-run evals; None is evals().
    notes: the run notes.
    metadata: extra metadata (str values).
    value_init: policy_kwargs's value_init.
    center_keys: policy_kwargs's center_keys.

  Returns:
    The runner.LeagueRunConfig.
  """
  meta = dict(metadata or {})
  cfg = recipes.ppo_magnet_self_play(
      name,
      hearts.HeartsEnv.name,
      learner(num_updates, schedule),
      policy_kwargs(value_init=value_init, center_keys=center_keys),
      num_envs=NUM_ENVS,
      num_steps=NUM_STEPS,
      num_updates=num_updates,
      seed=SEED,
      keep_every=KEEP_EVERY,
      env_kwargs=dict(hearts.OPTIONS),
      eval=evals() if eval is None else eval,
      notes=notes,
  )
  return dataclasses.replace(cfg, metadata={**dict(cfg.metadata), **meta})


def get_config() -> runner.LeagueRunConfig:
  """The run config (module doc).

  Returns:
    The runner.LeagueRunConfig.
  """
  return build()
