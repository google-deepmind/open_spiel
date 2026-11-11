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

r"""Euchre from scratch: one move transformer on all four seats, ppo_magnet.

OpenSpiel's corrected euchre at its default options (envs.euchre OPTIONS:
stick the dealer, no lone defender), two partnerships of two. One shared
network plays every seat from that seat's own view (partners share no hidden
information), in pure self-play (configs.recipes.ppo_magnet_self_play): no
pool, no snapshot, no fixed opponent.

  python -m open_spiel.python.scale.train \
      --config=euchre_move_transformer_self_play \
      --checkpoint_dir=/tmp/euchre_move_transformer

A smoke run on a CPU:

  python -m open_spiel.python.scale.train \
      --config=euchre_move_transformer_self_play \
      --checkpoint_dir=/tmp/emt_smoke --num_updates=2 --num_envs=8 \
      --num_steps=16 --set=learner.learner.minibatch_size=64 \
      --set=learner.learner.sgd_steps=2 --set=eval.0.deals=4

The recipe:
  * reward: the env's returns, which are already the team's (the makers'
    score for both makers, its negative for both defenders), so the loss's
    seat credit is the partnership's; the value predicts it over
    VALUE_SCALE (2: hands score -2, +1, +2 or +4), in [-1, 2].
  * network: catalog "euchre_card_move_transformer_bf16" (depth 8, 8 heads,
    D = 384, FFN 4D, 14,530,561 params; bf16 matmuls, f32 params) over the
    card planes encoders.euchre_card_planes [64, 24], one token per card
    plus a token per special move (pass, the four suits, alone, with
    partner); every move a card or special-move key dotted with the
    decision token's query; the value head's kernel and bias drawn from
    U(+-1/sqrt(D)) (VALUE_INIT) and each attention head's keys centred over
    the tokens (CENTER_KEYS).
  * learner: configs.recipes.ppo_magnet_config: PPO-clip 0.2 on raw
    advantages + 0.1 KL to the behaviour policy + an annealed uniform-magnet
    KL (temperature clip(0.05 / (n + 1)**0.3, 0.001, 0.1)) + value MSE, no
    entropy bonus; GAE(0.5) / TD(0.8), gamma 1; one epoch of SGD_STEPS (128)
    minibatches of at most 4,096 plies; Adam eps 1e-8, clip 0.267, step size
    clip(0.5 / (n + 1)**1.1, 5e-6, 1e-4), n = the update throughout.
  * value rows: besides its own decisions, a seat's value is trained where a
    round-robin row clock would have it wait for another seat's move
    (euchre_card_planes.waiting_row), linked into its TD / GAE chain.
  * rollout: NUM_ENVS (4,096) tables x NUM_STEPS (84) plies per update,
    every seat a perspective seat (16,384 batch rows): about 81 decisions
    and 3.1 hands per table per update, so about 331,500 decisions plus
    about 120,000 value rows per update (452,000 trained plies, measured on
    one H100), under 128 x 4,096 = 524,288.
  * length: NUM_UPDATES (3,000) updates; a keep every KEEP_EVERY (100).
  * in-run tracking eval: vs_uniform every UNIFORM_EVERY updates (the
    learner's partnership against two uniform-random seats).

Every in-run number is TRACKING tier.
"""

import dataclasses

from open_spiel.python.scale.configs import recipes
from open_spiel.python.scale.encoders import euchre_card_planes as planes
from open_spiel.python.scale.envs import euchre
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.networks import card_move_transformer
from open_spiel.python.scale.networks import catalog

NAME = "euchre_move_transformer_self_play"
NETWORK = "euchre_card_move_transformer_bf16"
NUM_ENVS = 4096
#: Plies per update: about 81 decisions and 3.3 terminal plies per table,
#: the decisions per table of a 4,096 x 128-row round-robin update whose rows
#: are about 35% waiting rows.
NUM_STEPS = 84
NUM_UPDATES = 3000
KEEP_EVERY = 100
#: 4 seats x (33 - 1) rows: SGD steps per update, each of at most 4,096 plies.
SGD_STEPS = 128
VALUE_SCALE = 2.0
VALUE_ROWS = f"{planes.__name__}:{planes.waiting_row.__name__}"
#: A perspective row decides on about a quarter of its plies (one seat of
#: four acts at each ply) and trains about one value row per eight plies
#: (0.373 of the plies trained in all under a uniform policy, the config
#: test): at most this share of the batch's plies is trained
#: (PGLearner.train_rows raises TrainRowsOverflowError above it; nothing is
#: dropped).
MAX_DECISION_SHARE = 0.45
#: The value head's init and attention key centring
#: (card_move_transformer.with_options): a uniform fan-in value-head init
#: (kernel and bias U(+-1/sqrt(D))) and per-head keys centred over the tokens
#: before the attention product.
VALUE_INIT = "torch_uniform"
CENTER_KEYS = True
UNIFORM_EVAL = "vs_uniform"
UNIFORM_EVERY, UNIFORM_DEALS, UNIFORM_SEED = 100, 512, 65535
SEED = 1

NOTES = (
    "euchre from scratch: one card move transformer on all four seats, "
    "ppo_magnet, team reward; in-run evals are TRACKING tier"
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
      euchre.EuchreEnv.name,
      planes.NAME,
      (planes.N_CHANNEL, planes.N_CARD),
      euchre.NUM_ACTIONS,
      euchre.NUM_PLAYERS,
      card_move_transformer.with_options(
          catalog.get(network), value_init=value_init, center_keys=center_keys
      ),
  )


def learner(num_updates: int = NUM_UPDATES):
  """The learner's PGConfig (module doc).

  Args:
    num_updates: the run length in updates.

  Returns:
    The pg.PGConfig.
  """
  return recipes.ppo_magnet_config(
      num_updates=num_updates,
      sgd_steps=SGD_STEPS,
      value_scale=VALUE_SCALE,
      value_rows=VALUE_ROWS,
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
      euchre.EuchreEnv.name,
      learner(num_updates),
      policy_kwargs(value_init=value_init, center_keys=center_keys),
      num_envs=NUM_ENVS,
      num_steps=NUM_STEPS,
      num_updates=num_updates,
      seed=SEED,
      keep_every=KEEP_EVERY,
      env_kwargs=dict(euchre.OPTIONS),
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
