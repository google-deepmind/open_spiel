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

r"""Gin MMD from scratch on gin_rummy, Oklahoma on every deal.

The recipe of the gin card-token transformer trained from random init:
OpenSpiel's standard gin_rummy rules with every deal Oklahoma (pyspiel
`gin_rummy(oklahoma=true)`), the public scripted endgame playing KNOCK /
LAYOFF / WALL (the network learns FIRST_UPCARD, DRAW and DISCARD), MMD
policy gradient against a snapshot pool of its own past selves. A full run
is 30,000 updates of 1,024 rows x 165 plies:

  python -m open_spiel.python.scale.train --config=gin_mmd_from_scratch \
      --checkpoint_dir=/tmp/gin_from_scratch

A smoke run on a CPU:

  python -m open_spiel.python.scale.train --config=gin_mmd_from_scratch \
      --checkpoint_dir=/tmp/gin_smoke --num_updates=2 --num_envs=8 \
      --num_steps=16 --set=eval.0.deals=4

The recipe (league.runner.snapshot_pool):
  * learner: MMD policy gradient (pg_config): KL 0.2 to a magnet re-taken
    from the learner's own params every 500 updates (the first magnet is the
    init), entropy 0.05, clip 0.1, LR 1e-4 annealed linearly to 0 over
    30,000 updates, an HL-Gauss value head, 4 epochs x 4 minibatches.
  * network: catalog "gin_tfv2_d192l3" (networks.gin_tfv2: d192, 3 layers,
    6 heads, bf16 compute, f32 params, HL-Gauss 101 bins on [-110, 110]) on
    the encoder gin_card_tokens (799 floats); random init.
  * opponents: controller scheduled_promotion, all argmax: 0.95 of the rows
    on the lagged learner snapshot (every 1,000 updates), the rest on a pool
    of snapshots harvested every 5,000 updates (newest 40), one seated per
    round.
  * endgame: the public scripted endgame "gin_rummy.scripted_endgame" (the
    gin_rummy GameSpec's endgame), routed for every slot.
  * in-run tracking eval: the scripted rule bot (gin_rummy.rule_bot) every
    1,000 updates on 2,048 seat-swapped duplicate deal pairs, both sides on
    the same endgame.

get_config checks the rules (check_standard_rules: the env plays exactly
OpenSpiel's default rule options on every deal Oklahoma, its EnvSpec is
ENV_KWARGS and pyspiel gin_rummy(oklahoma=true) serves onto it, with the
env's rules_sha recorded in the metadata) and that the run is from scratch
(check_from_scratch: no warm start, no fixed opponents, a pool of the
learner's own snapshots, the magnet re-taken from the learner's own params).
`build(eval=...)` returns the same config with other in-run evals (a
deployment adds its own reference head-to-heads). Equal deadwood after
layoffs is an undercut (the games package's gin_rummy game module);
OpenSpiel releases up to 2.0.2 score it 0-0.
"""

import dataclasses
import json
from typing import Any

from open_spiel.python.jax.games.gin_rummy import game as gin_game
from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import gin
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import kinds
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.learners import losses
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.networks import catalog

NAME = "gin_mmd_from_scratch"

#: OpenSpiel's gin_rummy rule options, written out (never read from a default).
RULES = {
    "allow_upcard_throwback": True,
    "wall_knock_pick": True,
    "spade_doubles": False,
}
OKLAHOMA_FRAC = 1.0  # every deal Oklahoma: pyspiel gin_rummy(oklahoma=true)
ENV_KWARGS = {"oklahoma_frac": OKLAHOMA_FRAC, **RULES}
#: The pyspiel game whose deals this env plays.
PYSPIEL_PARAMS = {"oklahoma": True}
#: The gin_rummy GameSpec's endgame.
STANDARD_ENDGAME = "gin_rummy.scripted_endgame"
#: The metadata's name for these rules (OpenSpiel's standard gin_rummy row).
RULES_ROW = "standard"

NUM_UPDATES = 30_000
NUM_ENVS = 1024
#: Plies per update: 1,024 x 165 rows, about 66,000 learner decisions per
#: update (the learner's share of an Arena ply is about 0.389).
NUM_STEPS = 165
#: PGConfig.max_decision_share.
MAX_DECISION_SHARE = 0.6
ENCODER = "gin_card_tokens"
OBS_DIM = 799
NUM_ACTIONS = 241
NETWORK_NAME = "gin_tfv2_d192l3"

RULE_BOT_EVAL = "gin_rummy.rule_bot"
RULE_BOT_EVERY, RULE_BOT_DEALS, RULE_BOT_SEED = 1000, 2048, 65535

NOTES = (
    "gin MMD from scratch on OpenSpiel's standard "
    "gin_rummy, Oklahoma on every deal, the public scripted endgame; in-run "
    "rule-bot and rules-mismatched h2h evals are TRACKING tier"
)
FROM_SCRATCH = (
    "random init; pool = own snapshots; magnet = own params; anchors eval-only"
)


def pg_config() -> pg.PGConfig:
  """The learner's PGConfig (MMD, module doc).

  Returns:
    The pg.PGConfig.
  """
  return pg.PGConfig(
      loss="mmd",
      lr=1e-4,
      lr_end=0.0,
      anneal_lr=True,
      num_updates=NUM_UPDATES,
      max_grad_norm=0.25,
      adam_eps=1e-5,
      use_adamw=False,
      weight_decay=0.0,
      update_epochs=4,
      num_minibatches=4,
      shuffle="flat",
      gamma=0.99,
      gae_lambda=0.95,
      pg=losses.PGLossConfig(
          clip_eps=0.1,
          vf_coef=0.5,
          clip_value=False,
          normalize_advantages=True,
          value_head="hl_gauss",
          kl_init_to_uniform=False,
          use_advantage_filter=False,
      ),
      ent_coef=schedules.Schedule("constant", 0.05, end=0.0, power=1.0),
      kl_coef=schedules.Schedule("constant", 0.2, end=0.0, power=0.0),
      reference_update_every=500,
      value_ema=False,
      value_ema_beta=0.925,
      max_decision_share=MAX_DECISION_SHARE,
  )


def pg_kind() -> kinds.PG:
  """The learner kind: pg_config with the public endgame routed.

  Returns:
    The kinds.PG.
  """
  return kinds.PG(pg_config(), scripted_endgame=STANDARD_ENDGAME)


def evals(
    rule_bot_every: int = RULE_BOT_EVERY, rule_bot_deals: int = RULE_BOT_DEALS
) -> tuple[runner.InLoopEval, ...]:
  """The run's in-loop tracking evals: the rule bot.

  Args:
    rule_bot_every: updates between evals.
    rule_bot_deals: seat-swapped deal pairs per eval.

  Returns:
    The evals.
  """
  return (
      runner.InLoopEval(
          RULE_BOT_EVAL,
          every=rule_bot_every,
          deals=rule_bot_deals,
          seed=RULE_BOT_SEED,
      ),
  )


def _builtin_gin_spec() -> env_registry.GameSpec:
  """The library's own gin_rummy GameSpec.

  Returns:
    The GameSpec.
  """
  rows = [g for g in env_registry.BUILTIN if g.name == "gin_rummy"]
  assert len(rows) == 1, rows
  return rows[0]


def check_standard_rules(cfg: runner.LeagueRunConfig) -> dict[str, str]:
  """Checks that cfg plays OpenSpiel's standard gin_rummy, every deal Oklahoma.

  Args:
    cfg: the run config.

  Returns:
    The rules record {gin_rules, rules_sha, env_spec}.

  Raises:
    AssertionError: the gin_rummy row is not the library's own, or the env
      plays other rules.
  """
  row = env_registry.game("gin_rummy")
  assert row == _builtin_gin_spec(), f"gin_rummy is not the built-in row: {row}"
  assert row.endgame == STANDARD_ENDGAME, row.endgame
  assert dict(RULES) == dict(gin_game.DEFAULT_RULES), (
      RULES,
      gin_game.DEFAULT_RULES,
  )
  assert cfg.env == "gin_rummy" and dict(cfg.env_kwargs) == ENV_KWARGS, (
      cfg.env,
      cfg.env_kwargs,
  )
  env = env_registry.make(cfg.env, **dict(cfg.env_kwargs))
  assert isinstance(env, gin.GinEnv), type(env)
  assert env.rules == RULES and env.oklahoma_frac == OKLAHOMA_FRAC, (
      env.rules,
      env.oklahoma_frac,
  )
  spec = env_registry.spec_of(env)
  assert spec.to_json() == {
      "name": "gin_rummy",
      "kwargs": dict(sorted(ENV_KWARGS.items())),
  }, spec.to_json()
  served = env_registry.spec(
      "gin_rummy", **row.serving.served_kwargs(PYSPIEL_PARAMS)
  )
  assert served.kwargs["oklahoma_frac"] == OKLAHOMA_FRAC, served.to_json()
  assert env_registry.covers(spec, served), (spec.to_json(), served.to_json())
  sha = env_registry.rules_sha(env)
  assert isinstance(sha, str) and len(sha) == 64, sha
  return {
      "gin_rules": RULES_ROW,
      "rules_sha": sha,
      "env_spec": json.dumps(spec.to_json(), sort_keys=True),
  }


def check_from_scratch(cfg: runner.LeagueRunConfig) -> None:
  """Checks that the run starts from random init and plays only itself.

  Args:
    cfg: the run config.

  Raises:
    AssertionError: a warm start, an external pool member, or a magnet that
      is not the learner's own params.
  """
  pgk, lc = cfg.learner, cfg.learner.learner
  assert isinstance(pgk, kinds.PG), type(pgk)
  assert pgk.init is None and not pgk.parents, (pgk.init, pgk.parents)
  assert cfg.num_initial == 0 and not cfg.fixed_opponents, (
      cfg.num_initial,
      cfg.fixed_opponents,
  )
  assert cfg.controller == "scheduled_promotion", cfg.controller
  assert set(cfg.controller_kwargs) == {
      "main",
      "snapshot_every",
      "harvest_every",
      "max_pool",
      "pool_slots",
      "self_weight",
      "seed",
  }, sorted(cfg.controller_kwargs)
  assert lc.loss == "mmd" and lc.reference_update_every >= 1, (
      lc.loss,
      lc.reference_update_every,
  )
  assert not lc.magnet_from_behavior and not lc.pg.kl_init_to_uniform


def build(eval: Any = None) -> runner.LeagueRunConfig:  # pylint: disable=redefined-builtin
  """The run config with in-run evals `eval` (module doc).

  Args:
    eval: the in-run evals; None is `evals()`.

  Returns:
    The runner.LeagueRunConfig, checked.
  """
  base = runner.snapshot_pool(
      NAME,
      "gin_rummy",
      pg_kind(),
      common.network_policy_kwargs(
          "gin_rummy",
          ENCODER,
          [OBS_DIM],
          NUM_ACTIONS,
          2,
          catalog.get(NETWORK_NAME),
      ),
      num_envs=NUM_ENVS,
      num_steps=NUM_STEPS,
      num_updates=NUM_UPDATES,
      snapshot_every=1000,
      harvest_every=5000,
      max_pool=40,
      pool_slots=1,
      self_weight=0.95,
      opponent_rule=types.ARGMAX,
      learner_rule=types.SAMPLE,
      keep_every=200,
      seed=0,
      env_kwargs=dict(ENV_KWARGS),
      eval=evals() if eval is None else eval,
      notes=NOTES,
  )
  rec = check_standard_rules(base)
  check_from_scratch(base)
  meta = dict(
      rec, rules=json.dumps(RULES, sort_keys=True), from_scratch=FROM_SCRATCH
  )
  return dataclasses.replace(base, metadata=meta)


def get_config() -> runner.LeagueRunConfig:
  """The run config (module doc).

  Returns:
    The runner.LeagueRunConfig.
  """
  return build()
