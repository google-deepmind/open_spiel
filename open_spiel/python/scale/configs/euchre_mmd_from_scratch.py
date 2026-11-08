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

r"""Euchre MMD from scratch: one learner partnership against its own snapshots.

OpenSpiel's corrected euchre (envs.euchre) at its two game parameters'
defaults (envs.euchre OPTIONS: no lone defender, stick the dealer), four
seats in two partnerships, North-South against East-West, zero-sum. The
learner holds one side, both of its seats (the side alternates by row); the
other side draws one opponent per row from the round's mixture
(league.sampler.draw_table), which plays both of its seats: the lagged
learner snapshot on SELF_WEIGHT of the draws, else one of the POOL_SLOTS
seated pool members. Returns are the makers' score for both makers (1, 2 or
4; -2 when euchred) and its negative for the defenders.

  python -m open_spiel.python.scale.train --config=euchre_mmd_from_scratch \
      --checkpoint_dir=/tmp/euchre_from_scratch

A smoke run on a CPU:

  python -m open_spiel.python.scale.train --config=euchre_mmd_from_scratch \
      --checkpoint_dir=/tmp/euchre_smoke --num_updates=2 --num_envs=8 \
      --num_steps=16 --set=eval.0.deals=4

The run length is sized to the throughput measured on one H100: about
0.26 s per update with the evals, so 180,000 updates is about 13 h (a
tracking read, not a benchmark).

The recipe (league.runner.snapshot_pool), hearts_mmd_from_scratch's adapted
to two partnerships:
  * learner: MMD policy gradient (pg_config): KL 0.1 to a magnet re-taken
    from the learner's own params every 1,000 updates (the first magnet is
    the init), entropy 0.01, clip 0.2, LR 3e-4 annealed linearly to 3e-5 over
    NUM_UPDATES, 4 epochs x 4 minibatches, gamma 1 (one hand is one
    episode), returns scaled by 1/4 for the value targets.
  * batch: 2,048 rows x 32 plies (a hand is at most 31 decisions), the same
    65,536 plies per update as hearts' 1,024 x 64.
  * network: a separate-torso MLP (1024, 512, 256), relu, on OpenSpiel's
    euchre information-state tensor (encoders.euchre, f32[959]); random
    init.
  * opponents: controller scheduled_promotion, all argmax: SELF_WEIGHT of
    the side draws on the lagged learner snapshot (every 1,000 updates),
    the rest on a pool of snapshots harvested every 5,000 updates (newest
    20), two seated per round.
  * in-run tracking eval: vs_uniform every 1,000 updates, the learner's side
    against a uniform-random side on 512 deals, seat-swapped (two sides:
    design auto); eval/vs_uniform/mean is the learner side's points per hand
    (-2..4 at these options, -4 needing a lone defender; a uniform side
    against a uniform side averages 0).

get_config checks the rules (check_standard_rules: the env plays euchre at
its default options, recorded with the env's rules_sha in the metadata) and
that the run is from scratch (check_from_scratch: no warm start, no fixed
opponents, a pool of the learner's own snapshots, the magnet re-taken from
the learner's own params). Every in-run number is TRACKING tier.
"""

import dataclasses
import json

from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.core import sides as sides_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.encoders import euchre as euchre_enc
from open_spiel.python.scale.envs import euchre
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import kinds
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.learners import losses
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.networks import catalog

NAME = "euchre_mmd_from_scratch"

#: OpenSpiel euchre's two game parameters at their defaults, written out.
RULES = {"allow_lone_defender": False, "stick_the_dealer": True}
RULES_ROW = "standard"

NUM_UPDATES = 180_000
NUM_ENVS = 2048
#: Plies per update: one hand is at most 31 player decisions (envs.euchre).
NUM_STEPS = 32
WIDTHS = (1024, 512, 256)
#: Returns are -2..4 at these options (the makers' score; -4 needs a lone
#: defender, off here); the scale is the largest magnitude, 4.
VALUE_SCALE = 4.0

SNAPSHOT_EVERY = 1000
HARVEST_EVERY = 5000
MAX_POOL = 20
POOL_SLOTS = 2
SELF_WEIGHT = 0.75
KEEP_EVERY = 5000
MAGNET_EVERY = 1000

UNIFORM_EVAL = "vs_uniform"
UNIFORM_EVERY, UNIFORM_DEALS, UNIFORM_SEED = 1000, 512, 65535
#: The in-loop read cadence a launcher declares (eval_every, the scale of an
#: alert on missing reads): the vs_uniform reads every
#: UNIFORM_EVERY updates were 4.4 min apart on one H100 (metrics.jsonl row
#: times), about 0.26 s per update.
LAUNCH_EVAL_EVERY = "15m"

NOTES = (
    "euchre MMD from scratch on OpenSpiel's corrected euchre at its default "
    "options; 2 partnerships, learner side v own snapshots; in-run "
    "vs_uniform eval is TRACKING tier"
)
FROM_SCRATCH = "random init; pool = own snapshots; magnet = own params"


def policy_kwargs() -> dict[str, object]:
  """The network's arch_kwargs (module doc).

  Returns:
    The arch_kwargs.
  """
  return common.network_policy_kwargs(
      euchre.EuchreEnv.name,
      euchre_enc.INFO_STATE.name,
      euchre_enc.INFO_STATE.shape,
      euchre.NUM_ACTIONS,
      euchre.NUM_PLAYERS,
      catalog.mlp(list(WIDTHS), "relu", "separate"),
  )


def pg_config(num_updates: int = NUM_UPDATES) -> pg.PGConfig:
  """The learner's PGConfig (MMD, module doc).

  max_decision_share is left at None (a bound no batch can exceed): a
  learner side holds two seats, and with a lone opponent it plays two cards
  of every three, so no tight share holds for every 32-ply window.

  Args:
    num_updates: the run length in updates (the LR anneal's horizon).

  Returns:
    The pg.PGConfig.
  """
  return pg.PGConfig(
      loss="mmd",
      lr=3e-4,
      lr_end=3e-5,
      anneal_lr=True,
      num_updates=int(num_updates),
      max_grad_norm=0.5,
      adam_eps=1e-5,
      use_adamw=False,
      weight_decay=0.0,
      update_epochs=4,
      num_minibatches=4,
      shuffle="flat",
      gamma=1.0,
      gae_lambda=0.95,
      pg=losses.PGLossConfig(
          clip_eps=0.2,
          vf_coef=0.5,
          clip_value=False,
          normalize_advantages=True,
          value_head="scalar",
          kl_init_to_uniform=False,
          use_advantage_filter=False,
      ),
      ent_coef=schedules.Schedule("constant", 0.01, end=0.0, power=1.0),
      kl_coef=schedules.Schedule("constant", 0.1, end=0.0, power=0.0),
      reference_update_every=MAGNET_EVERY,
      value_scale=VALUE_SCALE,
  )


def evals(
    every: int = UNIFORM_EVERY, deals: int = UNIFORM_DEALS
) -> tuple[runner.InLoopEval, ...]:
  """The run's in-loop tracking evals: the learner side against uniform.

  Args:
    every: updates between evals.
    deals: deals per eval (each played from both sides).

  Returns:
    The evals.
  """
  return (
      runner.InLoopEval(
          UNIFORM_EVAL, every=every, deals=deals, seed=UNIFORM_SEED
      ),
  )


def check_standard_rules(cfg: runner.LeagueRunConfig) -> dict[str, str]:
  """Checks that cfg plays euchre at its default options, two partnerships.

  Args:
    cfg: the run config.

  Returns:
    The rules record {euchre_rules, rules_sha, env_spec}.

  Raises:
    AssertionError: the env plays other rules or other sides.
  """
  assert dict(RULES) == dict(euchre.OPTIONS), (RULES, euchre.OPTIONS)
  assert cfg.env == "euchre" and dict(cfg.env_kwargs) == RULES, (
      cfg.env,
      cfg.env_kwargs,
  )
  env = env_registry.make(cfg.env, **dict(cfg.env_kwargs))
  assert isinstance(env, euchre.EuchreEnv), type(env)
  assert sides_lib.sides_of(env) == sides_lib.Sides.partnerships(
      4, 2, constant_sum=True
  ), sides_lib.sides_of(env)
  spec = env_registry.spec_of(env)
  assert spec.to_json() == {
      "name": "euchre",
      "kwargs": dict(sorted(RULES.items())),
  }, spec.to_json()
  sha = env_registry.rules_sha(env)
  assert isinstance(sha, str) and len(sha) == 64, sha
  assert sha == euchre.rules_sha(RULES), sha
  return {
      "euchre_rules": RULES_ROW,
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


def build(
    eval: object = None, num_updates: int = NUM_UPDATES  # pylint: disable=redefined-builtin
) -> runner.LeagueRunConfig:
  """The run config with in-run evals `eval` (module doc).

  Args:
    eval: the in-run evals; None is `evals()`.
    num_updates: the run length in updates.

  Returns:
    The runner.LeagueRunConfig, checked.
  """
  base = runner.snapshot_pool(
      NAME,
      euchre.EuchreEnv.name,
      kinds.PG(pg_config(num_updates)),
      policy_kwargs(),
      num_envs=NUM_ENVS,
      num_steps=NUM_STEPS,
      num_updates=int(num_updates),
      snapshot_every=SNAPSHOT_EVERY,
      harvest_every=HARVEST_EVERY,
      max_pool=MAX_POOL,
      self_weight=SELF_WEIGHT,
      pool_slots=POOL_SLOTS,
      opponent_rule=types.ARGMAX,
      learner_rule=types.SAMPLE,
      keep_every=KEEP_EVERY,
      seed=0,
      env_kwargs=dict(RULES),
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
