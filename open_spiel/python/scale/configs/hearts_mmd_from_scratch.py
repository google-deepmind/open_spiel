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

r"""Hearts MMD from scratch: one learner against a pool of its own snapshots.

OpenSpiel's hearts at pyspiel's default rule options (envs.hearts OPTIONS:
pass three cards, no points on the first trick, hearts must be broken, the
queen of spades breaks hearts), four seats, each for itself. The learner
holds one seat per row (row r: seat r % 4); the other three seats each draw
their own opponent from the round's mixture (league.sampler.draw_table): the
lagged learner snapshot on SELF_WEIGHT of the draws, else one of the
POOL_SLOTS seated pool members. Returns are 26 - points (envs.hearts).

  python -m open_spiel.python.scale.train --config=hearts_mmd_from_scratch \
      --checkpoint_dir=/tmp/hearts_from_scratch

A smoke run on a CPU:

  python -m open_spiel.python.scale.train --config=hearts_mmd_from_scratch \
      --checkpoint_dir=/tmp/hearts_smoke --num_updates=2 --num_envs=8 \
      --num_steps=16 --set=eval.0.deals=4

The run length is sized to the throughput measured on one H100: about
0.28 s per update, so 150,000 updates is about 12 h (a tracking read, not a
benchmark).

The recipe (league.runner.snapshot_pool):
  * learner: MMD policy gradient (pg_config): KL 0.1 to a magnet re-taken
    from the learner's own params every 1,000 updates (the first magnet is
    the init), entropy 0.01, clip 0.2, LR 3e-4 annealed linearly to 3e-5 over
    NUM_UPDATES (150,000), 4 epochs x 4 minibatches, gamma 1 (one hand is one
    episode), returns scaled by 1/26 for the value targets.
  * network: a separate-torso MLP (1024, 512, 256), relu, on pyspiel's
    hearts information-state tensor (encoders.hearts, f32[5088]); random
    init.
  * opponents: controller scheduled_promotion, all argmax: SELF_WEIGHT of
    the seat draws on the lagged learner snapshot (every 1,000 updates),
    the rest on a pool of snapshots harvested every 5,000 updates (newest
    20), two seated per round.
  * in-run tracking eval: vs_uniform every 1,000 updates, the learner on one
    seat against three uniform-random seats on 512 deals under the cyclic
    duplicate design (eval/vs_uniform/mean is the learner's 26 - points per
    hand; a uniform seat averages 19.5).

get_config checks the rules (check_standard_rules: the env plays pyspiel's
default hearts options, recorded with the env's rules_sha in the metadata)
and that the run is from scratch (check_from_scratch: no warm start, no fixed
opponents, a pool of the learner's own snapshots, the magnet re-taken from
the learner's own params). Every in-run number is TRACKING tier.
"""

import dataclasses
import json

from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.core import types
from open_spiel.python.scale.encoders import hearts as hearts_enc
from open_spiel.python.scale.envs import hearts
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import kinds
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.learners import losses
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.networks import catalog

NAME = "hearts_mmd_from_scratch"

#: pyspiel hearts' eight game parameters at their defaults, written out.
RULES = {
    "pass_cards": True,
    "no_pts_on_first_trick": True,
    "can_lead_any_club": False,
    "jd_bonus": False,
    "avoid_all_tricks_bonus": False,
    "must_break_hearts": True,
    "qs_breaks_hearts": True,
    "can_lead_hearts_instead_of_qs": False,
}
RULES_ROW = "standard"

NUM_UPDATES = 150_000
NUM_ENVS = 1024
#: Plies per update: one hand is at most 64 player decisions (envs.hearts).
NUM_STEPS = 64
#: PGConfig.max_decision_share: the learner holds one seat of four, so it
#: makes at most one play per trick and three passes per hand (a seat's three
#: passes are consecutive, seats in order). A 64-ply window that straddles
#: hands holds at most 19 of its decisions (0.297): the last 3 plies of a
#: pass phase (the learner's passes, as seat 3), a full 52-ply play phase (13)
#: and the first 9 plies of a no-pass hand's play phase (3 tricks touched).
#: PGLearner.train_rows raises TrainRowsOverflowError if a batch's decisions
#: ever exceed ceil(0.4 x B x T); nothing is dropped silently.
MAX_DECISION_SHARE = 0.4
WIDTHS = (1024, 512, 256)
VALUE_SCALE = 26.0

SNAPSHOT_EVERY = 1000
HARVEST_EVERY = 5000
MAX_POOL = 20
POOL_SLOTS = 2
SELF_WEIGHT = 0.75
KEEP_EVERY = 2500
MAGNET_EVERY = 1000

UNIFORM_EVAL = "vs_uniform"
UNIFORM_EVERY, UNIFORM_DEALS, UNIFORM_SEED = 1000, 512, 65535
#: The in-loop read cadence a launcher declares (eval_every, the scale of an
#: alert on missing reads): the vs_uniform reads every
#: UNIFORM_EVERY updates were 2.8-2.9 min apart on one H100 (metrics.jsonl
#: row times); the module doc's 0.28 s per update gives 4.7 min.
LAUNCH_EVAL_EVERY = "15m"

NOTES = (
    "hearts MMD from scratch on OpenSpiel's hearts at pyspiel's default "
    "options; in-run vs_uniform eval is TRACKING tier"
)
FROM_SCRATCH = "random init; pool = own snapshots; magnet = own params"


def policy_kwargs() -> dict[str, object]:
  """The network's arch_kwargs (module doc).

  Returns:
    The arch_kwargs.
  """
  return common.network_policy_kwargs(
      hearts.HeartsEnv.name,
      hearts_enc.INFO_STATE.name,
      hearts_enc.INFO_STATE.shape,
      hearts.NUM_CARDS,
      hearts.NUM_PLAYERS,
      catalog.mlp(list(WIDTHS), "relu", "separate"),
  )


def pg_config(num_updates: int = NUM_UPDATES) -> pg.PGConfig:
  """The learner's PGConfig (MMD, module doc).

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
      max_decision_share=MAX_DECISION_SHARE,
      value_scale=VALUE_SCALE,
  )


def evals(
    every: int = UNIFORM_EVERY, deals: int = UNIFORM_DEALS
) -> tuple[runner.InLoopEval, ...]:
  """The run's in-loop tracking evals: the learner against uniform seats.

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


def check_standard_rules(cfg: runner.LeagueRunConfig) -> dict[str, str]:
  """Checks that cfg plays pyspiel hearts at its default options.

  Args:
    cfg: the run config.

  Returns:
    The rules record {hearts_rules, rules_sha, env_spec}.

  Raises:
    AssertionError: the env plays other rules.
  """
  assert dict(RULES) == dict(hearts.OPTIONS), (RULES, hearts.OPTIONS)
  assert cfg.env == "hearts" and dict(cfg.env_kwargs) == RULES, (
      cfg.env,
      cfg.env_kwargs,
  )
  env = env_registry.make(cfg.env, **dict(cfg.env_kwargs))
  assert isinstance(env, hearts.HeartsEnv), type(env)
  spec = env_registry.spec_of(env)
  assert spec.to_json() == {
      "name": "hearts",
      "kwargs": dict(sorted(RULES.items())),
  }, spec.to_json()
  sha = env_registry.rules_sha(env)
  assert isinstance(sha, str) and len(sha) == 64, sha
  assert sha == hearts.rules_sha(RULES), sha
  return {
      "hearts_rules": RULES_ROW,
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
      hearts.HeartsEnv.name,
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
