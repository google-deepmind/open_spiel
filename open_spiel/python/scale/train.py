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

r"""The training entry point (the trainer).

    python -m open_spiel.python.scale.train --config=mmd_kuhn_poker_jax \
        --checkpoint_dir=/tmp/kuhn

trains the config's run to its length, checkpointing as it goes; started
again on the same --checkpoint_dir it resumes where the last process stopped.
Every run is a league run (`league.runner.LeagueRunConfig`): a plain PG, BC or
AlphaZero run is a league with one learner slot under a fixed sampler rule
(`configs.recipes`), and population methods such as PSRO are controllers
over the same loop.

Flags (default in brackets):

  --config=NAME                 config name or path to a .py config file
                                (required)
  --checkpoint_dir=DIR          checkpoint directory, local or gs:// [none:
                                no checkpoints]
  --set=dotted.path=value       config overrides, repeatable
                                (league.config_common.apply_overrides) [none]
  --num_updates, --num_envs, --num_steps, --seed
                                run length, batch geometry and seed, applied
                                everywhere the run holds them
                                (league.config_common.with_overrides)
                                [the config's]
  --stop_after=N                end this process after iteration N with the
                                run length unchanged; a later process resumes
                                [none]
  --run_id=ID                   the run's identity, recorded in artifacts; a
                                relaunch of a preempted run passes the
                                original run's id [the checkpoint directory's
                                run on a resume, else this attempt's id]
  --preemption_sources=a,b      the notice sources that end the run cleanly
                                after the iteration in flight
                                (launch.preemption) [the registered default
                                sources]
  --resume_every_min=M          minutes between resume checkpoints, each
                                overwriting the last; 0 turns them off [3]
  --wall_budget=<N>s|m|h        end the run at the first iteration boundary
                                past this wall time (below) [none]
  --eval_grace=<N>s|m|h         with --wall_budget, how long an eval may run
                                past the budget [30m]
  --collectives_preflight       check every device can run a collective
                                before training (parallel.collectives_check)
                                [off]
  --num_devices=N               shard over the first N devices [all]
  --metric_sinks=a,b            extra metric sinks (metrics) [none]
  --tensorboard                 also write TensorBoard scalars [off]
  --compilation_cache_dir=DIR   JAX's persistent compilation cache: "auto"
                                (a registered shared-dir rule, else
                                <checkpoint_dir>/xla_cache), a directory, or
                                "" for off [auto]
  --compilation_cache_min_compile_secs=S
                                cache programs that took at least S seconds
                                to compile [JAX's default]
  --plugins=pkg.mod,...         plugin modules to import (plugins) [none]

A learner kind adds its own flags (`kinds.base.KindCLI.add_flags`): BC adds
--data_dir to train from recorded shards and --record_shards /
--record_batches to record a teacher's games instead of training.

Plugins named in $OPEN_SPIEL_SCALE_PLUGINS (plugins.ENV) are imported when this
module is imported, before flags are parsed, so they may add flags
(`plugins.register_flags`); --plugins are imported after parsing.

Preemption: when a notice source fires, the run finishes the iteration in
flight, writes a resume checkpoint and exits with `PREEMPTED_EXIT` (143,
never 0), so a supervisor relaunches it.

num_envs is the global row count and must divide by the device count. A
checkpoint written on any device count resumes on any other.

Wall budget: --wall_budget=<N>s|m|h ends the run at the first iteration
boundary past that much wall time from this process's start. The last
update gets every in-loop eval, the iteration is committed with
train_state "wall_budget", and the process exits 0 with the run incomplete
(league.runner.run). An eval still running --eval_grace after the budget
raises core.deadline.EvalOverBudgetError and the process fails.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from absl import app
from absl import flags
from absl import logging

from open_spiel.python.scale import metrics as metrics_lib
from open_spiel.python.scale import plugins
from open_spiel.python.scale.ckpt import provenance
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.launch import compile_cache
from open_spiel.python.scale.launch import hardware
from open_spiel.python.scale.launch import preemption
from open_spiel.python.scale.league import config_common
from open_spiel.python.scale.league import kinds
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.parallel import collectives_check
from open_spiel.python.scale.parallel import data_parallel

RunResult = runner.LeagueRunResult
FLAGS = flags.FLAGS

_CONFIG = flags.DEFINE_string(
    "config", None, "Config name or path to a .py config file."
)
_CHECKPOINT_DIR = flags.DEFINE_string(
    "checkpoint_dir",
    None,
    "Checkpoint directory (local or gs://); none: no checkpoints.",
)
_NUM_UPDATES = flags.DEFINE_integer(
    "num_updates",
    None,
    "Run length in league iterations "
    "(one update each for a one-slot run); moves the "
    "learner's schedule horizon with it.",
)
_NUM_ENVS = flags.DEFINE_integer(
    "num_envs", None, "Arena rows per update (global)."
)
_NUM_STEPS = flags.DEFINE_integer("num_steps", None, "Plies per update.")
_SEED = flags.DEFINE_integer("seed", None, "The run's seed.")
_SET = flags.DEFINE_multi_string(
    "set", [], "Config override dotted.path=value (repeatable)."
)
_TENSORBOARD = flags.DEFINE_bool(
    "tensorboard", False, "Also write TensorBoard scalars."
)
_METRIC_SINKS = flags.DEFINE_list(
    "metric_sinks", [], "Extra metric sinks by registered name (metrics)."
)
_PLUGINS = flags.DEFINE_list(
    "plugins",
    [],
    f"Plugin modules to import after flag parsing (besides ${plugins.ENV}).",
)
_RUN_ID = flags.DEFINE_string(
    "run_id",
    None,
    "The run's id, recorded in artifacts; a relaunch "
    "of a preempted run passes the original run's. Default: the "
    "checkpoint directory's run on a resume, else this attempt's id.",
)
_STOP_AFTER = flags.DEFINE_integer(
    "stop_after",
    None,
    "End this process after iteration N "
    "without changing the run length; a later run resumes it.",
)
_NUM_DEVICES = flags.DEFINE_integer(
    "num_devices", None, "Devices to shard over (default all)."
)
_RESUME_EVERY_MIN = flags.DEFINE_float(
    "resume_every_min",
    3.0,
    "Minutes between resume checkpoints, each overwriting the last (0: none).",
)
_PREEMPTION_SOURCES = flags.DEFINE_list(
    "preemption_sources",
    None,
    "Preemption notice sources "
    "(launch.preemption); default: the registered "
    "default sources.",
)
_COLLECTIVES_PREFLIGHT = flags.DEFINE_bool(
    "collectives_preflight",
    False,
    "Check that every device can run a collective before training.",
)
_WALL_BUDGET = flags.DEFINE_string(
    "wall_budget",
    None,
    "End the run at the first iteration "
    "past this wall time (<N>s|m|h), its evals run, committed.",
)
_EVAL_GRACE = flags.DEFINE_string(
    "eval_grace",
    "30m",
    "With --wall_budget: an eval still "
    "running this long past the budget raises EvalOverBudgetError.",
)
_COMPILATION_CACHE_DIR = flags.DEFINE_string(
    "compilation_cache_dir",
    compile_cache.AUTO,
    "Persistent XLA compilation cache: 'auto' "
    "(a registered shared-dir rule, else <checkpoint_dir>/xla_cache), a dir, "
    "or '' (off).",
)
_COMPILATION_CACHE_MIN_SECS = flags.DEFINE_float(
    "compilation_cache_min_compile_secs",
    None,
    "Cache programs that took at least this long to compile (default: JAX's).",
)


def run(
    cfg: Any,
    checkpoint_dir: Optional[str] = None,
    *,
    run_id: Optional[str | int] = None,
    attempt_id: Optional[str | int] = None,
    **runner_kwargs: Any,
) -> RunResult:
  """Trains run config cfg to its length, resuming from checkpoint_dir.

  Args:
    cfg: a league.runner.LeagueRunConfig.
    checkpoint_dir: where checkpoints go (local or gs://); a directory that
      holds a committed run state continues that run if the config matches.
      None: no checkpoints.
    run_id: the run's id: a string, or an int when the deployment's id
      parser returns one (ckpt.provenance.resolve_run_ids). The run's records
      keep the value as given; an artifact's manifest stores the run id as its
      string.
    attempt_id: this process's id, a string or an int as run_id.
    **runner_kwargs: passed to league.runner.run (mesh, stop_after,
      git_commit, tensorboard, resume_every_s, preempted, sinks, heldout and
      a kind's extras such as BC's data_dir).

  Raises:
    TypeError: cfg is not a run config.

  Returns:
    the RunResult of league.runner.run.
  """
  x64.enable_x64()
  if not isinstance(cfg, runner.LeagueRunConfig):
    raise TypeError(
        "the trainer runs league.runner.LeagueRunConfig, got "
        f"{type(cfg).__name__}"
    )
  return runner.run(
      cfg, checkpoint_dir, run_id=run_id, attempt_id=attempt_id, **runner_kwargs
  )


def duration_s(s: str) -> float:
  """Parses a duration: '150m' / '2h' / '90s' -> seconds (> 0).

  Args:
    s: the duration.

  Returns:
    the seconds.

  Raises:
    app.UsageError: s is not a positive <N>s|m|h.
  """
  m = re.fullmatch(r"(\d+(?:\.\d+)?)([smh])", str(s).strip())
  if not m or float(m.group(1)) <= 0:
    raise app.UsageError(f"duration {s!r} is not a positive <N>s|m|h")
  return float(m.group(1)) * {"s": 1.0, "m": 60.0, "h": 3600.0}[m.group(2)]


def config_from_flags() -> Any:
  """The run config --config names, with --set and the size flags applied.

  Returns:
    the run config.

  Raises:
    app.UsageError: --config is not set.
  """
  if not _CONFIG.value:
    raise app.UsageError("--config is required")
  cfg = config_common.load_config(_CONFIG.value)
  cfg = config_common.apply_overrides(cfg, _SET.value)
  return config_common.with_overrides(
      cfg,
      num_updates=_NUM_UPDATES.value,
      num_envs=_NUM_ENVS.value,
      num_steps=_NUM_STEPS.value,
      seed=_SEED.value,
  )


def main(argv):
  """The trainer's entry point: trains the config the flags name.

  Args:
    argv: the command line left after flag parsing.

  Raises:
    app.UsageError: positional arguments were given.
  """
  if len(argv) > 1:
    raise app.UsageError(f"unexpected arguments {argv[1:]}")
  x64.enable_x64()
  plugins.load_env()
  plugins.load(_PLUGINS.value)
  cfg = config_from_flags()
  ckpt = _CHECKPOINT_DIR.value
  cache_dir = compile_cache.cache_dir_for(_COMPILATION_CACHE_DIR.value, ckpt)
  # Enabled before the first compile; turned off again when main returns.
  cache_off = (
      compile_cache.enable(cache_dir, _COMPILATION_CACHE_MIN_SECS.value)
      if cache_dir
      else None
  )
  try:
    _main(cfg, ckpt, cache_dir)
  finally:
    if cache_off is not None:
      cache_off()


def _check_kind_flags(active: str) -> None:
  """Raises UsageError if a flag of another learner kind is set."""
  for name, owned in _KIND_FLAGS.items():
    if name == active:
      continue
    for f in owned:
      if FLAGS[f].present:
        raise app.UsageError(
            f"--{f} is a flag of the {name!r} learner kind, not {active!r}"
        )


def _main(cfg: Any, ckpt: Optional[str], cache_dir: Optional[str]) -> None:
  """Runs training of cfg from the flags.

  Args:
    cfg: the run config.
    ckpt: the checkpoint directory, or None.
    cache_dir: the compile cache directory, or None.

  Raises:
    app.UsageError: a bad flag or config.
    SystemExit: a preemption notice arrived (preemption.PREEMPTED_EXIT).
  """
  run_id, attempt_id = provenance.resolve_run_ids(_RUN_ID.value)
  kind_name = config_common.learner_kind(cfg)
  logging.info(
      "train: %s (%s learner, controller %s), checkpoint_dir=%s, run id=%s, "
      "attempt id=%s",
      cfg.name,
      kind_name,
      cfg.controller,
      ckpt,
      run_id,
      attempt_id,
  )
  _check_kind_flags(kind_name)
  try:
    prepare = getattr(
        kinds.get(kind_name), "prepare", None
    )  # the optional KindCLI hook
    prepared = (cfg, {}) if prepare is None else prepare(cfg, FLAGS)
  except ValueError as e:
    raise app.UsageError(str(e)) from e
  if prepared is None:
    return
  cfg, kind_kwargs = prepared
  mesh = (
      data_parallel.batch_mesh(_NUM_DEVICES.value)
      if _NUM_DEVICES.value is not None
      else None
  )
  num_devices = int(mesh.devices.size) if mesh is not None else None
  hw = hardware.record(ckpt, num_devices=num_devices, emit=False)
  # What the process got is a setting of this attempt, not a series.
  metrics_lib.register_params({
      hardware.DEVICES_PARAM: hw["num_devices"],
      "hw/accelerator": hw["accelerator"] or "unknown",
      "hw/platform": hw["platform"],
      "hw/device_kinds": ",".join(hw["device_kinds"]),
  })
  hardware.check_platform(hw)
  data_parallel.check_divides(
      cfg.league.num_envs, hw["num_devices"], f"{cfg.name} num_envs"
  )
  collectives_check.run_preflights(
      list(mesh.devices.flat) if mesh is not None else None,
      collectives=_COLLECTIVES_PREFLIGHT.value,
  )
  if _RESUME_EVERY_MIN.value < 0:
    raise app.UsageError(
        f"--resume_every_min must be >= 0, got {_RESUME_EVERY_MIN.value}"
    )
  every_s = (
      _RESUME_EVERY_MIN.value * 60.0 if _RESUME_EVERY_MIN.value > 0 else None
  )
  wall_s = duration_s(_WALL_BUDGET.value) if _WALL_BUDGET.value else None
  grace_s = duration_s(_EVAL_GRACE.value) if wall_s is not None else 0.0
  notice = preemption.Notice()
  sources = (
      _PREEMPTION_SOURCES.value
      if _PREEMPTION_SOURCES.value is not None
      else preemption.default_sources()
  )
  stop_sources = preemption.start_notice_sources(notice, sources)
  try:
    res = run(
        cfg,
        ckpt,
        run_id=run_id,
        attempt_id=attempt_id,
        tensorboard=_TENSORBOARD.value,
        git_commit=provenance.git_commit() if ckpt else None,
        mesh=mesh,
        stop_after=_STOP_AFTER.value,
        resume_every_s=every_s,
        preempted=notice,
        sinks=tuple(_METRIC_SINKS.value),
        wall_budget_s=wall_s,
        eval_grace_s=grace_s,
        **kind_kwargs,
    )
  finally:
    stop_sources()
  if cache_dir and ckpt:
    compile_cache.record(ckpt, cache_dir)
  if res.preempted:
    logging.warning(
        "train: preemption notice (%s): checkpointed at iteration %d; exit %d",
        notice.reason,
        res.league.iteration,
        preemption.PREEMPTED_EXIT,
    )
    raise SystemExit(preemption.PREEMPTED_EXIT)
  if res.wall_budget is not None:
    w = res.wall_budget
    logging.warning(
        "train: wall budget reached at %d of %d (%.0f s): committed; exit 0 "
        "(the "
        "run is INCOMPLETE)",
        w["reached_at"],
        w["of"],
        w["budget_s"],
    )


def _define_extension_flags() -> dict[str, tuple[str, ...]]:
  """Defines each learner kind's flags and each plugin's.

  Returns:
    kind -> its flag names.
  """
  plugins.load_env()
  owned = {}
  for name in kinds.names():
    before = set(FLAGS)
    add_flags = getattr(
        kinds.get(name), "add_flags", None
    )  # the optional KindCLI hook
    if add_flags is not None:
      add_flags(FLAGS)
    owned[name] = tuple(sorted(set(FLAGS) - before))
  for fn in plugins.flag_extensions():
    fn(FLAGS)
  return owned


_KIND_FLAGS = _define_extension_flags()

if __name__ == "__main__":
  # The command line: x64 on before flag parsing, which may exit (--help)
  # before main() runs; main() and run() turn it on too, for a caller that
  # imports this module.
  x64.enable_x64()
  app.run(main)
