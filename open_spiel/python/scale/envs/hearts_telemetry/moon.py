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


"""Moon-shot rates of hearts rollouts, logged every update of a training run.

TRAINING ONLY. open_spiel.python.scale.train imports this module by name for a
hearts run (train.TRAINING_TELEMETRY) and wraps the run in observing(cfg);
nothing the framework serves, plays or evaluates with imports it, and no
framework module changes for it. Euchre, gin and every other game have none
of these keys.

observing(cfg) observes, never changes, the run: for a PG run it installs an
ArenaSource.next override (core.arena.override_source_next) that lets the
learner's rollout run as it always does and returns that very batch, after
dispatching batch_metrics on it in a jitted program of its own; a run hook
(league.runner.register_run_hook) adds the values to the update's train row,
which metrics.jsonl writes as train/<learner>/rollout/... The learner's update
and the rollout compile and run exactly as without it, so training is bit for
bit unchanged.

batch_metrics(TrajectoryBatch [B, T]) counts the hands that end in the batch:
a hand's terminal ply (to_move -1) is in exactly one segment of its row, so
each hand of a row counts once. Its InfoState holds every seat's final points
(public), and moon_outcome reads who shot the moon from them.
Each row is one perspective seat's view (in a league, the learner's seat);
the other three seats of the row are its opponents (league draws, or the
lagged learner). A row of a self-play batch that emits every seat of a game
counts that game's hand once per row, so its learner and opponent rates are
equal. Keys (every rate is 0 with no hand):

  rollout/hands                    hands ended in the batch (int64)
  rollout/moons/learner            hands the perspective seat shot the moon
  rollout/moons/opponents          hands another seat of the row shot it
  rollout/moon_rate/learner        moons/learner over hands (per seat-hand)
  rollout/moon_rate/opponents      moons/opponents over 3 x hands
  rollout/near_moons/learner       hands the perspective seat ended with
                                   NEAR_MOON_POINTS (20) or more and no moon
  rollout/near_moons/opponents     the same over the other seats
  rollout/near_moon_rate/learner   near_moons/learner over hands
  rollout/near_moon_rate/opponents near_moons/opponents over 3 x hands

A run that is not PG (BC reads no rollout) logs none, and says so.
"""

import contextlib
from typing import Any, Iterator, Mapping, Optional

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import hearts as hearts_game
from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import hearts
from open_spiel.python.scale.league import kinds
from open_spiel.python.scale.league import runner

PREFIX = "rollout/"
_F32 = jnp.float32
_I64 = jnp.int64
#: A seat that ends a hand with at least this many points, with no moon shot,
#: came close to shooting it (moon_outcome): the queen of spades and seven or
#: more hearts, a failed attempt or a hand that went badly wrong.
NEAR_MOON_POINTS = 20


def moon_outcome(points: Any) -> tuple[jax.Array, jax.Array]:
  """Who shot the moon, and who came close, from final points.

  A moon moves 26 points onto each other seat, so the hand's total exceeds
  26 exactly when the moon was shot (at any options: without a moon the
  total is at most 26; with one it is at least 68), and the shooter is the
  seat with the fewest points (at most 0, every other seat at least 16).

  Args:
    points: int[..., 4], each seat's final points of a hand (the terminal
      InfoState's `points`, or 26 - returns).

  Returns:
    (shooter i32[...]: the seat that shot the moon, or -1;
     near bool[..., 4]: no moon and the seat has NEAR_MOON_POINTS or more).
  """
  points = jnp.asarray(points).astype(jnp.int32)
  assert points.shape[-1:] == (hearts.NUM_PLAYERS,), points.shape
  moon = jnp.sum(points, axis=-1) > hearts_game.TOTAL_POSITIVE_POINTS
  shooter = jnp.where(
      moon, jnp.argmin(points, axis=-1).astype(jnp.int32), jnp.int32(-1)
  )
  near = ~moon[..., None] & (points >= NEAR_MOON_POINTS)
  return shooter, near


def batch_metrics(batch: types.TrajectoryBatch) -> Mapping[str, jax.Array]:
  """The module doc's moon metrics of a [B, T] hearts batch (jittable).

  Args:
    batch: The training batch; steps.info is the hearts InfoState.

  Returns:
    {name: int64 count or f32 rate}.
  """
  steps = batch.steps
  term = jnp.asarray(steps.to_move, jnp.int32) == -1  # [B, T]
  seat = jnp.asarray(steps.seat, jnp.int32)  # [B, T]
  shooter, near = moon_outcome(steps.info["points"])  # [B, T], [., 4]
  mine = (
      jnp.arange(hearts.NUM_PLAYERS, dtype=jnp.int32) == seat[..., None]
  )  # [B, T, 4]
  hands = jnp.sum(term, dtype=_I64)
  moons_mine = jnp.sum(term & (shooter == seat), dtype=_I64)
  moons_other = jnp.sum(term & (shooter >= 0) & (shooter != seat), dtype=_I64)
  near_mine = jnp.sum(term[..., None] & near & mine, dtype=_I64)
  near_other = jnp.sum(term[..., None] & near & ~mine, dtype=_I64)
  per_mine = jnp.maximum(hands, 1).astype(_F32)
  per_other = jnp.maximum((hearts.NUM_PLAYERS - 1) * hands, 1).astype(_F32)
  return {
      f"{PREFIX}hands": hands,
      f"{PREFIX}moons/learner": moons_mine,
      f"{PREFIX}moons/opponents": moons_other,
      f"{PREFIX}moon_rate/learner": moons_mine.astype(_F32) / per_mine,
      f"{PREFIX}moon_rate/opponents": moons_other.astype(_F32) / per_other,
      f"{PREFIX}near_moons/learner": near_mine,
      f"{PREFIX}near_moons/opponents": near_other,
      f"{PREFIX}near_moon_rate/learner": near_mine.astype(_F32) / per_mine,
      f"{PREFIX}near_moon_rate/opponents": near_other.astype(_F32) / per_other,
  }


_METRICS_JIT = jax.jit(batch_metrics)


class _Observer:
  """One observed run: the override that sees each batch, the hook that logs.

  The ArenaSource.next override is re-entered by the rollout it starts (it
  calls source.next itself); `inside` lets that inner call run the rollout.
  Each update's train step calls its source's next once, and the runner calls
  after_update right after that update (before any other source's next), so
  the newest batch is the update's own.

  Attributes:
    inside: whether the override's own rollout is running.
    pending: the newest batch's metrics, until after_update logs them.
  """

  def __init__(self):
    self.inside = False
    self.pending: Optional[Mapping[str, jax.Array]] = None

  def source_next(self, source: Any, key: Any):
    """The override: the source's own rollout, observed.

    Args:
      source: the ArenaSource.
      key: the batch's key.

    Returns:
      None inside the observed rollout (it runs as usual), else its batch.
    """
    if self.inside:
      return None
    self.inside = True
    try:
      batch = source.next(key)
    finally:
      self.inside = False
    self.pending = _METRICS_JIT(batch)
    return batch

  def after_update(self, lg, ls, u, metrics, row):
    """Run hook: adds the update's moon metrics to its train row.

    Args:
      lg: the League (unused).
      ls: the learner slot (unused).
      u: the update number.
      metrics: the update's metrics (a name here may not repeat one).
      row: the train row, written after the hooks.

    Raises:
      RuntimeError: no batch was observed since the last update.
      ValueError: a name clashes with the update's own.
    """
    del lg, ls
    if self.pending is None:
      raise RuntimeError(
          f"hearts moon telemetry: update {u} had no observed rollout batch"
      )
    out = {k: float(np.asarray(v)) for k, v in self.pending.items()}
    self.pending = None
    clash = sorted(set(out) & (set(metrics) | set(row)))
    if clash:
      raise ValueError(f"hearts moon telemetry: {clash} clash with the row")
    row.update(out)


#: The observer of the run in progress (observing), or none.
_ACTIVE: list[_Observer] = []


def _hook_factory(cfg: Any) -> Optional[_Observer]:
  """league.runner run hook factory: the active observer, if any.

  Args:
    cfg: the run config (unused).

  Returns:
    The observer of the run observing(cfg) wraps, else None.
  """
  del cfg
  return _ACTIVE[-1] if _ACTIVE else None


runner.register_run_hook(_hook_factory)


@contextlib.contextmanager
def observing(cfg: Any) -> Iterator[None]:
  """Within the block, a PG run of cfg logs the moon metrics every update.

  Args:
    cfg: the run config (league.runner.LeagueRunConfig) of a hearts run.

  Yields:
    None.

  Raises:
    ValueError: cfg is not a hearts run.
    RuntimeError: another run is already observed.
  """
  if cfg.env != hearts.HeartsEnv.name:
    raise ValueError(f"hearts moon telemetry on a {cfg.env!r} run")
  if not isinstance(cfg.learner, kinds.PG):
    logging.info(
        "hearts moon telemetry: off (the %s learner reads no rollout)",
        type(cfg.learner).__name__,
    )
    yield
    return
  if _ACTIVE:
    raise RuntimeError("hearts moon telemetry: a run is already observed")
  obs = _Observer()
  with arena_lib.override_source_next(obs.source_next):
    _ACTIVE.append(obs)
    try:
      yield
    finally:
      _ACTIVE.remove(obs)
