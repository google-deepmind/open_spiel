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

"""DAgger (Ross et al.

2011) as a trajectory source: a student and a teacher share the learner slot, a
beta schedule decides who acts, the teacher labels every decision.

    src = dagger.DaggerSource(env, student, teacher, student_params=sp,
                              teacher_params=tp, table=table, num_steps=64,
                              beta=dagger.beta_schedule(
                                  1.0, 0.2, decay_iters=2000))
    batch = src.next(key)          # targets: the teacher's distribution and
                                   # Monte Carlo returns

Teacher self-play is beta 1 (behaviour cloning), student-driven play beta 0,
and an annealed beta interpolates (distillation).

  DaggerPolicy   one Policy over (student, teacher). Both observe every ply;
                 at each decision a per-row coin (uniform(key) < beta) picks
                 whose logits the slot's DecisionRule acts on, so under SAMPLE
                 the teacher's sampled action is played with probability beta.
                 aux[AUX_TARGET_POLICY] is the teacher's distribution (f64
                 probabilities at temperature 1, cast to f32),
                 aux[AUX_TARGET_VALUE] the teacher's value (with emit_value
                 only: the Arena then writes target_value and
                 labels[TARGET_VALUE_VALID]), aux[AUX_TEACHER_ACTED] the coin.
                 behavior_logits / logp are those of the branch that acted.
  DaggerSource   DaggerPolicy over core.arena.ArenaSource: each `next(key)`
                 writes beta(iteration) and the current student params into
                 the slot, rolls a segment, keeps the student's part of
                 init_carry, and writes Monte Carlo value targets
                 (`sources.targets`), by default with a one-segment lookahead.

The table puts every perspective seat in slot 0
(`league.sampler.entries_table`, `core.arena.self_play_table`). Counters
(iteration, frames, decisions) are np.int64; `decisions` counts the valid
rows after the masks below (what BC trains on).

Options (each off by default):
  opening_plies   the first N plies of every hand are uniform random legal
                  moves, labelled by the teacher (`policies.opening`).
  first_hand_only every `next` restarts every row from a fresh deal and keeps
                  only each row's first hand, truncated at the segment's
                  plies (an unfinished hand keeps its policy labels and has no
                  value target).
  endgame         EndgameRoute(policy, params, route): the rows the route
                  sends to `policy` are played by it and are not learner
                  decisions (valid cleared).
"""

from typing import Any, NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import decide as decide_lib
from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.policies import opening as opening_lib
from open_spiel.python.scale.policies import routed
from open_spiel.python.scale.sources import targets

# PolicyOutput.aux / TrajectoryBatch.aux key: bool, the teacher's logits acted.
AUX_TEACHER_ACTED = "dagger/teacher_acted"
# a training-time composite, never published (the student is)
ARCH_NAME = "dagger_mix"


def beta_schedule(
    start: float, end: float, decay_iters: int = 0
) -> schedules.Schedule:
  """The DAgger mixing probability beta(it), linear then flat.

  beta(it) = start + clip(it / max(decay_iters, 1), 0, 1) * (end - start):
  linear from start to end over decay_iters segments, then flat.

  Args:
    start: beta at iteration 0.
    end: beta after decay_iters.
    decay_iters: segments of the linear decay (0: constant start).

  Returns:
    The schedules.Schedule.
  """
  assert 0.0 <= start <= 1.0 and 0.0 <= end <= 1.0, (start, end)
  assert int(decay_iters) >= 0, decay_iters
  if start == end:
    return schedules.Schedule("constant", float(start))
  return schedules.Schedule(
      "linear",
      float(start),
      float(end),
      decay_steps=float(max(int(decay_iters), 1)),
  )


class DaggerPolicy(types.Policy):
  """The student or the teacher acts per row; the teacher labels (module doc).

  params {"student", "teacher", "beta": f32[]}; carry {"student", "teacher"}.

  Attributes:
    student: the student's in-graph policy.
    teacher: the teacher's in-graph policy.
    emit_value: whether the teacher's value is emitted as
      aux[AUX_TARGET_VALUE].
    spec: the mix's PolicySpec.
  """

  # its targets are checked for information-set invariance
  emits_target_policy = True

  def __init__(
      self,
      student: types.Policy,
      teacher: types.Policy,
      emit_value: bool = True,
  ):
    """Builds the mix of a student and a teacher.

    Args:
      student: an in-graph policy.
      teacher: an in-graph policy of the same game.
      emit_value: emit the teacher's value as aux[AUX_TARGET_VALUE].

    Raises:
      ValueError: the parts play different games, or a part is HOST_DRIVEN.
    """
    ss, ts = student.spec, teacher.spec
    for f in ("game", "num_actions", "num_players"):
      if getattr(ss, f) != getattr(ts, f):
        raise ValueError(
            f"DaggerPolicy: student {f} {getattr(ss, f)!r} != teacher's "
            f"{getattr(ts, f)!r}"
        )
    caps = {ss.capability, ts.capability}
    if types.Capability.HOST_DRIVEN in caps:
      raise ValueError(
          "DaggerPolicy runs in-graph; a HOST_DRIVEN part is not supported"
      )
    cap = (
        types.Capability.CALLBACK_BATCHED
        if types.Capability.CALLBACK_BATCHED in caps
        else types.Capability.NATIVE_JAX
    )
    self.student, self.teacher = student, teacher
    self.emit_value = bool(emit_value)
    self.spec = types.PolicySpec(
        game=ss.game,
        obs_format=ss.obs_format,
        num_actions=ss.num_actions,
        num_players=ss.num_players,
        stateful=ss.stateful or ts.stateful,
        capability=cap,
        framework="jax",
        arch_name=ARCH_NAME,
        arch_kwargs=dict(student=ss.arch_name, teacher=ts.arch_name),
    )

  @staticmethod
  def make_params(student_params, teacher_params, beta) -> dict[str, Any]:
    """The mix's params around the student's and the teacher's.

    Args:
      student_params: the student's params.
      teacher_params: the teacher's params.
      beta: the scalar probability the teacher acts.

    Returns:
      {student, teacher, beta}.

    Raises:
      ValueError: beta is not a scalar.
    """
    beta = jnp.asarray(beta, jnp.float32)
    if beta.shape:
      raise ValueError(f"beta is a scalar, got shape {beta.shape}")
    return {"student": student_params, "teacher": teacher_params, "beta": beta}

  def init_params(self, key):
    """Fresh student and teacher params, beta 1.

    Args:
      key: the PRNG key.

    Returns:
      The mix's params.
    """
    ks, kt = jax.random.split(key)
    return self.make_params(
        self.student.init_params(ks), self.teacher.init_params(kt), 1.0
    )

  def encode(self, info):
    """The student's encoding of an InfoState.

    Args:
      info: the InfoState.

    Returns:
      The student's encoded observation.
    """
    return self.student.encode(info)

  def init_carry(self, params, batch, seat, key):
    """The student's and the teacher's initial carries.

    Args:
      params: the mix's params.
      batch: the batch size.
      seat: the seats.
      key: the PRNG key.

    Returns:
      {student, teacher} carries.
    """
    ks, kt = jax.random.split(key)
    return {
        "student": self.student.init_carry(params["student"], batch, seat, ks),
        "teacher": self.teacher.init_carry(params["teacher"], batch, seat, kt),
    }

  def observe(self, params, carry, step):
    """Both parts observe the step.

    Args:
      params: the mix's params.
      carry: {student, teacher} carries.
      step: the SeatStep.

    Returns:
      The new {student, teacher} carries.
    """
    return {
        "student": self.student.observe(
            params["student"], carry["student"], step
        ),
        "teacher": self.teacher.observe(
            params["teacher"], carry["teacher"], step
        ),
    }

  def evaluate(self, params, carry, step, key):
    """The teacher's output with probability beta, else the student's.

    Args:
      params: the mix's params.
      carry: {student, teacher} carries.
      step: the SeatStep.
      key: the PRNG key (the beta coin).

    Returns:
      The PolicyOutput.

    Raises:
      ValueError: key is None.
    """
    if key is None:
      raise ValueError("DaggerPolicy.evaluate needs a key (the beta coin)")
    kc, ks, kt = jax.random.split(key, 3)
    s_out = self.student.evaluate(params["student"], carry["student"], step, ks)
    t_out = self.teacher.evaluate(params["teacher"], carry["teacher"], step, kt)
    b = s_out.logits.shape[0]
    coin = jax.random.uniform(kc, (b,), jnp.float32) < params["beta"].astype(
        jnp.float32
    )
    logits = jnp.where(
        coin[:, None],
        t_out.logits.astype(jnp.float32),
        s_out.logits.astype(jnp.float32),
    )
    value = jnp.where(
        coin[:, None],
        t_out.value.astype(jnp.float32),
        s_out.value.astype(jnp.float32),
    )
    _, t_probs = decide_lib.decide(
        t_out.logits, step.legal_mask, types.ARGMAX, None
    )
    aux = {
        types.AUX_TARGET_POLICY: t_probs.astype(jnp.float32),
        AUX_TEACHER_ACTED: coin,
    }
    if self.emit_value:
      aux[types.AUX_TARGET_VALUE] = t_out.value.astype(jnp.float32)
    return types.PolicyOutput(logits=logits, value=value, aux=aux)


def _stack1(tree):
  """Adds a leading member axis of 1.

  Args:
    tree: A pytree.

  Returns:
    The stacked tree.
  """
  return jax.tree.map(lambda x: jnp.asarray(x)[None], tree)


class EndgameRoute(NamedTuple):
  """Rows a scripted policy plays instead of the learner.

  The rows where `route(info)` holds are played by `policy` (with
  `params`) and are not learner decisions.

  Attributes:
    policy: the scripted policy.
    params: its params.
    route: "module:attr" of InfoState[B] -> bool[B], as RoutedPolicy takes it.
  """

  policy: types.Policy
  params: Any
  route: str


def first_hand_mask(new_hand) -> jax.Array:
  """bool[B, T]: the ply belongs to the first hand of its row.

  Rows must start a hand at t = 0 (as after a reset).

  Args:
    new_hand: bool[B, T] hand starts.

  Returns:
    The mask.

  Raises:
    ValueError: a row does not start a hand at t = 0.
  """
  new_hand = jnp.asarray(new_hand, jnp.bool_)
  if not bool(jnp.all(new_hand[:, 0])):
    raise ValueError("first_hand_mask needs rows that start a hand at t = 0")
  return jnp.cumsum(new_hand.astype(jnp.int32), axis=1) <= 1


class DaggerSource:
  """A TrajectorySource whose learner slot is DaggerPolicy(student, teacher).

  Slot 0 is the learner (perspective) slot, acting with `rule`; `opponents`
  are further slots with their stacked params. `table` must put every
  perspective seat in slot 0 (league.sampler.entries_table,
  arena.self_play_table).

  Attributes:
    policy: the DaggerPolicy of the learner slot.
    opening_plies: uniform random opening plies per hand (module doc).
    first_hand_only: whether only each row's first hand is kept (module doc).
    endgame: the EndgameRoute, or None (module doc).
    slot_policy: the learner slot's policy (the DaggerPolicy, routed and
      opened as the options ask).
    arena: the Arena the segments are rolled in.
    table: the slot table.
    beta: the schedule of the probability the teacher acts.
    gamma: the discount of the Monte Carlo return.
    value_target: "return" or "teacher".
    return_lookahead: whether the return looks one segment ahead.
    student_params: the student's current params.
    teacher_params: the teacher's params.
    opponent_params: the opponent slots' stacked params.
    iteration: segments rolled (np.int64).
    decisions: valid rows emitted, after the masks (np.int64; settable for a
      resume).
  """

  def __init__(
      self,
      env: types.Env,
      student: types.Policy,
      teacher: types.Policy,
      *,
      student_params: Any,
      teacher_params: Any,
      table: arena_lib.SlotTable,
      num_steps: int,
      beta: schedules.Schedule,
      opponents: Sequence[arena_lib.Slot] = (),
      opponent_params: Sequence[Any] = (),
      rule: types.DecisionRule = types.SAMPLE,
      gamma: float = 1.0,
      value_target: str = "return",
      return_lookahead: bool = True,
      hands_per_match: int = 1,
      mesh: Optional[Any] = None,
      opening_plies: int = 0,
      first_hand_only: bool = False,
      endgame: Optional[EndgameRoute] = None,
  ):
    """Builds the source.

    Args:
      env: the env.
      student: the student policy.
      teacher: the teacher policy (same game).
      student_params: the student's initial params.
      teacher_params: the teacher's params.
      table: the slot table; it puts every perspective seat in slot 0.
      num_steps: the plies per segment.
      beta: the schedule of the probability the teacher acts, over
        iterations.
      opponents: further slots.
      opponent_params: one stacked params entry per opponent slot.
      rule: the learner slot's DecisionRule.
      gamma: the discount of the Monte Carlo return, in [0, 1].
      value_target: "return" (the Monte Carlo return with discount gamma, with
        a one-segment lookahead when return_lookahead) or "teacher" (the
        teacher's value head, written by the Arena with
        labels[TARGET_VALUE_VALID] on the plies the learner slot acted).
      return_lookahead: look one segment ahead for the returns.
      hands_per_match: hands per match in the Arena.
      mesh: the device mesh of the Arena, or None.
      opening_plies: the opening_plies option of the module doc.
      first_hand_only: the first_hand_only option of the module doc.
      endgame: the endgame option of the module doc.

    Raises:
      ValueError: inconsistent arguments (module doc for the options).
    """
    x64.assert_x64("DaggerSource")
    problems = [
        (
            not isinstance(beta, schedules.Schedule),
            f"beta must be a Schedule, got {type(beta)}",
        ),
        (
            len(opponents) != len(opponent_params),
            "one opponent_params entry per opponent slot",
        ),
        (
            value_target not in ("return", "teacher"),
            f"value_target {value_target!r}",
        ),
        (not 0.0 <= float(gamma) <= 1.0, f"gamma {gamma} not in [0, 1]"),
        (
            int(opening_plies) != opening_plies or opening_plies < 0,
            f"opening_plies {opening_plies} is not an int >= 0",
        ),
        (
            endgame is not None and not isinstance(endgame, EndgameRoute),
            f"endgame must be an EndgameRoute, got {type(endgame)}",
        ),
        (
            first_hand_only and hands_per_match != 1,
            "first_hand_only keeps one hand per row",
        ),
        (
            first_hand_only and value_target == "return" and return_lookahead,
            (
                "first_hand_only restarts every segment, so there is no next"
                " segment to look ahead into (pass return_lookahead=False)"
            ),
        ),
    ]
    for bad, what in problems:
      if bad:
        raise ValueError(f"DaggerSource: {what}")
    self.policy = DaggerPolicy(
        student, teacher, emit_value=value_target == "teacher"
    )
    self.opening_plies = int(opening_plies)
    self.first_hand_only = bool(first_hand_only)
    self.endgame = endgame
    slot_policy: types.Policy = self.policy
    self._routed: Optional[routed.RoutedPolicy] = None
    if endgame is not None:
      slot_policy = self._routed = routed.RoutedPolicy(
          slot_policy, endgame.policy, endgame.route
      )
    if self.opening_plies:
      slot_policy = opening_lib.OpeningPolicy(slot_policy, self.opening_plies)
    self.slot_policy = slot_policy
    self.arena = arena_lib.Arena(
        env,
        [arena_lib.Slot(slot_policy, 1, rule), *opponents],
        perspective_slot=0,
        hands_per_match=hands_per_match,
        mesh=mesh,
    )
    self.table = table
    self.beta = beta
    self.gamma = float(gamma)
    self.value_target = value_target
    self.return_lookahead = bool(return_lookahead)
    self.student_params = student_params
    self.teacher_params = teacher_params
    self.opponent_params = list(opponent_params)
    self.iteration = np.int64(0)  # segments rolled
    self.decisions = np.int64(
        0
    )  # valid rows emitted, after the masks (settable: resume)
    self._inner = arena_lib.ArenaSource(
        self.arena, self._params(), table, num_steps
    )
    self._pending: Optional[types.TrajectoryBatch] = None

  @property
  def inner(self) -> arena_lib.ArenaSource:
    """The ArenaSource the segments are rolled from.

    Its state is the hands in flight; a resume restores it (league.kinds.bc).
    """
    return self._inner

  @inner.setter
  def inner(self, src: arena_lib.ArenaSource) -> None:
    """Replaces the ArenaSource (a resume).

    Args:
      src: the ArenaSource.
    """
    assert isinstance(src, arena_lib.ArenaSource), type(src)
    self._inner = src

  @property
  def pending(self) -> Optional[types.TrajectoryBatch]:
    """The rolled lookahead segment the next `next` returns, or None.

    Return targets with return_lookahead.
    """
    return self._pending

  @pending.setter
  def pending(self, b: Optional[types.TrajectoryBatch]) -> None:
    """Replaces the lookahead segment (a resume).

    Args:
      b: the segment, or None.
    """
    self._pending = b

  @property
  def frames(self) -> np.int64:
    """Frames the inner ArenaSource has rolled (int64)."""
    return self._inner.frames

  @frames.setter
  def frames(self, v) -> None:
    """Sets the inner ArenaSource's frame count (a resume).

    Args:
      v: the frame count.
    """
    self._inner.frames = np.int64(v)

  def beta_now(self) -> float:
    """The beta at the current iteration.

    Returns:
      The probability the teacher acts.

    Raises:
      ValueError: the schedule's value is not a probability.
    """
    b = float(self.beta.value(np.int64(self.iteration)))
    if not 0.0 <= b <= 1.0:
      raise ValueError(
          f"beta {b} at iteration {int(self.iteration)} is not a probability"
      )
    return b

  def slot_params(self, mix):
    """The learner slot's params (unstacked) around the DAgger mix params.

    Args:
      mix: the DaggerPolicy params.

    Returns:
      The slot's params.
    """
    p = mix
    if self.endgame is not None:
      p = {"base": p, "override": self.endgame.params}
    if self.opening_plies:
      p = opening_lib.OpeningPolicy.make_params(p)
    return p

  def _student_carry(self, carry):
    """The student's carry inside the routed carries.

    Args:
      carry: The slot carry.

    Returns:
      The student's carry.
    """
    c = carry
    if self.opening_plies:
      c = c["base"]
    if self.endgame is not None:
      c = c["base"]
    return c["student"]

  def _params(self):
    """The slot params at the current beta.

    Returns:
      The slot params.
    """
    mix = self.policy.make_params(
        self.student_params, self.teacher_params, self.beta_now()
    )
    return [_stack1(self.slot_params(mix)), *self.opponent_params]

  def set_student_params(self, params) -> None:
    """Sets the student params the next segment plays.

    Args:
      params: the student's params.
    """
    self.student_params = params

  def restart(self) -> None:
    """Restarts the inner source's hands and drops the pending segment."""
    self._inner.restart()
    self._pending = None

  def _roll(self, key) -> types.TrajectoryBatch:
    """Rolls one segment with the current mix and counts its decisions.

    Args:
      key: the PRNG key of the roll.

    Returns:
      The TrajectoryBatch, its valid rows restricted to the trained ones.
    """
    self._inner.set_params(self._params())
    if self.first_hand_only:
      self._inner.restart()
    b = self._inner.next(key)
    assert isinstance(self.iteration, np.int64)
    self.iteration = np.int64(self.iteration + np.int64(1))
    valid = b.valid
    if self.first_hand_only:
      valid = valid & first_hand_mask(b.steps.new_hand)
    if self.endgame is not None:
      valid = valid & ~routed.override_rows(self._routed, b.steps.info)
    assert isinstance(self.decisions, np.int64)
    self.decisions = np.int64(
        self.decisions + np.int64(jnp.sum(valid, dtype=jnp.int64))
    )
    return b._replace(init_carry=self._student_carry(b.init_carry), valid=valid)

  def next(self, key) -> types.TrajectoryBatch:
    """The next segment with its value targets (TrajectorySource protocol).

    Args:
      key: the PRNG key of the roll.

    Returns:
      The TrajectoryBatch.
    """
    if self.value_target == "teacher":
      b = self._roll(key)
      assert (
          targets.TARGET_VALUE_VALID in b.labels
      ), "the Arena writes the teacher's value targets and their label"
      return b
    if not self.return_lookahead:
      b = self._roll(key)
      return targets.with_mc_value_targets(b, gamma=self.gamma)
    if self._pending is None:
      key, k0 = jax.random.split(key)
      self._pending = self._roll(k0)
    cur = self._roll(key)
    b = self._pending
    self._pending = cur
    return targets.with_mc_value_targets(b, gamma=self.gamma, lookahead=cur)
