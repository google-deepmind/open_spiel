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

"""The Arena: the rollout driver that every learner, evaluator and source uses.

An Arena plays B rows of one Env in parallel. Each row maps every seat to a
(slot, member): a slot holds one Policy (one architecture) with its params
stacked [K, ...], and the member picks the row's params. Self-play, frozen
opponents, populations and policy mixtures are therefore slot tables, never
special env classes; the env is dynamics only.

  arena = Arena(env, [Slot(policy, num_members=1, rule=types.SAMPLE)])
  table = self_play_table(batch=64, num_players=env.num_players)
  deal_key, policy_key = independent_keys(key, 64, env.num_players)
  state, batch, log = arena.collect([stacked_params], table, deal_key,
                                    policy_key, num_steps=32)

`Arena` runs the whole rollout in one jit and scan and takes NATIVE_JAX and
CALLBACK_BATCHED slots. `HostArena` (core.host_arena) drives the same plies
from a Python loop over the same Slot and SlotTable, and also takes
HOST_DRIVEN slots.

The ply. Every seat observes its SeatStep; the actor's slot evaluates the
actor's step and `decide` picks the action; the env applies it and resolves
chance. A hand ends with one observe-only terminal ply (to_move = -1, no
evaluate), so every seat sees the terminal InfoState and the hand-ending
action before the next hand is dealt (types.SeatStep).

Row ownership. A slot's observe and evaluate see only the rows it owns: rows
are gathered owned-first into a chunk padded with copies of an owned row, and
a slot that owns no row at a ply is skipped. A host policy therefore never
receives another slot's seats or an out-of-range member. A policy that
implements types.RowMaskedPolicy also never has the padding counted as real
rows (`gates_padding`). In an Arena of several slots a NATIVE_JAX slot runs
in the smallest of a few fixed chunk sizes that holds its rows
(`slot_chunks`), and env work that only some rows need (the chance resolve
after an action, the next hand's deal) runs on those rows only (`on_rows`).
A slot whose carry has no array leaves skips observe entirely.

Keys and duplicate deals. Each row has a deal key and per-seat policy keys:
  hand chance key = fold_in(fold_in64(deal_key[row], hand_index), 0)
  chance event n of the hand uses fold_in64(hand chance key, n)
    (core.chance.resolve; n is carried per row)
  policy keys of seat p at row ply t: core.plies.ply_keys(policy_key[row, p], t)
so two rows with the same deal key see the same chance outcomes in the same
order, whatever the seats play. Every per-row computation is vmapped with
that row's keys, so a row's result does not depend on the other rows.

Sharding. The batch axis is sharded over a 1-D `jax.sharding.Mesh` named
"batch", and reset and rollout each run under one `jax.shard_map`: a device
holds B/N rows and runs the single-device program on them, with params
replicated. The rollout contains no collective, so the sharded rollout is the
unsharded one computed in slices, equal bit for bit at any device count that
divides B. A CALLBACK_BATCHED slot's callback runs once per device on that
device's rows.

x64. The Arena asserts jax_enable_x64 at construction. Every counter
(hand_index, hand_in_match, chance_count, row_ply, ArenaSource.frames) is
int64 and folded into keys with core.x64.fold_in64, so nothing wraps at 2**31
or repeats keys at 2**32.

Temperature drops. A Slot may carry a TemperatureDrop(ply, temperature): its
rows act at the rule's temperature while the hand's ply (every seat's plies
counted, from 0) is below `ply`, and at `temperature` from then on. The
actor's logits are divided by the temperature in f64 and decide runs at
temperature 1, which is decide's own arithmetic, so a drop to T and
DecisionRule(kind, T) give the same probabilities and samples bit for bit.
behavior_logits stay the policy's own logits; behavior_logp is the tempered
rule's. Policies own no temperature.
"""

import functools
from typing import Any, NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec
import numpy as np

from open_spiel.python.scale.core import auxiliary as aux_lib
from open_spiel.python.scale.core import chance as chance_lib
from open_spiel.python.scale.core import decide as decide_lib
from open_spiel.python.scale.core import plies
from open_spiel.python.scale.core import slots as slots_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64

tree_map = jax.tree_util.tree_map

# The slot records live in core.slots, which imports no jax.
IN_GRAPH = slots_lib.IN_GRAPH
TemperatureDrop = slots_lib.TemperatureDrop
Slot = slots_lib.Slot
SlotTable = slots_lib.SlotTable


class ArenaState(NamedTuple):
  """Everything a rollout carries from one ply to the next.

  Attributes:
    env_state: [B] env states.
    carries: per slot, a carry with leaves [B, P, ...] (host slots: see
      HostArena).
    deal_key: [B] deal keys.
    policy_key: [B, P] policy keys.
    hand_index: i64[B], hands started.
    hand_in_match: i64[B], hands completed in the current match.
    chance_count: i64[B], chance events resolved in the current hand.
    row_ply: i64[B], plies since reset.
    hand_ply: i64[B], plies of the current hand before this one (0 at its
      first ply).
    new_hand: bool[B].
    new_match: bool[B].
    last_action: i32[B, P], the last action as each seat sees it.
    last_actor: i32[B], the last action's actor.
  """

  env_state: Any  # [B]
  # per slot: carry with leaves [B, P, ...] (host slots: see HostArena)
  carries: tuple[Any, ...]
  deal_key: Any  # [B]
  policy_key: Any  # [B, P]
  hand_index: Any  # i64[B] hands started
  hand_in_match: Any  # i64[B] hands completed in the current match
  chance_count: Any  # i64[B] chance events resolved in the current hand
  row_ply: Any  # i64[B] plies since reset
  # i64[B] plies of the current hand before this one (0 at its first ply)
  hand_ply: Any
  new_hand: Any  # bool[B]
  new_match: Any  # bool[B]
  last_action: Any  # i32[B, P] as each seat sees it
  last_actor: Any  # i32[B]


class PlyLog(NamedTuple):
  """A seat-independent record of each ply, [B, T, ...] (evaluators read it).

  Attributes:
    actor: i32, -1 on a terminal ply.
    action: i32, -1 on a terminal ply.
    rewards: f32[P].
    done: bool, the ply's action ended a hand.
    terminal_ply: bool, the observe-only terminal ply.
    slot: i32 slot of the actor (-1 on a terminal ply).
    member: i32 member of the actor (-1 on a terminal ply).
    bad: bool, the actor had no legal action, chose an illegal one, or got
      NaN probs.
  """

  actor: Any  # i32, -1 on a terminal ply
  action: Any  # i32, -1 on a terminal ply
  rewards: Any  # f32[P]
  done: Any  # bool: the ply's action ended a hand
  terminal_ply: Any  # bool: the observe-only terminal ply
  slot: Any  # i32 slot of the actor (-1 on a terminal ply)
  member: Any  # i32 member of the actor (-1 on a terminal ply)
  # bool: the actor had no legal action, chose an illegal one, or got NaN probs
  bad: Any


class SlotEval(NamedTuple):
  """One slot's evaluate+decide over the B rows (meaningful where `mine`).

  Attributes:
    mine: bool[B], the rows whose actor the slot owns.
    action: i32[B].
    logits: f32[B, A].
    value: f32[B, P].
    logp: f32[B].
    finite: bool[B], whether decide's probs are finite.
    aux: Mapping[str, [B, ...]].
  """

  mine: Any  # bool[B]
  action: Any  # i32[B]
  logits: Any  # f32[B, A]
  value: Any  # f32[B, P]
  logp: Any  # f32[B]
  finite: Any  # bool[B] decide's probs are finite
  aux: Any  # Mapping[str, [B, ...]]


def _take_seat(x, seat):
  """Each row's entry at its seat: x [B, P, ...], seat i32[B] -> [B, ...].

  Args:
    x: [B, P, ...] values.
    seat: i32[B], each row's seat.

  Returns:
    [B, ...], each row's entry at its seat.
  """
  return x[jnp.arange(x.shape[0], dtype=jnp.int32), seat]


def _take_seats(x, seats):
  """Rows' entries at seats: x [B, P, ...], seats i32[B, M] -> [B, M, ...].

  Args:
    x: [B, P, ...] values.
    seats: i32[B, M], each row's seats.

  Returns:
    [B, M, ...], each row's entries at its seats.
  """
  return x[jnp.arange(x.shape[0], dtype=jnp.int32)[:, None], seats]


def _tree_where(cond, a, b):
  """Leafwise select of two PyTrees, a where cond, else b.

  Args:
    cond: bool[...], broadcast over each leaf's trailing axes.
    a: the PyTree taken where cond.
    b: the PyTree taken elsewhere, same structure as a.

  Returns:
    The selected PyTree.
  """

  def sel(x, y):
    """One leaf of the selection, cond broadcast over trailing axes.

    Args:
      x: the leaf where cond.
      y: the leaf elsewhere.

    Returns:
      x where cond, else y.
    """
    c = cond.reshape(cond.shape + (1,) * (x.ndim - cond.ndim))
    return jnp.where(c, x, y)

  return tree_map(sel, a, b)


def _as_2d_perspective(persp):
  """The perspective seats as [B, M].

  Args:
    persp: i32[B] or i32[B, M] perspective seats.

  Returns:
    `persp` with a member axis.
  """
  return persp[:, None] if persp.ndim == 1 else persp


def owned_only(mine, fn, inputs, sizes=None, with_real=False):
  """Calls fn on the rows where `mine` only; returns fn's output per row.

  inputs and fn's output are pytrees with leading [R]. Rows are gathered
  owned-first into a chunk and the chunk's tail is padded with a copy of the
  first owned row, so fn never sees a row it does not own; when no row is
  owned fn is not called and zeros are returned. Non-owned rows of the
  result hold zeros or the padding's values: the caller masks them with
  `mine`.

  sizes: the chunk sizes, ascending, the last R (default (R,): one chunk of
  every row). With several, the smallest that holds the owned rows runs
  (lax.switch; exactly one), so a slot owning a quarter of the rows pays a
  quarter of a full evaluation; fn must then be a per-row program (each
  row's outputs depend on that row's inputs only) for a row's result not to
  depend on the chunk it ran in (up to an ulp of a transcendental, see
  on_rows).

  with_real: fn is called as fn(inputs, real) with real bool[chunk] True on
  the owned rows and False on the padding, so a batched host callback can
  keep the padding out of its module: the real rows' outputs and the
  policy's state are then those of an unpadded batch.

  Args:
    mine: bool[R], the owned rows.
    fn: the per-chunk program.
    inputs: a pytree with leading [R].
    sizes: the chunk sizes (below).
    with_real: pass fn the real-row mask (below).

  Returns:
    fn's output per row, a pytree with leading [R].
  """
  r = mine.shape[0]
  sizes = (r,) if sizes is None else tuple(sizes)
  assert sizes[-1] == r and list(sizes) == sorted(set(sizes)), (sizes, r)
  n = jnp.sum(mine, dtype=jnp.int32)
  order = jnp.argsort(jnp.logical_not(mine), stable=True).astype(
      jnp.int32
  )  # owned first
  if with_real:
    base_fn = fn
    # Rebinds `fn` to take (inputs, real) as one argument.
    fn = lambda inp: base_fn(inp[0], inp[1])  # pylint: disable=unnecessary-lambda-assignment
  if sizes == (r,):  # one chunk of every row
    real = jnp.arange(r, dtype=jnp.int32) < n
    idx = jnp.where(real, order, order[0])
    gathered = tree_map(lambda x: x[idx], inputs)
    if with_real:
      gathered = (gathered, real)
    out_shape = jax.eval_shape(fn, gathered)

    def skip(_):
      """The identity order when no row is owned.

      Args:
        _: the cond operand (unused).

      Returns:
        zeros shaped like fn's output on the chunk.
      """
      return tree_map(lambda sd: jnp.zeros(sd.shape, sd.dtype), out_shape)

    res = jax.lax.cond(n > 0, fn, skip, gathered)
    inv = jnp.argsort(order).astype(jnp.int32)
    return tree_map(lambda x: x[inv], res)
  one = tree_map(lambda x: x[:1], inputs)
  out_shape = jax.eval_shape(
      fn, (one, jnp.ones((1,), jnp.bool_)) if with_real else one
  )

  def zeros():
    """Zeros in the output's structure, r rows.

    Returns:
      The zeros.
    """
    return tree_map(
        lambda sd: jnp.zeros((r,) + sd.shape[1:], sd.dtype), out_shape
    )

  def chunk(c):
    """The switch branch for chunk size c.

    Args:
      c: the chunk size.

    Returns:
      the branch: owned rows gathered, evaluated, scattered back.
    """

    def run(operand):
      """Evaluates the owned rows in one chunk of c rows.

      Args:
        operand: The switch operand (unused).

      Returns:
        fn's output per row (zeros where not owned).
      """
      del operand
      pos = jnp.arange(c, dtype=jnp.int32)
      idx = jnp.where(pos < n, order[:c], order[0])
      gathered = tree_map(lambda x: x[idx], inputs)
      res = fn((gathered, pos < n) if with_real else gathered)
      if c == r:  # every row: un-permute
        inv = jnp.argsort(order).astype(jnp.int32)
        return tree_map(lambda x: x[inv], res)
      dst = jnp.where(pos < n, order[:c], jnp.int32(r))  # r: dropped
      return tree_map(lambda z, x: z.at[dst].set(x, mode="drop"), zeros(), res)

    return run

  branch = jnp.where(
      n == 0,
      jnp.int32(0),
      jnp.int32(1)
      + jnp.sum(jnp.asarray(sizes, jnp.int32) < n, dtype=jnp.int32),
  )
  return jax.lax.switch(
      branch, [lambda _: zeros()] + [chunk(c) for c in sizes], None
  )


def slot_chunks(rows: int) -> tuple[int, ...]:
  """Chunk sizes of a NATIVE_JAX slot's evaluation in a multi-slot Arena.

  ceil(R/4), ceil(5R/8), R. A learner seat against opponents owns about half
  the rows per ply (5/8 leaves room for the noise), a pool member a small
  share.

  Args:
    rows: R, the Arena's row count.

  Returns:
    the distinct chunk sizes, ascending, the last R.
  """
  r = int(rows)
  return tuple(sorted({max(1, -(-r // 4)), max(1, -(-5 * r // 8)), r}))


def slot_rows(
    arena_slots: int, slot: "Slot", mine, fn, inputs, with_real=False
):
  """The output of fn per row for the rows `slot` owns (`mine`).

  Other rows hold values the caller masks with `mine`.

  A NATIVE_JAX slot in an Arena of several slots runs in the smallest of
  `slot_chunks(R)` that holds its rows: its fn is a per-row program (every
  policy call is vmapped one row at a time with that row's params and keys),
  so the owned rows' outputs are the same bits at any chunk size. A
  CALLBACK_BATCHED slot (a host kernel may depend on the batch size) and the
  slot of a one-slot Arena run in one chunk of R rows.

  Args:
    arena_slots: the number of slots in the Arena.
    slot: the slot that evaluates.
    mine: bool[R], the rows the slot owns.
    fn: the per-row program (owned_only).
    inputs: fn's inputs, a pytree with leading [R].
    with_real: pass fn the chunk's real-row mask (owned_only).

  Returns:
    fn's output per row, a pytree with leading [R].
  """
  if (
      arena_slots > 1
      and slot.policy.spec.capability == types.Capability.NATIVE_JAX
  ):
    return owned_only(
        mine, fn, inputs, slot_chunks(mine.shape[0]), with_real=with_real
    )
  return owned_only(mine, fn, inputs, with_real=with_real)


def gates_padding(slot: "Slot") -> bool:
  """Whether `slot` evaluates with its chunk's padding marked.

  It does when its policy is a types.RowMaskedPolicy: its evaluate_rows sees
  which rows are real, so neither its real rows' outputs nor its state depend on
  the padding. The padding's outputs are dropped either way.

  Args:
    slot: the slot.

  Returns:
    True if the slot's policy is a types.RowMaskedPolicy.
  """
  return isinstance(slot.policy, types.RowMaskedPolicy)


def _per_row(fn, params, num_members, member, args, key=None):
  """Calls a batched policy fn once per row (batch of 1) with per-row params.

  K == 1: params[0] is broadcast (no gather). K > 1: params[member[row]].
  args are batched [R, ...]; key is [R] (one key per row) or None. Under vmap
  a CALLBACK_BATCHED policy's pure_callback receives all R rows at once, one
  call per device under the shard_map.

  Args:
    fn: the batched policy function.
    params: the slot's stacked params [K, ...].
    num_members: K.
    member: i32[R], each row's member.
    args: the batched arguments [R, ...].
    key: [R] keys, or None.

  Returns:
    fn's output per row, with leading [R].
  """
  if num_members == 1:
    p, p_axis = tree_map(lambda x: x[0], params), None
  else:
    p, p_axis = tree_map(lambda x: x[member], params), 0

  def one(pp, a, k):
    """Calls fn on one row as a batch of one.

    Args:
      pp: the row's params.
      a: the row's arguments.
      k: the row's key, or None.

    Returns:
      fn's output for the row, without the batch axis.
    """
    a1 = tree_map(lambda x: x[None], a)
    out = fn(pp, *a1) if k is None else fn(pp, *a1, k)
    return tree_map(lambda x: x[0], out)

  if key is None:
    return jax.vmap(lambda pp, a: one(pp, a, None), in_axes=(p_axis, 0))(
        p, args
    )
  return jax.vmap(one, in_axes=(p_axis, 0, 0))(p, args, key)


def tempered(slot: Slot, logits, hand_ply):
  """The decision rule and the logits `slot` decides on at each row's hand ply.

  Without a TemperatureDrop: the slot rule and the logits unchanged; else
  rule.kind at temperature 1 over f64 logits / f64 T(hand_ply), decide's own
  arithmetic (module doc).

  Args:
    slot: the deciding slot.
    logits: the slot's logits [B, A].
    hand_ply: i64[B], each row's ply within its hand.

  Returns:
    (decision rule, logits it decides on).
  """
  drop = slot.temperature_drop
  if drop is None:
    return slot.rule, logits
  x64.assert_x64("arena.tempered")
  t = jnp.where(
      jnp.asarray(hand_ply, jnp.int64) < jnp.int64(drop.ply),
      jnp.float64(slot.rule.temperature),
      jnp.float64(drop.temperature),
  )
  scaled = jnp.asarray(logits).astype(jnp.float64) / t[:, None]
  return types.DecisionRule(slot.rule.kind, 1.0), scaled.astype(jnp.float64)


def decide_rows(rule: types.DecisionRule, logits, legal_mask, decide_key):
  """Decides per row, each row with its own key.

  logp is log of decide's f64 probability of the action, cast to f32 for the
  batch; finite is checked on the f64 probabilities. logits reach decide in
  their own dtype (decide widens them to f64 itself), so tempered's f64 logits
  are not narrowed.

  logp is log of decide's f64 probability of the action, cast to f32 for the
  batch; finite is checked on the f64 probabilities. logits reach decide in
  their own dtype (decide widens them to f64 itself), so tempered's f64
  logits are not narrowed.

  Args:
    rule: the decision rule.
    logits: logits [B, A].
    legal_mask: bool[B, A].
    decide_key: one key per row [B].

  Returns:
    (action i32[B], logp f32[B], finite bool[B]).
  """

  def one(lg, m, k):
    """Decides one row.

    Args:
      lg: the row's logits.
      m: the row's legal mask.
      k: the row's decide key.

    Returns:
      (action i32, logp f32, whether the probabilities are finite).
    """
    a, probs = decide_lib.decide(lg[None], m[None], rule, k)
    a, probs = a[0], probs[0]
    if rule.kind == "argmax":
      lp = jnp.float64(0.0)
    else:
      lp = jnp.log(probs[jnp.maximum(a, 0)])
    return a, lp.astype(jnp.float32), jnp.all(jnp.isfinite(probs))

  return jax.vmap(one)(jnp.asarray(logits), legal_mask, decide_key)


class Arena:
  """The in-graph rollout driver over one Env and a fixed list of slots.

  See the module doc.

  Attributes:
    env: the Env.
    slots: one Slot per architecture; every slot's policy matches the env's
      num_actions and num_players.
    perspective_slot: the slot whose seats the TrajectoryBatch is emitted for.
    hands_per_match: hands before a match restarts (new_match, carry reset).
    mesh: a 1-D mesh named "batch" (default: every device).
  """

  accepted = IN_GRAPH

  def __init__(
      self,
      env: types.Env,
      slots: Sequence[Slot],
      perspective_slot: int = 0,
      hands_per_match: int = 1,
      mesh: Optional[Mesh] = None,
  ):
    x64.assert_x64(type(self).__name__)
    assert slots, "need at least one slot"
    assert 0 <= perspective_slot < len(slots)
    assert hands_per_match >= 1
    for sl in slots:
      assert sl.policy.spec.capability in self.accepted, (
          f"{type(self).__name__} takes "
          f"{sorted(c.value for c in self.accepted)} "
          f"slots; got {sl.policy.spec.capability} (HOST_DRIVEN slots run in "
          "HostArena)"
      )
      assert sl.policy.spec.num_actions == env.num_actions, (
          sl.policy.spec,
          env.num_actions,
      )
      assert sl.policy.spec.num_players == env.num_players
      types.check_policy_admitted(env, sl.policy.spec)
    self.env = env
    self.slots = tuple(slots)
    self.perspective_slot = perspective_slot
    self.hands_per_match = hands_per_match
    self.mesh = (
        mesh if mesh is not None else Mesh(np.array(jax.devices()), ("batch",))
    )
    assert self.mesh.axis_names == ("batch",), self.mesh.axis_names
    self._row_sharding = NamedSharding(self.mesh, PartitionSpec("batch"))
    self._replicated = NamedSharding(self.mesh, PartitionSpec())
    rows, rep = PartitionSpec("batch"), PartitionSpec()
    self._reset_jit = jax.jit(
        jax.shard_map(
            self._reset,
            mesh=self.mesh,
            in_specs=(rep, rows, rows, rows),
            out_specs=rows,
            check_vma=False,
        )
    )
    self._rollout_jit = jax.jit(
        self._rollout_sharded, static_argnames=("num_steps",)
    )

  # -- host-side helpers -------------------------------------------------------
  def validate(self, params: Sequence[Any], table: SlotTable) -> None:
    """Host-side checks of a params list and a slot table.

    Raises:
      ValueError: params or the table do not fit this Arena.

    Args:
      params: one params entry per slot.
      table: the slot table.
    """

    def need(ok, msg):
      if not ok:
        raise ValueError(f"Arena.validate: {msg}")

    need(
        len(params) == len(self.slots),
        f"{len(params)} params for {len(self.slots)} slots",
    )
    for j, (sl, p) in enumerate(zip(self.slots, params)):
      if sl.in_graph:
        for leaf in jax.tree_util.tree_leaves(p):
          need(
              leaf.shape[:1] == (sl.num_members,),
              f"slot {j}: params must be stacked [K={sl.num_members}, ...], "
              "got leaf "
              f"{leaf.shape}",
          )
      else:
        need(
            isinstance(p, (list, tuple)) and len(p) == sl.num_members,
            f"slot {j} (HOST_DRIVEN): params must be a sequence of "
            f"K={sl.num_members} "
            "per-member host objects",
        )
    seat_slot = np.asarray(table.seat_slot)
    seat_member = np.asarray(table.seat_member)
    persp = np.asarray(table.perspective_seat)
    need(
        seat_slot.ndim == 2, f"seat_slot must be [B, P], got {seat_slot.shape}"
    )
    b, p = seat_slot.shape
    need(
        p == self.env.num_players,
        f"seat_slot {seat_slot.shape} for {self.env.num_players} players",
    )
    need(
        seat_member.shape == (b, p),
        f"seat_member {seat_member.shape} != {(b, p)}",
    )
    need(
        persp.ndim in (1, 2) and persp.shape[0] == b,
        f"perspective_seat must be [B] or [B, S], got {persp.shape}",
    )
    persp2 = persp[:, None] if persp.ndim == 1 else persp
    need(
        ((seat_slot >= 0) & (seat_slot < len(self.slots))).all(),
        "slot out of range",
    )
    k = np.array([sl.num_members for sl in self.slots])[seat_slot]
    need(((seat_member >= 0) & (seat_member < k)).all(), "member out of range")
    need(((persp2 >= 0) & (persp2 < p)).all(), "perspective seat out of range")
    for row in persp2:
      need(
          len(set(row.tolist())) == len(row), f"repeated perspective seat {row}"
      )
    need(
        (
            seat_slot[np.arange(b)[:, None], persp2] == self.perspective_slot
        ).all(),
        "every row's perspective seat must sit in the perspective slot",
    )
    ndev = self.mesh.devices.size
    need(b % ndev == 0, f"batch {b} not divisible by {ndev} devices")

  def _shard_rows(self, tree):
    """Places `tree` sharded by row.

    Args:
      tree: A pytree.

    Returns:
      The sharded tree.
    """
    return jax.device_put(tree, self._row_sharding)

  def _replicate(self, tree):
    """Places `tree` replicated.

    Args:
      tree: A pytree.

    Returns:
      The replicated tree.
    """
    return jax.device_put(tree, self._replicated)

  def reset(self, params, table: SlotTable, deal_key, policy_key) -> ArenaState:
    """New match in every row.

    Args:
      params: one params entry per slot.
      table: the slot table.
      deal_key: deal keys [B].
      policy_key: policy keys [B, P].

    Returns:
      the ArenaState at the first ply.
    """
    self.validate(params, table)
    b = np.asarray(table.seat_slot).shape[0]
    assert deal_key.shape[0] == b and policy_key.shape[:2] == (
        b,
        self.env.num_players,
    )
    return self._reset_jit(
        self._replicate(list(params)),
        self._shard_rows(table),
        self._shard_rows(deal_key),
        self._shard_rows(policy_key),
    )

  def rollout(
      self, params, table: SlotTable, state: ArenaState, num_steps: int
  ):
    """Runs num_steps plies from `state`.

    Raises if any ply was bad (an actor with no legal action, an illegal action,
    or NaN decide probabilities, e.g. from NaN logits).

    Raises if any ply was bad (an actor with no legal action, an illegal
    action, or NaN decide probabilities, e.g. from NaN logits).

    Args:
      params: one params entry per slot.
      table: the slot table.
      state: the ArenaState to continue from.
      num_steps: plies to run (>= 1).

    Returns:
      (state, TrajectoryBatch, PlyLog).
    """
    self.validate(params, table)
    out = self._rollout_jit(
        self._replicate(list(params)),
        self._shard_rows(table),
        self._shard_rows(state),
        num_steps=num_steps,
    )
    check_log(out[2])
    return out

  def collect(self, params, table, deal_key, policy_key, num_steps: int):
    """Reset then rollout.

    Args:
      params: one params entry per slot.
      table: the slot table.
      deal_key: deal keys [B].
      policy_key: policy keys [B, P].
      num_steps: plies to run (>= 1).

    Returns:
      (state, TrajectoryBatch, PlyLog).
    """
    state = self.reset(params, table, deal_key, policy_key)
    return self.rollout(params, table, state, num_steps)

  # -- traced building blocks (shared with HostArena) --------------------------
  def _hand_chance_key(self, deal_key, hand_index):
    """The chance key of hand `hand_index`.

    Args:
      deal_key: The deal key.
      hand_index: The hand index.

    Returns:
      The key.
    """
    return hand_chance_key(deal_key, hand_index)

  def _deal(self, deal_key, hand_index):
    """(env_state [B], chance_count [B]) of hand `hand_index`.

    Args:
      deal_key: The deal key.
      hand_index: The hand index of each row.

    Returns:
      (env_state [B], chance_count [B]).
    """
    return deal_states(self.env, deal_key, hand_index)

  def _deal_rows(self, need, deal_key, hand_index, env_state, chance_count):
    """Deals hand `hand_index` where `need` (bool[R]).

    The given values are kept elsewhere. A deal is a pure function of (deal_key,
    hand_index) per row, so only the rows that need one are dealt (`on_rows`): a
    deal can be many chance events, and few rows end a hand at any one ply.

    Args:
      need: bool[R], the rows to deal.
      deal_key: [R] deal keys.
      hand_index: i64[R], each row's hand.
      env_state: the rows' env states.
      chance_count: i64[R], the rows' chance counts.

    Returns:
      (env_state, chance_count), dealt where `need`.
    """

    def deal(rows):
      """Deals the given rows (None: all rows).

      Args:
        rows: the row indices to deal, or None for all rows.

      Returns:
        (env_state, chance_count) of the rows.
      """
      pick = (lambda x: x) if rows is None else (lambda x: x[rows])
      return self._deal(pick(deal_key), pick(hand_index))

    return on_rows(need, deal, (env_state, chance_count))

  def _resolve_rows(self, need, deal_key, hand_index, env_state, chance_count):
    """Resolves chance with each row's hand chance key where `need`.

    Only the rows where `need` run core.chance.resolve (their state is a chance
    node: resolve would loop), through `on_rows`; resolve leaves every other row
    as it is, so the result is the same.

    Args:
      need: bool[R], the rows at a chance node.
      deal_key: [R] deal keys.
      hand_index: i64[R], each row's hand.
      env_state: the rows' env states.
      chance_count: i64[R], the rows' chance counts.

    Returns:
      (env_state, chance_count), resolved where `need`.
    """
    env = self.env

    def resolve(rows):
      """Resolves chance on the given rows (None: all rows).

      Args:
        rows: the row indices to resolve, or None for all rows.

      Returns:
        (env_state, chance_count) of the rows.
      """
      pick = (
          (lambda x: x)
          if rows is None
          else (lambda x: tree_map(lambda y: y[rows], x))
      )
      ck = self._hand_chance_key(pick(deal_key), pick(hand_index))
      return jax.vmap(lambda s, k, n: chance_lib.resolve(env, s, k, n))(
          pick(env_state), ck, pick(chance_count)
      )

    return on_rows(need, resolve, (env_state, chance_count))

  def _init_slot(self, j, params_j, table, policy_key):
    """Slot j's carry [B, P, ...] (init_carry on the seats it owns only).

    Args:
      j: The slot index.
      params_j: The slot's params.
      table: The SlotTable.
      policy_key: The policy keys [B, P].

    Returns:
      The slot's carry, leaves [B, P, ...].
    """
    sl = self.slots[j]
    b, p = table.seat_slot.shape
    seats = jnp.broadcast_to(jnp.arange(p, dtype=jnp.int32), (b, p)).reshape(-1)
    keys = jax.vmap(jax.vmap(plies.init_key))(policy_key).reshape(-1)
    member = table.seat_member.reshape(-1)
    mine = table.seat_slot.reshape(-1) == j

    def fn(inp):
      """Runs init_carry on the owned rows.

      Args:
        inp: (keys, seats, members) of the owned rows.

      Returns:
        the rows' carries.
      """
      s, k, m = inp
      return _per_row(
          lambda pp, s1, k1: sl.policy.init_carry(pp, 1, s1, k1),
          params_j,
          sl.num_members,
          m,
          (s,),
          key=k,
      )

    c = owned_only(mine, fn, (seats, keys, member))  # once per reset: one chunk
    return tree_map(lambda x: x.reshape((b, p) + x.shape[1:]), c)

  def _reset(self, params, table, deal_key, policy_key):
    """The ArenaState of a new match: hands dealt, carries initialised.

    Args:
      params: Each slot's params.
      table: The SlotTable.
      deal_key: The deal key.
      policy_key: The policy keys.

    Returns:
      The ArenaState.
    """
    b, _ = table.seat_slot.shape
    hand_index = jnp.zeros((b,), jnp.int64)
    env_state, chance_count = self._deal(deal_key, hand_index)
    carries = tuple(
        self._init_slot(j, params[j], table, policy_key) if sl.in_graph else ()
        for j, sl in enumerate(self.slots)
    )
    return ArenaState(
        env_state=env_state,
        carries=carries,
        deal_key=deal_key,
        policy_key=policy_key,
        hand_index=hand_index,
        hand_in_match=jnp.zeros((b,), jnp.int64),
        chance_count=chance_count,
        row_ply=jnp.zeros((b,), jnp.int64),
        hand_ply=jnp.zeros((b,), jnp.int64),
        new_hand=jnp.ones((b,), jnp.bool_),
        new_match=jnp.ones((b,), jnp.bool_),
        last_action=jnp.full(
            (b, self.env.num_players), types.NO_ACTION, jnp.int32
        ),
        last_actor=jnp.full((b,), types.NO_ACTOR, jnp.int32),
    )

  def _steps(self, st: ArenaState) -> types.SeatStep:
    """SeatStep for every seat, leaves [B, P, ...].

    Args:
      st: The ArenaState.

    Returns:
      The SeatSteps, leaves [B, P, ...].
    """
    return plies.seat_steps(
        self.env,
        st.env_state,
        st.new_hand,
        st.new_match,
        st.last_action,
        st.last_actor,
    )

  def _keys(self, st: ArenaState, steps: types.SeatStep):
    """(eval_key [B], decide_key [B]) of each row's actor at this ply.

    Args:
      st: The ArenaState.
      steps: The ply's SeatSteps.

    Returns:
      (eval_key [B], decide_key [B]).
    """
    actor = jnp.maximum(steps.to_move[:, 0], 0)
    return jax.vmap(plies.ply_keys)(
        _take_seat(st.policy_key, actor), st.row_ply
    )

  def _observe_slot(self, j, params_j, table, steps, carry_j):
    """Slot j observes every seat it owns. Returns its carry [B, P, ...].

    A carry with no array leaves (a stateless policy's ()) carries nothing an
    observe could write, so the slot's observe is skipped: no per-seat
    InfoState gather and no policy call for seats that do not act.

    Args:
      j: the slot index.
      params_j: the slot's params.
      table: the SlotTable.
      steps: the SeatStep [B, P] of this ply.
      carry_j: the slot's carry [B, P, ...].

    Returns:
      the slot's carry [B, P, ...] after the observe.
    """
    if not jax.tree_util.tree_leaves(carry_j):
      return carry_j
    sl = self.slots[j]
    b, p = table.seat_slot.shape

    def flat(x):
      """Merges the row and seat axes.

      Args:
        x: A leaf, [B, P, ...].

      Returns:
        The leaf, [B * P, ...].
      """
      return x.reshape((b * p,) + x.shape[2:])

    c = tree_map(flat, carry_j)
    mine = table.seat_slot.reshape(-1) == j

    def fn(inp):
      """Runs observe on the owned rows.

      Args:
        inp: (carries, steps, members) of the owned rows.

      Returns:
        the rows' updated carries.
      """
      cc, ss, m = inp
      return _per_row(sl.policy.observe, params_j, sl.num_members, m, (cc, ss))

    c2 = slot_rows(
        len(self.slots),
        sl,
        mine,
        fn,
        (c, tree_map(flat, steps), table.seat_member.reshape(-1)),
    )
    c2 = _tree_where(mine, c2, c)
    return tree_map(lambda x: x.reshape((b, p) + x.shape[1:]), c2)

  def _actor_view(self, j, table, steps):
    """The actor of each row and what slot j sees of it.

    Args:
      j: The slot index.
      table: The SlotTable.
      steps: The SeatSteps, leaves [B, P, ...].

    Returns:
      (actor seat, rows slot j acts on, the actor's SeatStep, its member).
    """
    actor = steps.to_move[:, 0]
    actor_safe = jnp.maximum(actor, 0)
    mine = (actor >= 0) & (_take_seat(table.seat_slot, actor_safe) == j)
    actor_step = tree_map(lambda x: _take_seat(x, actor_safe), steps)
    member = _take_seat(table.seat_member, actor_safe)
    return actor_safe, mine, actor_step, member

  def _evaluate_slot(
      self, j, params_j, table, steps, carry_j, eval_key, decide_key, hand_ply
  ) -> SlotEval:
    """Slot j evaluates and decides for the rows whose actor it owns.

    hand_ply is i64[B]: ArenaState.hand_ply, read by a TemperatureDrop.

    Args:
      j: the slot index.
      params_j: the slot's params.
      table: the SlotTable.
      steps: the SeatStep [B, P] of this ply.
      carry_j: the slot's carry [B, P, ...].
      eval_key: [B] evaluate keys.
      decide_key: [B] decide keys.
      hand_ply: i64[B], ArenaState.hand_ply, read by a TemperatureDrop.

    Returns:
      the slot's SlotEval over the B rows.
    """
    sl = self.slots[j]
    actor_safe, mine, actor_step, member = self._actor_view(j, table, steps)
    c_act = tree_map(lambda x: _take_seat(x, actor_safe), carry_j)
    gate = gates_padding(sl)

    def fn(inp, real=None):
      """Evaluates and decides the owned rows.

      Args:
        inp: (carries, steps, members, eval keys, decide keys, hand plies)
          of the owned rows.
        real: bool[chunk], the real rows, when the policy gates padding.

      Returns:
        (action, logits f32, value f32, logp, finite, aux) per row.
      """
      cc, ss, m, ek, dk, hp = inp
      if gate:  # the policy sees which rows are padding (gates_padding)

        def ev(p, c, s, r, k):
          """evaluate_rows with the arguments in _per_row's order.

          Args:
            p: The params.
            c: The carry.
            s: The SeatStep.
            r: The rows.
            k: The key.

          Returns:
            The PolicyOutput.
          """
          return sl.policy.evaluate_rows(p, c, s, k, r)

        out = _per_row(ev, params_j, sl.num_members, m, (cc, ss, real), key=ek)
      else:
        out = _per_row(
            sl.policy.evaluate, params_j, sl.num_members, m, (cc, ss), key=ek
        )
      rule, dlogits = tempered(sl, out.logits, hp)
      a, lp, fin = decide_rows(rule, dlogits, ss.legal_mask, dk)
      return (
          a,
          out.logits.astype(jnp.float32),
          out.value.astype(jnp.float32),
          lp,
          fin,
          dict(out.aux),
      )

    a, lg, v, lp, fin, aux = slot_rows(
        len(self.slots),
        sl,
        mine,
        fn,
        (c_act, actor_step, member, eval_key, decide_key, hand_ply),
        with_real=gate,
    )
    return SlotEval(mine, a, lg, v, lp, fin, aux)

  def _advance(
      self, table, st: ArenaState, steps, carries, evals: Sequence[SlotEval]
  ):
    """Combines the slots' decisions, steps the env, emits the ply's records.

    Args:
      table: The SlotTable.
      st: The ArenaState.
      steps: The SeatSteps of the ply.
      carries: Each slot's carry after observing.
      evals: Each slot's SlotEval.

    Returns:
      (the next ArenaState, the ply's record, the ply's log).

    Raises:
      TypeError: A target_value aux is not float32.
      ValueError: A target_value aux has the wrong shape, or the env's labels
        emit the reserved key.
    """
    env = self.env
    b, p = table.seat_slot.shape
    rows = jnp.arange(b, dtype=jnp.int32)
    a_dim = env.num_actions
    actor = steps.to_move[:, 0]
    actor_safe = jnp.maximum(actor, 0)
    at_term = actor < 0

    action = jnp.full((b,), -1, jnp.int32)
    logits = jnp.zeros((b, a_dim), jnp.float32)
    value = jnp.zeros((b, p), jnp.float32)
    logp = jnp.zeros((b,), jnp.float32)
    finite = jnp.ones((b,), jnp.bool_)
    for ev in evals:
      action = jnp.where(ev.mine, ev.action, action)
      logits = jnp.where(ev.mine[:, None], ev.logits, logits)
      value = jnp.where(ev.mine[:, None], ev.value, value)
      logp = jnp.where(ev.mine, ev.logp, logp)
      finite = jnp.where(ev.mine, ev.finite, finite)

    legal_actor = _take_seat(steps.legal_mask, actor_safe)
    a_safe = jnp.maximum(action, 0)
    bad = ~at_term & ((action < 0) | ~legal_actor[rows, a_safe] | ~finite)
    pev = evals[self.perspective_slot]
    # Registered aux keys (core.auxiliary) and a teacher's AUX_TARGET_VALUE:
    # dtype and shape are checked at trace time, and a non-finite value on a row
    # the perspective slot decided is a bad ply (check_log raises), so a NaN
    # target never reaches a learner.
    check_reserved_aux(pev.aux, b, a_dim)
    bad = bad | (pev.mine & ~reserved_aux_finite(pev.aux, b))
    if types.AUX_TARGET_VALUE in pev.aux:
      tv_in = pev.aux[types.AUX_TARGET_VALUE]
      if jnp.dtype(tv_in.dtype) != jnp.dtype(jnp.float32):
        raise TypeError(
            f"aux {types.AUX_TARGET_VALUE!r}: dtype {tv_in.dtype}, the "
            "contract is float32"
        )
      if tuple(tv_in.shape) != (b, p):
        raise ValueError(
            f"aux {types.AUX_TARGET_VALUE!r}: shape {tuple(tv_in.shape)}, "
            f"the contract is {(b, p)}"
        )
      bad = bad | (pev.mine & ~jnp.all(jnp.isfinite(tv_in), axis=1))

    # Env: apply, resolve chance with the hand's key and event counter (on the
    # rows the action left at a chance node; a terminal ply's row is dealt
    # below instead).
    applied = jax.vmap(env.apply)(st.env_state, a_safe)
    if (
        env.max_chance_outcomes == 0
    ):  # no chance nodes (chance.resolve is the identity)
      s2, cc2 = applied, st.chance_count
    else:
      at_chance = (
          ~at_term
          & jax.vmap(env.is_chance)(applied)
          & ~jax.vmap(env.terminal)(applied)
      )
      s2, cc2 = self._resolve_rows(
          at_chance, st.deal_key, st.hand_index, applied, st.chance_count
      )
    done = ~at_term & jax.vmap(env.terminal)(s2)
    rewards = jnp.where(
        done[:, None],
        jax.vmap(env.returns)(s2).astype(jnp.float32),
        jnp.float32(0.0),
    )
    visible = plies.visible_actions(env, st.env_state, a_safe)

    # Terminal ply: deal the next hand (only the rows at their terminal ply).
    hand_index = st.hand_index + at_term.astype(jnp.int64)
    env_state, chance_count = self._deal_rows(
        at_term, st.deal_key, hand_index, s2, cc2
    )
    match_over = at_term & (st.hand_in_match >= self.hands_per_match)
    hand_in_match = jnp.where(
        match_over, jnp.int64(0), st.hand_in_match + done.astype(jnp.int64)
    )
    new_state = ArenaState(
        env_state=env_state,
        carries=tuple(carries),
        deal_key=st.deal_key,
        policy_key=st.policy_key,
        hand_index=hand_index,
        hand_in_match=hand_in_match,
        chance_count=chance_count,
        row_ply=st.row_ply + jnp.int64(1),
        hand_ply=jnp.where(at_term, jnp.int64(0), st.hand_ply + jnp.int64(1)),
        new_hand=at_term,
        new_match=match_over,
        last_action=jnp.where(at_term[:, None], types.NO_ACTION, visible),
        last_actor=jnp.where(at_term, types.NO_ACTOR, actor),
    )

    # Emit from the perspective seats [B, M].
    ps = _as_2d_perspective(table.perspective_seat)
    m = ps.shape[1]
    valid = actor[:, None] == ps

    def bm(x):
      """Broadcasts x [B, ...] over the member axis.

      Args:
        x: A leaf, [B, ...].

      Returns:
        [B, M, ...].
      """
      return jnp.broadcast_to(x[:, None], (b, m) + x.shape[1:])

    aux = {
        k: jnp.where(
            valid.reshape(valid.shape + (1,) * (v.ndim - 1)),
            bm(v),
            jnp.zeros((), v.dtype),
        )
        for k, v in pev.aux.items()
    }
    emits_value = types.AUX_TARGET_VALUE in aux
    tp = aux.pop(types.AUX_TARGET_POLICY, jnp.zeros((b, m, a_dim), jnp.float32))
    tv = aux.pop(types.AUX_TARGET_VALUE, jnp.zeros((b, m, p), jnp.float32))
    assert tp.shape == (b, m, a_dim) and tv.shape == (b, m, p), (
        tp.shape,
        tv.shape,
    )
    labels = dict(
        jax.vmap(lambda s, qs: jax.vmap(lambda q: env.labels(s, q))(qs))(
            st.env_state, ps
        )
    )
    if types.TARGET_VALUE_VALID in labels:
      raise ValueError(
          f"{env.name}.labels emits {types.TARGET_VALUE_VALID!r}: that key "
          "marks "
          "the plies whose target_value a value-target writer filled (types "
          "doc)"
      )
    if emits_value:
      # The perspective slot's AUX_TARGET_VALUE (a teacher's value) is the
      # target where it acted, which makes the Arena this batch's one
      # target_value writer; a later writer refuses (sources.targets).
      labels[types.TARGET_VALUE_VALID] = valid
    record = dict(
        steps=tree_map(lambda x: _take_seats(x, ps), steps),
        action=bm(action),
        behavior_logits=jnp.where(
            valid[..., None], bm(logits), jnp.float32(0.0)
        ),
        behavior_logp=jnp.where(valid, bm(logp), jnp.float32(0.0)),
        value_pred=jnp.where(valid[..., None], bm(value), jnp.float32(0.0)),
        target_policy=tp.astype(jnp.float32),
        target_value=tv.astype(jnp.float32),
        reward=_take_seats(rewards, ps),
        done=bm(done),
        valid=valid,
        labels=labels,
        aux=aux,
    )
    slot = jnp.where(
        at_term, jnp.int32(-1), _take_seat(table.seat_slot, actor_safe)
    )
    member = jnp.where(
        at_term, jnp.int32(-1), _take_seat(table.seat_member, actor_safe)
    )
    log = PlyLog(
        actor=actor,
        action=action,
        rewards=rewards,
        done=done,
        terminal_ply=at_term,
        slot=slot,
        member=member,
        bad=bad,
    )
    return new_state, record, log

  def _ply(self, params, table, st: ArenaState):
    """One ply of every row: each slot observes, evaluates and decides; advance.

    Args:
      params: the per-slot params, indexed by slot.
      table: the SlotTable.
      st: the ArenaState before the ply.

    Returns:
      (new ArenaState, (the ply's record, its PlyLog)).
    """
    steps = self._steps(st)
    carries = [
        self._observe_slot(j, params[j], table, steps, st.carries[j])
        for j in range(len(self.slots))
    ]
    eval_key, decide_key = self._keys(st, steps)
    evals = [
        self._evaluate_slot(
            j,
            params[j],
            table,
            steps,
            carries[j],
            eval_key,
            decide_key,
            st.hand_ply,
        )
        for j in range(len(self.slots))
    ]
    new_state, record, log = self._advance(table, st, steps, carries, evals)
    return new_state, (record, log)

  def _init_carry_of(self, table, carry):
    """The perspective seats' carries as rows.

    Args:
      table: The SlotTable.
      carry: The perspective slot's carry, [B, P, ...].

    Returns:
      The init carry, leaves [B * M, ...].
    """
    ps = _as_2d_perspective(table.perspective_seat)
    c = tree_map(lambda x: _take_seats(x, ps), carry)
    return tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), c)

  def _bootstrap_of(self, table, state):
    """The perspective seats' SeatSteps at `state` as rows.

    Args:
      table: The SlotTable.
      state: The ArenaState.

    Returns:
      The bootstrap steps, [B * M, ...].
    """
    ps = _as_2d_perspective(table.perspective_seat)
    st = tree_map(lambda x: _take_seats(x, ps), self._steps(state))
    return tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), st)

  @staticmethod
  def _assemble(init_carry, rec, bootstrap):
    """Rec leaves [B, T, M, ...] -> TrajectoryBatch [B*M, T, ...].

    Args:
      init_carry: The rows' initial carry.
      rec: The records, leaves [B, T, M, ...].
      bootstrap: The bootstrap SeatSteps.

    Returns:
      The TrajectoryBatch, rows [B * M, T, ...].
    """

    def rows(x):
      """One leaf [B, T, M, ...] as [B*M, T, ...].

      Args:
        x: the leaf.

      Returns:
        the reshaped leaf.
      """
      b, t, m = x.shape[:3]
      return jnp.swapaxes(x, 1, 2).reshape((b * m, t) + x.shape[3:])

    rec = tree_map(rows, rec)
    return types.TrajectoryBatch(
        steps=rec["steps"],
        init_carry=init_carry,
        action=rec["action"],
        behavior_logits=rec["behavior_logits"],
        behavior_logp=rec["behavior_logp"],
        value_pred=rec["value_pred"],
        target_policy=rec["target_policy"],
        target_value=rec["target_value"],
        reward=rec["reward"],
        done=rec["done"],
        valid=rec["valid"],
        labels=rec["labels"],
        aux=rec["aux"],
        bootstrap_step=bootstrap,
    )

  def _rollout_sharded(self, params, table, state, num_steps):
    """_rollout under one shard_map over "batch" (no collective inside).

    Args:
      params: Each slot's params.
      table: The SlotTable.
      state: The ArenaState.
      num_steps: The plies to roll.

    Returns:
      _rollout's (state, batch, log).
    """
    rows, rep = PartitionSpec("batch"), PartitionSpec()
    fn = jax.shard_map(
        functools.partial(self._rollout, num_steps=num_steps),
        mesh=self.mesh,
        in_specs=(rep, rows, rows),
        out_specs=rows,
        check_vma=False,
    )
    return fn(params, table, state)

  def _rollout(self, params, table, state, num_steps):
    """The single-device program over the rows it is given (a device's B/N).

    Args:
      params: Each slot's params.
      table: The SlotTable.
      state: The ArenaState.
      num_steps: The plies to roll.

    Returns:
      (the final ArenaState, the TrajectoryBatch, the logs).
    """
    assert num_steps >= 1
    init_carry = self._init_carry_of(
        table, state.carries[self.perspective_slot]
    )

    def body(st, _):
      """One ply of the scan.

      Args:
        st: the ArenaState before the ply.
        _: unused (the scan's xs).

      Returns:
        (state, (records, PlyLog)) of the ply.
      """
      return self._ply(params, table, st)

    state, (rec, log) = jax.lax.scan(body, state, None, length=num_steps)

    def bt(x):
      """Swaps the time and row axes.

      Args:
        x: A leaf, [T, B, ...].

      Returns:
        The leaf, [B, T, ...].
      """
      return jnp.swapaxes(x, 0, 1)  # [T, B, ...] -> [B, T, ...]

    batch = self._assemble(
        init_carry, tree_map(bt, rec), self._bootstrap_of(table, state)
    )
    log = tree_map(bt, log)
    return state, batch, log


def row_chunks(rows: int) -> tuple[int, ...]:
  """The chunk sizes `on_rows` runs a per-row program at, ascending.

  The last is every row: ceil(R/32), ceil(R/4), ceil(R/2), R (duplicates
  dropped).

  Args:
    rows: R, the row count.

  Returns:
    the chunk sizes, ascending.
  """
  r = int(rows)
  return tuple(sorted({max(1, -(-r // d)) for d in (32, 4, 2)} | {r}))


def on_rows(need, compute, current):
  """`current` with compute's values written at the rows where `need`.

  `current` is a pytree with leading [R] and `need` a bool[R]; compute(rows)
  returns the new values of those rows (rows i32[c], or None = all R rows in
  order).

  Only the needed rows are computed: they are gathered first, in row order,
  into the smallest chunk of `row_chunks(R)` that holds them all, one
  lax.switch branch per chunk size (so exactly one size runs; branch 0, no
  row needed, computes nothing), and scattered back (the chunk's tail past
  the needed count is dropped). compute must be per-row (each row's values
  depend on that row's inputs only, as the Arena's vmapped deal and chance
  programs are). Then a row's result does not depend on which rows share
  its chunk, up to how XLA:CPU rounds a transcendental at different array
  sizes (a vectorized body vs its scalar tail: an ulp), the same caveat as
  the sharded rollout's B/N rows vs B; arena_golden_test pins every shipped
  env and policy family bit for bit.

  Args:
    need: bool[R], the rows to compute.
    compute: rows -> those rows' new values.
    current: the values of every row.

  Returns:
    `current` with the needed rows replaced.
  """
  r = need.shape[0]
  sizes = row_chunks(r)
  n = jnp.sum(need, dtype=jnp.int32)
  order = jnp.argsort(jnp.logical_not(need), stable=True).astype(jnp.int32)

  def chunk(c):
    """The switch branch for chunk size c.

    Args:
      c: the chunk size.

    Returns:
      the branch: needed rows gathered, computed, scattered back.
    """

    def run(cur):
      """Computes the needed rows in one chunk of c rows.

      Args:
        cur: the values of every row.

      Returns:
        cur with the needed rows replaced.
      """
      if c == r:
        return _tree_where(need, compute(None), cur)
      rows = order[:c]
      dst = jnp.where(
          jnp.arange(c, dtype=jnp.int32) < n, rows, jnp.int32(r)
      )  # r: dropped
      return tree_map(
          lambda x, y: x.at[dst].set(y, mode="drop"), cur, compute(rows)
      )

    return run

  branch = jnp.where(
      n == 0,
      jnp.int32(0),
      jnp.int32(1)
      + jnp.sum(jnp.asarray(sizes, jnp.int32) < n, dtype=jnp.int32),
  )
  return jax.lax.switch(
      branch, [lambda cur: cur] + [chunk(c) for c in sizes], current
  )


def hand_chance_key(deal_key, hand_index):
  """The hand chance key of each row (module doc, Keys).

  fold_in(fold_in64(deal_key, hand_index), 0).

  Args:
    deal_key: deal keys [B].
    hand_index: i64[B], each row's hand.

  Returns:
    one chance key per row [B].
  """

  def one(k, h):
    """One row's hand chance key.

    Args:
      k: the row's deal key.
      h: the row's hand index.

    Returns:
      fold_in(fold_in64(k, h), 0).
    """
    return jax.random.fold_in(x64.fold_in64(k, h), 0)

  return jax.vmap(one)(deal_key, hand_index)


def deal_states(env: types.Env, deal_key, hand_index):
  """(env_state [B], chance_count [B]) of hand `hand_index` of each row.

  The Arena's deal, a pure function of (deal_key, hand_index) (also what
  sources.deals hashes for an eval set's deals).

  the Arena's deal, a pure function of (deal_key, hand_index) (also what
  sources.deals hashes for an eval set's deals).

  Args:
    env: the env.
    deal_key: deal keys [B].
    hand_index: i64[B], each row's hand.

  Returns:
    (env_state [B], chance_count [B]).
  """
  ck = hand_chance_key(deal_key, hand_index)
  root = env.root()
  return jax.vmap(lambda k: chance_lib.resolve(env, root, k, 0))(ck)


def check_reserved_aux(aux, rows: int, num_actions: int) -> None:
  """Trace-time check of the registered keys in a slot's aux (core.auxiliary).

  Each present key has its registered dtype and shape [rows, ...], and a key in
  a reserved namespace is registered.

  each present key has its registered dtype and shape [rows, ...], and a key
  in a reserved namespace is registered.

  Raises:
    TypeError: a wrong dtype.
    ValueError: a wrong shape, or an unregistered key in a reserved namespace.

  Args:
    aux: the slot's aux dict.
    rows: the leading dimension each value must have.
    num_actions: A, for keys shaped by the action count.
  """
  for k, v in aux.items():
    spec = aux_lib.get(k)
    if spec is None:
      if aux_lib.is_reserved(k):
        raise ValueError(
            f"aux {k!r} is in a reserved namespace but not registered "
            f"({aux_lib.names()})"
        )
      continue
    want = (rows,) + tuple(num_actions if d == "A" else d for d in spec.shape)
    if jnp.dtype(v.dtype) != jnp.dtype(spec.dtype):
      raise TypeError(
          f"aux {k!r}: dtype {v.dtype}, the contract is {spec.dtype} "
          "(core.auxiliary)"
      )
    if tuple(v.shape) != want:
      raise ValueError(
          f"aux {k!r}: shape {tuple(v.shape)}, the contract is {want} "
          "(core.auxiliary)"
      )


def reserved_aux_finite(aux, rows: int):
  """Whether every float registered aux value of each row is finite.

  Args:
    aux: the slot's aux dict.
    rows: the row count.

  Returns:
    bool[rows].
  """
  ok = jnp.ones((rows,), jnp.bool_)
  for k, v in aux.items():
    if aux_lib.get(k) is not None and jnp.issubdtype(v.dtype, jnp.floating):
      fin = jnp.isfinite(v)
      ok = ok & (
          fin if fin.ndim == 1 else jnp.all(fin, axis=tuple(range(1, fin.ndim)))
      )
  return ok


class BadPlyError(RuntimeError):
  """A rollout had a bad ply (PlyLog.bad; check_log)."""


def check_log(log: PlyLog) -> None:
  """Raises BadPlyError if any ply was bad (one host sync on a scalar).

  Args:
    log: the rollout's PlyLog.

  Raises:
    BadPlyError: a ply was bad.
  """
  n_bad = int(jnp.sum(log.bad, dtype=jnp.int64))
  if n_bad:
    raise BadPlyError(
        f"{n_bad} bad plies: an actor had no legal action, decide chose an "
        "illegal "
        "action, decide's probabilities were not finite (NaN logits), or the "
        "perspective "
        "slot emitted a non-finite registered aux value (core.auxiliary)"
    )


# -- slot-table builders
# ---------------------------------------------------------
def self_play_table(
    batch: int, num_players: int, all_seats: bool = False
) -> SlotTable:
  """A table with every seat in slot 0, member 0.

  all_seats=False: one perspective seat per row, cycling over rows (B rows out).
  all_seats=True: every seat is a perspective seat (B*P rows out), so no
  self-play decision is discarded.

  all_seats=False: one perspective seat per row, cycling over rows (B rows out).
  all_seats=True: every seat is a perspective seat (B*P rows out), so no
  self-play decision is discarded.

  Args:
    batch: B, the row count.
    num_players: P.
    all_seats: make every seat a perspective seat.

  Returns:
    the SlotTable.
  """
  if all_seats:
    persp = np.broadcast_to(
        np.arange(num_players, dtype=np.int32), (batch, num_players)
    ).copy()
  else:
    persp = (np.arange(batch) % num_players).astype(np.int32)
  return SlotTable(
      seat_slot=np.zeros((batch, num_players), np.int32),
      seat_member=np.zeros((batch, num_players), np.int32),
      perspective_seat=persp,
  )


def rotated_table(
    n_deals: int,
    seat_slots: Sequence[int],
    seat_members: Sequence[int],
    perspective: int = 0,
) -> SlotTable:
  """A duplicate-deal table of n_deals x P rows.

  Row (i, r) rotates the line-up by r.

  seat_slots/seat_members give the line-up for rotation 0 (index = seat);
  `perspective` is the line-up position whose seat the batch is emitted for.
  Pair with `duplicate_keys` so the P rotations of deal i share a deal key.

  Args:
    n_deals: the number of deals.
    seat_slots: the slot of each seat in rotation 0.
    seat_members: the member of each seat in rotation 0.
    perspective: the line-up position the batch is emitted for.

  Returns:
    the SlotTable of n_deals x P rows.
  """
  p = len(seat_slots)
  assert len(seat_members) == p and 0 <= perspective < p
  slot = np.zeros((n_deals * p, p), np.int32)
  member = np.zeros((n_deals * p, p), np.int32)
  persp = np.zeros((n_deals * p,), np.int32)
  for i in range(n_deals):
    for r in range(p):
      row = i * p + r
      for pos in range(p):
        seat = (pos + r) % p
        slot[row, seat] = seat_slots[pos]
        member[row, seat] = seat_members[pos]
      persp[row] = (perspective + r) % p
  return SlotTable(slot, member, persp)


def duplicate_keys(key, n_deals: int, num_players: int):
  """(deal_key [n*P], policy_key [n*P, P]) for rotated_table.

  The P rotations of a deal share its deal key and nothing else.

  the P rotations of a deal share its deal key and nothing else.

  Args:
    key: the source key.
    n_deals: the number of deals.
    num_players: P.

  Returns:
    (deal_key [n*P], policy_key [n*P, P]).
  """
  dk, pk = jax.random.split(key)
  deal = jax.random.split(dk, n_deals)
  deal = jnp.repeat(deal, num_players, axis=0)
  pol = jax.random.split(pk, n_deals * num_players * num_players)
  return deal, pol.reshape((n_deals * num_players, num_players) + pol.shape[1:])


def independent_keys(key, batch: int, num_players: int):
  """Deal and policy keys with every row and seat independent.

  Args:
    key: the source key.
    batch: B.
    num_players: P.

  Returns:
    (deal_key [B], policy_key [B, P]).
  """
  dk, pk = jax.random.split(key)
  pol = jax.random.split(pk, batch * num_players)
  return jax.random.split(dk, batch), pol.reshape(
      (batch, num_players) + pol.shape[1:]
  )


class ArenaSource:
  """TrajectorySource over an Arena (or HostArena) and a slot table.

  Stateful: the first `next(key)` resets every row from `key`; later calls
  continue the same hands (a hand longer than T is cut, not restarted;
  `bootstrap_step` carries the step after the cut, and `done` stays False
  there) with fresh policy keys drawn from that
  call's key, so sampling differs per call while each hand's chance sequence
  continues. `set_params` swaps params between calls (on-policy learners);
  `restart()` drops the state so the next call resets.

  `frames`: np.int64 count of Arena plies, B * T per `next`, cumulative
  across restarts. It counts every ply of the batch, including the
  observe-only plies at a hand's end, where the env is not stepped, so it
  is at least the number of env steps, not equal to it.
  `decisions`: np.int64 count of the batches' valid rows (perspective-seat
  decisions, what a learner trains on), cumulative the same way.

  Attributes:
    arena: the Arena (or HostArena).
    params: the slots' params.
    table: the SlotTable.
    num_steps: T, plies per batch.
    batch: B, rows per batch.
    state: the ArenaState the next call continues, or None (the next call
      resets).
    frames: np.int64 count of Arena plies (above).
    decisions: np.int64 count of valid rows (above).
  """

  def __init__(self, arena: Arena, params, table: SlotTable, num_steps: int):
    assert num_steps >= 1
    self.arena, self.params, self.table, self.num_steps = (
        arena,
        params,
        table,
        num_steps,
    )
    self.batch = np.asarray(table.seat_slot).shape[0]
    self.state: Optional[ArenaState] = None
    self.frames = np.int64(0)
    self.decisions = np.int64(0)

  def set_params(self, params) -> None:
    """Validates and sets the params of the next batches.

    Args:
      params: one params entry per slot.
    """
    self.arena.validate(params, self.table)
    self.params = params

  def restart(self) -> None:
    """Drops the state so the next call resets."""
    self.state = None

  def next(self, key) -> types.TrajectoryBatch:
    """Rolls out num_steps plies from the kept state (reset if none).

    Args:
      key: this batch's key.

    Returns:
      the TrajectoryBatch.
    """
    dk, pk = independent_keys(key, self.batch, self.arena.env.num_players)
    if self.state is None:
      state = self.arena.reset(self.params, self.table, dk, pk)
    else:
      state = self.state._replace(policy_key=pk)
    self.state, batch, _ = self.arena.rollout(
        self.params, self.table, state, self.num_steps
    )
    assert isinstance(self.frames, np.int64), type(self.frames)
    self.frames = np.int64(
        self.frames + np.int64(self.batch) * np.int64(self.num_steps)
    )
    assert isinstance(self.decisions, np.int64), type(self.decisions)
    self.decisions = np.int64(
        self.decisions + np.int64(jnp.sum(batch.valid, dtype=jnp.int64))
    )
    return batch
