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

"""HostArena: the Arena driven from a Python loop, for HOST_DRIVEN policies.

Some policies cannot run inside a JAX graph: a model in another framework, a
search engine with its own process, anything that needs Python per call. A
HostArena plays them against JAX policies with the Arena's semantics: the
same SlotTable, Slot, keys, SeatSteps, decide and env stepping (the jitted
building blocks are shared, not copied); only the loop over plies is Python.

  arena = HostArena(env, [Slot(host_policy, num_members=2), Slot(jax_policy)])
  state, batch, log = arena.collect(params, table, deal_key, policy_key,
                                    num_steps=16)

In-graph slots (NATIVE_JAX, CALLBACK_BATCHED) run through the Arena's jitted
per-slot observe and evaluate, so a HostArena with only in-graph slots
reproduces the Arena exactly. HOST_DRIVEN slots are called on the host with
numpy SeatSteps and raw uint32[..., 2] keys, grouped by member, so a host
policy only ever sees the seats it owns. Their params are a sequence of K
per-member host objects. Their logits go through the jitted `decide`, after
the slot's TemperatureDrop if it has one, as for in-graph slots.

A HOST_DRIVEN slot's carry is a host pytree with leading [B*P] (row-major
(row, seat)); rows of seats it does not own stay zero.
"""

import functools
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import plies
from open_spiel.python.scale.core import types
from open_spiel.python.scale.interop import host

tree_map = jax.tree_util.tree_map


def _host_decide(
    slot: arena_lib.Slot, logits, legal_mask, decide_key, hand_ply
):
  """A HOST_DRIVEN slot's decision.

  its TemperatureDrop (arena.tempered, f64) on the row's hand ply, then
  decide_rows, as Arena._evaluate_slot.

  Args:
    slot: the HOST_DRIVEN slot.
    logits: the rows' logits [B, A].
    legal_mask: bool[B, A].
    decide_key: one key per row [B].
    hand_ply: i64[B], each row's hand ply.

  Returns:
    (action i32[B], logp f32[B], finite bool[B]), as decide_rows.
  """
  rule, dlogits = arena_lib.tempered(slot, logits, hand_ply)
  return arena_lib.decide_rows(rule, dlogits, legal_mask, decide_key)


def _fill_missing_keys(recs) -> None:
  """Fills the aux keys a ply's record lacks with zeros, in place.

  A HOST_DRIVEN perspective slot that acts on no row at a ply returns no aux
  there (it is not called), so that ply's record lacks the aux keys (and any
  label the Arena writes from them) the other plies have. They are zeros,
  exactly what the Arena emits off the slot's rows; fill them so the plies stack
  (in place, per record dict).

  Args:
    recs: the plies' record dicts.
  """
  for field in ("aux", "labels"):
    keys = {}
    for r in recs:
      for k, v in r[field].items():
        keys.setdefault(k, v)
    for r in recs:
      for k, like in keys.items():
        if k not in r[field]:
          r[field][k] = jnp.zeros_like(like)


class HostArena(arena_lib.Arena):
  """The Python-loop rollout driver; it takes slots of every Capability.

  Arguments are Arena's. HOST_DRIVEN slots' params are a sequence of K
  per-member host objects.
  """

  accepted = frozenset(types.Capability)

  def __init__(
      self,
      env: types.Env,
      slots: Sequence[arena_lib.Slot],
      perspective_slot: int = 0,
      hands_per_match: int = 1,
      mesh: Optional[Mesh] = None,
  ):
    super().__init__(env, slots, perspective_slot, hands_per_match, mesh)
    self._steps_jit = jax.jit(self._steps)
    self._keys_jit = jax.jit(self._keys)
    self._advance_jit = jax.jit(self._advance)
    self._obs_jit = {
        j: jax.jit(functools.partial(self._observe_slot, j))
        for j, sl in enumerate(self.slots)
        if sl.in_graph
    }
    self._eval_jit = {
        j: jax.jit(functools.partial(self._evaluate_slot, j))
        for j, sl in enumerate(self.slots)
        if sl.in_graph
    }
    self._decide_jit = {
        j: jax.jit(functools.partial(_host_decide, sl))
        for j, sl in enumerate(self.slots)
        if not sl.in_graph
    }
    self._init_carry_jit = jax.jit(self._init_carry_of)
    self._bootstrap_jit = jax.jit(self._bootstrap_of)

  def _in_graph_params(self, params):
    """`params` with every host slot's entry replaced by ().

    Args:
      params: Each slot's params.

    Returns:
      The params, host slots' entries ().
    """
    return [p if sl.in_graph else () for sl, p in zip(self.slots, params)]

  def _jax_state(self, st: arena_lib.ArenaState) -> arena_lib.ArenaState:
    """`st` with the host slots' carries replaced by ().

    Args:
      st: The ArenaState.

    Returns:
      The ArenaState.
    """
    return st._replace(
        carries=tuple(
            c if sl.in_graph else () for sl, c in zip(self.slots, st.carries)
        )
    )

  # -- host slot calls ---------------------------------------------------------
  def _groups(self, j, owned_flat, member_flat):
    sl = self.slots[j]
    for m in range(sl.num_members):
      idx = np.flatnonzero(owned_flat & (member_flat == m))
      if idx.size:
        yield m, idx

  def _host_init(self, j, params_j, table_np, policy_key):
    """Slot j's carry on the host: init_carry per member over its owned seats.

    Args:
      j: The slot index.
      params_j: The slot's params.
      table_np: The SlotTable (numpy).
      policy_key: The policy keys.

    Returns:
      The carry, leaves [B * P, ...].
    """
    sl = self.slots[j]
    b, p = table_np.seat_slot.shape
    seats = np.broadcast_to(np.arange(p, dtype=np.int32), (b, p)).reshape(-1)
    keys = host.raw_keys(
        jax.vmap(jax.vmap(plies.init_key))(policy_key)
    ).reshape(b * p, -1)
    owned = table_np.seat_slot.reshape(-1) == j
    member = table_np.seat_member.reshape(-1)
    carry = None
    for m, idx in self._groups(j, owned, member):
      part = sl.policy.init_carry(params_j[m], idx.size, seats[idx], keys[idx])
      if carry is None:
        carry = host.alloc_rows(part, b * p)
      carry = host.put(carry, idx, part)
    return carry

  def _host_observe(self, j, params_j, table_np, steps_np, carry):
    """Slot j observes on the host, per member over its owned seats.

    Args:
      j: The slot index.
      params_j: The slot's params.
      table_np: The SlotTable (numpy).
      steps_np: The SeatSteps (numpy).
      carry: The slot's carry.

    Returns:
      The updated carry.
    """
    sl = self.slots[j]
    b, p = table_np.seat_slot.shape
    flat = tree_map(lambda x: x.reshape((b * p,) + x.shape[2:]), steps_np)
    owned = table_np.seat_slot.reshape(-1) == j
    for m, idx in self._groups(j, owned, table_np.seat_member.reshape(-1)):
      part = sl.policy.observe(
          params_j[m], host.take(carry, idx), host.take(flat, idx)
      )
      carry = host.put(carry, idx, part)
    return carry

  def _host_evaluate(
      self,
      j,
      params_j,
      table_np,
      steps_np,
      carry,
      eval_key,
      decide_key,
      hand_ply,
  ):
    """Slot j evaluates and decides on the host for the rows it acts on.

    Args:
      j: The slot index.
      params_j: The slot's params.
      table_np: The SlotTable (numpy).
      steps_np: The SeatSteps (numpy).
      carry: The slot's carry.
      eval_key: The eval keys.
      decide_key: The decide keys.
      hand_ply: Each row's ply in its hand.

    Returns:
      The slot's SlotEval.
    """
    sl = self.slots[j]
    b, p = table_np.seat_slot.shape
    a_dim = self.env.num_actions
    rows = np.arange(b)
    actor = steps_np.to_move[:, 0]
    actor_safe = np.maximum(actor, 0)
    mine = (actor >= 0) & (table_np.seat_slot[rows, actor_safe] == j)
    member = table_np.seat_member[rows, actor_safe]
    actor_step = tree_map(lambda x: x[rows, actor_safe], steps_np)
    raw_eval = host.raw_keys(eval_key)
    logits = np.zeros((b, a_dim), np.float32)
    value = np.zeros((b, p), np.float32)
    aux = None
    for m, idx in self._groups(j, mine, member):
      out = sl.policy.evaluate(
          params_j[m],
          host.take(carry, idx * p + actor_safe[idx]),
          host.take(actor_step, idx),
          raw_eval[idx],
      )
      assert isinstance(out, types.PolicyOutput), type(out)
      out = host.to_numpy(out)
      assert out.logits.shape == (idx.size, a_dim), out.logits.shape
      assert out.value.shape == (idx.size, p), out.value.shape
      logits[idx] = out.logits.astype(np.float32)
      value[idx] = out.value.astype(np.float32)
      if aux is None:
        aux = host.alloc_rows(dict(out.aux), b)
      aux = host.put(aux, idx, dict(out.aux))
    a, lp, fin = self._decide_jit[j](
        jnp.asarray(logits),
        jnp.asarray(actor_step.legal_mask),
        decide_key,
        hand_ply,
    )
    return arena_lib.SlotEval(
        jnp.asarray(mine),
        a,
        jnp.asarray(logits),
        jnp.asarray(value),
        lp,
        fin,
        tree_map(jnp.asarray, aux or {}),
    )

  # -- driver ------------------------------------------------------------------
  def reset(self, params, table: arena_lib.SlotTable, deal_key, policy_key):
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
    st = self._reset_jit(
        self._replicate(self._in_graph_params(params)),
        self._shard_rows(table),
        self._shard_rows(deal_key),
        self._shard_rows(policy_key),
    )
    table_np = arena_lib.SlotTable(*(np.asarray(x) for x in table))
    carries = list(st.carries)
    for j, sl in enumerate(self.slots):
      if not sl.in_graph:
        carries[j] = self._host_init(j, params[j], table_np, policy_key)
    return st._replace(carries=tuple(carries))

  def rollout(
      self,
      params,
      table: arena_lib.SlotTable,
      state: arena_lib.ArenaState,
      num_steps: int,
  ):
    """Runs num_steps plies from `state`, the host driving each ply.

    Args:
      params: one params entry per slot.
      table: the slot table.
      state: the ArenaState to continue from.
      num_steps: plies to run.

    Returns:
      (state, TrajectoryBatch, PlyLog).
    """
    assert num_steps >= 1
    self.validate(params, table)
    jparams = self._replicate(self._in_graph_params(params))
    jtable = self._shard_rows(table)
    table_np = arena_lib.SlotTable(*(np.asarray(x) for x in table))
    ps2 = np.asarray(table_np.perspective_seat)
    ps2 = ps2[:, None] if ps2.ndim == 1 else ps2
    p = self.env.num_players
    j0 = self.perspective_slot
    if self.slots[j0].in_graph:
      init_carry = self._init_carry_jit(jtable, state.carries[j0])
    else:
      rows = np.arange(ps2.shape[0])[:, None]
      init_carry = host.take(state.carries[j0], (rows * p + ps2).reshape(-1))

    st = state
    recs, logs = [], []
    for _ in range(num_steps):
      jst = self._jax_state(st)
      steps = self._steps_jit(jst)
      steps_np = None
      if not all(sl.in_graph for sl in self.slots):
        steps_np = host.to_numpy(steps)
      carries = list(st.carries)
      for j, sl in enumerate(self.slots):
        if sl.in_graph:
          carries[j] = self._obs_jit[j](
              jparams[j], jtable, steps, st.carries[j]
          )
        elif st.carries[j] is not None:
          carries[j] = self._host_observe(
              j, params[j], table_np, steps_np, st.carries[j]
          )
      eval_key, decide_key = self._keys_jit(jst, steps)
      evals = []
      for j, sl in enumerate(self.slots):
        if sl.in_graph:
          evals.append(
              self._eval_jit[j](
                  jparams[j],
                  jtable,
                  steps,
                  carries[j],
                  eval_key,
                  decide_key,
                  jst.hand_ply,
              )
          )
        else:
          evals.append(
              self._host_evaluate(
                  j,
                  params[j],
                  table_np,
                  steps_np,
                  carries[j],
                  eval_key,
                  decide_key,
                  jst.hand_ply,
              )
          )
      jcarries = [
          c if sl.in_graph else () for sl, c in zip(self.slots, carries)
      ]
      new_jst, rec, log = self._advance_jit(jtable, jst, steps, jcarries, evals)
      st = new_jst._replace(carries=tuple(carries))
      recs.append(rec)
      logs.append(log)

    _fill_missing_keys(recs)

    def stack(*xs):
      """Stacks the plies on the time axis.

      Args:
        *xs: The plies' leaves, each [B, ...].

      Returns:
        [B, T, ...].
      """
      return jnp.stack(xs, axis=1)  # -> [B, T, ...]

    rec = tree_map(stack, *recs)
    log = tree_map(stack, *logs)
    bootstrap = self._bootstrap_jit(jtable, self._jax_state(st))
    batch = self._assemble(init_carry, rec, bootstrap)
    arena_lib.check_log(log)
    return st, batch, log
