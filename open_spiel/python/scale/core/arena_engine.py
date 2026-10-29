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

"""ArenaEngine: the Evaluator's default MatchEngine, over the Arena.

Evaluation plays through the same Arena as training, so a policy is scored
exactly as it plays. `Arena` runs when every slot is in-graph (NATIVE_JAX or
CALLBACK_BATCHED), `HostArena` when one is HOST_DRIVEN; one Arena per (slot
layout, perspective) is built and cached. In-graph slots stack their
members' params [K, ...]; a HOST_DRIVEN slot takes the list of its K host
objects.

  engine = arena_engine.ArenaEngine(env, hands_per_match=1)
  ev = match.Evaluator(env, engine=engine)

Keys: deal i of a job is k_i = fold_in64(job key, i); duplicate deals give
both rows of k_i one deal key and separate policy keys
(core.arena.duplicate_keys), otherwise each row gets its own deal
(core.arena.independent_keys). A numpy uint32[2] job key (core.rng) is
wrapped as a typed threefry key with the same bits.

Ply segments: a rollout materialises its TrajectoryBatch, roughly rows x
plies x num_actions values several times over (behaviour logits, target
policy, legal masks), while the Evaluator keeps only rewards and done. A
long match in a game with many actions (512 rows x 1001 plies x 8192
actions is about 66 GiB at BYTES_PER_ACTION) therefore rolls in segments of
segment_plies() plies, each continuing from the returned ArenaState, keeps
only (rewards, done) of each segment, and stops once every row has finished
its hands. Per-ply keys are functions of (policy_key, row_ply) in the state,
so the result equals one rollout of num_steps plies.

While an eval deadline is set (`core.deadline`), segments are at most
deadline.SEGMENT_PLIES plies and the deadline is checked before each, so an
eval stuck in a long match raises EvalOverBudgetError instead of running on.
"""

from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import deadline as deadline_lib
from open_spiel.python.scale.core import host_arena
from open_spiel.python.scale.core import rng
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64

tree_map = jax.tree_util.tree_map

#: The TrajectoryBatch bytes one rollout segment may materialise.
LOG_BUDGET_BYTES = 1 << 30
#: Estimated bytes per (row, ply) of a TrajectoryBatch, per action, plus a
#: fixed part for the observation, value and labels (an over-estimate:
#: f32 behaviour logits and target policy, the bool legal mask per seat).
BYTES_PER_ACTION = 16
BYTES_FIXED = 8192


def segment_plies(
    rows: int, num_steps: int, num_actions: int, budget: int = LOG_BUDGET_BYTES
) -> int:
  """Plies per rollout segment under a log byte budget.

  Plies per rollout segment: num_steps when the whole log fits `budget`,
  else the most plies that fit (at least 1).

  Args:
    rows: rows in the rollout batch.
    num_steps: plies of the whole rollout.
    num_actions: actions per ply.
    budget: the log byte budget of one segment.

  Returns:
    num_steps when the whole log fits `budget`, else the most plies that fit (at
    least 1).
  """
  assert rows >= 1 and num_steps >= 1 and num_actions >= 1, (
      rows,
      num_steps,
      num_actions,
  )
  per_ply = int(rows) * (BYTES_PER_ACTION * int(num_actions) + BYTES_FIXED)
  return int(max(1, min(int(num_steps), int(budget) // per_ply)))


def stack_members(members: Sequence[Any]):
  """In-graph members' params stacked [K, ...] (one tree, shapes and dtypes).

  Args:
    members: in-graph members' params, one tree each.

  Returns:
    one tree whose leaves are the members' leaves stacked on a new axis 0.
  """
  s0 = jax.tree_util.tree_structure(members[0])
  sig0 = [
      (jnp.shape(x), jnp.result_type(x))
      for x in jax.tree_util.tree_leaves(members[0])
  ]
  for m in members[1:]:
    assert (
        jax.tree_util.tree_structure(m) == s0
    ), "slot members have different param trees"
    sig = [
        (jnp.shape(x), jnp.result_type(x)) for x in jax.tree_util.tree_leaves(m)
    ]
    assert (
        sig == sig0
    ), f"slot members differ in param shapes/dtypes: {sig} vs {sig0}"
  return tree_map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *members)


def deal_keys(key, n: int, duplicate: bool):
  """(deal_key [2n], policy_key [2n, 2]) for rows (i, r), i < n, r in {0, 1}.

  Args:
    key: the job's key.
    n: deal pairs.
    duplicate: whether the two rows of a pair share the deal key.

  Returns:
    (deal_key [2n], policy_key [2n, 2]).
  """
  key = rng.to_jax_key(key)
  ks = jax.vmap(lambda i: x64.fold_in64(key, i))(jnp.arange(n, dtype=jnp.int64))
  if duplicate:
    deal, pol = jax.vmap(lambda k: arena_lib.duplicate_keys(k, 1, 2))(ks)
  else:
    deal, pol = jax.vmap(lambda k: arena_lib.independent_keys(k, 2, 2))(ks)
  return deal.reshape((2 * n,)), pol.reshape((2 * n, 2))


class ArenaEngine:
  """eval.payoff.MatchEngine over the Arena / HostArena (module doc).

  Attributes:
    env: the Env.
    hands_per_match: hands per match.
    mesh: the ("batch",) mesh rows shard over (default every device).
    log_budget_bytes: the TrajectoryBatch bytes one rollout segment may
      materialise.
  """

  def __init__(
      self,
      env: types.Env,
      hands_per_match: int = 1,
      mesh=None,
      log_budget_bytes: int = LOG_BUDGET_BYTES,
  ):
    x64.assert_x64("Evaluator")
    assert log_budget_bytes >= 1, log_budget_bytes
    self.env, self.hands_per_match, self.mesh = env, int(hands_per_match), mesh
    self.log_budget_bytes = int(log_budget_bytes)
    self._arenas = {}

  def row_unit(self) -> int:
    """Rows are dealt in multiples of this: two per device.

    Returns:
      2 * the number of devices.
    """
    ndev = (
        self.mesh.devices.size if self.mesh is not None else len(jax.devices())
    )
    return 2 * ndev

  def stack(self, slot: arena_lib.Slot, members: Sequence[Any]) -> Any:
    """A slot's members as the arena takes them.

    Args:
      slot: the slot.
      members: its members' params.

    Returns:
      the params stacked [K, ...] for an in-graph slot, else a list.
    """
    return stack_members(members) if slot.in_graph else list(members)

  def job_keys(
      self, keys: Sequence[Any], n: int, duplicate: bool
  ) -> tuple[Any, Any]:
    """The deal and policy keys of every job's rows, concatenated.

    Args:
      keys: one key per job.
      n: deal pairs per job.
      duplicate: whether the two rows of a pair share the deal.

    Returns:
      (deal keys, policy keys) over all jobs' rows.
    """
    dks, pks = zip(*(deal_keys(k, n, duplicate) for k in keys))
    return jnp.concatenate(dks), jnp.concatenate(pks)

  def _arena(self, slot_list, cache_key, perspective):
    """The compiled arena for `cache_key` and `perspective`, built once.

    A HostArena when any slot runs outside the graph, else an Arena.

    Args:
      slot_list: the arena's slots.
      cache_key: the key of the compiled arena.
      perspective: the seat the batch is emitted for.

    Returns:
      the cached arena.
    """
    ak = (cache_key, perspective)
    if ak not in self._arenas:
      cls = (
          host_arena.HostArena
          if not all(s.in_graph for s in slot_list)
          else arena_lib.Arena
      )
      self._arenas[ak] = cls(
          self.env,
          slot_list,
          perspective_slot=perspective,
          hands_per_match=self.hands_per_match,
          mesh=self.mesh,
      )
    return self._arenas[ak]

  def play(
      self,
      slot_list,
      params,
      table,
      deal_key,
      policy_key,
      num_steps: int,
      *,
      perspective: int,
      cache_key: Any,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Plays the rows of `table` for num_steps plies.

    Args:
      slot_list: the arena's slots.
      params: each slot's stacked params.
      table: the slot table of the rows.
      deal_key: per-row deal keys.
      policy_key: per-row policy keys.
      num_steps: plies to play.
      perspective: the seat the batch is emitted for.
      cache_key: the key of the compiled arena.

    Returns:
      (rewards f64[B, T, P], done bool[B, T]) on the host.
    """
    ar = self._arena(slot_list, cache_key, perspective)
    st = ar.reset(params, table, deal_key, policy_key)
    rows = int(np.asarray(table.seat_slot).shape[0])
    seg = segment_plies(
        rows, num_steps, self.env.num_actions, self.log_budget_bytes
    )
    if deadline_lib.active():
      seg = min(seg, deadline_lib.SEGMENT_PLIES)
      deadline_lib.check(f"{self.env.name} eval rollout")
    if seg >= num_steps:
      _, _, log = ar.rollout(params, table, st, num_steps)
      return np.asarray(log.rewards, np.float64), np.asarray(log.done)
    # Segments of one length (one compile); the last may overshoot num_steps
    # and is trimmed, so a row that needs more than num_steps plies still
    # makes payoff.payoffs raise.
    hands = self.hands_per_match
    rewards, dones, n_done, played = [], [], np.zeros((rows,), np.int64), 0
    while played < num_steps:
      deadline_lib.check(
          f"{self.env.name} eval rollout at ply {played} of {num_steps}"
      )
      st, _, log = ar.rollout(params, table, st, seg)
      d = np.asarray(log.done)[:, : num_steps - played]
      rewards.append(
          np.asarray(log.rewards, np.float64)[:, : num_steps - played]
      )
      dones.append(d)
      n_done += d.sum(1)
      played += d.shape[1]
      if (n_done >= hands).all():
        break
    # Stopped early: every row has its hands, the plies not played count
    # nothing.
    return np.concatenate(rewards, 1), np.concatenate(dones, 1)
