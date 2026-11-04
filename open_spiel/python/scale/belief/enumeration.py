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

"""The exact Bayesian posterior over hidden worlds, by enumerating the tree.

For a small imperfect-information game the whole tree fits in memory, so the
posterior over the histories of an information set can be computed exactly
instead of approximated. Given the opponent's behaviour as a policy table,
the weight of history h in the searching seat's information set is

  P(h | infoset) ∝ chance_reach(h) * prod over the opponents' decisions on h
                   of table[opponent infoset, action]

The seat's own action probabilities are the same for every history of its
information set under perfect recall, so they cancel.

  sampler = registry.build("enumeration", env)
  params = sampler.table_for(opponent, opponent_params, types.SAMPLE)
  carry = sampler.init_carry(params, batch, seat, key)
  worlds, logw = sampler.sample_worlds(params, carry, step.info, k, key)

The tree is built once on the host (`core.game_tree.GameTree`) and turned
into dense arrays, so `sample_worlds` is jittable: it looks the information
set up by exact equality of the flattened InfoState, weights its histories
and draws K of them. Draws are exact samples, so the log-weights are 0.

Known limitations: the tree must have at most `max_nodes` nodes (larger games
raise TooLargeError; use the particle sampler). A row whose InfoState is not a
decision information set of the carry's seat (for example a row where the
seat is not to move) gets the root state with log-weight -inf; use
`infoset_index` to check rows explicitly.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.belief import registry
from open_spiel.python.scale.core import game_tree as game_tree_lib
from open_spiel.python.scale.core import types

DEFAULT_MAX_NODES = game_tree_lib.DEFAULT_MAX_NODES
F32, F64, I32 = jnp.float32, jnp.float64, jnp.int32


def _flat_info(info: Any, lead: int) -> np.ndarray:
  """f64[lead, F]: every InfoState leaf flattened and concatenated.

  The InfoState leaves are bools, small integers and f32, all exactly
  representable in f64, so equality of the flattened rows is equality of the
  InfoStates.

  Args:
    info: the InfoState, leaves [lead, ...].
    lead: the number of rows.

  Returns:
    f64[lead, F].
  """
  leaves = jax.tree_util.tree_leaves(info)
  return np.concatenate(
      [np.asarray(x).reshape(lead, -1).astype(np.float64) for x in leaves],
      axis=1,
  )


class EnumerationSampler:
  """Exact posterior over the histories of an information set (module doc).

  Attributes:
    env: the Env.
    tree: the enumerated GameTree.
    kwargs: the sampler's constructor kwargs, as recorded.
    num_infosets: I, the number of decision information sets (all seats).
    num_actions: A, the game's number of actions.
  """

  capability = types.Capability.NATIVE_JAX
  stateful = False
  is_perfect_information = False
  params_source = "opponent"

  def __init__(self, env: types.Env, *, max_nodes: int = DEFAULT_MAX_NODES):
    """Enumerates the tree.

    Args:
      env: a small imperfect-information Env with perfect recall.
      max_nodes: the most tree nodes to enumerate.

    Raises:
      TooLargeError: the tree has more than `max_nodes` nodes.
    """
    tree = game_tree_lib.GameTree(env, int(max_nodes))
    self.env, self.tree = env, tree
    self.kwargs = {"max_nodes": int(max_nodes)}
    self.num_infosets = int(tree.num_infosets)
    self.num_actions = int(tree.num_actions)
    n, a, p = tree.num_nodes, self.num_actions, int(tree.num_players)

    # chance_reach[n]: the product of the chance probabilities on n's path.
    reach = np.ones(n)
    for node in range(1, n):  # BFS order: a parent precedes its children
      par = tree.parent[node]
      reach[node] = reach[par] * (
          tree.edge_p[node] if tree.in_chance[node] else 1.0
      )

    # opp_edges[seat, n, :]: flat table indices (infoset * A + action) of the
    # decisions of seats other than `seat` on n's path; -1 padded.
    paths: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for node in range(1, n):
      par = int(tree.parent[node])
      paths[node] = list(paths[par])
      if not tree.in_chance[node]:
        paths[node].append((
            int(tree.actor[par]),
            int(tree.infoset[par]) * a + int(tree.in_action[node]),
        ))
    depth = max(1, max(len(x) for x in paths))
    opp = np.full((p, n, depth), -1, np.int64)
    for node, path in enumerate(paths):
      for seat in range(p):
        idx = []
        for who, e in path:
          if who != seat:
            idx.append(e)
        opp[seat, node, : len(idx)] = idx

    width = max(len(m) for m in tree.members)
    members = np.full((self.num_infosets, width), -1, np.int64)
    for row, m in enumerate(tree.members):
      members[row, : len(m)] = m

    self._reach = jnp.asarray(reach, F64)
    self._opp_edges = jnp.asarray(opp, I32)
    self._members = jnp.asarray(members, I32)
    self._infoset_player = jnp.asarray(tree.player, I32)
    self._keys = jnp.asarray(_flat_info(tree.info, self.num_infosets), F64)
    states = tree.node_states()
    self._states = jax.tree_util.tree_map(jnp.asarray, states)

  # -- params ------------------------------------------------------------------
  def init_params(self, key: Any = None) -> jax.Array:
    """The uniform policy over each information set's legal actions.

    Args:
      key: unused (the table is deterministic).

    Returns:
      f32[I, A], the uniform opponent table.
    """
    del key
    return jnp.asarray(self.tree.uniform_table(), F32)

  def table_for(
      self,
      policy: types.Policy,
      params: Any,
      rule: types.DecisionRule = types.SAMPLE,
      key: Any = None,
  ) -> jax.Array:
    """`policy`'s action distribution at every information set.

    This is the params of the sampler when that policy is the opponent model.
    `policy` must be an in-graph JAX policy (GameTree.policy_table).

    Args:
      policy: the opponent model.
      params: the policy's params.
      rule: the decision rule that turns its logits into a distribution.
      key: the PRNG key of a stochastic policy, or None.

    Returns:
      f32[I, A], the opponent table.
    """
    return jnp.asarray(
        self.tree.policy_table(policy, params, rule, key=key), F32
    )

  # -- carry -------------------------------------------------------------------
  def init_carry(self, params, batch, seat, key) -> jax.Array:
    """The searching seat of each row.

    The posterior depends only on the current information set, so the carry
    holds nothing else.

    Args:
      params: unused.
      batch: the number of rows B.
      seat: the searching seat, a scalar or i32[B].
      key: unused.

    Returns:
      i32[B], the seat of each row.
    """
    del params, key
    return jnp.broadcast_to(jnp.asarray(seat, I32), (batch,))

  def observe(self, params, carry, step):
    """The carry unchanged: the posterior needs no history.

    Args:
      params: unused.
      carry: the carry.
      step: unused.

    Returns:
      `carry`.
    """
    del params, step
    return carry

  # -- the posterior -----------------------------------------------------------
  def infoset_index(self, info: Any, seat: jax.Array) -> jax.Array:
    """The row of each InfoState among the seat's decision information sets.

    Args:
      info: the InfoStates, leaves [B, ...].
      seat: i32[B], the seat of each row.

    Returns:
      i32[B], the information set row, -1 where it is none of them.
    """
    b = jnp.shape(jnp.asarray(seat))[0]
    leaves = jax.tree_util.tree_leaves(info)
    flat = jnp.concatenate(
        [jnp.reshape(x, (b, -1)).astype(F64) for x in leaves], axis=1
    )
    hit = jnp.all(flat[:, None, :] == self._keys[None, :, :], axis=-1)
    hit = hit & (
        self._infoset_player[None, :] == jnp.asarray(seat, I32)[:, None]
    )
    return jnp.where(jnp.any(hit, axis=1), jnp.argmax(hit, axis=1), -1).astype(
        I32
    )

  def log_posterior(
      self, params: jax.Array, seat: jax.Array, info: Any
  ) -> jax.Array:
    """Log P(member m | infoset) over the members of each row's information set.

    Args:
      params: f32[I, A], the opponent table.
      seat: i32[B], the seat of each row.
      info: the InfoStates, leaves [B, ...].

    Returns:
      f64[B, M] over `members_of`; -inf on padding and on unmatched rows.
    """
    table = jnp.asarray(params, F64).reshape(-1)
    seat = jnp.asarray(seat, I32)
    row = self.infoset_index(info, seat)
    nodes = self._members[jnp.maximum(row, 0)]  # [B, M]
    edges = self._opp_edges[seat[:, None], jnp.maximum(nodes, 0)]  # [B, M, D]
    probs = jnp.where(edges >= 0, table[jnp.maximum(edges, 0)], F64(1.0))
    logw = jnp.log(self._reach[jnp.maximum(nodes, 0)]) + jnp.sum(
        jnp.log(probs), axis=-1
    )
    live = (nodes >= 0) & (row >= 0)[:, None]
    return jnp.where(live, logw, -jnp.inf)

  def members_of(self, info: Any, seat: jax.Array) -> jax.Array:
    """The tree nodes of each row's information set.

    Args:
      info: the InfoStates, leaves [B, ...].
      seat: the seat of each row.

    Returns:
      i32[B, M], node indices, -1 padded.
    """
    row = self.infoset_index(info, jnp.asarray(seat, I32))
    return jnp.where(
        (row >= 0)[:, None], self._members[jnp.maximum(row, 0)], -1
    )

  def sample_worlds(self, params, carry, info, k: int, key):
    """Draws K worlds per row from the exact posterior.

    Each world is an exact draw from the posterior, so its log-weight is 0;
    rows without a matching information set get the root state and -inf.

    Args:
      params: f32[I, A], the opponent table.
      carry: i32[B], the seat of each row.
      info: the InfoStates, leaves [B, ...].
      k: worlds per row.
      key: a PRNG key.

    Returns:
      (env states with leaves [B, K, ...], log-weights f64[B, K]).

    Raises:
      ValueError: `k` < 1.
    """
    if k < 1:
      raise ValueError(f"k must be >= 1, got {k}")
    seat = jnp.asarray(carry, I32)
    logp = self.log_posterior(params, seat, info)
    nodes = self.members_of(info, seat)
    b = logp.shape[0]
    found = jnp.any(jnp.isfinite(logp), axis=1)
    safe = jnp.where(found[:, None], logp, F64(0.0))
    keys = jax.random.split(key, b)
    pick = jax.vmap(lambda kk, lp: jax.random.categorical(kk, lp, shape=(k,)))(
        keys, safe
    )
    node = jnp.take_along_axis(nodes, pick.astype(I32), axis=1)
    node = jnp.where(found[:, None], jnp.maximum(node, 0), 0)
    worlds = jax.tree_util.tree_map(lambda x: x[node], self._states)
    logw = jnp.broadcast_to(
        jnp.where(found, F64(0.0), -jnp.inf)[:, None], (b, k)
    )
    return worlds, logw


registry.register("enumeration", EnumerationSampler)
