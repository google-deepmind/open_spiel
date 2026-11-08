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

"""The full game tree of a small Env, in flat numpy arrays.

Exact evaluation of a policy (expected returns, best responses, NashConv) and
exact Bayesian belief over hidden state both need every history of the game
and the information set each one belongs to. GameTree expands an Env from
its root, level by level with a jitted and vmapped expansion, and stores the
result as flat f64 / int arrays in breadth-first order. It needs only the
Env's raw layer (`root`, `apply`, `is_chance`, `apply_chance`,
`chance_probs`) and `info_state`, so any small Env works without pyspiel.

  tree = game_tree.GameTree(kuhn.KuhnEnv())
  table = tree.policy_table(policy, params, types.SAMPLE)   # f64[I, A]
  tree.expected_returns([table, table])                     # f64[P]
  tree.best_response_value(table, player=0)
  tree.nash_conv(table)

Concepts:
  * Nodes are numbered in breadth-first order; node 0 is the root. A node is
    TERMINAL, CHANCE or DECISION (`kind`).
  * An infoset is the set of decision nodes whose actor and InfoState are
    byte-identical. Infosets are numbered globally (`infoset`, rows of a
    policy table); `seat_infosets(seat)` lists one seat's rows, which the
    seat-local methods index.
  * A policy table is f64[num_infosets, A]: each row a distribution over the
    infoset's legal actions.
  * Chance probabilities are the Env's f32 `chance_probs` renormalised in f64,
    which is exact for equal-probability outcomes.

The tree assumes perfect recall and does not check it; best_response_value
is correct only for perfect-recall games. Construction checks only that an
infoset's histories share their legal actions (the actor is part of the
infoset's key). Best responses are pure and take the lowest action within
TIE_TOL of the best; the value does not depend on the choice.
"""

import sys
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.core import decide as decide_lib
from open_spiel.python.scale.core import plies
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64

tree_map = jax.tree_util.tree_map

TERMINAL, CHANCE, DECISION = 0, 1, 2
TIE_TOL = 1e-12
CONSISTENCY_ATOL = 1e-12
DEFAULT_MAX_NODES = 200_000
#: Rows per call of a per-infoset-keys policy table; a fixed size compiles once.
PER_ROW_CHUNK = 1024
IN_GRAPH = (types.Capability.NATIVE_JAX, types.Capability.CALLBACK_BATCHED)


class TooLargeError(ValueError):
  """The game has more than `max_nodes` nodes, so no tree is built."""


def _bucket(n: int) -> int:
  """Padded batch size (powers of two bound the number of compiles).

  Args:
    n: The rows.

  Returns:
    The power of two at least `n` (at least 8).
  """
  b = 8
  while b < n:
    b *= 2
  return b


def _pad_take(tree, idx: np.ndarray, size: int):
  """Takes rows `idx` of `tree`, padded to `size` with the first.

  Args:
    tree: A pytree.
    idx: The row indices.
    size: The padded size.

  Returns:
    The rows.
  """
  full = np.concatenate([idx, np.full(size - idx.size, idx[0], idx.dtype)])
  return tree_map(lambda x: x[full], tree)


class GameTree:
  """The full game tree of a small Env in flat arrays (module doc).

  Per-node arrays (length num_nodes, breadth-first order): kind, parent,
  in_action (the action or outcome on the edge from the parent), in_chance,
  edge_p (the chance probability of that edge; 1 for decision edges), actor
  (-1 off decision nodes), node_legal bool[N, A], node_returns f64[N, P]
  (zero off terminal nodes), child_index i64[N, max(A, C)] (-1 for no
  child), level, infoset (-1 off decision nodes).

  Per-infoset arrays (length num_infosets): members (node ids), player,
  legal bool[I, A], info (the InfoState pytree, stacked).

  Attributes:
    env: the Env the tree expands.
    game: the env's registered name.
    num_players: P.
    num_actions: A.
    num_nodes: N, the number of nodes.
    kind: each node's kind (decision, chance or terminal).
    parent: each node's parent (-1 at the root).
    in_action: the action or outcome on the edge from the parent.
    in_chance: whether that edge is a chance outcome.
    edge_p: the chance probability of that edge (1 for decision edges).
    actor: the seat to move (-1 off decision nodes).
    node_legal: bool[N, A], the legal actions.
    node_returns: f64[N, P], the returns (zero off terminal nodes).
    child_index: i64[N, max(A, C)], the children (-1 for no child).
    level: each node's depth.
    infoset: each decision node's infoset (-1 off decision nodes).
    num_infosets: I, the number of infosets.
    members: each infoset's node ids.
    player: each infoset's seat.
    legal: bool[I, A], each infoset's legal actions.
    info: each infoset's InfoState pytree, stacked.
  """

  def __init__(self, env: types.Env, max_nodes: int = DEFAULT_MAX_NODES):
    """Expands `env` from its root.

    Args:
      env: the Env.
      max_nodes: the most nodes the tree may have.

    Raises:
      TooLargeError: the tree has more than `max_nodes` nodes.
      ValueError: a decision node has no legal action, a chance node no
        positive outcome, or an infoset's histories disagree on the legal
        actions (the InfoState is not a valid infoset key).
    """
    x64.assert_x64("GameTree")
    self.env = env
    self.game = env.name
    self.num_players = int(env.num_players)
    self.num_actions = int(env.num_actions)
    width = max(self.num_actions, int(env.max_chance_outcomes))
    self._node_fn = jax.jit(self._node_info)
    self._child_fn = jax.jit(self._children)

    kind, parent, in_action, in_chance, edge_p = [], [], [], [], []
    actor, legal, returns, child_index, level_of = [], [], [], [], []
    state_levels, info_levels, level_slices = [], [], []
    frontier = tree_map(lambda x: jnp.asarray(x)[None], env.root())
    f_parent = np.array([-1], np.int64)
    f_action = np.array([-1], np.int32)
    f_chance = np.array([False])
    f_prob = np.array([1.0])
    n0, level = 0, 0
    while True:
      n = int(f_parent.shape[0])
      if n0 + n > max_nodes:
        raise TooLargeError(
            f"{self.game}: more than max_nodes={max_nodes} tree nodes; a "
            "GameTree is for "
            "small games only"
        )
      size = _bucket(n)
      padded = _pad_take(frontier, np.arange(n), size)
      term, ch, act, lg, cp, rets, info = self._node_fn(padded)
      term, ch, act, lg, cp, rets = (
          np.asarray(x)[:n] for x in (term, ch, act, lg, cp, rets)
      )
      info = tree_map(lambda x: np.asarray(x)[:n], info)
      k = np.where(term, TERMINAL, np.where(ch, CHANCE, DECISION)).astype(
          np.int8
      )
      kind.append(k)
      parent.append(f_parent)
      in_action.append(f_action)
      in_chance.append(f_chance)
      edge_p.append(f_prob)
      actor.append(np.where(k == DECISION, act, -1).astype(np.int32))
      legal.append(np.where((k == DECISION)[:, None], lg, False))
      returns.append(
          np.where((k == TERMINAL)[:, None], rets.astype(np.float64), 0.0)
      )
      level_of.append(np.full(n, level, np.int32))
      state_levels.append(tree_map(lambda x: np.asarray(x)[:n], padded))
      info_levels.append(info)
      level_slices.append((n0, n0 + n))

      # Children of this level, contiguous, in (node, action/outcome) order.
      cidx = np.full((n, width), -1, np.int64)
      c_parent, c_action, c_chance, c_prob = [], [], [], []
      nxt = n0 + n
      for i in range(n):
        if k[i] == DECISION:
          acts = np.flatnonzero(lg[i])
          if acts.size == 0:
            raise ValueError(
                f"{self.game}: a decision node with no legal action"
            )
          probs = np.ones(acts.size)
          is_ch = False
        elif k[i] == CHANCE:
          p = cp[i].astype(np.float64)
          acts = np.flatnonzero(p > 0)
          total = p[acts].sum()
          if acts.size == 0 or not np.isfinite(total) or total <= 0:
            raise ValueError(
                f"{self.game}: a chance node with no positive outcome"
            )
          probs = p[acts] / total
          is_ch = True
        else:
          continue
        for a, q in zip(acts, probs):
          cidx[i, a] = nxt
          nxt += 1
          c_parent.append(n0 + i)
          c_action.append(a)
          c_chance.append(is_ch)
          c_prob.append(q)
      child_index.append(cidx)
      if not c_parent:
        break
      c_parent = np.asarray(c_parent, np.int64)
      local = c_parent - n0
      c_action = np.asarray(c_action, np.int32)
      c_chance = np.asarray(c_chance, bool)
      size = _bucket(local.size)

      def pad(x, v):
        """Pads `x` to `size` with `v`.

        Args:
          x: The array.
          v: The fill value.

        Returns:
          The padded array.
        """
        return np.concatenate([x, np.full(size - x.size, v, x.dtype)])

      frontier = self._child_fn(
          padded,
          jnp.asarray(pad(local, 0)),
          jnp.asarray(pad(c_action, 0)),
          jnp.asarray(pad(c_chance, False)),
      )
      frontier = tree_map(lambda x: x[: local.size], frontier)
      f_parent, f_action, f_chance = c_parent, c_action, c_chance
      f_prob = np.asarray(c_prob, np.float64)
      n0 += n
      level += 1

    self.kind = np.concatenate(kind)
    self.parent = np.concatenate(parent)
    self.in_action = np.concatenate(in_action)
    self.in_chance = np.concatenate(in_chance)
    self.edge_p = np.concatenate(edge_p)
    self.actor = np.concatenate(actor)
    self.node_legal = np.concatenate(legal)
    self.node_returns = np.concatenate(returns)
    self.child_index = np.concatenate(child_index)
    self.level = np.concatenate(level_of)
    self.num_nodes = int(self.kind.shape[0])
    self._states = tree_map(lambda *x: np.concatenate(x), *state_levels)
    node_info = tree_map(lambda *x: np.concatenate(x), *info_levels)
    self._index_infosets(node_info)

  # -- expansion ---------------------------------------------------------------
  def _node_info(self, states):
    """The per-node facts of a batch of states.

    Args:
      states: the states [B].

    Returns:
      (terminal, is chance, actor, legal mask, chance probs, returns, the
      actor's InfoState), each batched over B.
    """
    env = self.env
    term = jax.vmap(env.terminal)(states)
    ch = jax.vmap(env.is_chance)(states)
    actor = jax.vmap(env.actor)(states)
    legal = jax.vmap(env.legal_mask)(states)
    b = term.shape[0]
    if env.max_chance_outcomes:
      cp = jax.vmap(env.chance_probs)(states)
    else:
      cp = jnp.zeros((b, 0), jnp.float32)
    rets = jax.vmap(env.returns)(states)
    info = jax.vmap(env.info_state)(
        states, jnp.maximum(actor, 0).astype(jnp.int32)
    )
    return term, ch, actor, legal, cp, rets, info

  def _children(self, states, parent, action, is_chance):
    """The child states reached along a batch of edges.

    Args:
      states: the parents' states.
      parent: each edge's index into `states`.
      action: each edge's action or chance outcome.
      is_chance: whether each edge leaves a chance node.

    Returns:
      the child states, one per edge.
    """
    env = self.env
    s = tree_map(lambda x: x[parent], states)
    played = jax.vmap(env.apply)(s, action)
    if env.max_chance_outcomes:
      dealt = jax.vmap(env.apply_chance)(s, action)

      def sel(x, y):
        """`x` where the node is a chance node, else `y`.

        Args:
          x: the chance branch.
          y: the decision branch.

        Returns:
          the selected array.
        """
        c = is_chance.reshape(is_chance.shape + (1,) * (x.ndim - 1))
        return jnp.where(c, x, y)

      return tree_map(sel, dealt, played)
    return played

  def _index_infosets(self, node_info):
    """Groups the decision nodes into infosets by actor and InfoState.

    Sets infoset, members, num_infosets, player, legal and info.

    Args:
      node_info: every node's InfoState.

    Raises:
      ValueError: an infoset's histories have different legal actions.
    """
    dec = np.flatnonzero(self.kind == DECISION)
    leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(node_info)]
    keys = {}
    infoset = np.full(self.num_nodes, -1, np.int64)
    members: list[list[int]] = []
    for n in dec:
      key = (int(self.actor[n]),) + tuple(x[n].tobytes() for x in leaves)
      row = keys.get(key)
      if row is None:
        row = len(members)
        keys[key] = row
        members.append([])
      members[row].append(int(n))
      infoset[n] = row
    self.infoset = infoset
    self.members = [np.asarray(m, np.int64) for m in members]
    rep = np.array([m[0] for m in members], np.int64)
    self._rep = rep
    self.num_infosets = len(members)
    self.player = self.actor[rep].astype(np.int32)
    self.legal = self.node_legal[rep]
    for row, m in enumerate(self.members):
      if not (self.node_legal[m] == self.legal[row]).all():
        raise ValueError(
            f"{self.game}: infoset {row} has histories with different legal "
            "actions (InfoState is not a valid infoset key)"
        )
    self.info = tree_map(lambda x: np.asarray(x)[rep], node_info)

  # -- infosets ----------------------------------------------------------------
  def history(self, row: int) -> tuple[tuple[bool, int], ...]:
    """((is_chance, action), ...) from the root to `row`'s first history.

    Args:
      row: an infoset row.

    Returns:
      ((is_chance, action), ...) of its first history.
    """
    n = int(self._rep[row])
    path = []
    while self.parent[n] >= 0:
      path.append((bool(self.in_chance[n]), int(self.in_action[n])))
      n = int(self.parent[n])
    return tuple(reversed(path))

  def uniform_table(self) -> np.ndarray:
    """f64[I, A]: the uniform distribution over each infoset's legal actions.

    Returns:
      the uniform table f64[I, A].
    """
    return self.legal / self.legal.sum(1, keepdims=True)

  def check_table(self, table, what: str = "policy table") -> np.ndarray:
    """`table` as f64[I, A].

    Raises ValueError unless every row is a finite distribution over the
    infoset's legal actions.

    Args:
      table: a policy table.
      what: what the table is, for messages.

    Returns:
      the table as f64[I, A].

    Raises:
      ValueError: the shape is wrong or a row is not a finite distribution over
          the infoset's legal actions.
    """
    t = np.asarray(table, np.float64)
    if t.shape != (self.num_infosets, self.num_actions):
      raise ValueError(
          f"{what}: shape {t.shape}, want "
          f"{(self.num_infosets, self.num_actions)}"
      )
    if not np.all(np.isfinite(t)) or np.any(t < 0):
      raise ValueError(f"{what}: non-finite or negative probability")
    if np.any(t[~self.legal] != 0):
      raise ValueError(f"{what}: mass on an illegal action")
    if np.any(np.abs(t.sum(1) - 1.0) > 1e-9):
      raise ValueError(f"{what}: a row does not sum to 1")
    return t

  def _tables(self, tables) -> list[np.ndarray]:
    """Checks the per-seat tables, one table meaning every seat's.

    Args:
      tables: One table or one per seat.

    Returns:
      The checked tables, one per seat.
    """
    if isinstance(tables, np.ndarray) and tables.ndim == 2:
      tables = [tables] * self.num_players
    assert len(tables) == self.num_players, len(tables)
    return [
        self.check_table(t, f"seat {p} table") for p, t in enumerate(tables)
    ]

  def _edge_weights(
      self, tables, skip_player: Optional[int] = None
  ) -> np.ndarray:
    """Per node: the probability of the edge from its parent.

    Per node: probability of the edge from its parent (chance prob, or the
    acting seat's table entry; 1 for `skip_player`'s own actions).

    Args:
      tables: per seat, its policy table f64[I, A] over infosets.
      skip_player: a seat whose own action edges weigh 1, or None.

    Returns:
      f64[N], each node's edge probability (1 at the root).
    """
    w = np.ones(self.num_nodes)
    kids = np.arange(1, self.num_nodes)
    chance = self.in_chance[kids]
    w[kids[chance]] = self.edge_p[kids[chance]]
    dec = kids[~chance]
    pd = self.parent[dec]
    rows = self.infoset[pd]
    acts = self.in_action[dec]
    who = self.actor[pd]
    vals = np.empty(dec.size)
    for p in range(self.num_players):
      sel = who == p
      vals[sel] = 1.0 if p == skip_player else tables[p][rows[sel], acts[sel]]
    w[dec] = vals
    return w

  # -- exact values ------------------------------------------------------------
  def expected_returns(self, tables) -> np.ndarray:
    """f64[P]: every seat's expected return when seat p plays tables[p].

    Args:
      tables: one table f64[I, A] per seat.

    Returns:
      f64[P] expected returns.
    """
    tables = self._tables(tables)
    w = self._edge_weights(tables)
    v = self.node_returns.copy()
    for lvl in range(int(self.level.max()), 0, -1):
      nodes = np.flatnonzero(self.level == lvl)
      np.add.at(v, self.parent[nodes], w[nodes][:, None] * v[nodes])
    return v[0].copy()

  def best_response_value(self, tables, player: int) -> float:
    """The best value `player` can get against the other seats' tables.

    Args:
      tables: one table per seat, or one table playing every seat.
      player: the best-responding seat.

    Returns:
      its best-response value.
    """
    tables = self._tables(tables)
    p = int(player)
    w = self._edge_weights(tables, skip_player=p)
    reach = np.ones(self.num_nodes)
    for lvl in range(1, int(self.level.max()) + 1):
      nodes = np.flatnonzero(self.level == lvl)
      reach[nodes] = reach[self.parent[nodes]] * w[nodes]
    node_v: dict[int, float] = {}
    choice: dict[int, int] = {}

    def node_value(n: int) -> float:
      """The best-response value of node n for `player`, memoised.

      Args:
        n: a node.

      Returns:
        its value.
      """
      got = node_v.get(n)
      if got is not None:
        return got
      k = self.kind[n]
      if k == TERMINAL:
        out = float(self.node_returns[n, p])
      else:
        kids = self.child_index[n]
        kids = kids[kids >= 0]
        if k == DECISION and self.actor[n] == p:
          out = node_value(
              int(self.child_index[n, infoset_action(int(self.infoset[n]))])
          )
        else:
          out = float(sum(w[c] * node_value(int(c)) for c in kids))
      node_v[n] = out
      return out

    def infoset_action(row: int) -> int:
      """`player`'s best action at infoset `row`, memoised.

      Args:
        row: an infoset row of `player`.

      Returns:
        the action (the first within TIE_TOL of the best).
      """
      got = choice.get(row)
      if got is not None:
        return got
      acts = np.flatnonzero(self.legal[row])
      q = np.zeros(acts.size)
      for n in self.members[row]:
        for i, a in enumerate(acts):
          q[i] += reach[n] * node_value(int(self.child_index[n, a]))
      best = int(acts[int(np.flatnonzero(q >= q.max() - TIE_TOL)[0])])
      choice[row] = best
      return best

    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, 10_000))
    try:
      return node_value(0)
    finally:
      sys.setrecursionlimit(limit)

  def br_and_on_policy_values(
      self, table
  ) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """The best-response and on-policy values per seat of one shared table.

    (BR_p(table) per seat, value_p(table) per seat), with one table f64[I, A]
    playing every seat: the two halves of NashConv, which
    eval.exploitability.table_result reports both of. The table is not
    checked (nash_conv checks it).

    Args:
      table: the table f64[I, A] every seat plays.

    Returns:
      (BR_p(table) per seat, value_p(table) per seat).
    """
    t = np.asarray(table, np.float64)
    on = self.expected_returns([t] * self.num_players)
    br = tuple(
        float(self.best_response_value(t, p)) for p in range(self.num_players)
    )
    return br, tuple(float(x) for x in on)

  @staticmethod
  def nash_conv_of(br_values, on_policy_values) -> float:
    """sum_p [BR_p - value_p]: NashConv from br_and_on_policy_values.

    Args:
      br_values: BR_p per seat.
      on_policy_values: value_p per seat.

    Returns:
      NashConv.
    """
    return float(
        sum(b - o for b, o in zip(br_values, on_policy_values, strict=True))
    )

  def nash_conv(self, table) -> float:
    """sum_p [BR_p(table) - value_p(table)], one table playing every seat.

    Args:
      table: the table f64[I, A] every seat plays.

    Returns:
      NashConv of the table.
    """
    return self.nash_conv_of(
        *self.br_and_on_policy_values(self.check_table(table))
    )

  def exploitability(self, table) -> float:
    """NashConv / num_players.

    NashConv / num_players: for 2-player zero-sum games, OpenSpiel's
    exploitability(game, policy) of the same tabular policy.

    Args:
      table: the table f64[I, A] every seat plays.

    Returns:
      the exploitability of the table.
    """
    return self.nash_conv(table) / self.num_players

  # -- arrays for exact belief -------------------------------------------------
  @staticmethod
  def flatten_info(info, batch_ndim: int = 0):
    """An InfoState as one f64 vector.

    An InfoState as one f64 vector: every leaf raveled, cast to f64 and
    concatenated in tree_leaves order.

    Args:
      info: an InfoState pytree, numpy or jax, with `batch_ndim` leading batch
        dimensions on every leaf.
      batch_ndim: the number of leading batch dimensions.

    Returns:
      f64[*batch, F]. Equal InfoStates give equal vectors; the cast is exact
      for bools, integers below 2**53 and f32 values. Jittable.
    """
    leaves = jax.tree_util.tree_leaves(info)
    assert leaves, "an InfoState with no leaves"
    lead = jnp.shape(leaves[0])[:batch_ndim]
    return jnp.concatenate(
        [
            jnp.reshape(jnp.asarray(x), lead + (-1,)).astype(jnp.float64)
            for x in leaves
        ],
        axis=-1,
    )

  def seat_infosets(self, seat: int) -> np.ndarray:
    """i64[I_seat]: the global infoset rows at which `seat` acts, ascending.

    Position i in this array is the seat-local infoset index i.

    Args:
      seat: a seat.

    Returns:
      i64[I_seat] global infoset rows.
    """
    return np.flatnonzero(self.player == int(seat)).astype(np.int64)

  def infoset_nodes(self, seat: int, infoset: int) -> np.ndarray:
    """i64[M]: the decision nodes of `seat`'s seat-local infoset `infoset`.

    Args:
      seat: a seat.
      infoset: a seat-local infoset index of `seat`.

    Returns:
      i64[M] node ids.

    Raises:
      IndexError: `infoset` is out of range for `seat`.
    """
    rows = self.seat_infosets(seat)
    if not 0 <= int(infoset) < rows.size:
      raise IndexError(
          f"seat {seat} has {rows.size} infosets, asked for {infoset}"
      )
    return self.members[int(rows[int(infoset)])].copy()

  def infoset_members(self, seat: int) -> np.ndarray:
    """i64[I_seat, M]: the nodes of each of `seat`'s infosets, -1 padded.

    i64[I_seat, M]: the nodes of each of `seat`'s infosets, -1 padded to the
    largest infoset (a dense form of infoset_nodes).

    Args:
      seat: a seat.

    Returns:
      i64[I_seat, M] node ids, -1 padded.
    """
    rows = self.seat_infosets(seat)
    width = max([self.members[r].size for r in rows] or [1])
    out = np.full((rows.size, width), -1, np.int64)
    for i, r in enumerate(rows):
      out[i, : self.members[r].size] = self.members[r]
    return out

  def infoset_keys(self, seat: int) -> np.ndarray:
    """f64[I_seat, F]: flatten_info of each of `seat`'s infosets.

    f64[I_seat, F]: flatten_info of each of `seat`'s infosets, in
    seat-local order. A seat's current infoset is found by exact equality of
    flatten_info(step.info) against these rows.

    Args:
      seat: a seat.

    Returns:
      f64[I_seat, F] infoset keys.
    """
    rows = self.seat_infosets(seat)
    info = tree_map(lambda x: np.asarray(x)[rows], self.info)
    return np.asarray(self.flatten_info(info, batch_ndim=1), np.float64)

  def chance_reach(self) -> np.ndarray:
    """f64[N]: the chance reach probability of each node.

    f64[N]: the product of the chance probabilities on each node's path
    from the root (1 at the root).

    Returns:
      f64[N] chance reach probabilities.
    """
    reach = np.ones(self.num_nodes)
    for lvl in range(1, int(self.level.max()) + 1):
      nodes = np.flatnonzero(self.level == lvl)
      w = np.where(self.in_chance[nodes], self.edge_p[nodes], 1.0)
      reach[nodes] = reach[self.parent[nodes]] * w
    return reach

  def player_edges_on_path(self, player: int) -> np.ndarray:
    """i32[N, D]: per node, the decision edges `player` takes from the root.

    i32[N, D]: per node, the decision edges `player` takes on the path
    from the root, root first, -1 padded to the longest such path.

    An edge id is `infoset_row * num_actions + action`, a flat index into a
    policy table f[num_infosets, A]. The probability that `player`'s table
    produces the path to node n is therefore
    prod(table.reshape(-1)[e] for e in edges[n] if e >= 0).

    Args:
      player: a seat.

    Returns:
      i32[N, D] edge ids, -1 padded.
    """
    p = int(player)
    paths = [[] for _ in range(self.num_nodes)]
    for n in range(1, self.num_nodes):
      parent = int(self.parent[n])
      paths[n] = list(paths[parent])
      if not self.in_chance[n] and int(self.actor[parent]) == p:
        paths[n].append(
            int(self.infoset[parent]) * self.num_actions
            + int(self.in_action[n])
        )
    depth = max(len(x) for x in paths)
    out = np.full((self.num_nodes, depth), -1, np.int32)
    for n, edges in enumerate(paths):
      out[n, : len(edges)] = edges
    return out

  def node_states(self) -> Any:
    """The env state of every node, stacked [N, ...] (numpy leaves).

    Returns:
      the stacked env states (a copy).
    """
    return tree_map(np.copy, self._states)

  # -- policies ----------------------------------------------------------------
  def _seat_paths(self):
    """Per decision node: its decision ancestors and the actions taken.

    (decision ancestors incl. itself, the action taken at each of them on the
    path to the node).

    Returns:
      {node: (decision ancestors incl. itself, the action taken at each)}.
    """
    out = {}
    for n in np.flatnonzero(self.kind == DECISION):
      path, acts = [], []
      m, below = int(n), None
      while m >= 0:
        if self.kind[m] == DECISION:
          path.append(m)
          acts.append(-1 if below is None else int(self.in_action[below]))
        below, m = m, int(self.parent[m])
      out[int(n)] = (path[::-1], acts[::-1])
    return out

  def _build_steps(self, states, dec_idx, seat, prev_idx, prev_act):
    """SeatStep [G, T] of `seat` at the decision plies of each path.

    Args:
      states: The tree's states.
      dec_idx: int[G, T], the decision state of each ply.
      seat: The seat.
      prev_idx: int[G, T], the state before each ply.
      prev_act: int[G, T], the action into each ply.

    Returns:
      The SeatSteps, leaves [G, T, ...].
    """
    env = self.env
    g, t = dec_idx.shape
    per_t = []
    for i in range(t):
      s = tree_map(lambda x, idx=dec_idx[:, i]: x[idx], states)
      if i == 0:
        la = jnp.full((g,), types.NO_ACTION, jnp.int32)
        lr = jnp.full((g,), types.NO_ACTOR, jnp.int32)
      else:
        sp = tree_map(lambda x, idx=prev_idx[:, i]: x[idx], states)
        la = jax.vmap(env.visible_action)(sp, prev_act[:, i], seat).astype(
            jnp.int32
        )
        lr = jax.vmap(env.actor)(sp).astype(jnp.int32)
      per_t.append(plies.seat_step(env, s, seat, i == 0, i == 0, la, lr))
    return tree_map(lambda *x: jnp.stack(x, axis=1), *per_t)

  def policy_table(
      self,
      policy: types.Policy,
      params: Any,
      rule: types.DecisionRule,
      key: Optional[Any] = None,
      check_consistent: bool = True,
      per_infoset_keys: bool = False,
  ) -> np.ndarray:
    """f64[num_infosets, A]: the action distribution of `policy` per infoset.

    Every decision node is evaluated through the same SeatSteps the Arena
    emits (a stateful policy observes the seat's whole path) and the one
    `decide`: the row is decide's f64 probabilities, or the one-hot of its
    action under ARGMAX.

    Args:
      policy: an in-graph JAX policy (NATIVE_JAX or CALLBACK_BATCHED).
      params: one member's params.
      rule: the decision rule.
      key: the evaluation key (default key 0).
      check_consistent: require every history of an infoset to give the same
        row (within CONSISTENCY_ATOL).
      per_infoset_keys: evaluate every history on its own with keys folded
        with its infoset row r (init_carry under fold_in64(k_init, r),
        evaluate under fold_in64(k_eval, r), decide under fold_in64(k_dec, r),
        where k_init, k_eval, k_dec = split(key, 3)). A stochastic policy
        such as a search policy is then one deterministic table at a fixed
        key, the same for every history of an infoset and independent of
        batching. For a key-independent policy it equals the default table.

    Returns:
      the checked table f64[num_infosets, A].

    Raises:
      NotImplementedError: the policy is host-driven or not a JAX policy.
      TypeError: rule is not a DecisionRule.
      ValueError: the policy's action or player count is not the game's, the
        policy is stateful and the game refuses stateful policies, or
        check_consistent and two histories of one infoset give different rows
        (the policy reads something other than the infoset).
    """
    x64.assert_x64("policy_table")
    spec = policy.spec
    if not isinstance(rule, types.DecisionRule):
      raise TypeError(f"rule {rule!r} is not a DecisionRule")
    if spec.framework != "jax" or spec.capability not in IN_GRAPH:
      raise NotImplementedError(
          "policy_table drives in-graph JAX policies; got "
          f"{spec.capability.value}/"
          f"{spec.framework}"
      )
    if (spec.num_actions, spec.num_players) != (
        self.num_actions,
        self.num_players,
    ):
      raise ValueError(
          f"{spec.arch_name}: {spec.num_actions} actions / {spec.num_players} "
          f"players, the game has {self.num_actions} / {self.num_players}"
      )
    types.check_policy_admitted(self.env, spec)
    key = jax.random.key(0) if key is None else key
    k_init, k_eval, k_dec = jax.random.split(key, 3)
    paths = self._seat_paths()
    by_len: dict[int, list[int]] = {}
    for n, (path, _) in paths.items():
      by_len.setdefault(len(path), []).append(n)
    states = tree_map(jnp.asarray, self._states)
    node_probs = np.full((self.num_nodes, self.num_actions), np.nan)

    def run(
        params, states, dec_idx, seat, prev_idx, prev_act, k_init, k_eval, k_dec
    ):
      """The rows of one group of equal-length paths, batch-shared keys.

      Args:
        params: the policy's params.
        states: the node states.
        dec_idx: the decision node ids along each path.
        seat: each path's acting seat.
        prev_idx: each ply's previous node id.
        prev_act: each ply's previous action.
        k_init: the init_carry key.
        k_eval: the evaluate key.
        k_dec: the decide key.

      Returns:
        f64[G, A] probabilities.
      """
      steps = self._build_steps(states, dec_idx, seat, prev_idx, prev_act)
      g, t = dec_idx.shape
      carry = policy.init_carry(params, g, seat, k_init)
      _, out = policy.evaluate_seq(
          params, carry, steps, jax.random.split(k_eval, t)
      )
      logits = out.logits[:, -1].astype(jnp.float32)
      legal = steps.legal_mask[:, -1]
      action, probs = decide_lib.decide(logits, legal, rule, k_dec)
      if rule.kind == "argmax":
        probs = jax.nn.one_hot(action, self.num_actions, dtype=jnp.float64)
      return probs

    def run_rows(
        params,
        states,
        dec_idx,
        seat,
        prev_idx,
        prev_act,
        rows,
        lengths,
        k_init,
        k_eval,
        k_dec,
    ):
      """The rows of padded paths, each with keys folded with its infoset row.

      Args:
        params: the policy's params.
        states: the node states.
        dec_idx: the decision node ids along each path.
        seat: each path's acting seat.
        prev_idx: each ply's previous node id.
        prev_act: each ply's previous action.
        rows: each path's infoset row.
        lengths: each path's length.
        k_init: the init_carry key.
        k_eval: the evaluate key.
        k_dec: the decide key.

      Returns:
        f64[G, A] probabilities.
      """
      steps = self._build_steps(states, dec_idx, seat, prev_idx, prev_act)
      t_max = dec_idx.shape[1]

      def one(step_row, seat_r, row, length):
        """One path's row.

        Args:
          step_row: the path's SeatSteps.
          seat_r: its seat.
          row: its infoset row.
          length: its length.

        Returns:
          f64[A] probabilities.
        """

        def at(t):
          """Ply `t` of the row, with a batch axis of 1.

          Args:
            t: The ply.

          Returns:
            The SeatStep.
          """
          return tree_map(lambda x: x[t][None], step_row)

        carry = policy.init_carry(
            params, 1, seat_r[None], x64.fold_in64(k_init, row)
        )

        def body(c, t):
          """Observes ply t, freezing the carry past the path's length.

          Args:
            c: the carry.
            t: the ply.

          Returns:
            (the next carry, None).
          """
          c2 = policy.observe(params, c, at(t))
          return tree_map(lambda a, b: jnp.where(t < length, a, b), c2, c), None

        carry, _ = jax.lax.scan(body, carry, jnp.arange(t_max, dtype=jnp.int64))
        last = at(length - 1)
        out = policy.evaluate(params, carry, last, x64.fold_in64(k_eval, row))
        action, probs = decide_lib.decide(
            out.logits.astype(jnp.float32),
            last.legal_mask,
            rule,
            x64.fold_in64(k_dec, row),
        )
        if rule.kind == "argmax":
          probs = jax.nn.one_hot(action, self.num_actions, dtype=jnp.float64)
        return probs[0]

      return jax.vmap(one)(steps, seat, rows, lengths)

    if per_infoset_keys:
      # One padded group (paths padded to the longest, the carry frozen past a
      # row's length) in fixed-size chunks: one compile.
      path_nodes = []
      for ns in by_len.values():
        path_nodes.extend(ns)
      every = np.asarray(sorted(path_nodes), np.int64)
      groups = [(max(by_len), every)]
      chunk = min(_bucket(every.size), PER_ROW_CHUNK)
      run_jit = jax.jit(run_rows)
    else:
      groups = [
          (t, np.asarray(ns, np.int64)) for t, ns in sorted(by_len.items())
      ]
      chunk = None
      run_jit = jax.jit(run)
    for t_len, all_nodes in groups:
      step = chunk or all_nodes.size
      for c0 in range(0, all_nodes.size, step):
        nodes = all_nodes[c0 : c0 + step]
        g = nodes.size
        size = chunk or _bucket(g)
        dec_idx = np.zeros((size, t_len), np.int64)
        prev_idx = np.zeros((size, t_len), np.int64)
        prev_act = np.zeros((size, t_len), np.int32)
        seat = np.zeros((size,), np.int32)
        lengths = np.zeros((size,), np.int64)
        for r in range(size):
          n = int(nodes[min(r, g - 1)])
          path, acts = paths[n]
          ln = len(path)
          dec_idx[r, :ln] = path
          dec_idx[r, ln:] = path[-1]  # padding, never observed
          prev_idx[r, 1:ln] = path[:-1]
          prev_act[r, 1:ln] = acts[:-1]
          seat[r] = self.actor[n]
          lengths[r] = ln
        args = (
            params,
            states,
            jnp.asarray(dec_idx),
            jnp.asarray(seat),
            jnp.asarray(prev_idx),
            jnp.asarray(prev_act),
        )
        if per_infoset_keys:
          rows = self.infoset[nodes[np.minimum(np.arange(size), g - 1)]].astype(
              np.int64
          )
          args = args + (
              jnp.asarray(rows, jnp.int64),
              jnp.asarray(lengths, jnp.int64),
          )
        probs = run_jit(*args, k_init, k_eval, k_dec)
        node_probs[nodes] = np.asarray(probs, np.float64)[:g]

    table = node_probs[self._rep].copy()
    if check_consistent:
      for row, m in enumerate(self.members):
        if not np.allclose(
            node_probs[m], table[row], atol=CONSISTENCY_ATOL, rtol=0.0
        ):
          raise ValueError(
              f"{spec.arch_name}: output differs between histories of one "
              f"infoset (row {row}, history {self.history(row)}): the policy "
              "reads something other than the infoset"
          )
    return self.check_table(table, f"{spec.arch_name} policy table")
