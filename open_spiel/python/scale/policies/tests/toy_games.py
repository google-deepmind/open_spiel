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

"""Tiny explicit game trees as Envs, for search known-answer tests.

TreeGameEnv implements the Env protocol over a table of nodes: each node is a
decision node of seat 0/1, a chance node with explicit outcome probabilities,
or a terminal node with explicit returns. Perfect information: the InfoState
is the node id, and world_from_info rebuilds the state. Each tree is
registered in envs.registry (a GameSpec with no options, never served), so a
SearchPolicy over it records and rebuilds it through its EnvSpec like any game.
"""

from typing import Any, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.core import chance
from open_spiel.python.scale.envs import registry as env_registry

CHANCE = -2
TERMINAL = -1


class TreeState(NamedTuple):
  """A toy game state: the current tree node.

  Attributes:
    node: i32, the node index.
  """

  node: jax.Array  # i32


class TreeInfo(NamedTuple):
  """A toy game InfoState: perfect information, the seat and the node.

  Attributes:
    seat: i32, the observing seat.
    node: i32, the node index.
  """

  seat: jax.Array  # i32
  node: jax.Array  # i32


def node(kind, children=(), probs=(), returns=(0.0, 0.0)):
  """One node of a toy game tree.

  Args:
    kind: "p0", "p1", "chance" or "term".
    children: the child node index of each action or chance outcome.
    probs: a chance node's outcome probabilities.
    returns: a terminal node's returns per seat.

  Returns:
    The node dict.
  """
  return dict(
      kind=kind,
      children=tuple(children),
      probs=tuple(probs),
      returns=tuple(returns),
  )


class TreeGameEnv:
  """A two-player zero-sum Env over an explicit game tree, for known answers.

  Attributes:
    name: the game name.
    ctor: 'module:factory' that rebuilds this env (its GameSpec ctor).
    num_actions: the number of actions.
    max_chance_outcomes: the most outcomes of a chance node.
    max_len: the longest play.
    scale: the return multiplier (set by coin_scaled only).
  """

  num_players = 2
  zero_sum = True

  def __init__(
      self,
      name: str,
      ctor: str,
      nodes: Sequence[dict[str, Any]],
      num_actions: int,
      max_chance_outcomes: int,
      max_len: int,
  ):
    self.name = name
    # 'module:factory' that rebuilds this env (its envs.registry GameSpec ctor)
    self.ctor = ctor
    self.num_actions = num_actions
    self.max_chance_outcomes = max_chance_outcomes
    self.max_len = max_len
    n = len(nodes)
    width = max(num_actions, max_chance_outcomes, 1)
    kind = np.zeros((n,), np.int32)
    child = np.full((n, width), -1, np.int32)
    probs = np.zeros((n, max(max_chance_outcomes, 1)), np.float32)
    rets = np.zeros((n, 2), np.float32)
    for i, nd in enumerate(nodes):
      kind[i] = {"p0": 0, "p1": 1, "chance": CHANCE, "term": TERMINAL}[
          nd["kind"]
      ]
      for j, c in enumerate(nd["children"]):
        child[i, j] = c
      for j, p in enumerate(nd["probs"]):
        probs[i, j] = p
      if nd["kind"] == "chance":
        assert abs(sum(nd["probs"]) - 1.0) < 1e-6, nd
      if nd["kind"] == "term":
        rets[i] = nd["returns"]
    self._kind, self._child = jnp.asarray(kind), jnp.asarray(child)
    self._probs, self._rets = jnp.asarray(probs), jnp.asarray(rets)

  def _kind_of(self, s):
    """The node kind of `s`.

    Args:
      s: The state.

    Returns:
      The kind.
    """
    return self._kind[s.node]

  def root(self):
    """The root state.

    Returns:
      The TreeState at node 0.
    """
    return TreeState(node=jnp.int32(0))

  def apply(self, s, a):
    """Takes action `a`; an action with no child leaves the state.

    Args:
      s: the state.
      a: the action.

    Returns:
      The next state.
    """
    c = self._child[s.node, jnp.asarray(a, jnp.int32)]
    return TreeState(node=jnp.where(c >= 0, c, s.node).astype(jnp.int32))

  def is_chance(self, s):
    """Whether `s` is a chance node.

    Args:
      s: the state.

    Returns:
      bool.
    """
    return self._kind_of(s) == CHANCE

  def apply_chance(self, s, outcome):
    """Takes chance outcome `outcome`.

    Args:
      s: the state.
      outcome: the outcome index.

    Returns:
      The next state.
    """
    return self.apply(s, outcome)

  def init(self, deal_key):
    """The root with its chance nodes resolved.

    Args:
      deal_key: a PRNG key for the chance nodes.

    Returns:
      The first decision (or terminal) state.
    """
    return chance.resolve(self, self.root(), deal_key)[0]

  def step(self, s, a, chance_key):
    """Takes action `a`, then resolves the chance nodes after it.

    Args:
      s: the state.
      a: the action.
      chance_key: a PRNG key for the chance nodes.

    Returns:
      The next decision (or terminal) state.
    """
    return chance.resolve(self, self.apply(s, a), chance_key)[0]

  def chance_probs(self, s):
    """The outcome probabilities at a chance node, zeros elsewhere.

    Args:
      s: the state.

    Returns:
      f32[max_chance_outcomes].
    """
    return jnp.where(self.is_chance(s), self._probs[s.node], 0.0).astype(
        jnp.float32
    )

  def actor(self, s):
    """The seat to act, -1 at chance and terminal nodes.

    Args:
      s: the state.

    Returns:
      i32.
    """
    k = self._kind_of(s)
    return jnp.where(k >= 0, k, jnp.int32(-1)).astype(jnp.int32)

  def legal_mask(self, s):
    """The actions with a child at a decision node.

    Args:
      s: the state.

    Returns:
      bool[num_actions].
    """
    return (self._child[s.node, : self.num_actions] >= 0) & (
        self._kind_of(s) >= 0
    )

  def terminal(self, s):
    """Whether `s` is terminal.

    Args:
      s: the state.

    Returns:
      bool.
    """
    return self._kind_of(s) == TERMINAL

  def returns(self, s):
    """The returns per seat at a terminal node, zeros elsewhere.

    Args:
      s: the state.

    Returns:
      f32[2].
    """
    return jnp.where(self.terminal(s), self._rets[s.node], 0.0).astype(
        jnp.float32
    )

  def info_state(self, s, seat):
    """The seat's (perfect-information) view.

    Args:
      s: the state.
      seat: the observing seat.

    Returns:
      The TreeInfo.
    """
    return TreeInfo(
        seat=jnp.asarray(seat, jnp.int32), node=jnp.asarray(s.node, jnp.int32)
    )

  def labels(self, s, seat):
    """No labels.

    Args:
      s: the state.
      seat: the observing seat.

    Returns:
      An empty dict.
    """
    del s, seat
    return {}

  def visible_action(self, s, a, seat):
    """Every action is visible.

    Args:
      s: the state.
      a: the action.
      seat: the observing seat.

    Returns:
      `a` as i32.
    """
    del s, seat
    return jnp.asarray(a, jnp.int32)

  def world_from_info(self, info):
    """The state an InfoState shows (perfect information).

    Args:
      info: the TreeInfo.

    Returns:
      The TreeState.
    """
    return TreeState(node=jnp.asarray(info.node, jnp.int32))


def coin():
  """A one-decision game with a gamble.

  p0: safe (+0.1) or gamble (chance: +1 w.p. .3, -1 w.p. .7; EV -0.4).

  Returns:
    The TreeGameEnv.
  """
  return TreeGameEnv(
      "toy_coin",
      f"{__name__}:coin",
      [
          node("p0", (1, 2)),
          node("term", returns=(0.1, -0.1)),
          node("chance", (3, 4), (0.3, 0.7)),
          node("term", returns=(1.0, -1.0)),
          node("term", returns=(-1.0, 1.0)),
      ],
      num_actions=2,
      max_chance_outcomes=2,
      max_len=1,
  )


def coin_reply():
  """A chance node followed by the opponent's reply.

  p0: a0 -> chance(.5/.5) -> p1 replies (p1 minimises p0); a1 -> 0.
  Exact: Q(a0) = .5 * min(1, -.5) + .5 * min(.2, .8) = -0.15, Q(a1) = 0.
  With the wrong discount sign (p1 maximising p0) Q(a0) would be 0.9.

  Returns:
    The TreeGameEnv.
  """
  return TreeGameEnv(
      "toy_coin_reply",
      f"{__name__}:coin_reply",
      [
          node("p0", (1, 2)),
          node("chance", (3, 4), (0.5, 0.5)),
          node("term", returns=(0.0, 0.0)),
          node("p1", (5, 6)),
          node("p1", (7, 8)),
          node("term", returns=(1.0, -1.0)),
          node("term", returns=(-0.5, 0.5)),
          node("term", returns=(0.2, -0.2)),
          node("term", returns=(0.8, -0.8)),
      ],
      num_actions=2,
      max_chance_outcomes=2,
      max_len=2,
  )


def coin_again():
  """A chance node after which the same seat moves again.

  p0: a0 -> chance(.5/.5) -> p0 moves AGAIN (the seat does not change, as
  leduc's round 2 opens with seat 0); a1 -> -0.3.
  Exact: Q(a0) = .5 * max(-1, .4) + .5 * max(-1, -.6) = -0.1, Q(a1) = -0.3.

  Returns:
    The TreeGameEnv.
  """
  return TreeGameEnv(
      "toy_coin_again",
      f"{__name__}:coin_again",
      [
          node("p0", (1, 2)),
          node("chance", (3, 4), (0.5, 0.5)),
          node("term", returns=(-0.3, 0.3)),
          node("p0", (5, 6)),
          node("p0", (7, 8)),
          node("term", returns=(-1.0, 1.0)),
          node("term", returns=(0.4, -0.4)),
          node("term", returns=(-1.0, 1.0)),
          node("term", returns=(-0.6, 0.6)),
      ],
      num_actions=2,
      max_chance_outcomes=2,
      max_len=2,
  )


def reply():
  """A deterministic two-ply game.

  Deterministic: p0 a0 -> p1 {+1, -1}; a1 -> p1 {+.5, +.3} (p0's returns).
  Minimax: Q(a0) = -1, Q(a1) = .3. Opponent folded into the dynamics with an
  argmax-of-uniform (first legal action) policy: Q(a0) = +1, Q(a1) = +.5.

  Returns:
    The TreeGameEnv.
  """
  return TreeGameEnv(
      "toy_reply",
      f"{__name__}:reply",
      [
          node("p0", (1, 2)),
          node("p1", (3, 4)),
          node("p1", (5, 6)),
          node("term", returns=(1.0, -1.0)),
          node("term", returns=(-1.0, 1.0)),
          node("term", returns=(0.5, -0.5)),
          node("term", returns=(0.3, -0.3)),
      ],
      num_actions=2,
      max_chance_outcomes=0,
      max_len=2,
  )


def coin_scaled(scale: float = 1.0):
  """coin() with every return multiplied by `scale`.

  A game with a constructor option (its GameSpec default_kwargs
  {"scale": 1.0}), so an EnvSpec round trip is observable. Exact:
  Q(safe) = 0.1 * scale, the gamble's afterstate value -0.4 * scale.

  Args:
    scale: the return multiplier.

  Returns:
    The TreeGameEnv.
  """
  sc = float(scale)
  env = TreeGameEnv(
      "toy_coin_scaled",
      f"{__name__}:coin_scaled",
      [
          node("p0", (1, 2)),
          node("term", returns=(0.1 * sc, -0.1 * sc)),
          node("chance", (3, 4), (0.3, 0.7)),
          node("term", returns=(1.0 * sc, -1.0 * sc)),
          node("term", returns=(-1.0 * sc, 1.0 * sc)),
      ],
      num_actions=2,
      max_chance_outcomes=2,
      max_len=1,
  )
  env.scale = sc
  return env


env_registry.register(
    env_registry.GameSpec(
        "toy_coin_scaled",
        f"{__name__}:coin_scaled",
        default_kwargs={"scale": 1.0},
    )
)

for _name, _fn in (
    ("toy_coin", "coin"),
    ("toy_coin_reply", "coin_reply"),
    ("toy_coin_again", "coin_again"),
    ("toy_reply", "reply"),
):
  env_registry.register(env_registry.GameSpec(_name, f"{__name__}:{_fn}"))
