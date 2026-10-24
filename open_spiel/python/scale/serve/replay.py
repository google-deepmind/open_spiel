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

"""Replaying a recorded match into the SeatSteps one seat saw.

Serving a stateless request means rebuilding the seat's view of the match
from its history. This is the host-side twin of the Arena's ply loop, built
from the same blocks (core.plies.seat_step, core.plies.visible_actions), so a
replayed history yields exactly the SeatSteps the Arena fed that seat:

  rep = replay.replay_seat_steps(env, hands, seat)
  rep.steps        # SeatStep, leaves [1, T]
  rep.state        # the env state at the last ply

  * a ply is every player decision node (all seats observe every ply) plus
    ONE observe-only terminal ply at the end of each completed hand;
  * chance nodes are not plies: their outcomes come from the history through
    the env's raw layer (`apply_chance`), and a seat learns them only through
    its info_state;
  * new_hand is True on each hand's first ply, new_match on the match's
    first ply only (the history is one match: the caller passes the hands of
    the current match);
  * last_action / last_actor are the previous ply's action as `seat` sees it
    (Env.visible_action) and its actor; NO_ACTION / NO_ACTOR on a hand's
    first ply.

Input: `hands`, a sequence of hands, each a raw action history from the
env's root (chance outcomes and player actions interleaved, in the env's
chance-outcome ids; serve.wire translates a pyspiel history). Every hand but the
last must end at a terminal state; the last may end at a decision node or a
terminal state, never at a chance node. Illegal actions, zero-probability
chance outcomes and actions after the end of a hand raise ValueError.
"""

from typing import Any, Callable, NamedTuple, Optional, Sequence
import weakref

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.core import plies
from open_spiel.python.scale.core import types

tree_map = jax.tree_util.tree_map


class SeatReplay(NamedTuple):
  """A seat's replayed SeatSteps of one match.

  Attributes:
    steps: SeatStep leaves [1, T]: batch of one row, T plies.
    state: the env state at the last ply (unbatched).
    num_plies: T.
    hand_starts: ply index of each hand's first ply.
  """

  steps: types.SeatStep  # leaves [1, T]: batch of one row, T plies
  state: Any  # the env state at the last ply (unbatched)
  num_plies: int  # T
  hand_starts: tuple[int, ...]  # ply index of each hand's first ply


class _Fns:
  """Jitted env calls for one env.

  jit caches per function object, so they are built once per env, not per
  request. They hold the env through a weak reference, so the cache entry (keyed
  weakly by the env) goes when the env does.

  Attributes:
    root: Jitted env.root.
    apply: Jitted env.apply.
    apply_chance: Jitted env.apply_chance.
    node: Jitted (is_chance, terminal, legal_mask, chance_probs) of a state.
    decision_node: Jitted (actor, legal_mask, terminal) of a state.
    visible: Jitted plies.visible_actions.
    actors: Jitted, vmapped env.actor.
    seat_step: Jitted plies.seat_step.
  """

  def __init__(self, env: types.Env):
    ref = weakref.ref(env)

    def _env() -> types.Env:
      """The env behind the weak reference.

      Returns:
        The env.
      """
      e = ref()
      assert e is not None, "a jitted env call outlived its env"
      return e

    self.root = jax.jit(lambda *a: _env().root(*a))
    self.apply = jax.jit(lambda *a: _env().apply(*a))
    self.apply_chance = jax.jit(lambda *a: _env().apply_chance(*a))
    self.node = jax.jit(
        lambda s: (
            _env().is_chance(s),
            _env().terminal(s),
            _env().legal_mask(s),
            _env().chance_probs(s),
        )
    )
    self.decision_node = jax.jit(
        lambda s: (_env().actor(s), _env().legal_mask(s), _env().terminal(s))
    )
    self.visible = jax.jit(
        lambda states, actions: plies.visible_actions(_env(), states, actions)
    )
    self.actors = jax.jit(jax.vmap(lambda s: _env().actor(s)))
    self.seat_step = jax.jit(
        lambda states, seat, nh, nm, la, lact: plies.seat_step(
            _env(), states, seat, nh, nm, la, lact
        )
    )


_FNS: "weakref.WeakKeyDictionary[Any, _Fns]" = weakref.WeakKeyDictionary()


def _fns(env: types.Env) -> _Fns:
  """The jitted helpers of `env`, built once per env.

  Args:
    env: The env.

  Returns:
    The env's _Fns.
  """
  f = _FNS.get(env)
  if f is None:
    f = _Fns(env)
    _FNS[env] = f
  return f


def decision_node_fn(env: types.Env) -> Callable[[Any], tuple[Any, Any, Any]]:
  """The jitted (actor, legal mask, terminal) of an env state.

  Built once per env and dropped with the env.

  Args:
    env: the env.

  Returns:
    The jitted function.
  """
  return _fns(env).decision_node


def replay_hand(
    env: types.Env, hand: Sequence[int], fns: Optional[_Fns] = None
):
  """(ply states, ply actions, final state) of one hand from the env root.

  Ply states are the decision nodes in order, then the final node (decision
  or terminal); ply actions are the action taken at each decision node (the
  final node has none).

  Args:
    env: the env.
    hand: the hand's raw actions.
    fns: the env's jitted calls (default the cached ones).

  Returns:
    (ply states, ply actions, final state).

  Raises:
    ValueError: an action is illegal, comes after the hand is over, or the hand
      does not reach a node.
  """
  fns = fns or _fns(env)
  s = fns.root()
  states, actions = [], []
  for i, a in enumerate(hand):
    a = int(a)
    is_chance, terminal, legal, probs = fns.node(s)
    if bool(terminal):
      raise ValueError(f"action {i} ({a}) comes after the hand is over")
    if bool(is_chance):
      probs = np.asarray(probs)
      if not (0 <= a < probs.shape[0]) or not probs[a] > 0:
        raise ValueError(
            f"chance outcome {a} has probability 0 (or is out of range) at "
            f"history index {i}"
        )
      s = fns.apply_chance(s, jnp.int32(a))
    else:
      legal = np.asarray(legal)
      if not (0 <= a < legal.shape[0]) or not legal[a]:
        raise ValueError(f"illegal action {a} at history index {i}")
      states.append(s)
      actions.append(a)
      s = fns.apply(s, jnp.int32(a))
  if bool(fns.node(s)[0]):
    raise ValueError(
        "the history ends at a chance node; there is no ply to replay"
    )
  states.append(s)
  return states, actions, s


def replay_seat_steps(
    env: types.Env, hands: Sequence[Sequence[int]], seat: int
) -> SeatReplay:
  """SeatStep [1, T] of `seat` over the plies of `hands` (one match).

  Args:
    env: the env.
    hands: the match's raw hands.
    seat: the perspective seat.

  Returns:
    The SeatReplay.

  Raises:
    ValueError: seat is out of range, there are no hands, or a hand but the last
      is not terminal.
  """
  if not 0 <= int(seat) < env.num_players:
    raise ValueError(f"seat {seat} is not in [0, {env.num_players})")
  if not hands:
    raise ValueError("no hands to replay")
  fns = _fns(env)
  states, new_hand, last_action, last_actor = [], [], [], []
  hand_starts = []
  for h, hand in enumerate(hands):
    hs, acts, final = replay_hand(env, hand, fns)
    if h < len(hands) - 1 and not bool(fns.node(final)[1]):
      raise ValueError(
          f"hand {h} of {len(hands)} is not terminal; only the last hand may "
          "be "
          "in progress"
      )
    hand_starts.append(len(states))
    n = len(hs)
    la = np.full((n,), types.NO_ACTION, np.int32)
    lact = np.full((n,), types.NO_ACTOR, np.int32)
    if acts:
      prev = tree_map(lambda *xs: jnp.stack(xs), *hs[:-1])
      vis = np.asarray(
          fns.visible(prev, jnp.asarray(acts, jnp.int32))
      )  # [n-1, P]
      la[1:] = vis[:, int(seat)]
      lact[1:] = np.asarray(fns.actors(prev))
    states.extend(hs)
    new_hand.extend([True] + [False] * (n - 1))
    last_action.append(la)
    last_actor.append(lact)
  t = len(states)
  batched = tree_map(lambda *xs: jnp.stack(xs), *states)
  new_match = np.zeros((t,), np.bool_)
  new_match[0] = True
  steps = fns.seat_step(
      batched,
      jnp.int32(seat),
      jnp.asarray(new_hand, jnp.bool_),
      jnp.asarray(new_match),
      jnp.asarray(np.concatenate(last_action)),
      jnp.asarray(np.concatenate(last_actor)),
  )
  steps = tree_map(lambda x: x[None], steps)
  return SeatReplay(
      steps=steps, state=states[-1], num_plies=t, hand_starts=tuple(hand_starts)
  )
