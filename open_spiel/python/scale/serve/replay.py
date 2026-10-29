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

Length buckets. jit compiles once per input shape, so a replay stacked to
exactly T plies would compile its batched calls (visible actions, actors,
seat_step) once per history length, and a long game would compile at nearly
every move. Instead the plies are stacked on the host and padded to
bucket_size(T), the next power of two, by repeating the last ply's state
(every batched call is a vmap over plies, so a padded ply cannot change a
real one): a match of up to N plies compiles O(log N) shapes, and
warmup(env, seat, max_plies) compiles all of them ahead of play.

  rep = replay.replay_seat_steps_padded(env, hands, seat)
  # SeatStep, leaves [1, L], L = bucket_size(T); plies >= T are padding
  rep.steps
  rep.last         # SeatStep [1]: ply T - 1 (numpy, sliced on the host)
  rep.exact()      # the SeatReplay [1, T] (sliced on the host: no compile)

replay_seat_steps is replay_seat_steps_padded(...).exact(): the same SeatSteps,
bit for bit, as stacking exactly T plies (serve/tests/replay_bucket_test.py).

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


class IllegalHistoryActionError(ValueError):
  """A history action the env does not allow where it was played.

  A ValueError: the request is refused. A serving shell that tolerates the
  env and pyspiel disagreeing (server.PolicyServer.decide, repair) catches
  it: the env's rules and pyspiel's diverged earlier in the hand.
  """


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
    hand_chunk: Jitted _hand_chunk (replay_hand_last).
    last_step: Jitted _last_step (replay_last_seat_step).
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
    # SeatStep [1, B]: the batch-of-one axis added in the jitted call (one
    # trace per length; an eager x[None] would trace once per leaf shape).
    self.seat_step = jax.jit(
        lambda states, seat, nh, nm, la, lact: tree_map(
            lambda x: x[None],
            plies.seat_step(_env(), states, seat, nh, nm, la, lact),
        )
    )
    self.hand_chunk = jax.jit(
        lambda carry, actions, valid: _hand_chunk(_env(), carry, actions, valid)
    )
    self.last_step = jax.jit(
        lambda final, prev, a, has_prev, seat, nh, nm: _last_step(
            _env(), final, prev, a, has_prev, seat, nh, nm
        )
    )


# Actions one replay_hand_last call applies per jitted call (a lax.scan).
HAND_CHUNK = 32


def _at(x: jax.Array, a: jax.Array) -> jax.Array:
  """x[a], False (0) when a is out of range (x may be empty).

  Args:
    x: a 1-D array.
    a: an i32 index.

  Returns:
    x[a] or a zero of x's dtype.
  """
  n = x.shape[0]
  if n == 0:
    return jnp.zeros((), x.dtype)
  inside = (a >= 0) & (a < n)
  return jnp.where(inside, x[jnp.clip(a, 0, n - 1)], jnp.zeros((), x.dtype))


def _hand_chunk(env: types.Env, carry, actions, valid):
  """Applies up to HAND_CHUNK actions of a hand in one scan (replay_hand_last).

  Args:
    env: the env.
    carry: (state, state at the last decision node, its action, decisions).
    actions: i32[K], the actions.
    valid: bool[K]; False steps (padding) leave the carry unchanged.

  Returns:
    (carry, (terminal, is_chance, chance_ok, legal_ok)), flags bool[K] of the
    node each action was applied at.
  """

  def body(c, xs):
    """One action of the scan.

    Args:
      c: The carry.
      xs: (action, valid).

    Returns:
      (carry, flags).
    """
    st, prev, last_a, n_dec = c
    a, v = xs
    is_chance = env.is_chance(st)
    terminal = env.terminal(st)
    legal = env.legal_mask(st)
    probs = env.chance_probs(st)
    chance_ok = _at(probs, a) > 0
    legal_ok = _at(legal, a)
    branch = jnp.where(v, jnp.where(is_chance, 1, 2), 0)
    # An env without chance outcomes (an empty chance_probs) never reaches
    # branch 1 with a valid action (chance_ok is False and the host raises);
    # its apply_chance is not traced.
    nxt = jax.lax.switch(
        branch,
        (
            lambda x: x,
            (lambda x: env.apply_chance(x, a))
            if probs.shape[0]
            else (lambda x: x),
            lambda x: env.apply(x, a),
        ),
        st,
    )
    dec = v & jnp.logical_not(is_chance)
    kept = jax.lax.cond(dec, lambda x, y: x, lambda x, y: y, st, prev)
    last_a = jnp.where(dec, a, last_a)
    n_dec = n_dec + dec.astype(jnp.int32)
    flags = (terminal, is_chance, chance_ok, legal_ok)
    return (nxt, kept, last_a, n_dec), flags

  return jax.lax.scan(body, carry, (actions, valid))


def _last_step(env: types.Env, final, prev, a, has_prev, seat, nh, nm):
  """The SeatStep [1, 1] of `seat` at `final` (replay_last_seat_step).

  Args:
    env: the env.
    final: the state of the ply.
    prev: the state of the hand's previous ply (when has_prev).
    a: i32, the action taken there.
    has_prev: bool, whether the ply is not its hand's first.
    seat: i32, the seat.
    nh: bool, new_hand.
    nm: bool, new_match.

  Returns:
    The SeatStep, leaves [1, 1].
  """
  prev1 = tree_map(lambda x: x[None], prev)
  vis = plies.visible_actions(env, prev1, a[None])[0, seat]
  actor = jax.vmap(env.actor)(prev1)[0].astype(jnp.int32)
  la = jnp.where(has_prev, vis, jnp.int32(types.NO_ACTION))
  lact = jnp.where(has_prev, actor, jnp.int32(types.NO_ACTOR))
  return tree_map(
      lambda x: x[None],
      plies.seat_step(
          env,
          tree_map(lambda x: x[None], final),
          seat,
          nh[None],
          nm[None],
          la[None],
          lact[None],
      ),
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
    # One device-to-host transfer of the node's four leaves (per-leaf bool()
    # and np.asarray each synchronise and copy).
    is_chance, terminal, legal, probs = jax.device_get(fns.node(s))
    if bool(terminal):
      raise ValueError(f"action {i} ({a}) comes after the hand is over")
    if bool(is_chance):
      if not (0 <= a < probs.shape[0]) or not probs[a] > 0:
        raise ValueError(
            f"chance outcome {a} has probability 0 (or is out of range) at "
            f"history index {i}"
        )
      s = fns.apply_chance(s, np.int32(a))
    else:
      if not (0 <= a < legal.shape[0]) or not legal[a]:
        raise IllegalHistoryActionError(
            f"illegal action {a} at history index {i}"
        )
      states.append(s)
      actions.append(a)
      s = fns.apply(s, np.int32(a))
  if bool(fns.node(s)[0]):
    raise ValueError(
        "the history ends at a chance node; there is no ply to replay"
    )
  states.append(s)
  return states, actions, s


def replay_hand_last(
    env: types.Env, hand: Sequence[int], fns: Optional[_Fns] = None
):
  """(plies, last decision state, its action, final state) of one hand.

  replay_hand's result reduced to what the last ply needs: the number of ply
  states (decision nodes and the final node), the state at the last decision
  node and the action taken there (None, None when the hand has no decision
  yet), and the final node. The actions are applied HAND_CHUNK at a time,
  each chunk one jitted scan, and checked as replay_hand checks them, with
  the same errors.

  Args:
    env: the env.
    hand: the hand's raw actions.
    fns: the env's jitted calls (default the cached ones).

  Returns:
    (plies, last decision state or None, its action or None, final state).

  Raises:
    ValueError: an action is illegal, comes after the hand is over, or the hand
      does not reach a node.
  """
  fns = fns or _fns(env)
  root = fns.root()
  carry = (root, root, jnp.int32(types.NO_ACTION), jnp.int32(0))
  hand = [int(a) for a in hand]
  for c0 in range(0, len(hand), HAND_CHUNK):
    part = hand[c0 : c0 + HAND_CHUNK]
    k = len(part)
    actions = np.zeros((HAND_CHUNK,), np.int32)
    actions[:k] = part
    valid = np.zeros((HAND_CHUNK,), np.bool_)
    valid[:k] = True
    carry, flags = fns.hand_chunk(carry, actions, valid)
    terminal, is_chance, chance_ok, legal_ok = jax.device_get(flags)
    for j in range(k):
      i, a = c0 + j, part[j]
      if terminal[j]:
        raise ValueError(f"action {i} ({a}) comes after the hand is over")
      if is_chance[j]:
        if not chance_ok[j]:
          raise ValueError(
              f"chance outcome {a} has probability 0 (or is out of range) at "
              f"history index {i}"
          )
      elif not legal_ok[j]:
        raise IllegalHistoryActionError(
            f"illegal action {a} at history index {i}"
        )
  final, prev, last_a, n_dec = carry
  if bool(fns.node(final)[0]):
    raise ValueError(
        "the history ends at a chance node; there is no ply to replay"
    )
  n_dec = int(n_dec)
  if n_dec == 0:
    return 1, None, None, final
  return n_dec + 1, prev, int(last_a), final


def bucket_size(n: int) -> int:
  """The padded length of n >= 1 plies: the smallest power of two >= n.

  Args:
    n: the plies.

  Returns:
    The bucket length.

  Raises:
    ValueError: n < 1.
  """
  n = int(n)
  if n < 1:
    raise ValueError(f"a bucket holds at least one ply, got {n}")
  return 1 << (n - 1).bit_length()


def buckets(max_plies: int) -> tuple[int, ...]:
  """Every padded length a match of at most max_plies plies uses.

  1, 2, 4, ...

  Args:
    max_plies: the longest match in plies.

  Returns:
    The bucket lengths.
  """
  return tuple(1 << k for k in range(bucket_size(max_plies).bit_length()))


def _stack_padded(xs: Sequence[Any], length: int):
  """The pytrees `xs` stacked on a new leading axis on the host.

  Padded to `length` by repeating the last one, as device arrays.

  Args:
    xs: the pytrees, 1 <= len(xs) <= length.
    length: the stacked length.

  Returns:
    The stacked pytree, leaves [length, ...].
  """
  assert 1 <= len(xs) <= length, (len(xs), length)

  def one(*leaves):
    """One leaf of the stacked pytree.

    Args:
      *leaves: the leaf of each pytree.

    Returns:
      The stacked, padded leaf.
    """
    if jax.dtypes.issubdtype(leaves[0].dtype, jax.dtypes.prng_key):
      impl = jax.random.key_impl(leaves[0])
      return jax.random.wrap_key_data(
          one(*(jax.random.key_data(x) for x in leaves)), impl=impl
      )
    a = np.stack([np.asarray(x) for x in leaves])
    if length > a.shape[0]:
      a = np.concatenate([a, np.repeat(a[-1:], length - a.shape[0], axis=0)])
    return jnp.asarray(a)

  return tree_map(one, *xs)


def _pad1(a: np.ndarray, length: int, fill) -> jax.Array:
  """`a` padded along its first axis to `length` with `fill`.

  Args:
    a: An array.
    length: The padded length.
    fill: The pad value.

  Returns:
    The padded array, on device.
  """
  out = np.full((length,) + a.shape[1:], fill, a.dtype)
  out[: a.shape[0]] = a
  return jnp.asarray(out)


class PaddedSeatReplay(NamedTuple):
  """A replay whose plies are padded to a bucket length (module doc)."""

  steps: types.SeatStep  # leaves [1, L]; plies num_plies..L-1 are padding
  state: Any  # the env state at the last ply (unbatched)
  num_plies: int  # T
  hand_starts: tuple[Any, ...]  # ply index of each hand's first ply

  @property
  def length(self) -> int:
    """L, the padded length."""
    return int(self.steps.seat.shape[1])

  @property
  def last(self) -> types.SeatStep:
    """The SeatStep [1] of the last real ply, as numpy."""
    t = self.num_plies - 1
    return tree_map(lambda x: np.asarray(x)[:, t], self.steps)

  def exact(self) -> SeatReplay:
    """The SeatReplay of the real plies, [1, T] (sliced on the host).

    Returns:
      The SeatReplay.
    """
    t = self.num_plies
    steps = tree_map(lambda x: jnp.asarray(np.asarray(x)[:, :t]), self.steps)
    return SeatReplay(
        steps=steps, state=self.state, num_plies=t, hand_starts=self.hand_starts
    )


def _hand_last(
    fns: _Fns, hs: list[Any], acts: list[Any], seat: int
) -> tuple[np.ndarray, np.ndarray]:
  """(last_action, last_actor) i32[n] of a hand's n plies, as `seat` sees.

  Args:
    fns: The env's compiled functions.
    hs: The hand's n states.
    acts: Its n - 1 actions.
    seat: The viewing seat.

  Returns:
    (last_action, last_actor), i32[n] each.
  """
  n = len(hs)
  la = np.full((n,), types.NO_ACTION, np.int32)
  lact = np.full((n,), types.NO_ACTOR, np.int32)
  if acts:
    m = len(acts)  # n - 1
    length = bucket_size(m)
    prev = _stack_padded(hs[:-1], length)
    a = _pad1(np.asarray(acts, np.int32), length, acts[-1])
    vis = np.asarray(fns.visible(prev, a))[:m]  # [n-1, P]
    la[1:] = vis[:, int(seat)]
    lact[1:] = np.asarray(fns.actors(prev))[:m]
  return la, lact


def _seat_step_padded(
    fns: _Fns,
    states: list[Any],
    seat: int,
    new_hand: np.ndarray,
    last_action: np.ndarray,
    last_actor: np.ndarray,
    new_match: Optional[np.ndarray] = None,
) -> types.SeatStep:
  """seat_step over `states` padded to bucket_size(T), leaves [1, L].

  Padded plies repeat the last state with new_hand False and no last action.

  Args:
    fns: the env's jitted calls.
    states: the T states.
    seat: the seat observing.
    new_hand: bool[T].
    last_action: i32[T], each ply's last visible action.
    last_actor: i32[T], each ply's last actor.
    new_match: bool[T]; None: True on the first ply only (states are a whole
      match from its first ply).

  Returns:
    The SeatStep, leaves [1, L].
  """
  t = len(states)
  length = bucket_size(t)
  if new_match is None:
    new_match = np.zeros((t,), np.bool_)
    new_match[0] = True
  assert new_match.shape == (t,), (new_match.shape, t)
  return fns.seat_step(
      _stack_padded(states, length),
      jnp.int32(seat),
      _pad1(new_hand, length, False),
      _pad1(new_match, length, False),
      _pad1(last_action, length, types.NO_ACTION),
      _pad1(last_actor, length, types.NO_ACTOR),
  )


def replay_seat_steps_padded(
    env: types.Env, hands: Sequence[Sequence[int]], seat: int
) -> PaddedSeatReplay:
  """SeatStep [1, bucket_size(T)] of `seat` over the plies of `hands`.

  One match.

  Args:
    env: the env.
    hands: the match's raw hands.
    seat: the perspective seat.

  Returns:
    The PaddedSeatReplay.

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
          "be in progress"
      )
    hand_starts.append(len(states))
    la, lact = _hand_last(fns, hs, acts, int(seat))
    states.extend(hs)
    new_hand.extend([True] + [False] * (len(hs) - 1))
    last_action.append(la)
    last_actor.append(lact)
  steps = _seat_step_padded(
      fns,
      states,
      int(seat),
      np.asarray(new_hand, np.bool_),
      np.concatenate(last_action),
      np.concatenate(last_actor),
  )
  return PaddedSeatReplay(
      steps=steps,
      state=states[-1],
      num_plies=len(states),
      hand_starts=tuple(hand_starts),
  )


class LastSeatStep(NamedTuple):
  """The last ply of a seat's replay, all a stateless policy reads.

  Attributes:
    last: the SeatStep [1] of the last ply (numpy), bit for bit
      replay_seat_steps_padded(...).last.
    state: the env state at the last ply (unbatched).
    num_plies: T, the plies of the match.
  """

  last: types.SeatStep
  state: Any
  num_plies: int


def replay_last_seat_step(
    env: types.Env, hands: Sequence[Sequence[int]], seat: int
) -> LastSeatStep:
  """The last ply's SeatStep of `seat` over `hands` (one match).

  What a stateless policy is served on: the same SeatStep as
  replay_seat_steps_padded(env, hands, seat).last (every batched call is a
  vmap over plies, so one ply computed alone is that ply of the stack), with
  seat_step, visible actions and actors run on that ply alone (one jitted
  call) instead of on every ply of the match. The history is still applied
  action by action from the root, and checked, HAND_CHUNK actions per jitted
  call (replay_hand_last).

  Args:
    env: the env.
    hands: the match's raw hands.
    seat: the perspective seat.

  Returns:
    The LastSeatStep.

  Raises:
    ValueError: seat is out of range, there are no hands, or a hand but the last
      is not terminal.
  """
  if not 0 <= int(seat) < env.num_players:
    raise ValueError(f"seat {seat} is not in [0, {env.num_players})")
  if not hands:
    raise ValueError("no hands to replay")
  fns = _fns(env)
  num_plies = 0
  n, prev, last_a, final = 0, None, None, None
  for h, hand in enumerate(hands):
    n, prev, last_a, final = replay_hand_last(env, hand, fns)
    if h < len(hands) - 1 and not bool(fns.node(final)[1]):
      raise ValueError(
          f"hand {h} of {len(hands)} is not terminal; only the last hand may "
          "be in progress"
      )
    num_plies += n
  has_prev = prev is not None
  steps = fns.last_step(
      final,
      final if prev is None else prev,
      np.int32(types.NO_ACTION if last_a is None else last_a),
      np.bool_(has_prev),
      np.int32(seat),
      np.bool_(n == 1),
      np.bool_(num_plies == 1),
  )
  return LastSeatStep(
      last=tree_map(lambda x: np.asarray(x)[:, 0], steps),
      state=final,
      num_plies=num_plies,
  )


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
  """
  return replay_seat_steps_padded(env, hands, seat).exact()


def first_decision_state(env: types.Env) -> Any:
  """The env's first decision node.

  Reached from the root by the lowest chance outcome of positive probability
  at every chance node.

  Args:
    env: the env.

  Returns:
    The state.

  Raises:
    RuntimeError: no decision node after 100000 chance nodes.
  """
  fns = _fns(env)
  s = fns.root()
  for _ in range(100_000):
    is_chance, _, _, probs = fns.node(s)
    if not bool(is_chance):
      return s
    s = fns.apply_chance(
        s, np.int32(int(np.flatnonzero(np.asarray(probs) > 0)[0]))
    )
  raise RuntimeError(f"{env.name}: no decision node after 100000 chance nodes")


def first_decision_step(env: types.Env, seat: int) -> types.SeatStep:
  """The SeatStep [1] (numpy) of `seat` at first_decision_state(env).

  As the first ply of a match: a last ply of the shapes and dtypes a replay
  gives.

  Args:
    env: the env.
    seat: the seat.

  Returns:
    The SeatStep [1].
  """
  steps = _seat_step_padded(
      _fns(env),
      [first_decision_state(env)],
      int(seat),
      np.ones((1,), np.bool_),
      np.full((1,), types.NO_ACTION, np.int32),
      np.full((1,), types.NO_ACTOR, np.int32),
  )
  return tree_map(lambda x: np.asarray(x)[:, 0], steps)


def warmup(
    env: types.Env, seat: int, max_plies: int, *, last_only: bool = False
) -> None:
  """Compiles the batched calls of every bucket a match may use.

  Every bucket a match of at most max_plies plies uses, for `seat`, so no
  replay of such a match compiles. The calls run on copies of the env's root
  state: jit compiles by shape and dtype, so these are the compilations a real
  replay of that length reuses. With last_only, only what
  replay_last_seat_step uses (one shape each), whatever max_plies.

  Args:
    env: the env.
    seat: the seat.
    max_plies: the longest match in plies.
    last_only: compile for replay_last_seat_step only.

  Raises:
    ValueError: seat is out of range.
  """
  if not 0 <= int(seat) < env.num_players:
    raise ValueError(f"seat {seat} is not in [0, {env.num_players})")
  fns = _fns(env)
  # compiles root, node and (with chance) apply_chance
  node = first_decision_state(env)
  # The per-action calls of replay_hand and the node check, whatever the
  # values (a chance outcome applied at a decision node is computed and
  # dropped: only the shapes are compiled).
  legal = np.asarray(fns.node(node)[2])
  jax.block_until_ready((
      fns.apply(node, np.int32(int(np.flatnonzero(legal)[0]))),
      fns.apply_chance(node, np.int32(0)),
      fns.decision_node(node),
  ))
  if last_only:
    root = fns.root()
    carry = (root, root, jnp.int32(types.NO_ACTION), jnp.int32(0))
    jax.block_until_ready((
        fns.hand_chunk(
            carry,
            np.zeros((HAND_CHUNK,), np.int32),
            np.zeros((HAND_CHUNK,), np.bool_),
        ),
        fns.last_step(
            node,
            node,
            np.int32(types.NO_ACTION),
            np.bool_(False),
            np.int32(seat),
            np.bool_(True),
            np.bool_(True),
        ),
    ))
    return
  for length in buckets(max_plies):
    states = [node] * length
    _seat_step_padded(
        fns,
        states,
        int(seat),
        np.zeros((length,), np.bool_),
        np.full((length,), types.NO_ACTION, np.int32),
        np.full((length,), types.NO_ACTOR, np.int32),
    )
    # visible / actors run on a hand's plies but its last (no longer a hand).
    prev = _stack_padded(states, length)
    a = jnp.asarray(np.zeros((length,), np.int32))
    jax.block_until_ready((fns.visible(prev, a), fns.actors(prev)))
