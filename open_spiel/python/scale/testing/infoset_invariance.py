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

"""The infoset-invariance test: a policy's output sees only its seat's view.

A search operator, a teacher or any policy that emits training targets must
not read hidden state: in an imperfect-information game that silently turns
it into an oracle. Comparing two InfoStates proves nothing for an in-graph
policy, because it only ever receives InfoStates. The test therefore works at
the level of env states and goes through the production path:

  1. Play seeded random playouts s_0..s_T.
  2. Build a partner history h_t = hide_fn(env, s_t, seat, key_t) with the
     env's privacy hook, which re-deals everything hidden from `seat` (for
     example the opponent's cards and the stock order).
  3. Turn both histories into SeatSteps with core.plies.seat_step, run the
     policy's observe_seq over the whole history and evaluate at the last
     ply under the same keys.

check_infoset_invariance asserts, in order:
  1. the seat's SeatSteps are byte-identical on both histories (otherwise the
     ENV leaks hidden information into the InfoState);
  2. the final hidden states differ on most rows (otherwise hide_fn is a
     no-op and the test is vacuous);
  3. the policy's logits, value and every aux leaf are byte-identical.

For host-driven policies, `interleave=True` evaluates real, partner, real and
requires the two real outputs equal (catching a cache across calls), and
`fresh=` evaluates the real side again on a freshly built policy.

  case = infoset_invariance.InfosetCase(
      "search/leduc", env, policy, params, leduc.hide_opponent_card,
      plies=(1, 2, 3), rows=24)
  infoset_invariance.check_infoset_invariance(case)

Coverage. `walk()` lists every registered architecture whose class declares
`decider == "search"` or `emits_target_policy`, plus the runtime target
emitters, and returns those with no registered fixture; a test fails on any.
`register_fixture(key, fn)` adds cases for an architecture (or a runtime
emitter's class name); several fixtures may share a key.
"""

from __future__ import annotations

import os
from typing import Any, Callable, NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale import plugins
from open_spiel.python.scale.core import plies
from open_spiel.python.scale.core import types
from open_spiel.python.scale.interop import host
from open_spiel.python.scale.policies import operators
from open_spiel.python.scale.policies import registry
from open_spiel.python.scale.sources import dagger
from open_spiel.python.scale.testing.errors import ContractError

tree_map = jax.tree_util.tree_map

# "real" or "partner" while that side is evaluated, else None. Not part of the
# API: only this module's own negative controls (tests/infoset_invariance_test)
# read it, to build a policy that sees which side runs and so must fail. One
# evaluation runs at a time per process (it is a module global, not
# thread-safe).
_ACTIVE_SIDE: Optional[str] = None
#: The environment variable that, set to "1", asks fixtures for their large
#: position counts (read when a fixture is built: full()).
FULL_ENV = plugins.ENV_PREFIX + "INFOSET_INVARIANCE_FULL"


def full() -> bool:
  """Returns whether FULL_ENV asks fixtures for their large position counts.

  Returns:
    True when FULL_ENV is "1".
  """
  return os.environ.get(FULL_ENV, "") == "1"


class NotInvariantError(ContractError):
  """The infoset-invariance test does not hold (module doc)."""


class InfosetCase(NamedTuple):
  """One infoset-invariance case.

  Attributes:
    name: reported in failures.
    env: the Env.
    policy: the policy under test.
    params: its params (one member).
    hide_fn: (env, state, seat, key) -> a state the seat cannot tell apart.
    plies: history lengths T; the seat decides at ply T.
    rows: playouts per history length.
    seed: the playout seed.
    fresh: builds a new instance of the policy; its output must equal the
      tested instance's.
    interleave: evaluate real, partner, real and require equal real outputs.
  """

  name: str
  env: types.Env
  policy: types.Policy
  params: Any
  hide_fn: Callable[[Any, Any, Any, Any], Any]
  plies: Sequence[int]
  rows: int
  seed: int = 0
  fresh: Optional[Callable[[], types.Policy]] = None
  interleave: bool = False


def _bytes_equal(a, b) -> bool:
  """Whether two pytrees are byte-identical.

  Args:
    a: A pytree.
    b: The other pytree.

  Returns:
    True when equal.
  """
  la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
  if len(la) != len(lb):
    return False
  for x, y in zip(la, lb):
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape or x.dtype != y.dtype or x.tobytes() != y.tobytes():
      return False
  return True


def _row_bytes_equal(a, b) -> np.ndarray:
  """bool[N]: per row, every leaf of a and b byte-identical.

  Args:
    a: A pytree, leaves [N, ...].
    b: Another such pytree.

  Returns:
    bool[N].
  """
  la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
  assert len(la) == len(lb), (len(la), len(lb))
  n = np.asarray(la[0]).shape[0] if la else 0
  ok = np.ones((n,), bool)
  for x, y in zip(la, lb):
    x, y = np.asarray(x), np.asarray(y)
    assert x.shape == y.shape and x.dtype == y.dtype, (
        x.shape,
        y.shape,
        x.dtype,
        y.dtype,
    )
    xb = np.ascontiguousarray(x).view(np.uint8).reshape(n, -1)
    yb = np.ascontiguousarray(y).view(np.uint8).reshape(n, -1)
    ok &= (xb == yb).all(axis=1)
  return ok


def playouts(env: types.Env, n: int, t_max: int, seed: int):
  """Seeded playouts with uniformly random legal actions.

  Args:
    env: The env.
    n: The number of playouts.
    t_max: The number of plies played.
    seed: The seed.

  Returns:
    (states [n, t_max + 1, ...], actor i32[n, t_max + 1],
    visible i32[n, t_max + 1, P]); visible[:, t] is the action into ply t as
    each seat sees it (NO_ACTION at t = 0 and after the hand ended). A
    finished hand stays at its terminal state.
  """
  p = env.num_players
  kd, kp = jax.random.split(jax.random.key(seed))
  deal = jax.random.split(kd, n)

  def one(dk, pk):
    """Plays one seeded playout.

    Args:
      dk: The deal key.
      pk: The play key.

    Returns:
      (states, actor, visible) over the playout's plies.
    """
    s0 = env.init(dk)

    def body(c, k):
      """Plays one uniformly random legal action; a no-op after the end.

      Args:
        c: (state,).
        k: The ply's key.

      Returns:
        ((next state,), (next state, its actor, the visible action per seat)).
      """
      (s,) = c
      ka, kc = jax.random.split(k)
      legal = env.legal_mask(s)
      term = env.terminal(s)
      a = jax.random.categorical(
          ka, jnp.where(legal, jnp.float32(0.0), -jnp.inf)
      )
      a = jnp.where(term, 0, a).astype(jnp.int32)
      vis = jax.vmap(lambda q: env.visible_action(s, a, q))(
          jnp.arange(p, dtype=jnp.int32)
      )
      s2 = env.step(s, a, kc)
      s2 = tree_map(lambda x, y: jnp.where(term, x, y), s, s2)
      vis = jnp.where(term, types.NO_ACTION, vis).astype(jnp.int32)
      return (s2,), (s2, env.actor(s2).astype(jnp.int32), vis)

    _, (ss, act, vis) = jax.lax.scan(body, (s0,), jax.random.split(pk, t_max))

    def cat(a0, rest):
      """Prepends a0 to `rest`.

      Args:
        a0: The first entry.
        rest: The rest, stacked.

      Returns:
        The concatenation.
      """
      return jnp.concatenate([a0[None], rest])

    states = tree_map(cat, s0, ss)
    actor = cat(env.actor(s0).astype(jnp.int32), act)
    visible = cat(jnp.full((p,), types.NO_ACTION, jnp.int32), vis)
    return states, actor, visible

  return jax.jit(jax.vmap(one))(deal, jax.random.split(kp, n))


def seat_steps(env, states, seat, actor, visible, t_len):
  """Returns the SeatSteps of `seat` over the playout states.

  They are what the Arena would emit (core.plies.seat_step).

  Args:
    env: The env.
    states: The playout states [N, t_len].
    seat: i32[N], each row's seat.
    actor: i32[N, T], the actor at each ply.
    visible: i32[N, T, P], the action into each ply as each seat sees it.
    t_len: The number of plies.

  Returns:
    SeatStep [N, t_len].
  """
  n = seat.shape[0]
  steps = []
  for t in range(t_len):
    s_t = tree_map(lambda x, t=t: x[:, t], states)
    first = jnp.full((n,), t == 0)
    la = (
        jnp.full((n,), types.NO_ACTION, jnp.int32)
        if t == 0
        else visible[jnp.arange(n), t][jnp.arange(n), seat]
    )
    lr = (
        jnp.full((n,), types.NO_ACTOR, jnp.int32) if t == 0 else actor[:, t - 1]
    )
    steps.append(plies.seat_step(env, s_t, seat, first, first, la, lr))
  return tree_map(lambda *xs: jnp.stack(xs, axis=1), *steps)


def _is_jax(policy) -> bool:
  """Whether `policy` runs in JAX (not host-driven).

  Args:
    policy: The policy.

  Returns:
    True for a JAX policy.
  """
  return policy.spec.framework == "jax" and (
      policy.spec.capability != types.Capability.HOST_DRIVEN
  )


def _evaluate(policy, params, steps, seat, key, side: str):
  """observe_seq over every ply, then evaluate at the last ply.

  A JAX policy runs under a fresh jit per call, so a trace-time read of
  _ACTIVE_SIDE also differs per side.

  Args:
    policy: The policy.
    params: Its params.
    steps: The SeatSteps of every ply, stacked on axis 1.
    seat: i32[B] the seat of each row.
    key: The key of the carry and the evaluation.
    side: Which side of the comparison this run is (_ACTIVE_SIDE).

  Returns:
    The PolicyOutput at the last ply, as numpy.
  """
  global _ACTIVE_SIDE
  n = seat.shape[0]
  last = tree_map(lambda x: x[:, -1], steps)
  k_init, k_eval = jax.random.split(key)
  _ACTIVE_SIDE = side
  try:
    if _is_jax(policy):

      def run(p, st, s, ki, ke):
        """Initializes the carry, observes every ply and evaluates the last.

        Args:
          p: The params.
          st: The SeatSteps [N, T].
          s: The seats.
          ki: The init_carry key.
          ke: The evaluate key.

        Returns:
          The PolicyOutput at the last ply.
        """
        c = policy.init_carry(p, n, s, ki)
        c = policy.observe_seq(p, c, st)
        return policy.evaluate(p, c, tree_map(lambda x: x[:, -1], st), ke)

      out = jax.jit(run)(params, steps, seat, k_init, k_eval)
    else:
      st_np, seat_np = host.to_numpy(steps), np.asarray(seat)
      c = policy.init_carry(
          params, n, seat_np, host.raw_keys(jax.random.split(k_init, n))
      )
      c = policy.observe_seq(params, c, st_np)
      out = policy.evaluate(
          params,
          c,
          host.to_numpy(last),
          host.raw_keys(jax.random.split(k_eval, n)),
      )
  finally:
    _ACTIVE_SIDE = None
  return host.to_numpy(out)


def check_infoset_invariance(
    case: InfosetCase, only_plies: Optional[Sequence[int]] = None
) -> dict[str, Any]:
  """Runs the infoset-invariance test of one case (module doc).

  Args:
    case: the case.
    only_plies: the history lengths to check, a subset of case.plies
      (default: all). Each is checked exactly as in the full check (same
      playouts and keys), so the checks of a partition of case.plies are the
      full check split up (a test runner can spread them over files).

  Returns:
    {positions, per_phase, rows_hidden_differ}: positions tested, their count
    per InfoState `phase` value when the InfoState has one, and the rows
    whose hidden state differed.

  Raises:
    NotInvariantError: on the first property that does not hold.
    ValueError: only_plies names a history length the case does not have.
  """
  env, pol = case.env, case.policy
  if only_plies is not None and not set(only_plies) <= set(case.plies):
    raise ValueError(
        f"{case.name}: plies {sorted(only_plies)} are not all "
        f"among {case.plies}"
    )
  t_max = max(case.plies)
  states, actor, visible = playouts(env, case.rows, t_max, case.seed)
  positions, per_phase, differ = 0, {}, 0
  for i, t in enumerate(sorted(set(case.plies))):
    if only_plies is not None and t not in only_plies:
      continue
    a_t = np.asarray(actor[:, t])
    term = np.asarray(
        jax.vmap(env.terminal)(tree_map(lambda x, t=t: x[:, t], states))
    )
    rows = np.flatnonzero((a_t >= 0) & ~term)
    if rows.size == 0:
      continue
    sub = tree_map(lambda x, rows=rows, t=t: x[rows, : t + 1], states)
    seat = jnp.asarray(a_t[rows], jnp.int32)
    act, vis = actor[rows], visible[rows]
    m = rows.size
    keys = jax.random.split(
        jax.random.fold_in(jax.random.key(case.seed + 1), i), m * (t + 1)
    )
    keys = keys.reshape((m, t + 1) + keys.shape[1:])
    partner = jax.jit(
        jax.vmap(
            jax.vmap(
                lambda s, q, k: case.hide_fn(env, s, q, k), in_axes=(0, None, 0)
            ),
            in_axes=(0, 0, 0),
        )
    )(sub, seat, keys)
    real_steps = seat_steps(env, sub, seat, act, vis, t + 1)
    partner_steps = seat_steps(env, partner, seat, act, vis, t + 1)
    if not _bytes_equal(real_steps, partner_steps):
      ok = _row_bytes_equal(real_steps, partner_steps)
      raise NotInvariantError(
          f"{case.name}: the seat's SeatStep differs between a history and its "
          "hide_fn "
          f"partner at history length {t} on {int((~ok).sum())} of {m} rows: "
          "the env leaks "
          "hidden information (info_state reads something hide_fn re-deals)"
      )
    last_real = tree_map(lambda x: x[:, -1], sub)
    last_partner = tree_map(lambda x: x[:, -1], partner)
    same_hidden = _row_bytes_equal(last_real, last_partner)
    differ += int((~same_hidden).sum())
    if same_hidden.mean() > 0.5:
      raise NotInvariantError(
          f"{case.name}: hide_fn left {int(same_hidden.sum())} of {m} final "
          "states unchanged "
          f"at history length {t}: a no-op hide_fn makes the test vacuous"
      )
    key = jax.random.fold_in(jax.random.key(case.seed + 2), i)
    out_r = _evaluate(pol, case.params, real_steps, seat, key, "real")
    out_p = _evaluate(pol, case.params, partner_steps, seat, key, "partner")
    if not _bytes_equal(out_r, out_p):
      bad = ~_row_bytes_equal(out_r, out_p)
      raise NotInvariantError(
          f"{case.name}: the policy's output (logits, value or aux) differs "
          "between a history "
          f"and its hide_fn partner on {int(bad.sum())} of {m} rows at history "
          f"length {t}: it reads something outside the seat's InfoState"
      )
    if case.interleave:
      again = _evaluate(pol, case.params, real_steps, seat, key, "real")
      if not _bytes_equal(out_r, again):
        raise NotInvariantError(
            f"{case.name}: real, partner, real gave a different second real "
            "output: the policy carries state across calls"
        )
    if case.fresh is not None:
      other = _evaluate(
          case.fresh(), case.params, real_steps, seat, key, "real"
      )
      if not _bytes_equal(out_r, other):
        raise NotInvariantError(
            f"{case.name}: a freshly built policy gives a different output"
        )
    positions += m
    ph = getattr(real_steps.info, "phase", None)
    if ph is not None:
      for v, c in zip(*np.unique(np.asarray(ph)[:, -1], return_counts=True)):
        per_phase[int(v)] = per_phase.get(int(v), 0) + int(c)
  if positions == 0:
    checked = case.plies if only_plies is None else sorted(only_plies)
    raise NotInvariantError(
        f"{case.name}: no decision position at history lengths {checked}"
    )
  return dict(
      positions=positions, per_phase=per_phase, rows_hidden_differ=differ
  )


# -- coverage ------------------------------------------------------------------

#: key (an architecture name or a runtime emitter's class name) -> case
#: builders.
FIXTURES: dict[str, list[Callable[[], Sequence[InfosetCase]]]] = {}


def register_fixture(key: str, fn: Callable[[], Sequence[InfosetCase]]) -> None:
  """Adds fn() -> [InfosetCase, ...] to the fixtures of `key`.

  Args:
    key: An architecture name or a runtime emitter's class name.
    fn: The case builder.

  Raises:
    ValueError: `fn` is already registered for `key`.
  """
  fns = FIXTURES.setdefault(key, [])
  if fn in fns:
    raise ValueError(
        f"infoset-invariance fixture {fn!r} is already registered for {key!r}"
    )
  fns.append(fn)


def cases(key: str) -> list[InfosetCase]:
  """Returns every InfosetCase registered for `key`, in registration order.

  Args:
    key: An architecture name or a runtime emitter's class name.

  Returns:
    The cases.

  Raises:
    KeyError: no fixture is registered for `key`.
  """
  if key not in FIXTURES:
    raise KeyError(
        f"no infoset-invariance fixture for {key!r}; registered: "
        f"{sorted(FIXTURES)}"
    )
  out = []
  for fn in FIXTURES[key]:
    out.extend(fn())
  return out


def needs_fixture(cls: type[Any]) -> bool:
  """Returns whether `cls` needs an infoset-invariance fixture.

  Args:
    cls: The policy class.

  Returns:
    True when the class declares decider == "search" or emits_target_policy.
  """
  return getattr(cls, "decider", None) == "search" or bool(
      getattr(cls, "emits_target_policy", False)
  )


def runtime_emitters() -> dict[str, type[Any]]:
  """Returns the target emitters built at run time, by class name.

  They are built at run time rather than through the registry: the DAgger
  mixture and the behaviour/label operator mixture.

  Returns:
    {class name: class}.
  """
  return {
      cls.__name__: cls
      for cls in (dagger.DaggerPolicy, operators.BehaviorMixture)
  }


_Fixtures = dict[str, list[Callable[[], Sequence[InfosetCase]]]]


def walk(fixtures: Optional[_Fixtures] = None) -> list[str]:
  """Returns the architectures that need a fixture and have none.

  They are registered architectures (and runtime emitters).

  Args:
    fixtures: The fixtures by key; defaults to FIXTURES.

  Returns:
    Their keys, sorted.

  Raises:
    ContractError: an architecture whose class the registry cannot name (the
      registered ctor is neither a class nor has `policy_class`).
  """
  fixtures = FIXTURES if fixtures is None else fixtures
  need, unknown = {}, []
  for name in registry.registered():
    cls = registry.policy_class(name)
    if cls is None:
      unknown.append(name)
    elif needs_fixture(cls):
      need[name] = cls
  if unknown:
    raise ContractError(
        "registered architectures with no policy class (register a class, or "
        f"a ctor with .policy_class): {unknown}"
    )
  need.update(runtime_emitters())
  return sorted(k for k in need if k not in fixtures)
