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

"""An approximate belief by sequential importance resampling over particles.

When a game is too large to enumerate, the posterior over hidden worlds can
still be approximated from the Env contract and an opponent model alone. The
sampler keeps `num_particles` complete game states per row and updates them
at every SeatStep the seat observes:

  1. At the first ply of a hand, particles are dealt with `env.init` and kept
     only if `env.info_state(particle, seat)` equals the observed InfoState;
     dead particles are re-dealt up to `deal_attempts` times (rejection from
     the deal prior, which is exact).
  2. At every later ply each particle replays the transition the seat saw.
     If the seat itself acted, its recorded action is applied. If an opponent
     acted, its action is drawn from the opponent model restricted to the
     actions whose `visible_action` matches what the seat saw, and the
     particle's log-weight gains the log of that restricted mass. If the
     action leads to a chance node, its outcomes are enumerated and one is
     drawn from those after which the particle agrees with the observed
     InfoState, weighted by their chance probability; further chance nodes
     before the next ply are sampled.
  3. A particle whose InfoState differs from the observed one in any leaf is
     killed (log-weight -inf).
  4. When the effective sample size falls below `resample_ess *
     num_particles`, particles are resampled in proportion to their weights.
  5. If every particle dies, the sampler restarts: fresh particles are dealt
     and replayed along the hand's recorded actions, up to `max_restarts`
     times. A row where every restart fails is degenerate
     (`degenerate(carry)`), and `sample_worlds` returns its particles with
     log-weight -inf.

  sampler = registry.build("particle", env, opponent=model, num_particles=1024)
  carry = sampler.init_carry(model_params, batch, seat, key)
  carry = sampler.observe(model_params, carry, step)  # once per SeatStep
  worlds, logw = sampler.sample_worlds(model_params, carry, step.info, k, key)

The belief is approximate: it converges to the exact posterior under the
opponent model as the number of particles grows, and its accuracy depends on
how much of the opponent's support the particles cover. With one chance node
between consecutive plies the importance weights are exact; with more, the
chance nodes after the first are sampled and inconsistent draws are killed.
Every method is jittable (fixed shapes). The carry holds a PRNG key per row,
because `observe` takes no key.
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple, Union

import jax
import jax.numpy as jnp

from open_spiel.python.scale.belief import registry
from open_spiel.python.scale.core import chance as chance_lib
from open_spiel.python.scale.core import decide as decide_lib
from open_spiel.python.scale.core import plies
from open_spiel.python.scale.core import types
from open_spiel.python.scale.policies import registry as policy_registry

F32, F64, I32 = jnp.float32, jnp.float64, jnp.int32
tree_map = jax.tree_util.tree_map


class ParticleCarry(NamedTuple):
  """Per-row particle state, leaves [B, ...].

  Attributes:
    states: env states [B, N, ...].
    opp_carry: the model's carries for every seat [B, N, P, ...].
    logw: f64[B, N], the log-weights.
    seat: i32[B], the searching seat.
    key: u32[B, 2], raw PRNG key data.
    log_action: i32[B, L], the hand's observed actions, as the seat saw them.
    log_actor: i32[B, L], their actors.
    log_len: i32[B], the length of the log.
    degenerate: bool[B], every particle died and no restart was left.
  """

  states: Any  # env states [B, N, ...]
  opp_carry: Any  # the model's carries for every seat [B, N, P, ...]
  logw: jax.Array  # f64[B, N]
  seat: jax.Array  # i32[B]
  key: jax.Array  # u32[B, 2] raw PRNG key data
  # i32[B, L] the hand's observed actions, as the seat saw them
  log_action: jax.Array
  log_actor: jax.Array  # i32[B, L] their actors
  log_len: jax.Array  # i32[B]
  degenerate: jax.Array  # bool[B]


def _rows_where(cond, a, b):
  """Selects pytree `a` where `cond` holds, else `b`, row by row.

  Args:
    cond: the bool rows, broadcast over each leaf's trailing axes.
    a: the pytree taken where `cond` holds.
    b: the pytree taken elsewhere.

  Returns:
    The selected pytree.
  """

  def sel(x, y):
    """Selects `x` where `cond` holds, broadcasting `cond` over trailing axes.

    Args:
      x: the leaf taken where `cond` holds.
      y: the leaf taken elsewhere.

    Returns:
      The selected leaf.
    """
    c = jnp.reshape(
        cond, jnp.shape(cond) + (1,) * (jnp.ndim(x) - jnp.ndim(cond))
    )
    return jnp.where(c, x, y)

  return tree_map(sel, a, b)


def _same_info(env, states, seat, info) -> jax.Array:
  """bool[N]: env.info_state(particle, seat) equals `info` in every leaf.

  Args:
    env: The env.
    states: The particles [N].
    seat: The seat.
    info: The InfoState.

  Returns:
    bool[N].
  """
  got = jax.vmap(lambda s: env.info_state(s, seat))(states)
  eq = [
      jnp.all(jnp.reshape(g == w[None], (jnp.shape(g)[0], -1)), axis=1)
      for g, w in zip(
          jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(info)
      )
  ]
  return jnp.all(jnp.stack(eq, axis=0), axis=0)


class ParticleSampler:
  """Sequential importance resampling over complete game states (module doc).

  Attributes:
    env: the Env.
    opponent: the opponent model, a NATIVE_JAX Policy.
    num_particles: N, particles per row.
    resample_ess: the effective-sample-size fraction of N below which the
      particles are resampled.
    max_restarts: how many times a row whose particles all died is re-dealt
      and replayed before it is marked degenerate.
    deal_attempts: how many times a particle inconsistent with the first ply
      of a hand is re-dealt.
    kwargs: the sampler's constructor kwargs, as recorded.
  """

  capability = types.Capability.NATIVE_JAX
  stateful = True
  is_perfect_information = False
  params_source = "opponent"

  def __init__(
      self,
      env: types.Env,
      opponent: Union[types.Policy, Mapping[str, Any]],
      *,
      num_particles: int = 256,
      resample_ess: float = 0.5,
      max_restarts: int = 8,
      deal_attempts: int = 64,
  ):
    """Builds the sampler.

    Args:
      env: any Env.
      opponent: the opponent model, a NATIVE_JAX Policy (or its
        `{"arch_name", "arch_kwargs"}` record), used for every seat other than
        the searching one.
      num_particles: N, particles per row.
      resample_ess: resample when the effective sample size is below this
        fraction of N; in (0, 1].
      max_restarts: how many times a row whose particles all died is re-dealt
        and replayed before it is marked degenerate.
      deal_attempts: how many times a particle inconsistent with the first ply
        of a hand is re-dealt.

    Raises:
      ValueError: invalid sizes, or an opponent model that is not NATIVE_JAX
        or does not play `env`.
    """
    if isinstance(opponent, Mapping):
      opponent = policy_registry.build(
          opponent["arch_name"], opponent["arch_kwargs"]
      )
    spec = opponent.spec
    if spec.capability != types.Capability.NATIVE_JAX:
      raise ValueError(
          f"the opponent model must be NATIVE_JAX, got {spec.capability}"
      )
    if spec.game != env.name or spec.num_actions != env.num_actions:
      raise ValueError(
          f"the opponent model plays {spec.game}; the env is {env.name}"
      )
    if (
        int(num_particles) < 1
        or not 0.0 < float(resample_ess) <= 1.0
        or int(max_restarts) < 0
        or int(deal_attempts) < 1
    ):
      raise ValueError(
          f"num_particles={num_particles} (>= 1), resample_ess={resample_ess} "
          f"(in (0, 1]), max_restarts={max_restarts} (>= 0), "
          f"deal_attempts={deal_attempts} (>= 1)"
      )
    self.env, self.opponent = env, opponent
    self.num_particles = int(num_particles)
    self.resample_ess = float(resample_ess)
    self.max_restarts = int(max_restarts)
    self.deal_attempts = int(deal_attempts)
    self._log_len = int(env.max_len) + 1
    self.kwargs = {
        "opponent": {
            "arch_name": spec.arch_name,
            "arch_kwargs": dict(spec.arch_kwargs),
        },
        "num_particles": self.num_particles,
        "resample_ess": self.resample_ess,
        "max_restarts": self.max_restarts,
        "deal_attempts": self.deal_attempts,
    }

  def init_params(self, key: Any = None):
    """The opponent model's params.

    Args:
      key: the PRNG key for the model's init; None uses key 0.

    Returns:
      The opponent model's initial params.
    """
    return self.opponent.init_params(jax.random.key(0) if key is None else key)

  # -- one row's particles -----------------------------------------------------
  def _fresh_carries(self, params, key):
    """The model's initial carries for every seat of N particles [N, P, ...].

    Args:
      params: The model's params.
      key: The key.

    Returns:
      The carries, leaves [N, P, ...].
    """
    n, p = self.num_particles, self.env.num_players
    seats = jnp.broadcast_to(jnp.arange(p, dtype=I32), (n, p)).reshape(-1)
    c = self.opponent.init_carry(params, n * p, seats, key)
    return tree_map(lambda x: x.reshape((n, p) + x.shape[1:]), c)

  def _observe_all(
      self, params, carries, states, visible, last_actor, new_hand
  ):
    """Every seat's carry observes the ply at `states` [N].

    Args:
      params: The model's params.
      carries: The carries [N, P, ...].
      states: The particles [N].
      visible: The visible last action.
      last_actor: The last actor.
      new_hand: Whether a hand starts.

    Returns:
      The carries after observing.
    """
    n, p = self.num_particles, self.env.num_players
    flag = jnp.full((n,), new_hand, jnp.bool_)
    steps = plies.seat_steps(
        self.env,
        states,
        flag,
        flag,
        visible,
        jnp.broadcast_to(last_actor, (n,)).astype(I32),
    )

    def flat(x):
      """Merges the particle and seat axes.

      Args:
        x: A leaf, [n, p, ...].

      Returns:
        The leaf, [n * p, ...].
      """
      return x.reshape((n * p,) + x.shape[2:])

    c = self.opponent.observe(
        params, tree_map(flat, carries), tree_map(flat, steps)
    )
    return tree_map(lambda x: x.reshape((n, p) + x.shape[1:]), c)

  def _deal(self, params, key):
    """N fresh first-ply particles and the model carries after observing them.

    Args:
      params: the opponent model's params.
      key: a PRNG key.

    Returns:
      (states [N], carries [N, P, ...]).
    """
    kd, kc = jax.random.split(key)
    states = jax.vmap(self.env.init)(jax.random.split(kd, self.num_particles))
    carries = self._fresh_carries(params, kc)
    none = jnp.full(
        (self.num_particles, self.env.num_players), types.NO_ACTION, I32
    )
    carries = self._observe_all(
        params, carries, states, none, types.NO_ACTOR, True
    )
    return states, carries

  def _deal_consistent(self, params, seat, info, key):
    """First-ply particles consistent with `info`: (states, carries, alive).

    Args:
      params: The model's params.
      seat: The seat.
      info: The InfoState.
      key: The key.

    Returns:
      (states, carries, alive).
    """
    states, carries = self._deal(params, jax.random.fold_in(key, 0))
    alive = _same_info(self.env, states, seat, info)

    def body(i, c):
      """Re-deals the rows not yet consistent with `info`.

      Args:
        i: the attempt.
        c: (states, carries, alive).

      Returns:
        The updated (states, carries, alive).
      """
      st, ca, al = c
      st2, ca2 = self._deal(params, jax.random.fold_in(key, i))
      ok2 = _same_info(self.env, st2, seat, info)
      take = ~al & ok2
      return _rows_where(take, st2, st), _rows_where(take, ca2, ca), al | ok2

    return jax.lax.fori_loop(
        1, self.deal_attempts, body, (states, carries, alive)
    )

  def _resolve_chance(self, states, seat, info, key):
    """Resolves the chance nodes after an action.

    The first is resolved by enumeration over the outcomes consistent with
    `info`, the rest by sampling.

    Args:
      states: the particles' states [N].
      seat: the searching seat.
      info: the seat's InfoState.
      key: a PRNG key.

    Returns:
      (states, log chance mass of the consistent outcomes f64[N]).
    """
    env, n = self.env, self.num_particles
    if env.max_chance_outcomes == 0:
      return states, jnp.zeros((n,), F64)
    c = int(env.max_chance_outcomes)
    k_pick, k_rest = jax.random.split(key)
    at_chance = jax.vmap(env.is_chance)(states) & ~jax.vmap(env.terminal)(
        states
    )
    probs = jax.vmap(env.chance_probs)(states).astype(F64)  # [N, C]
    rest_keys = jax.random.split(k_rest, n)

    def outcome(o):
      """Applies chance outcome `o` to every particle and resolves the rest.

      Args:
        o: the chance outcome.

      Returns:
        (states [N], whether each agrees with `info` bool[N]).
      """
      s_o = jax.vmap(env.apply_chance)(states, jnp.full((n,), o, I32))
      s_o = jax.vmap(lambda s, k: chance_lib.resolve(env, s, k, 0)[0])(
          s_o, rest_keys
      )
      return s_o, _same_info(env, s_o, seat, info)

    # [N, C]
    outs, match = jax.vmap(outcome, out_axes=(1, 1))(jnp.arange(c, dtype=I32))
    w = jnp.where(match & (probs > 0), probs, F64(0.0))
    mass = jnp.sum(w, axis=1)
    logits = jnp.where(w > 0, jnp.log(jnp.where(w > 0, w, F64(1.0))), -jnp.inf)
    safe = jnp.where((mass > 0)[:, None], logits, F64(0.0))
    pick = jax.vmap(jax.random.categorical)(
        jax.random.split(k_pick, n), safe
    ).astype(I32)
    chosen = tree_map(lambda x: x[jnp.arange(n), pick], outs)
    states = _rows_where(at_chance, chosen, states)
    logp = jnp.where(
        mass > 0, jnp.log(jnp.where(mass > 0, mass, F64(1.0))), -jnp.inf
    )
    return states, jnp.where(at_chance, logp, F64(0.0))

  def _advance(self, params, states, carries, seat, action, actor, info, key):
    """Every particle takes the transition the seat saw.

    The transition is the action as seen by the seat, and its actor. With
    `info` the chance nodes after the action are resolved consistently with it
    (_resolve_chance); with None they are sampled from their prior.

    Args:
      params: the opponent model's params.
      states: the particles' states [N].
      carries: the model carries [N, P, ...].
      seat: the searching seat.
      action: the action as the seat saw it.
      actor: its actor.
      info: the seat's InfoState after the transition, or None.
      key: a PRNG key.

    Returns:
      (states, carries, log-weight increment f64[N]).
    """
    env, n = self.env, self.num_particles
    k_eval, k_pick, k_chance = jax.random.split(key, 3)
    acts = jnp.arange(env.num_actions, dtype=I32)
    visible = jax.vmap(
        lambda s: jax.vmap(lambda a: env.visible_action(s, a, seat))(acts)
    )(states)
    legal = jax.vmap(env.legal_mask)(states)
    match = legal & (visible == action)
    mover = jnp.maximum(jax.vmap(env.actor)(states), 0).astype(I32)
    step = plies.seat_step(
        env, states, mover, False, False, types.NO_ACTION, types.NO_ACTOR
    )
    own = tree_map(lambda x: x[jnp.arange(n), mover], carries)
    out = self.opponent.evaluate(params, own, step, k_eval)
    _, probs = decide_lib.decide(out, step.legal_mask, types.SAMPLE, k_eval)
    probs = jnp.asarray(probs, F64)
    by_model = mover != seat
    # The seat's own action carries no information about the hidden world
    # (perfect recall), so it is weighted 1; only the model's choices weigh.
    w = jnp.where(
        match, jnp.where(by_model[:, None], probs, F64(1.0)), F64(0.0)
    )
    mass = jnp.sum(w, axis=1)
    logits = jnp.where(w > 0, jnp.log(jnp.where(w > 0, w, F64(1.0))), -jnp.inf)
    safe = jnp.where((mass > 0)[:, None], logits, F64(0.0))
    a = jax.vmap(jax.random.categorical)(
        jax.random.split(k_pick, n), safe
    ).astype(I32)
    if info is None:
      nxt = jax.vmap(env.step)(states, a, jax.random.split(k_chance, n))
      chance_logp = jnp.zeros((n,), F64)
    else:
      applied = jax.vmap(env.apply)(states, a)
      nxt, chance_logp = self._resolve_chance(applied, seat, info, k_chance)
    seen = plies.visible_actions(env, states, a)
    carries = self._observe_all(params, carries, nxt, seen, actor, False)
    logp = jnp.where(
        by_model, jnp.log(jnp.where(mass > 0, mass, F64(1.0))), F64(0.0)
    )
    return nxt, carries, jnp.where(mass > 0, logp + chance_logp, -jnp.inf)

  def _replay(self, params, seat, info, log_action, log_actor, log_len, key):
    """Fresh particles dealt and replayed along the logged actions.

    Only the final InfoState `info` is checked.

    Args:
      params: the opponent model's params.
      seat: the searching seat.
      info: the seat's final InfoState.
      log_action: i32[L], the logged actions, as the seat saw them.
      log_actor: i32[L], their actors.
      log_len: the length of the log.
      key: a PRNG key.

    Returns:
      (states, carries, log-weights f64[N]); -inf where inconsistent with
      `info`.
    """
    k_deal, k_play = jax.random.split(key)
    states, carries = self._deal(params, k_deal)
    logw = jnp.zeros((self.num_particles,), F64)

    def body(t, c):
      """Replays logged ply `t` on the rows whose log reaches it.

      Args:
        t: the ply.
        c: (states, carries, log-weights).

      Returns:
        The updated (states, carries, log-weights).
      """
      st, ca, lw = c
      # The InfoStates of intermediate plies are not kept, so chance is drawn
      # from its prior and `info` is checked after the last transition.
      st2, ca2, lp = self._advance(
          params,
          st,
          ca,
          seat,
          log_action[t],
          log_actor[t],
          None,
          jax.random.fold_in(k_play, t),
      )
      on = t < log_len
      return (
          _rows_where(on, st2, st),
          _rows_where(on, ca2, ca),
          jnp.where(on, lw + lp, lw),
      )

    states, carries, logw = jax.lax.fori_loop(
        0, self._log_len, body, (states, carries, logw)
    )
    alive = _same_info(self.env, states, seat, info)
    return states, carries, jnp.where(alive, logw, -jnp.inf)

  def _resample(self, states, carries, logw, key):
    """Resamples the particles when their effective sample size is low.

    Below `resample_ess * num_particles`, with any particle alive, the particles
    are drawn by weight and the log-weights reset to 0; otherwise they are
    returned unchanged.

    Args:
      states: the particles' states [N].
      carries: the model carries [N, ...].
      logw: the log-weights f64[N].
      key: a PRNG key.

    Returns:
      (states, carries, log-weights).
    """
    n = self.num_particles
    finite = jnp.isfinite(logw)
    lw = jnp.where(
        finite, logw - jnp.max(jnp.where(finite, logw, -jnp.inf)), -jnp.inf
    )
    w = jnp.exp(lw)
    ess = jnp.square(jnp.sum(w)) / jnp.maximum(
        jnp.sum(jnp.square(w)), F64(1e-300)
    )
    do = jnp.any(finite) & (ess < self.resample_ess * n)
    idx = jax.random.categorical(
        key, jnp.where(finite, lw, -jnp.inf), shape=(n,)
    ).astype(I32)
    idx = jnp.where(do, idx, jnp.arange(n, dtype=I32))

    def pick(x):
      """The resampled particles.

      Args:
        x: A particle leaf.

      Returns:
        The leaf at idx.
      """
      return x[idx]

    return (
        tree_map(pick, states),
        tree_map(pick, carries),
        jnp.where(do, jnp.zeros_like(logw), logw),
    )

  def _deal_row(self, params, c: ParticleCarry, step):
    """The fresh particles of a new hand at a row: (states, carries, alive).

    Args:
      params: The model's params.
      c: The row's ParticleCarry.
      step: The row's SeatStep.

    Returns:
      (states, carries, alive).
    """
    _, k_deal, _, _, _ = jax.random.split(jax.random.wrap_key_data(c.key), 5)
    return self._deal_consistent(params, c.seat, step.info, k_deal)

  def _observe_row(
      self, params, c: ParticleCarry, step, dealt
  ) -> ParticleCarry:
    """One row's particles after its seat's next observation.

    A new hand takes the dealt particles; any other ply advances them. When
    every particle dies the row restarts from fresh particles replayed along its
    action log (at most max_restarts times), then resamples.

    Args:
      params: the opponent model's params.
      c: the row's carry.
      step: the row's step.
      dealt: (states, carries, alive), the row's fresh particles of a new hand.

    Returns:
      the row's next carry.
    """
    key = jax.random.wrap_key_data(c.key)
    key, _, k_step, k_res, k_restart = jax.random.split(key, 5)
    seat = c.seat
    new_hand = jnp.asarray(step.new_hand, jnp.bool_)

    # A new hand deals fresh particles (`dealt`, _deal_row); any other ply
    # advances them.
    d_states, d_carries, d_alive = dealt
    a_states, a_carries, a_logp = self._advance(
        params,
        c.states,
        c.opp_carry,
        seat,
        step.last_action,
        step.last_actor,
        step.info,
        k_step,
    )
    a_alive = _same_info(self.env, a_states, seat, step.info)
    states = _rows_where(new_hand, d_states, a_states)
    carries = _rows_where(new_hand, d_carries, a_carries)
    logw = jnp.where(
        new_hand,
        jnp.where(d_alive, F64(0.0), -jnp.inf),
        jnp.where(a_alive, c.logw + a_logp, -jnp.inf),
    )

    pos = jnp.where(new_hand, 0, c.log_len)
    log_action = jnp.where(
        new_hand, jnp.full_like(c.log_action, types.NO_ACTION), c.log_action
    )
    log_actor = jnp.where(
        new_hand, jnp.full_like(c.log_actor, types.NO_ACTOR), c.log_actor
    )
    write = ~new_hand & (pos < self._log_len)
    slot = jnp.minimum(pos, self._log_len - 1)
    log_action = jnp.where(
        write, log_action.at[slot].set(step.last_action), log_action
    )
    log_actor = jnp.where(
        write, log_actor.at[slot].set(step.last_actor), log_actor
    )
    log_len = jnp.where(new_hand, 0, jnp.minimum(c.log_len + 1, self._log_len))

    def retry(x):
      """Restarts the row: fresh particles replayed along the log.

      Args:
        x: (states, carries, log-weights, restarts so far).

      Returns:
        (states, carries, log-weights, restarts so far).
      """
      st, ca, lw = self._replay(
          params,
          seat,
          step.info,
          log_action,
          log_actor,
          log_len,
          jax.random.fold_in(k_restart, x[3]),
      )
      return (st, ca, lw, x[3] + 1)

    def dead(x):
      """Whether every particle is dead and a restart is left.

      Args:
        x: (states, carries, log-weights, restarts so far).

      Returns:
        A scalar bool.
      """
      return ~jnp.any(jnp.isfinite(x[2])) & (x[3] < self.max_restarts)

    states, carries, logw, _ = jax.lax.while_loop(
        dead, retry, (states, carries, logw, 0)
    )
    degenerate = ~jnp.any(jnp.isfinite(logw))
    states, carries, logw = self._resample(states, carries, logw, k_res)
    return ParticleCarry(
        states=states,
        opp_carry=carries,
        logw=logw,
        seat=seat,
        key=jax.random.key_data(key),
        log_action=log_action,
        log_actor=log_actor,
        log_len=log_len,
        degenerate=degenerate,
    )

  # -- BeliefSampler -----------------------------------------------------------
  def init_carry(self, params, batch, seat, key) -> ParticleCarry:
    """Placeholder particles for `batch` rows of seat `seat`.

    Args:
      params: the opponent model's params.
      batch: the number of rows B.
      seat: the seat of each row, a scalar or i32[B].
      key: a PRNG key, split into one key per row.

    Returns:
      The initial ParticleCarry; the first ply of a hand deals real particles.
    """
    seat = jnp.broadcast_to(jnp.asarray(seat, I32), (batch,))
    states = jax.vmap(self.env.init)(
        jax.random.split(jax.random.key(0), self.num_particles)
    )
    carries = self._fresh_carries(params, jax.random.key(0))

    def rep(x):
      """Broadcasts a leaf over the batch.

      Args:
        x: A leaf.

      Returns:
        The leaf, [batch, ...].
      """
      return jnp.broadcast_to(x[None], (batch,) + x.shape)

    keys = jax.random.key_data(jax.random.split(key, batch))
    return ParticleCarry(
        states=tree_map(rep, states),
        opp_carry=tree_map(rep, carries),
        logw=jnp.full((batch, self.num_particles), -jnp.inf, F64),
        seat=seat,
        key=keys,
        log_action=jnp.full((batch, self._log_len), types.NO_ACTION, I32),
        log_actor=jnp.full((batch, self._log_len), types.NO_ACTOR, I32),
        log_len=jnp.zeros((batch,), I32),
        degenerate=jnp.zeros((batch,), jnp.bool_),
    )

  def observe(self, params, carry: ParticleCarry, step) -> ParticleCarry:
    """Advances every row's particles by one SeatStep.

    The rejection-sampled deal of a new hand (deal_attempts x num_particles
    env.init) runs only when some row starts a hand; a row that does not ignores
    it.

    Args:
      params: the opponent model's params.
      carry: the ParticleCarry.
      step: the SeatStep, leaves [B, ...].

    Returns:
      The updated ParticleCarry.
    """

    def deal():
      """Deals fresh particles for every row.

      Returns:
        (states, carries, alive) per row.
      """
      return jax.vmap(lambda c, s: self._deal_row(params, c, s))(carry, step)

    def no_deal():  # same structure as deal(); every row ignores it
      """The current particles, in the same structure as deal().

      Returns:
        (states, carries, alive) per row; every row ignores it.
      """
      return (
          carry.states,
          carry.opp_carry,
          jnp.zeros(carry.logw.shape, jnp.bool_),
      )

    dealt = jax.lax.cond(
        jnp.any(jnp.asarray(step.new_hand, jnp.bool_)), deal, no_deal
    )
    return jax.vmap(lambda c, s, d: self._observe_row(params, c, s, d))(
        carry, step, dealt
    )

  def degenerate(self, carry: ParticleCarry) -> jax.Array:
    """Rows whose particles all died and could not be restarted.

    Args:
      carry: the ParticleCarry.

    Returns:
      bool[B].
    """
    return carry.degenerate

  def belief(self, carry: ParticleCarry) -> jax.Array:
    """Each row's normalized particle weights.

    Args:
      carry: the ParticleCarry.

    Returns:
      f64[B, N], uniform on a degenerate row.
    """
    finite = jnp.isfinite(carry.logw)
    logits = jnp.where(
        jnp.any(finite, axis=1, keepdims=True),
        jnp.where(finite, carry.logw, -jnp.inf),
        F64(0.0),
    )
    return jax.nn.softmax(logits, axis=1)

  def sample_worlds(self, params, carry: ParticleCarry, info, k: int, key):
    """Draws K worlds per row from the particles.

    Worlds are drawn from the particles in proportion to their weights, so a
    draw's log-weight is 0; a degenerate row's draws are uniform over its
    particles and get -inf.

    Args:
      params: unused (the particles already carry the model).
      carry: the ParticleCarry.
      info: unused (the carry has observed it).
      k: worlds per row.
      key: a PRNG key.

    Returns:
      (env states with leaves [B, K, ...], log-weights f64[B, K]).

    Raises:
      ValueError: `k` < 1.
    """
    del params, info
    if k < 1:
      raise ValueError(f"k must be >= 1, got {k}")
    b = carry.logw.shape[0]
    logits = jnp.log(self.belief(carry))
    keys = jax.random.split(key, b)
    idx = jax.vmap(lambda kk, lw: jax.random.categorical(kk, lw, shape=(k,)))(
        keys, logits
    )
    rows = jnp.arange(b, dtype=I32)[:, None]
    worlds = tree_map(lambda x: x[rows, idx.astype(I32)], carry.states)
    logw = jnp.broadcast_to(
        jnp.where(carry.degenerate, -jnp.inf, F64(0.0))[:, None], (b, k)
    )
    return worlds, logw


registry.register("particle", ParticleSampler)
