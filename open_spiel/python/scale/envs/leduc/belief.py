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

"""The exact posterior over the opponent's hidden card in leduc poker.

Leduc hides one card per player. Given a seat's InfoState (its own card, the
public card and both betting rounds) and an opponent model pi (a stateless
policy over InfoStates), Bayes' rule gives

  P(c | history) ∝ 1[c != own card, c != public card]
                   * prod over the opponent's decisions t of pi(a_t |
                   I_opp_t(c))

because the deal prior is uniform over the cards the seat cannot rule out.
Each hypothesis c is checked by replaying the public history in the env, so
the opponent's InfoState at each of its decisions comes from
`Env.info_state`, and the replayed state for c is the world returned.

  sampler = registry.build("leduc", env, opponent=model)
  probs = sampler.posterior(model_params, info)          # f32[B, 6]
  worlds, logw = sampler.sample_worlds(model_params, (), info, k, key)

The sampler reads only the InfoState. If the model gives the observed history
probability 0 under every possible card, the posterior falls back to the
card-removal prior: a model that cannot explain the history carries no
information.

Params are the opponent model's params, and the sampler declares
`params_source = "own"`: under search it is fed the params of the network
being searched. The opponent model must therefore take that network's params
(the same architecture); SearchPolicy checks that the two param trees match.
"""

from typing import Any, Mapping, Union

import jax
import jax.numpy as jnp

from open_spiel.python.scale.belief import registry
from open_spiel.python.scale.core import decide as decide_lib
from open_spiel.python.scale.core import plies
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import leduc
from open_spiel.python.scale.policies import registry as policy_registry
from open_spiel.python.scale.policies import uniform

NUM_CARDS = leduc.NUM_CARDS
F64 = jnp.float64


def _tree_where(c, a, b):
  """Selects `a` where `c`, else `b`, leaf by leaf.

  Args:
    c: The condition.
    a: The tree where it holds.
    b: The tree elsewhere.

  Returns:
    The selected tree.
  """
  return jax.tree_util.tree_map(lambda x, y: jnp.where(c, x, y), a, b)


class LeducBeliefSampler:
  """The exact posterior over leduc's hidden opponent card. Stateless.

  Attributes:
    env: The leduc env.
    opponent: The opponent model (a built Policy).
    kwargs: The constructor options as a record: the opponent's
      `{"arch_name", "arch_kwargs"}`.
  """

  capability = types.Capability.NATIVE_JAX
  stateful = False
  is_perfect_information = False
  params_source = "own"

  def __init__(
      self, env: types.Env, opponent: Union[types.Policy, Mapping[str, Any]]
  ):
    """Builds the sampler.

    Args:
      env: The leduc env.
      opponent: The opponent model, a stateless NATIVE_JAX Policy over leduc
        InfoStates, or its `{"arch_name", "arch_kwargs"}` record. Under
        search it receives the searched network's params (module doc), so it
        must have that network's architecture.

    Raises:
      ValueError: a non-leduc env, or a stateful or non-NATIVE_JAX model.
    """
    if env.name != "leduc_poker" or not hasattr(env, "apply_chance"):
      raise ValueError(
          f"LeducBeliefSampler needs the leduc env, got {env.name!r}"
      )
    if isinstance(opponent, Mapping):
      opponent = policy_registry.build(
          opponent["arch_name"], opponent["arch_kwargs"]
      )
    spec = opponent.spec
    if spec.stateful:
      raise ValueError(
          "the opponent model must be stateless (it is evaluated on replayed "
          f"InfoStates without a history); {spec.arch_name} is stateful"
      )
    if spec.capability != types.Capability.NATIVE_JAX:
      raise ValueError(
          f"the opponent model must be NATIVE_JAX, got {spec.capability}"
      )
    if spec.game != env.name or spec.num_actions != env.num_actions:
      raise ValueError(
          f"the opponent model plays {spec.game} with {spec.num_actions} "
          f"actions; the env is {env.name} with {env.num_actions}"
      )
    self.env = env
    self.opponent = opponent
    self.kwargs = {
        "opponent": {
            "arch_name": spec.arch_name,
            "arch_kwargs": dict(spec.arch_kwargs),
        }
    }

  def init_params(self, key: Any = None):
    """The opponent model's params.

    Args:
      key: PRNG key (None: key 0).

    Returns:
      The opponent model's initial params.
    """
    return self.opponent.init_params(jax.random.key(0) if key is None else key)

  def init_carry(self, params, batch, seat, key):
    """The sampler's carry (it keeps none).

    Args:
      params: Unused.
      batch: Unused.
      seat: Unused.
      key: Unused.

    Returns:
      An empty tuple.
    """
    del params, batch, seat, key
    return ()

  def observe(self, params, carry, step):
    """Passes the carry through unchanged.

    Args:
      params: Unused.
      carry: The carry.
      step: Unused.

    Returns:
      The carry.
    """
    del params, step
    return carry

  def _opp_logp(self, params, s, opp, a) -> jax.Array:
    """f64 log pi(a | the opponent's InfoState at s).

    Args:
      params: The opponent's params.
      s: The state.
      opp: The opponent's seat.
      a: The action, -1 for none.

    Returns:
      f64[], the log-probability (0 for no action).
    """
    s1 = jax.tree_util.tree_map(lambda x: x[None], s)
    step = plies.seat_step(
        self.env, s1, opp, False, False, types.NO_ACTION, types.NO_ACTOR
    )
    # A stateless model is deterministic given its inputs (Policy contract), so
    # any fixed key gives the same probabilities.
    key = jax.random.key(0)
    c = self.opponent.init_carry(params, 1, step.seat, key)
    c = self.opponent.observe(params, c, step)
    out = self.opponent.evaluate(params, c, step, key)
    _, probs = decide_lib.decide(out, step.legal_mask, types.SAMPLE, key)
    p = probs[0, jnp.maximum(a, 0)]
    return jnp.where(
        p > 0, jnp.log(jnp.where(p > 0, p, F64(1.0))), -jnp.inf
    ).astype(F64)

  def _replay_one(self, params, info):
    """(states [6], log-likelihood f64[6]) for one unbatched LeducInfo.

    Args:
      params: The opponent's params.
      info: The LeducInfo.

    Returns:
      (states [6], log-likelihood f64[6]).
    """
    env = self.env
    seat = jnp.asarray(info.seat, jnp.int32)
    opp = jnp.int32(1) - seat
    own = jnp.asarray(info.card, jnp.int32)
    pub = jnp.asarray(info.public_card, jnp.int32)

    def act(carry, a):
      """Replays one action, accumulating the opponent's log-likelihood.

      Args:
        carry: (state, log-likelihood).
        a: The action, -1 past the end.

      Returns:
        ((state, log-likelihood), None).
      """
      s, logl = carry
      live = a >= 0
      lp = self._opp_logp(params, s, opp, a)
      logl = logl + jnp.where(live & (env.actor(s) == opp), lp, F64(0.0))
      s = _tree_where(live, env.apply(s, jnp.maximum(a, 0)), s)
      return (s, logl), None

    def one(c):
      """Replays the history with opponent card c.

      Args:
        c: The opponent card.

      Returns:
        (state, log-likelihood, -inf when c is impossible).
      """
      possible = (c != own) & (c != pub)
      # An impossible hypothesis is replayed with a legal stand-in card (any
      # card other than own and pub) so the replay stays a real game; it is
      # masked to -inf below.
      stand_in = jnp.int32(0)
      for alt in range(NUM_CARDS):
        stand_in = jnp.where(
            (stand_in == own) | (stand_in == pub), jnp.int32(alt), stand_in
        )
      c_play = jnp.where(possible, c, stand_in)
      first = jnp.where(seat == 0, own, c_play)
      second = jnp.where(seat == 0, c_play, own)
      s = env.apply_chance(env.apply_chance(env.root(), first), second)
      (s, logl), _ = jax.lax.scan(
          act, (s, F64(0.0)), jnp.asarray(info.round1, jnp.int32)
      )
      deal_pub = (pub >= 0) & env.is_chance(s)
      s = _tree_where(deal_pub, env.apply_chance(s, jnp.maximum(pub, 0)), s)
      (s, logl), _ = jax.lax.scan(
          act, (s, logl), jnp.asarray(info.round2, jnp.int32)
      )
      return s, jnp.where(possible, logl, -jnp.inf)

    return jax.vmap(one)(jnp.arange(NUM_CARDS, dtype=jnp.int32))

  def _replay(self, params, info):
    """_replay_one over a batch of InfoStates.

    Args:
      params: The opponent's params.
      info: The InfoStates [B].

    Returns:
      (states [B, 6], log-likelihood f64[B, 6]).
    """
    return jax.vmap(lambda i: self._replay_one(params, i))(info)

  def log_likelihood(self, params, info) -> jax.Array:
    """f64[B, 6]: log P(history | c) (with the prior's support), -inf off it.

    Args:
      params: The opponent model's params.
      info: A batched InfoState.

    Returns:
      f64[B, 6] log-likelihoods.
    """
    return self._replay(params, info)[1]

  @staticmethod
  def _log_posterior(info, logl):
    """The log posterior over the opponent's card.

    Args:
      info: The InfoStates [B].
      logl: f64[B, 6], the log-likelihoods.

    Returns:
      f64[B, 6]: the likelihoods, or the uniform prior when none is finite.
    """
    own = jnp.asarray(info.card, jnp.int32)[:, None]
    pub = jnp.asarray(info.public_card, jnp.int32)[:, None]
    ids = jnp.arange(NUM_CARDS, dtype=jnp.int32)[None]
    prior = jnp.where((ids != own) & (ids != pub), F64(0.0), -jnp.inf)
    explained = jnp.any(jnp.isfinite(logl), axis=-1, keepdims=True)
    return jnp.where(explained, logl, prior)

  def posterior(self, params, info) -> jax.Array:
    """f32[B, 6]: P(opponent card = c | the seat's InfoState).

    Args:
      params: The opponent model's params.
      info: A batched InfoState.

    Returns:
      f32[B, 6] posterior probabilities.
    """
    logp = self._log_posterior(info, self.log_likelihood(params, info))
    return jax.nn.softmax(logp, axis=-1).astype(jnp.float32)

  def sample_worlds(self, params, carry, info, k: int, key):
    """Worlds with the opponent card drawn from the posterior.

    (env states with leaves [B, K, ...], log-weights f64[B, K]): c ~ the
    posterior and the replayed world for c; exact draws, so log-weights 0.

    Args:
      params: The opponent model's params.
      carry: Unused.
      info: A batched InfoState.
      k: Worlds per row.
      key: PRNG key.

    Returns:
      (env states with leaves [B, K, ...], log-weights f64[B, K] of 0).

    Raises:
      ValueError: k < 1.
    """
    del carry
    if k < 1:
      raise ValueError(f"k must be >= 1, got {k}")
    states, logl = self._replay(params, info)
    logp = self._log_posterior(info, logl)
    b = logp.shape[0]
    keys = jax.random.split(key, b)
    idx = jax.vmap(lambda kk, lp: jax.random.categorical(kk, lp, shape=(k,)))(
        keys, logp
    )
    idx = idx.astype(jnp.int32)
    rows = jnp.arange(b, dtype=jnp.int32)[:, None]
    worlds = jax.tree_util.tree_map(lambda x: x[rows, idx], states)
    return worlds, jnp.zeros((b, k), F64)


class LeducCardRemovalSampler(LeducBeliefSampler):
  """The card-removal belief over leduc.

  Every card the seat cannot rule out is equally likely, and the opponent's
  actions carry no information.

  This is LeducBeliefSampler under a uniform opponent model, built from the
  env alone. Search over it hedges against opponents that do not play like
  the searching network, which the self-model belief (LeducBeliefSampler
  over the network being trained) cannot do: an opponent that deviates from
  the network is read as holding what the network would hold. Its params
  are the uniform model's ({}).

  Attributes:
    kwargs: The constructor options as a record (none: {}).
  """

  params_source = "none"

  def __init__(self, env: types.Env):
    super().__init__(
        env,
        uniform.UniformPolicy(
            env.name, int(env.num_actions), int(env.num_players)
        ),
    )
    self.kwargs = {}


registry.register("leduc", LeducBeliefSampler)
registry.register("leduc_card_removal", LeducCardRemovalSampler)
