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

"""A random-rollout value policy and the MCTS-N baseline bot built on it.

A fixed, network-free search opponent is a useful baseline for
perfect-information games: MCTS with uniform priors whose leaves are valued
by one uniformly random playout to the end of the game. `mcts_bot` builds it
from the shared SearchPolicy, so there is one MCTS implementation.

  bot, params = mcts_bot("connect_four", num_simulations=1000)

RandomRolloutPolicy (architecture "random_rollout") is a NATIVE_JAX policy
over a perfect-information Env:
  logits  zeros f32[B, A] (uniform priors; the search masks illegal actions);
  value   f32[B, P], every seat's return at the end of one random playout
          from the state (to the game's end, at most env.max_len plies,
          uniform over legal actions; chance nodes resolved by Env.step);
          zeros at a terminal state.
Each row draws its playout from its own key (a split of the evaluate key,
split again per ply).

The bot runs SearchPolicy in zero_sum mode over PerfectInformationSampler
roots with MCTSConfig(search="gumbel", gumbel_scale=0, act="mctx_action"):
deterministic Gumbel MuZero search (Danihelka et al. 2022) that plays its
sequential-halving winner. Illegal actions are passed to mctx as invalid, so
the root never visits one.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import jax
import jax.numpy as jnp

from open_spiel.python.scale.belief import perfect_information
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.policies import registry
from open_spiel.python.scale.policies import search

ARCH = "random_rollout"
F32 = jnp.float32


class RandomRolloutPolicy(types.Policy):
  """Uniform priors, random-playout values. Stateless; params {}.

  Attributes:
    env: the game's env.
    spec: the PolicySpec.
  """

  def __init__(self, env: str, env_kwargs: Optional[Mapping[str, Any]] = None):
    """Builds the rollout policy of a perfect-information game.

    Args:
      env: a registered game name.
      env_kwargs: its options when not the defaults (recorded in arch_kwargs
        only when given).

    Raises:
      ValueError: the game has hidden information.
    """
    self.env = env_registry.make(env, **dict(env_kwargs or {}))
    probe = self.env.info_state(self.env.init(jax.random.key(0)), jnp.int32(0))
    if self.env.world_from_info(probe) is None:
      raise ValueError(
          f"{env}: RandomRolloutPolicy needs a perfect-information env "
          "(world_from_info returned None)"
      )
    self.spec = types.PolicySpec(
        game=self.env.name,
        obs_format="none",
        num_actions=self.env.num_actions,
        num_players=self.env.num_players,
        stateful=False,
        capability=types.Capability.NATIVE_JAX,
        framework="jax",
        arch_name=ARCH,
        arch_kwargs=dict(
            env=env, **({"env_kwargs": dict(env_kwargs)} if env_kwargs else {})
        ),
    )

  def init_params(self, key=None):
    """No params.

    Args:
      key: unused.

    Returns:
      An empty dict.
    """
    del key
    return {}

  def encode(self, info):
    """No features: the policy reads the InfoState directly.

    Args:
      info: the InfoState, leaves [B, ...].

    Returns:
      f32[B, 0].
    """
    leaves = jax.tree_util.tree_leaves(info)
    return jnp.zeros((leaves[0].shape[0], 0), F32)

  def init_carry(self, params, batch, seat, key):
    """No carry.

    Args:
      params: unused.
      batch: unused.
      seat: unused.
      key: unused.

    Returns:
      An empty tuple.
    """
    del params, batch, seat, key
    return ()

  def observe(self, params, carry, step):
    """The carry unchanged.

    Args:
      params: unused.
      carry: the carry.
      step: unused.

    Returns:
      `carry`.
    """
    del params, step
    return carry

  def _ply(self, c):
    """One random ply of the rollout; finished rollouts stay put.

    Args:
      c: (state, key, done).

    Returns:
      The next (state, key, done).
    """
    st, k, done = c
    env = self.env
    k, ka, kc = jax.random.split(k, 3)
    legal = env.legal_mask(st)
    logits = jnp.where(legal, F32(0.0), F32(-1e9))
    a = jax.random.categorical(ka, logits).astype(jnp.int32)
    nxt = env.step(st, a, kc)
    st = jax.tree_util.tree_map(lambda n, o: jnp.where(done, o, n), nxt, st)
    return st, k, done | env.terminal(st)

  def _playout(self, s, key):
    """One random playout from s.

    At most env.max_len plies, stopping at the terminal state; returns the
    returns of the final state (zeros when s is already terminal).

    Args:
      s: the state to play out from.
      key: a PRNG key.

    Returns:
      f32[num_players], the returns of the final state.
    """
    env = self.env
    start_term = env.terminal(s)

    def cond(c):
      """Whether the playout goes on: plies left and not terminal.

      Args:
        c: (ply count, (state, key, done)).

      Returns:
        A scalar bool.
      """
      i, (_, _, done) = c
      return (i < env.max_len) & ~done

    def body(c):
      """One uniform-random ply.

      Args:
        c: (ply count, (state, key, done)).

      Returns:
        The next loop state.
      """
      i, inner = c
      return i + 1, self._ply(inner)

    _, (final, _, _) = jax.lax.while_loop(
        cond, body, (jnp.int32(0), (s, key, start_term))
    )
    rets = env.returns(final).astype(F32)
    return jnp.where(start_term, jnp.zeros_like(rets), rets)

  def evaluate(self, params, carry, step, key):
    """Zero logits and the returns of one random playout per row.

    Args:
      params: unused.
      carry: unused.
      step: the SeatStep, leaves [B, ...].
      key: a PRNG key, split per row.

    Returns:
      The PolicyOutput: logits zeros, value f32[B, P] playout returns.
    """
    del params, carry
    worlds = jax.vmap(self.env.world_from_info)(step.info)
    b = step.legal_mask.shape[0]
    value = jax.vmap(self._playout)(worlds, jax.random.split(key, b)).astype(
        F32
    )
    return types.PolicyOutput(
        logits=jnp.zeros((b, self.spec.num_actions), F32), value=value, aux={}
    )


registry.register(ARCH, RandomRolloutPolicy)


def _drop_empty_env_kwargs(arch_name: str, arch_kwargs: Mapping[str, Any]):
  """Drops an empty env_kwargs, which is the defaults.

  The spec does not record the defaults, so build("random_rollout",
  {"env": ..., "env_kwargs": {}}) rebuilds like {"env": ...}.

  Args:
    arch_name: the architecture name.
    arch_kwargs: its kwargs.

  Returns:
    The kwargs without env_kwargs, or None to leave them as they are.
  """
  if (
      arch_name == ARCH
      and "env_kwargs" in arch_kwargs
      and not arch_kwargs["env_kwargs"]
  ):
    return {k: v for k, v in arch_kwargs.items() if k != "env_kwargs"}
  return None


registry.register_kwargs_translator(_drop_empty_env_kwargs)


def mcts_config(num_simulations: int = 1000) -> search.MCTSConfig:
  """The baseline bot's search settings.

  Deterministic gumbel search that plays its own selected action.

  Args:
    num_simulations: simulations per search.

  Returns:
    The MCTSConfig.
  """
  return search.MCTSConfig(
      num_simulations=int(num_simulations),
      search="gumbel",
      gumbel_scale=0.0,
      mode="zero_sum",
      num_worlds=1,
      act="mctx_action",
  )


def mcts_bot(
    env: str,
    num_simulations: int = 1000,
    env_kwargs: Optional[Mapping[str, Any]] = None,
) -> tuple[search.SearchPolicy, Any]:
  """The random-rollout MCTS bot for game `env`.

  Args:
    env: a registered game name.
    num_simulations: simulations per search.
    env_kwargs: the game's options; None is the defaults.

  Returns:
    (SearchPolicy, params).
  """
  base = RandomRolloutPolicy(env, env_kwargs)
  sp = search.SearchPolicy(
      base,
      base.env,
      perfect_information.PerfectInformationSampler(base.env),
      mcts_config(num_simulations),
  )
  return sp, sp.init_params(jax.random.key(0))
