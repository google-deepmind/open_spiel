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

"""Search test fixtures: networks, configs, toy and leduc positions."""


import jax
import jax.numpy as jnp

from open_spiel.python.scale.core import plies
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import leduc
from open_spiel.python.scale.envs.leduc import belief as leduc_belief
from open_spiel.python.scale.networks import policy as network_policy
from open_spiel.python.scale.policies import search
from open_spiel.python.scale.policies import uniform

tree_map = jax.tree_util.tree_map


def uniform_policy(env):
  """The uniform policy over env's actions.

  Args:
    env: the env.

  Returns:
    The UniformPolicy.
  """
  return uniform.UniformPolicy(env.name, env.num_actions, env.num_players)


def root_step(env, s, seat):
  """The seat step at batched states s, as at a match's first ply.

  Args:
    env: the env.
    s: the batched states.
    seat: the seat to act.

  Returns:
    The SeatStep.
  """
  b = jax.tree_util.tree_leaves(s)[0].shape[0]
  return plies.seat_step(
      env,
      s,
      jnp.full((b,), seat, jnp.int32),
      True,
      True,
      types.NO_ACTION,
      types.NO_ACTOR,
  )


def run(policy, env, s, seat, key=0, params=None):
  """Evaluates at batched states s for `seat` (after observing the step).

  Args:
    policy: the policy.
    env: the env.
    s: the batched states.
    seat: the seat to act.
    key: the PRNG seed of the evaluation.
    params: the policy params (None: init_params(key(11))).

  Returns:
    The PolicyOutput.
  """
  params = policy.init_params(jax.random.key(11)) if params is None else params
  step = root_step(env, s, seat)
  b = step.seat.shape[0]
  carry = policy.init_carry(params, b, step.seat, jax.random.key(1))
  carry = policy.observe(params, carry, step)
  return policy.evaluate(params, carry, step, jax.random.key(key))


def toy_root(env, n=1):
  """The toy env's root, repeated n times.

  Args:
    env: the toy env.
    n: the batch size.

  Returns:
    The batched root states.
  """
  return jax.tree_util.tree_map(
      lambda x: jnp.broadcast_to(x, (n,) + x.shape), env.root()
  )


def cfg(**kw):
  """An MCTSConfig: 64 muzero simulations, no noise, then `kw`.

  Args:
    **kw: MCTSConfig overrides.

  Returns:
    The MCTSConfig.
  """
  base = dict(num_simulations=64, search="muzero", dirichlet_fraction=0.0)
  base.update(kw)
  return search.MCTSConfig(**base)


def toy_node(env, n=1):
  """A search Node at n toy roots, seat 0 to act.

  Args:
    env: the toy env.
    n: the batch size.

  Returns:
    The Node.
  """
  return search.Node(
      state=toy_root(env, n),
      carries=(),
      seat=jnp.zeros((n,), jnp.int32),
      actor=jnp.zeros((n,), jnp.int32),
      terminal=jnp.zeros((n,), jnp.bool_),
      chance_scores=jnp.zeros((n, 0), jnp.float32),
  )


def c4_net(seed_widths=(32, 32)):
  """A connect four MLP network policy.

  Args:
    seed_widths: the torso's layer widths.

  Returns:
    The NetworkPolicy.
  """
  return network_policy.NetworkPolicy(
      game="connect_four",
      obs_format="connect_four_flat_planes",
      num_actions=7,
      num_players=2,
      encoder="connect_four_flat_planes",
      obs_shape=[126],
      network={
          "compute_dtype": "float32",
          "torso": {
              "kind": "mlp",
              "widths": list(seed_widths),
              "layout": "shared",
          },
          "policy_head": {"kind": "dense"},
          "value_head": {"kind": "scalar", "tanh": True},
      },
      value_layout="zero_sum",
  )


def leduc_net():
  """A small leduc MLP network policy.

  Returns:
    The NetworkPolicy.
  """
  return network_policy.NetworkPolicy(
      game="leduc_poker",
      obs_format="leduc_poker_info_state",
      num_actions=3,
      num_players=2,
      encoder="leduc_poker_info_state",
      obs_shape=[30],
      network={
          "compute_dtype": "float32",
          "torso": {"kind": "mlp", "widths": [16], "layout": "shared"},
          "policy_head": {"kind": "dense"},
          "value_head": {"kind": "scalar", "tanh": True},
      },
      value_layout="zero_sum",
  )


def leduc_search(k=4, sims=16):
  """The leduc env and a search policy over leduc_net with k worlds.

  Args:
    k: the number of sampled worlds.
    sims: the number of simulations.

  Returns:
    (LeducEnv, SearchPolicy).
  """
  env = leduc.LeducEnv()
  base = leduc_net()
  sampler = leduc_belief.LeducBeliefSampler(env, base)
  return env, search.SearchPolicy(
      base, env, sampler, cfg(num_simulations=sims, num_worlds=k)
  )


def leduc_state(env, cards, actions):
  """The batch-of-one leduc state after `cards` and `actions`.

  Args:
    env: the LeducEnv.
    cards: the chance outcomes, in order.
    actions: the player actions, in order.

  Returns:
    The batched state.
  """
  s = env.root()
  for c in cards:
    s = env.apply_chance(s, jnp.int32(c))
  for a in actions:
    s = env.apply(s, jnp.int32(a))
  return tree_map(lambda x: x[None], s)
