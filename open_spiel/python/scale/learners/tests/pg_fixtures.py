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

"""Fixtures for the PG learner tests.

Fixtures for the PG learner tests: small NetworkPolicies and Arena batches
on kuhn / leduc self-play (real TrajectoryBatches from the one Arena).
"""

from typing import Any, Mapping, Optional

import jax
import jax.numpy as jnp

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.envs import leduc
from open_spiel.python.scale.networks import policy as policy_lib

tree_map = jax.tree_util.tree_map


def mlp(
    widths=(16, 16), layout="separate", value="scalar", dtype="float32"
) -> dict[str, Any]:
  """The network config of an MLP NetworkPolicy.

  Args:
    widths: the torso's hidden widths.
    layout: the torso layout.
    value: the value head kind.
    dtype: the compute dtype.

  Returns:
    The network config dict.
  """
  vh: dict[str, Any] = {"kind": value}
  if value == "hl_gauss":
    vh.update(num_bins=11, v_min=-3.0, v_max=3.0)
  return {
      "compute_dtype": dtype,
      "torso": {"kind": "mlp", "widths": list(widths), "layout": layout},
      "policy_head": {"kind": "dense"},
      "value_head": vh,
  }


def rnn(hidden=8, use_last_action=True) -> dict[str, Any]:
  """The network config of a recurrent NetworkPolicy.

  Args:
    hidden: the recurrent state size.
    use_last_action: feed the last action to the torso.

  Returns:
    The network config dict.
  """
  return {
      "compute_dtype": "float32",
      "torso": {
          "kind": "recurrent",
          "hidden_size": hidden,
          "use_last_action": use_last_action,
          "inner": {"kind": "mlp", "widths": [16], "layout": "shared"},
      },
      "policy_head": {"kind": "dense"},
      "value_head": {"kind": "scalar"},
  }


GAMES = {
    "kuhn_poker": (kuhn.KuhnEnv, "kuhn_poker_observation_tensor", (7,)),
    "leduc_poker": (leduc.LeducEnv, "leduc_poker_info_state", (30,)),
}


def make_policy(
    game: str = "kuhn_poker",
    network: Optional[Mapping[str, Any]] = None,
    encoder: Optional[str] = None,
    obs_shape=None,
) -> policy_lib.NetworkPolicy:
  """A NetworkPolicy for one of GAMES.

  Args:
    game: a GAMES key.
    network: the network config (default mlp()).
    encoder: the obs encoder (default the game's).
    obs_shape: the encoded obs shape (default the game's).

  Returns:
    The NetworkPolicy.
  """
  env_cls, enc, shape = GAMES[game]
  env = env_cls()
  return policy_lib.NetworkPolicy(
      game,
      encoder or enc,
      env.num_actions,
      env.num_players,
      encoder or enc,
      list(obs_shape or shape),
      dict(network or mlp()),
  )


def make_env(game: str = "kuhn_poker"):
  """The env of one of GAMES.

  Args:
    game: a GAMES key.

  Returns:
    The Env.
  """
  return GAMES[game][0]()


def stack1(params):
  """Params with a leading member axis of 1.

  Args:
    params: the params.

  Returns:
    The stacked params.
  """
  return tree_map(lambda x: x[None], params)


def self_play_batch(
    policy, params, game="kuhn_poker", b=8, t=8, seed=0, all_seats=True
):
  """One Arena self-play segment from a fresh reset: TrajectoryBatch.

  Args:
    policy: the policy.
    params: its params.
    game: a GAMES key.
    b: Arena rows.
    t: plies.
    seed: the key seed.
    all_seats: emit a row per seat.

  Returns:
    The TrajectoryBatch.
  """
  env = make_env(game)
  arena = arena_lib.Arena(env, [arena_lib.Slot(policy, 1, types.SAMPLE)])
  table = arena_lib.self_play_table(b, env.num_players, all_seats=all_seats)
  deal, pk = arena_lib.independent_keys(
      jax.random.key(seed), b, env.num_players
  )
  _, batch, _ = arena.collect([stack1(params)], table, deal, pk, t)
  return batch


def source(policy, params, game="kuhn_poker", b=8, t=8, all_seats=True):
  """An ArenaSource of self-play segments.

  Args:
    policy: the policy.
    params: its params.
    game: a GAMES key.
    b: Arena rows.
    t: plies per segment.
    all_seats: emit a row per seat.

  Returns:
    The ArenaSource.
  """
  env = make_env(game)
  arena = arena_lib.Arena(env, [arena_lib.Slot(policy, 1, types.SAMPLE)])
  table = arena_lib.self_play_table(b, env.num_players, all_seats=all_seats)
  return arena_lib.ArenaSource(arena, [stack1(params)], table, t)


def leaves_equal(a, b) -> bool:
  """Whether two trees have bitwise-equal leaves (dtypes and shapes too).

  Args:
    a: a tree.
    b: a tree.

  Returns:
    True when equal.
  """
  la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
  return len(la) == len(lb) and all(
      x.dtype == y.dtype and x.shape == y.shape and bool(jnp.array_equal(x, y))
      for x, y in zip(la, lb)
  )


def max_abs_diff(a, b) -> float:
  """The largest absolute difference between two trees' leaves (in f64).

  Args:
    a: a tree.
    b: a tree.

  Returns:
    The difference.
  """
  return max(
      float(
          jnp.max(
              jnp.abs(jnp.asarray(x, jnp.float64) - jnp.asarray(y, jnp.float64))
          )
      )
      for x, y in zip(
          jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
      )
  )
