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

"""Gin privacy: permuting the hidden cards changes nothing seat-visible.

Gin privacy: permuting the hidden hand and the stock in 200 states leaves
the InfoState, the gin encoders, the rule bot's action, the scripted
endgame's action and a network policy's output unchanged.

`check_infostate_views` and `check_network_policy` take the encoders and the
network to check, so other gin encoders reuse them.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import plies
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.encoders import registry as encoders
from open_spiel.python.scale.envs import gin
from open_spiel.python.scale.envs.gin import endgame as gin_endgame
from open_spiel.python.scale.envs.gin import rule_bot as gin_rule_bot
from open_spiel.python.scale.envs.tests import gin_games
from open_spiel.python.scale.league import config_common
from open_spiel.python.scale.networks import catalog
from open_spiel.python.scale.policies import registry

tree_map = jax.tree_util.tree_map
NUM_STATES = 200
NUM_PERMUTATIONS = 3


def _equal(a, b):
  """Whether two pytrees are equal, NaNs equal.

  Args:
    a: A pytree.
    b: The other pytree.

  Returns:
    True when equal.
  """
  la, ta = jax.tree_util.tree_flatten(a)
  lb, tb = jax.tree_util.tree_flatten(b)
  return ta == tb and all(
      np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True)
      for x, y in zip(la, lb)
  )


def _pick_states(env):
  """Decision states, half from the endgame phases.

  Args:
    env: The env.

  Returns:
    The picked states.
  """
  g = gin_games.games(64, 2)
  mask = gin_games.decision_mask(env, g)
  idx = np.stack(np.nonzero(mask), axis=1)
  rng = np.random.default_rng(0)
  # Half from the endgame phases, so the scripted endgame is exercised.
  phase = np.asarray(g.states.game["phase"])[mask]
  end = np.nonzero(np.isin(phase, [4, 5, 6]))[0]
  strat = np.nonzero(~np.isin(phase, [4, 5, 6]))[0]
  n_end = min(len(end), NUM_STATES // 2)
  assert n_end >= 50, f"only {len(end)} endgame decision states"
  pick = np.concatenate([
      rng.choice(end, n_end, replace=False),
      rng.choice(strat, NUM_STATES - n_end, replace=False),
  ])
  sel = idx[pick]
  return tree_map(lambda x: x[sel[:, 0], sel[:, 1]], g.states)


def check_infostate_views(encoder_names):
  """The InfoState, encoders, rule bot and endgame are blind to hidden cards.

  InfoState, the encoders, the rule bot and the endgame are blind to the
  hidden cards of 200 states.

  Args:
    encoder_names: The encoders to check.
  """
  env = gin.GinEnv(oklahoma_frac=1.0)
  states = _pick_states(env)
  n = jax.tree_util.tree_leaves(states)[0].shape[0]
  assert n == NUM_STATES
  actor = jax.vmap(env.actor)(states)
  encs = [encoders.get(name) for name in encoder_names]

  def views(s, seat):
    """Everything the seat's policies see at `s`.

    Args:
      s: The batched states.
      seat: The seats.

    Returns:
      (InfoState, encoder outputs, rule bot actions, endgame actions).
    """
    info = jax.vmap(env.info_state)(s, seat)
    return (
        info,
        [e.batch(info) for e in encs],
        gin_rule_bot.actions(info),
        gin_endgame.actions(info),
    )

  views = jax.jit(views)
  hide = jax.jit(jax.vmap(lambda s, q, k: gin.hide_hidden_cards(env, s, q, k)))
  changed = 0
  for seat_of in ("actor", "other"):
    seat = actor if seat_of == "actor" else 1 - actor
    ref = views(states, seat)
    for p in range(NUM_PERMUTATIONS):
      keys = jax.random.split(
          jax.random.key(1000 * p + (seat_of == "other")), n
      )
      hidden = hide(states, seat, keys)
      opp = jnp.where(
          (seat == 0)[:, None],
          hidden.game["player1_hand"],
          hidden.game["player0_hand"],
      )
      opp0 = jnp.where(
          (seat == 0)[:, None],
          states.game["player1_hand"],
          states.game["player0_hand"],
      )
      changed += int(np.any(np.asarray(opp) != np.asarray(opp0), axis=-1).sum())
      got = views(hidden, seat)
      assert _equal(
          got[0], ref[0]
      ), f"InfoState leaks hidden cards (seat={seat_of}, perm {p})"
      for e, x, y in zip(encs, got[1], ref[1]):
        np.testing.assert_array_equal(
            np.asarray(x).view(np.uint32),
            np.asarray(y).view(np.uint32),
            err_msg=e.name,
        )
      if seat_of == "actor":
        np.testing.assert_array_equal(np.asarray(got[2]), np.asarray(ref[2]))
        np.testing.assert_array_equal(np.asarray(got[3]), np.asarray(ref[3]))
  # The permutation is not vacuous: most states had the hidden hand changed.
  assert changed > NUM_STATES * NUM_PERMUTATIONS, changed


def check_network_policy(encoder: str, obs_shape, net):
  """A NetworkPolicy is blind to the hidden cards of 200 states.

  A NetworkPolicy's logits and value at 200 states (the actor's SeatStep)
  are bitwise unchanged when the hidden hand and stock are permuted.

  Args:
    encoder: The encoder name.
    obs_shape: The encoder's observation shape.
    net: The network.
  """
  env = gin.GinEnv(oklahoma_frac=1.0)
  states = _pick_states(env)
  n = jax.tree_util.tree_leaves(states)[0].shape[0]
  actor = jax.vmap(env.actor)(states)
  pol = registry.build(
      "network",
      config_common.network_policy_kwargs(
          "gin_rummy", encoder, list(obs_shape), env.num_actions, 2, net
      ),
  )
  params = pol.init_params(jax.random.key(3))

  @jax.jit
  def evaluate(s):
    """The policy's output at the actor's SeatStep.

    Args:
      s: The batched states.

    Returns:
      (logits, value).
    """
    st = plies.seat_step(
        env, s, actor, False, False, types.NO_ACTION, types.NO_ACTOR
    )
    out = pol.evaluate(
        params, pol.init_carry(params, n, st.seat, None), st, None
    )
    return out.logits, out.value

  ref = evaluate(states)
  hide = jax.jit(jax.vmap(lambda s, q, k: gin.hide_hidden_cards(env, s, q, k)))
  for p in range(NUM_PERMUTATIONS):
    got = evaluate(
        hide(states, actor, jax.random.split(jax.random.key(77 + p), n))
    )
    for x, y in zip(got, ref):
      np.testing.assert_array_equal(
          np.asarray(x).view(np.uint32), np.asarray(y).view(np.uint32)
      )
  assert float(jnp.std(ref[0][jnp.isfinite(ref[0])])) > 0.0


def test_hidden_cards_do_not_reach_infostate_encoders_or_scripted_policies():
  """Hidden cards do not reach infostate encoders or scripted policies."""
  check_infostate_views(["gin_rummy_observation_tensor"])


def test_hidden_cards_do_not_reach_a_network_policy_evaluate():
  """Hidden cards do not reach a network policy evaluate."""
  check_network_policy(
      "gin_rummy_observation_tensor", [644], catalog.mlp([32], "relu", "shared")
  )


if __name__ == "__main__":
  # The package runs with x64 on, as its entry points and conftest.py set
  # it; set here for `python <file>`, never at import (which changes no
  # process-global state).
  x64.enable_x64()
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
