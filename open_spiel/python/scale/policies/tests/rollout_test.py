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

"""Tests for RandomRolloutPolicy and the MCTS baseline bot.

The bot: uniform priors, random-rollout leaf values, deterministic gumbel
search that plays its own selected action.

Known answers: a one-empty-cell tic-tac-toe position has a deterministic
rollout; win-in-1 and block-in-1 positions in tic-tac-toe and a vertical
win-in-1 in connect four are found by the bot.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import decide as decide_lib
from open_spiel.python.scale.core import plies
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.eval import match
from open_spiel.python.scale.eval import mcts_baseline
from open_spiel.python.scale.policies import registry
from open_spiel.python.scale.policies import rollout
from open_spiel.python.scale.policies import search
from open_spiel.python.scale.policies import uniform


def _state(env, moves):
  """Applies `moves` from the root.

  Args:
    env: The env.
    moves: The moves.

  Returns:
    The state.
  """
  s = env.root()
  for m in moves:
    s = env.apply(s, jnp.int32(m))
  return s


def _step(env, s):
  """The batch-of-one SeatStep of the actor at `s`.

  Args:
    env: The env.
    s: The state.

  Returns:
    The SeatStep.
  """
  b = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[None], s)
  seat = jnp.maximum(env.actor(s), 0).astype(jnp.int32)
  return plies.seat_step(
      env, b, seat[None], True, True, types.NO_ACTION, types.NO_ACTOR
  )


def _act(env, pol, params, s, key=0):
  """The policy's argmax action at `s`.

  Args:
    env: The env.
    pol: The policy.
    params: Its params.
    s: The state.
    key: The seed.

  Returns:
    (action, PolicyOutput).
  """
  st = _step(env, s)
  c = pol.init_carry(params, 1, st.seat, jax.random.key(key))
  c = pol.observe(params, c, st)
  out = pol.evaluate(params, c, st, jax.random.key(key))
  a, _ = decide_lib.decide(
      out, st.legal_mask, types.ARGMAX, jax.random.key(key)
  )
  return int(a[0]), out


def test_rollout_value_known_answer():
  """Rollout value known answer."""
  env = env_registry.make("tic_tac_toe")
  # X {0,1,4,6}, O {2,3,5,7}, X to move; the only move (8) wins on 0-4-8.
  s = _state(env, [0, 2, 1, 3, 4, 5, 6, 7])
  pol = rollout.RandomRolloutPolicy("tic_tac_toe")
  st = _step(env, s)
  for k in range(3):
    out = pol.evaluate({}, (), st, jax.random.key(k))
    np.testing.assert_array_equal(np.asarray(out.value), [[1.0, -1.0]])
    assert out.value.dtype == jnp.float32 and out.logits.dtype == jnp.float32
    np.testing.assert_array_equal(
        np.asarray(out.logits), np.zeros((1, 9), np.float32)
    )


def test_an_empty_env_kwargs_rebuilds_like_none():
  """An empty env kwargs rebuilds like none."""
  pol = registry.build(rollout.ARCH, {"env": "connect_four", "env_kwargs": {}})
  assert dict(pol.spec.arch_kwargs) == {"env": "connect_four"}
  again = registry.build(pol.spec.arch_name, pol.spec.arch_kwargs)
  assert again.spec == pol.spec


def test_rollout_refuses_imperfect_information():
  """Rollout refuses imperfect information."""
  with pytest.raises(ValueError, match="perfect"):
    rollout.RandomRolloutPolicy("kuhn_poker")


@pytest.mark.parametrize(
    "game,moves,want,sims",
    [
        ("tic_tac_toe", [0, 3, 1, 4], 2, 64),  # X wins at 2
        ("tic_tac_toe", [0, 3, 8, 4], 5, 200),  # X must block O at 5
        ("connect_four", [0, 1, 0, 1, 0, 1], 0, 64),  # vertical win in column 0
    ],
)
def test_mcts_bot_known_moves(game, moves, want, sims):
  """Mcts bot known moves."""
  env = env_registry.make(game)
  pol, params = rollout.mcts_bot(game, num_simulations=sims)
  a, out = _act(env, pol, params, _state(env, moves))
  assert a == want
  # It plays the search's action under any rule: one finite logit.
  lg = np.asarray(out.logits[0])
  assert np.isfinite(lg).sum() == 1 and lg[want] == 0.0


def test_mcts_bot_is_deterministic_per_key():
  """Mcts bot is deterministic per key."""
  env = env_registry.make("connect_four")
  pol, params = rollout.mcts_bot("connect_four", num_simulations=16)
  s = _state(env, [3, 3, 2])
  a1, o1 = _act(env, pol, params, s, key=5)
  a2, o2 = _act(env, pol, params, s, key=5)
  assert a1 == a2
  np.testing.assert_array_equal(
      np.asarray(o1.aux[types.AUX_TARGET_POLICY]),
      np.asarray(o2.aux[types.AUX_TARGET_POLICY]),
  )


def test_mcts_bot_spec_round_trips_and_uses_deterministic_gumbel_search():
  """Mcts bot spec round trips and uses deterministic gumbel search."""
  pol, params = rollout.mcts_bot("tic_tac_toe", num_simulations=1000)
  assert isinstance(pol, search.SearchPolicy)
  mcts = pol.spec.arch_kwargs["mcts"]
  assert mcts["num_simulations"] == 1000 and mcts["search"] == "gumbel"
  assert mcts["gumbel_scale"] == 0.0 and mcts["act"] == "mctx_action"
  assert pol.spec.arch_kwargs["base"]["arch_name"] == rollout.ARCH
  again = registry.build(pol.spec.arch_name, pol.spec.arch_kwargs)
  assert again.spec == pol.spec
  assert jax.tree_util.tree_structure(params) == jax.tree_util.tree_structure(
      again.init_params(jax.random.key(0))
  )


def test_act_is_recorded_and_validated():
  """Act is recorded and validated."""
  assert search.MCTSConfig().to_json()["act"] == "weights"
  assert search.MCTSConfig(act="mctx_action").to_json()["act"] == "mctx_action"
  with pytest.raises(ValueError, match="num_worlds must be 1"):
    search.MCTSConfig(act="mctx_action", num_worlds=2)
  with pytest.raises(ValueError):
    search.MCTSConfig(act="bogus")


def test_mcts_baseline_beats_uniform_in_the_evaluator():
  """Mcts baseline beats uniform in the evaluator."""
  env = env_registry.make("tic_tac_toe")
  ev = match.Evaluator(
      env, mcts_baseline.vs_mcts_config("tic_tac_toe", num_simulations=32)
  )
  subject = match.Entrant(
      "uniform", uniform.UniformPolicy("tic_tac_toe", 9, 2), {}, types.SAMPLE
  )
  r = ev.vs_reference(subject, 32, jax.random.key(0))
  assert r.b == "mcts32"
  # Uniform random loses most tic-tac-toe games to MCTS-32 (it wins none in
  # practice); payoff per match is +1/0/-1.
  assert r.mean.mean < -0.5, r.mean


def _playout_scan(pol, s, key):
  """The fixed-length reference of RandomRolloutPolicy._playout (max_len).

  Args:
    pol: The rollout policy.
    s: The state.
    key: The key.

  Returns:
    f32[P], the returns (zeros from a terminal start).
  """
  start_term = pol.env.terminal(s)
  (final, _, _), _ = jax.lax.scan(
      # Reaches private _ply of the code under test.
      lambda c, _: (pol._ply(c), None),  # pylint: disable=protected-access
      (s, key, start_term),
      None,
      length=pol.env.max_len,
  )
  rets = pol.env.returns(final).astype(jnp.float32)
  return jnp.where(start_term, jnp.zeros_like(rets), rets)


@pytest.mark.parametrize(
    "game,kw",
    [
        ("connect_four", {}),
        ("breakthrough", {"rows": 6, "columns": 6}),
        ("tic_tac_toe", {}),
    ],
)
def test_playout_stops_at_the_game_end_with_the_scan_values(game, kw):
  """Playout stops at the game end with the scan values.

  The playout is a while loop that ends with the game, not a fixed
  env.max_len-ply scan; its values equal the scan's for every key (the plies and
  keys up to the end are the same, a finished game is frozen) from the root,
  mid-game and terminal states, batched.

  Args:
    game: the game name.
    kw: its options.
  """
  pol = rollout.RandomRolloutPolicy(game, kw or None)
  env = pol.env
  states = [env.init(jax.random.key(i)) for i in range(3)]
  s = states[0]
  for i in range(4):  # a mid-game state
    legal = np.flatnonzero(np.asarray(env.legal_mask(s)))
    assert not bool(env.terminal(s)) and legal.size
    s = env.apply(s, jnp.int32(legal[i % len(legal)]))
  states.append(s)
  keys = jax.random.split(jax.random.key(42), 16)
  for st in states:
    # Reaches private _playout of the code under test.
    got = jax.vmap(lambda k, st=st: pol._playout(st, k))(keys)  # pylint: disable=protected-access
    want = jax.vmap(lambda k, st=st: _playout_scan(pol, st, k))(keys)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
  # a terminal state: zeros, no ply played
  t = states[0]
  while not bool(env.terminal(t)):
    t = env.apply(
        t, jnp.int32(np.flatnonzero(np.asarray(env.legal_mask(t)))[0])
    )
  # Reaches private _playout of the code under test.
  np.testing.assert_array_equal(np.asarray(pol._playout(t, keys[0])), 0.0)  # pylint: disable=protected-access


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
