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

"""Serving liars_dice, phantom_ttt_ir and bargaining.

Serving liars_dice, phantom_ttt_ir and bargaining: on random pyspiel
playouts the replayed env node equals pyspiel's at every prefix (actor, legal
set, terminal) and the env's terminal returns equal pyspiel's (bargaining at
non-default options too, prob_end's translated chance ids included); variants
the env does not play are refused; HostAgent serves a network artifact on
each game's research encoder, and an artifact from disk on its training env.

The hidden-move games refuse stateful policies: seat 0's histories [0, 1]
and [0, 0, 1] of phantom_ttt and dark_hex give it the same information state
(pyspiel's and the env's) but a different number of plies, the opponent's
private retry. A carry observing every ply could count it, so the Arena,
HostArena, HostAgent and ShardReader refuse a recurrent network and a
history-counting policy on all four hidden-move envs, while a feed-forward
network plays them and answers the two histories identically.
"""

import sys

import jax
import numpy as np
import pytest

from open_spiel.python.scale.ckpt import artifact as ckpt
from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import host_arena
from open_spiel.python.scale.core import types
from open_spiel.python.scale.encoders import registry as enc_registry
from open_spiel.python.scale.envs import dark_hex
from open_spiel.python.scale.envs import phantom_ttt
from open_spiel.python.scale.envs import phantom_ttt_ir
from open_spiel.python.scale.envs import registry
from open_spiel.python.scale.networks import policy as policy_lib
from open_spiel.python.scale.policies import uniform
from open_spiel.python.scale.serve import games
from open_spiel.python.scale.serve import host_agent
from open_spiel.python.scale.serve import replay
from open_spiel.python.scale.serve import wire
from open_spiel.python.scale.serve.tests import helpers
from open_spiel.python.scale.sources import shards
from open_spiel.python.scale.testing import fixtures
import pyspiel

GAME_STRINGS = [
    "liars_dice(numdice=2,dice_sides=5)",
    "phantom_ttt_ir(gameversion=abrupt)",
    "phantom_ttt_ir(gameversion=abrupt,obstype=reveal-numturns)",
    "bargaining",
    "bargaining(max_turns=4,discount=0.9)",
    "bargaining(prob_end=0.4)",
]
ENCODER = {
    "liars_dice": "liars_dice_info_state",
    "phantom_ttt_ir": "phantom_ttt_ir_observation",
    "bargaining": "bargaining_info_state",
    "phantom_ttt": "phantom_ttt_info_state",
    "dark_hex": "dark_hex_info_state",
    "dark_hex_ir": "dark_hex_ir_view",
}
HIDDEN_MOVE_ENVS = {
    "phantom_ttt": phantom_ttt.PhantomTTTEnv,
    "phantom_ttt_ir": phantom_ttt_ir.PhantomTTTIREnv,
    "dark_hex": dark_hex.DarkHexEnv,
    "dark_hex_ir": dark_hex.DarkHexIREnv,
}
# Seat 0 claims cell 0; seat 1 plays cell 1, or first tries the held cell 0
# (fails, private, retries) and then plays 1. Seat 0 is to move in both.
SHORT, RETRY = (0, 1), (0, 0, 1)


def _playouts(game, n, seed):
  """Every prefix state of n random pyspiel games.

  Args:
    game: The pyspiel game.
    n: The games.
    seed: The seed.

  Returns:
    The states.
  """
  rng = np.random.default_rng(seed)
  out = []
  for _ in range(n):
    s = game.new_initial_state()
    while True:
      out.append(s.clone())
      if s.is_terminal():
        break
      if s.is_chance_node():
        acts, probs = zip(*s.chance_outcomes())
        s.apply_action(
            int(rng.choice(acts, p=np.asarray(probs) / np.sum(probs)))
        )
      else:
        s.apply_action(int(rng.choice(s.legal_actions())))
  return out


@pytest.mark.parametrize("game_string", GAME_STRINGS)
def test_replayed_node_and_returns_equal_pyspiel(game_string):
  """Replayed node and returns equal pyspiel."""
  game = pyspiel.load_game(game_string)
  name = game.get_type().short_name
  env = games.env_for(name, wire.game_params(game))
  assert env.name == name
  n_checked = n_term = 0
  for state in _playouts(game, 12, seed=3):
    if state.is_chance_node():
      continue
    hands = wire.hands_from_state(state)
    seat = 0 if state.is_terminal() else state.current_player()
    rep = replay.replay_seat_steps(env, hands, seat)
    wire.check_replayed_node(env, rep.state, state)  # raises on any mismatch
    n_checked += 1
    if state.is_terminal():
      np.testing.assert_allclose(
          np.asarray(env.returns(rep.state)),
          np.asarray(state.returns(), np.float32),
          rtol=1e-6,
      )
      n_term += 1
  assert n_checked > 30 and n_term == 12


def test_bargaining_prob_end_history_replays_untranslated():
  # The engine numbers the mid-hand continue / end outcomes as pyspiel does
  # (1000 / 1001), so a pyspiel history is the raw history.
  """Bargaining prob end history replays untranslated."""
  hist = [0, 9, 15, 1000, 5, 1001]
  assert wire.pyspiel_history_to_raw("bargaining", hist, {}) == [hist]
  assert wire.pyspiel_history_to_raw("bargaining", [999, 3], {}) == [[999, 3]]


def test_variants_the_env_does_not_play_are_refused():
  """Variants the env does not play are refused."""
  assert {"liars_dice", "phantom_ttt_ir", "bargaining"} <= set(
      games.served_games()
  )
  for game_string, match in (
      (
          "liars_dice",
          "dice_sides|numdice",
      ),  # pyspiel's default is 1 die, 6 sides
      ("phantom_ttt_ir", "gameversion"),  # pyspiel's default: classical
      ("bargaining(max_num_instances=10)", "max_num_instances"),
  ):
    game = pyspiel.load_game(game_string)
    with pytest.raises(ValueError, match=match):
      games.env_for(game.get_type().short_name, wire.game_params(game))
  assert games.env_spec_for("bargaining", {"max_turns": 4}) == registry.spec(
      "bargaining", max_turns=4
  )


def _mlp_policy(env, recurrent=False):
  """A small MLP (or recurrent) policy on the env's encoder.

  Args:
    env: The env.
    recurrent: Whether the torso is recurrent.

  Returns:
    (policy, params).
  """
  enc = enc_registry.get(ENCODER[env.name])
  torso = {"kind": "mlp", "widths": [16, 16], "layout": "shared"}
  if recurrent:
    torso = {
        "kind": "recurrent",
        "hidden_size": 8,
        "use_last_action": True,
        "inner": torso,
    }
  cfg = {
      "compute_dtype": "float32",
      "torso": torso,
      "policy_head": {"kind": "dense"},
      "value_head": {"kind": "scalar"},
  }
  pol = policy_lib.NetworkPolicy(
      env.name,
      enc.name,
      env.num_actions,
      env.num_players,
      enc.name,
      list(enc.shape),
      cfg,
  )
  return pol, pol.init_params(jax.random.key(1))


@pytest.mark.parametrize(
    "game_string",
    [
        "liars_dice(numdice=2,dice_sides=5)",
        "phantom_ttt_ir(gameversion=abrupt)",
        "bargaining",
    ],
)
def test_host_agent_serves_a_network(game_string):
  """Host agent serves a network."""
  game = pyspiel.load_game(game_string)
  name = game.get_type().short_name
  env = games.env_for(name, wire.game_params(game))
  pol, params = _mlp_policy(env)
  served = 0
  for state in _playouts(game, 3, seed=5):
    if state.is_chance_node() or state.is_terminal():
      continue
    seat = state.current_player()
    agent = host_agent.HostAgent(helpers.artifact(pol, params), seat, env=env)
    reply = agent.act(
        wire.hands_from_state(state), seat_key=jax.random.key(served)
    )
    assert reply.action in state.legal_actions()
    assert reply.probs.dtype == np.float64 and reply.probs.shape == (
        env.num_actions,
    )
    np.testing.assert_allclose(reply.probs.sum(), 1.0, atol=1e-12)
    assert reply.decider in ("network", "forced")
    served += 1
  assert served >= 6


def test_host_agent_serves_a_bargaining_artifact_on_its_training_env(tmp_path):
  """Host agent serves a bargaining artifact on its training env."""
  env = registry.make("bargaining", max_turns=4, discount=0.9)
  pol = uniform.UniformPolicy(env.name, env.num_actions, env.num_players)
  ckpt.write_artifact(
      str(tmp_path),
      1,
      pol,
      pol.init_params(jax.random.key(0)),
      rules_sha=registry.rules_sha(env),
      config_hash="c",
      algo="test",
      env_name=env.name,
      env_spec=registry.spec_of(env).to_json(),
  )
  art = ckpt.load_artifact(str(tmp_path), 1)
  agent = host_agent.HostAgent(art, 1)
  assert registry.spec_of(agent.env) == registry.spec(
      "bargaining", max_turns=4, discount=0.9
  )
  reply = agent.act([[0, 9]], seat_key=jax.random.key(0))
  assert reply.legal[reply.action]
  with pytest.raises(ValueError, match="does not cover"):
    host_agent.HostAgent(art, 1, env=registry.make("bargaining"))


def _played(name, actions):
  """The pyspiel state after legal `actions`, player 0 to move.

  Args:
    name: The game.
    actions: The actions.

  Returns:
    The state.
  """
  state = pyspiel.load_game(name).new_initial_state()
  for a in actions:
    assert a in state.legal_actions(), (name, actions, a)
    state.apply_action(a)
  assert state.current_player() == 0, (name, actions)
  return state


@pytest.mark.parametrize("name", ["phantom_ttt", "dark_hex"])
def test_a_private_retry_changes_the_ply_count_not_the_information_state(name):
  """A private retry changes the ply count not the information state."""
  env = HIDDEN_MOVE_ENVS[name]()
  short, retry = _played(name, SHORT), _played(name, RETRY)
  assert short.information_state_tensor(0) == retry.information_state_tensor(0)
  rs = replay.replay_seat_steps(env, wire.hands_from_state(short), 0)
  rr = replay.replay_seat_steps(env, wire.hands_from_state(retry), 0)

  def last(r):
    """The last ply's SeatStep of the first row, as numpy.

    Args:
      r: The replay result.

    Returns:
      The SeatStep.
    """
    return jax.tree.map(lambda x: np.asarray(x[0, -1]), r.steps)

  for a, b in zip(jax.tree.leaves(last(rs)), jax.tree.leaves(last(rr))):
    np.testing.assert_array_equal(a, b)
  assert (rs.num_plies, rr.num_plies) == (
      3,
      4,
  )  # what a per-ply carry would count


def _stateful_policies(env):
  """Stateful policies for `env`: a recurrent network and a history policy.

  Args:
    env: The env.

  Returns:
    [(policy, params)].
  """
  rnn, rnn_params = _mlp_policy(env, recurrent=True)
  hist = fixtures.HistoryPolicy(env.name, env.num_actions, env.num_players)
  return [(rnn, rnn_params), (hist, hist.init_params())]


@pytest.mark.parametrize("name", sorted(HIDDEN_MOVE_ENVS))
def test_hidden_move_games_refuse_stateful_policies(name, tmp_path):
  """Hidden move games refuse stateful policies."""
  env = HIDDEN_MOVE_ENVS[name]()
  for pol, params in _stateful_policies(env):
    assert pol.spec.stateful
    slot = arena_lib.Slot(pol, 1, types.SAMPLE)
    with pytest.raises(ValueError, match="refuses stateful policies"):
      arena_lib.Arena(env, [slot])
    with pytest.raises(ValueError, match="refuses stateful policies"):
      host_arena.HostArena(env, [slot])
    with pytest.raises(ValueError, match="refuses stateful policies"):
      host_agent.HostAgent(helpers.artifact(pol, params), 0, env=env)
    with pytest.raises(ValueError, match="refuses stateful policies"):
      shards.ShardReader(
          str(tmp_path / "none"),
          env,
          rules_sha="r",
          info_schema_version=1,
          student=pol.spec,
      )
  assert not env.admits_stateful_policies


@pytest.mark.parametrize("name", ["phantom_ttt", "dark_hex"])
def test_feed_forward_policy_plays_and_cannot_see_the_retry(name):
  """Feed forward policy plays and cannot see the retry."""
  env = HIDDEN_MOVE_ENVS[name]()
  pol, params = _mlp_policy(env)
  ar = arena_lib.Arena(env, [arena_lib.Slot(pol, 1, types.SAMPLE)])
  b = 8
  dk, pk = arena_lib.independent_keys(jax.random.key(1), b, 2)
  stacked = [jax.tree.map(lambda x: x[None], params)]
  _, _, log = ar.collect(
      stacked, arena_lib.self_play_table(b, 2), dk, pk, env.max_len + 1
  )
  assert np.asarray(log.done).sum() >= 1 and not np.asarray(log.bad).any()
  agent = host_agent.HostAgent(helpers.artifact(pol, params), 0, env=env)
  replies = [
      agent.act(
          wire.hands_from_state(_played(name, h)), seat_key=jax.random.key(2)
      )
      for h in (SHORT, RETRY)
  ]
  np.testing.assert_array_equal(replies[0].logits, replies[1].logits)
  np.testing.assert_array_equal(replies[0].probs, replies[1].probs)
  assert replies[0].action == replies[1].action


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
