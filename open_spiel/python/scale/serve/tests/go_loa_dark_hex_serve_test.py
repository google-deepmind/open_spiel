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

"""Serving go, lines_of_action, dark_hex and dark_hex_ir.

Serving go, lines_of_action, dark_hex and dark_hex_ir: the env table
builds the right env from pyspiel's game parameters and refuses variants
the env does not play; the replayed node equals pyspiel's on random
playouts (wire.check_replayed_node at every prefix); a HostAgent over a
random-weight network answers a legal move.
"""

import sys

import jax
import numpy as np
import pytest

from open_spiel.python.scale.encoders import registry as enc_registry
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.networks import catalog
from open_spiel.python.scale.networks import policy as policy_lib
from open_spiel.python.scale.serve import games
from open_spiel.python.scale.serve import host_agent
from open_spiel.python.scale.serve import replay
from open_spiel.python.scale.serve import wire
from open_spiel.python.scale.serve.tests import helpers
import pyspiel

# (pyspiel name, load params, the EnvSpec it must serve, encoder, max prefixes)
CASES = [
    ("go", {"board_size": 9}, env_registry.spec("go"), "go_planes", 60),
    (
        "go",
        {"board_size": 5, "komi": 0.5},
        env_registry.spec("go", board_size=5, komi=0.5),
        "go_planes@board_size=5",
        60,
    ),
    (
        "lines_of_action",
        {},
        env_registry.spec("lines_of_action"),
        "lines_of_action_planes",
        40,
    ),
    ("dark_hex", {}, env_registry.spec("dark_hex"), "dark_hex_info_state", 200),
    (
        "dark_hex_ir",
        {"gameversion": "adh"},
        env_registry.spec("dark_hex_ir"),
        "dark_hex_ir_view",
        200,
    ),
]


def _prefixes(game, n, seed, cap):
  """Prefix states of n random games, thinned to `cap` keeping terminals.

  Args:
    game: The pyspiel game.
    n: The games.
    seed: The seed.
    cap: The most states.

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
      s.apply_action(int(rng.choice(s.legal_actions())))
  if len(out) > cap:  # keep every terminal, thin the rest evenly
    keep = set(np.linspace(0, len(out) - 1, cap).astype(int))
    out = [s for i, s in enumerate(out) if i in keep or s.is_terminal()]
  return out


@pytest.mark.parametrize("name,params,want,enc,cap", CASES)
def test_env_for_pyspiel_parameters(name, params, want, enc, cap):
  """Env for pyspiel parameters."""
  del enc, cap
  game = pyspiel.load_game(name, params)
  env = games.env_for(name, wire.game_params(game))
  assert env.name == name and env_registry.spec_of(env) == want


@pytest.mark.parametrize("name,params,want,enc,cap", CASES)
def test_replayed_node_equals_pyspiel(name, params, want, enc, cap):
  """Replayed node equals pyspiel."""
  del want, enc
  game = pyspiel.load_game(name, params)
  env = games.env_for(name, wire.game_params(game))
  n = 0
  for state in _prefixes(
      game, 4 if name in ("go", "lines_of_action") else 20, 1, cap
  ):
    hands = wire.hands_from_state(state)
    seat = 0 if state.is_terminal() else state.current_player()
    rep = replay.replay_seat_steps(env, hands, seat)
    wire.check_replayed_node(env, rep.state, state)
    n += 1
  assert n >= 30


def test_refused_variants():
  """Refused variants."""
  with pytest.raises(
      ValueError, match="gameversion"
  ):  # pyspiel's dark_hex_ir default is cdh
    games.env_for(
        "dark_hex_ir", wire.game_params(pyspiel.load_game("dark_hex_ir"))
    )
  with pytest.raises(ValueError, match="gameversion"):
    games.env_for("dark_hex", {"gameversion": "adh"})
  with pytest.raises(ValueError, match="board_size"):
    games.env_for("dark_hex", {"board_size": 4})
  with pytest.raises(ValueError, match="handicap"):
    games.env_for("go", {"board_size": 9, "handicap": 2})
  with pytest.raises(ValueError, match="max_game_length"):
    games.env_for("go", {"board_size": 9, "max_game_length": 100})
  env = games.env_for("lines_of_action", {"max_game_length": 200})
  assert env.max_game_length == 200 and env.max_len == 200
  assert games.env_for("go", {"board_size": 13}).board_size == 13


def _network(env, enc_name):
  """A small network on `enc_name` (CNN for planes, else MLP).

  Args:
    env: The env.
    enc_name: The encoder.

  Returns:
    (policy, params).
  """
  enc = enc_registry.get(enc_name)
  if len(enc.shape) == 3:
    net = catalog.cnn(32, (0.25, 0.5), 1, 32)
  else:
    net = catalog.mlp((32, 32), "relu", "shared")
  pol = policy_lib.NetworkPolicy(
      env.name,
      enc_name,
      env.num_actions,
      env.num_players,
      enc_name,
      list(enc.shape),
      net,
  )
  return pol, pol.init_params(jax.random.key(1))


@pytest.mark.parametrize("name,params,want,enc,cap", CASES)
def test_host_agent_answers_a_legal_move(name, params, want, enc, cap):
  """Host agent answers a legal move."""
  del want, cap
  game = pyspiel.load_game(name, params)
  env = games.env_for(name, wire.game_params(game))
  pol, p = _network(env, enc)
  art = helpers.artifact(pol, p)
  rng = np.random.default_rng(7)
  state = game.new_initial_state()
  for _ in range(
      3 if name.startswith("dark_hex") else 6
  ):  # history to replay (hex ends >= 5)
    state.apply_action(int(rng.choice(state.legal_actions())))
  assert not state.is_terminal()
  seat = state.current_player()
  agent = host_agent.HostAgent(art, seat, env=env)
  reply = agent.act(
      wire.hands_from_state(state),
      seat_key=jax.random.key(3),
      node_check=lambda s: wire.check_replayed_node(env, s, state),
  )
  assert reply.action in state.legal_actions()
  assert set(reply.distribution()) <= set(state.legal_actions())


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
