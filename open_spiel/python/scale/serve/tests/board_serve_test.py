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

"""Breakthrough, gomoku, hex and othello are servable.

breakthrough / gomoku / hex / othello are servable: a pyspiel state's
history replays into the env at pyspiel's node (actor, legal set,
terminal), and HostAgent with a planes-encoder CNN network artifact answers
pyspiel-legal actions over a whole game, both seats, on pyspiel's state.
"""

import sys

import jax
import numpy as np
import pytest

from open_spiel.python.scale.core import types
from open_spiel.python.scale.encoders import board_games
from open_spiel.python.scale.encoders import registry as enc_registry
from open_spiel.python.scale.networks import catalog
from open_spiel.python.scale.networks import policy as policy_lib
from open_spiel.python.scale.serve import games
from open_spiel.python.scale.serve import host_agent
from open_spiel.python.scale.serve import replay
from open_spiel.python.scale.serve import wire
from open_spiel.python.scale.serve.tests import helpers

pyspiel = pytest.importorskip("pyspiel")

GAMES = ["breakthrough", "gomoku", "hex", "othello"]
SMALL = {
    "breakthrough": "breakthrough(rows=6,columns=6)",
    "gomoku": "gomoku(size=7,connect=4)",
    "hex": "hex(board_size=5)",
    "othello": "othello",
}


def _random_prefixes(game, seed, every):
  """Every `every`-th prefix of one random game, and its end.

  Args:
    game: The pyspiel game.
    seed: The seed.
    every: The prefix stride.

  Returns:
    The prefix states.
  """
  rng = np.random.default_rng(seed)
  s = game.new_initial_state()
  out = []
  while True:
    if len(s.history()) % every == 0 or s.is_terminal():
      out.append(s.clone())
    if s.is_terminal():
      return out
    s.apply_action(int(rng.choice(s.legal_actions())))


@pytest.mark.parametrize("name", GAMES)
def test_replayed_node_equals_pyspiel_at_default_size(name):
  """Replayed node equals pyspiel at default size."""
  game = pyspiel.load_game(name)
  env = games.env_for(name, wire.game_params(game))
  checked = 0
  for state in _random_prefixes(game, seed=3, every=9):
    hands = wire.hands_from_state(state)
    assert hands == [list(state.history())]
    seat = 0 if state.is_terminal() else state.current_player()
    rep = replay.replay_seat_steps(env, hands, seat)
    wire.check_replayed_node(env, rep.state, state)
    checked += 1
  assert checked >= 5


def _network_artifact(env):
  """A small CNN artifact on the env's board planes.

  Args:
    env: The env.

  Returns:
    The artifact.
  """
  enc = board_games.planes_name(env)
  cfg = catalog.cnn(16, [1.0], 1, 32)  # the planes through a small CNN torso
  shape = list(enc_registry.get(enc).shape)
  pol = policy_lib.NetworkPolicy(
      env.name, enc, env.num_actions, env.num_players, enc, shape, cfg
  )
  return helpers.artifact(pol, pol.init_params(jax.random.key(0)), types.SAMPLE)


@pytest.mark.parametrize("name", GAMES)
def test_host_agent_plays_a_whole_pyspiel_game(name):
  """Host agent plays a whole pyspiel game."""
  game = pyspiel.load_game(SMALL[name])
  env = games.env_for(name, wire.game_params(game))
  art = _network_artifact(env)
  agents = [host_agent.HostAgent(art, seat, env=env) for seat in (0, 1)]
  state = game.new_initial_state()
  n = 0
  while not state.is_terminal():
    seat = state.current_player()
    hands = wire.hands_from_state(state)
    rep = replay.replay_seat_steps(env, hands, seat)
    wire.check_replayed_node(env, rep.state, state)
    reply = agents[seat].act(hands, seat_key=jax.random.key(n))
    assert reply.action in state.legal_actions(), (
        reply.action,
        state.history(),
    )
    probs = np.asarray(reply.probs)
    assert probs.dtype == np.float64 and abs(probs.sum() - 1.0) < 1e-9
    illegal = np.ones(env.num_actions, bool)
    illegal[state.legal_actions()] = False
    assert (probs[illegal] == 0).all()
    state.apply_action(int(reply.action))
    n += 1
  assert n >= 5


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
