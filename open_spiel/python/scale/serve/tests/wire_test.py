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

"""Pyspiel state -> env raw history (wire.py) and the env table (games.py).

pyspiel state -> env raw history (wire.py) and the env table
(games.py): on random pyspiel playouts the replayed node equals pyspiel's
(actor, legal set, terminal) at every prefix, for every served game.
"""

import sys

import numpy as np
import pytest

from open_spiel.python.scale.serve import games
from open_spiel.python.scale.serve import replay
from open_spiel.python.scale.serve import wire
import pyspiel

GAMES = [
    "kuhn_poker",
    "leduc_poker",
    "connect_four",
    "tic_tac_toe",
    "phantom_ttt",
]


def _playouts(game, n, seed, max_prefixes=None):
  """Every prefix state of n random pyspiel playouts.

  Args:
    game: The pyspiel game.
    n: The games.
    seed: The seed.
    max_prefixes: The most states, or None for all.

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
  return out[:max_prefixes] if max_prefixes else out


@pytest.mark.parametrize("name", GAMES)
def test_replayed_node_equals_pyspiel(name):
  """Replayed node equals pyspiel."""
  game = pyspiel.load_game(name)
  env = games.env_for(name, wire.game_params(game))
  n_checked = 0
  for state in _playouts(game, 15, seed=1):
    if state.is_chance_node():
      continue
    hands = wire.hands_from_state(state)
    seat = 0 if state.is_terminal() else state.current_player()
    rep = replay.replay_seat_steps(env, hands, seat)
    wire.check_replayed_node(env, rep.state, state)  # raises on any mismatch
    n_checked += 1
  assert n_checked > 30


def test_mismatch_is_detected():
  """A history replayed into the wrong game's env is refused, not served."""
  kuhn_state = pyspiel.load_game("kuhn_poker").new_initial_state()
  for a in (0, 1, 0):
    kuhn_state.apply_action(a)
  leduc_env = games.env_for("leduc_poker", {})
  with pytest.raises((ValueError, AssertionError)):
    rep = replay.replay_seat_steps(
        leduc_env, wire.hands_from_state(kuhn_state), 1
    )
    wire.check_replayed_node(leduc_env, rep.state, kuhn_state)


def test_env_table():
  """Env table."""
  assert set(GAMES) <= set(games.served_games())
  assert games.env_for("kuhn_poker", {}).name == "kuhn_poker"
  with pytest.raises(KeyError, match="no env for game"):
    games.env_for("chess", {})
  with pytest.raises(ValueError, match="players"):
    games.env_for("kuhn_poker", {"players": 3})


def test_gin_replayed_node_equals_pyspiel():
  """Gin replayed node equals pyspiel."""
  game = pyspiel.load_game("gin_rummy")
  params = wire.game_params(game)
  env = games.env_for("gin_rummy", params)
  states = [s for s in _playouts(game, 2, seed=2) if not s.is_chance_node()]
  # Every 7th decision of two hands (each replay is a full host replay).
  for state in states[::7]:
    hands = wire.hands_from_state(state)
    seat = 0 if state.is_terminal() else state.current_player()
    rep = replay.replay_seat_steps(env, hands, seat)
    wire.check_replayed_node(env, rep.state, state)


def test_gin_prefix_known_answer():
  """Gin prefix known answer.

  The gin env has an explicit first-player chance node (and a variant node when
  0 < oklahoma_frac < 1); pyspiel's history has neither. The wire prefixes the
  first player (pyspiel's starting_player, default 0).
  """
  hist = [5, 17, 30]
  assert wire.pyspiel_history_to_raw("gin_rummy", hist, {}) == [[0] + hist]
  assert wire.pyspiel_history_to_raw(
      "gin_rummy", hist, {"starting_player": 1}
  ) == [[1] + hist]
  assert wire.pyspiel_history_to_raw(
      "kuhn_poker", [0, 1, 0], {"players": 2}
  ) == [[0, 1, 0]]
  with pytest.raises(ValueError, match="starting_player"):
    wire.pyspiel_history_to_raw("gin_rummy", hist, {"starting_player": 2})


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
