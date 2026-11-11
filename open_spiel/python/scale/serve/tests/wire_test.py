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

import jax
import numpy as np
import pytest

from open_spiel.python.scale.core import chance
from open_spiel.python.scale.envs import gin
from open_spiel.python.scale.envs import registry
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


def _env_playout(env, seed):
  """One uniformly random hand of a gin env from its root, on the host.

  Args:
    env: The gin env.
    seed: The seed.

  Returns:
    (raw history, actor per entry (-1 for chance), returns).
  """
  apply_chance, apply = jax.jit(env.apply_chance), jax.jit(env.apply)
  node = jax.jit(
      lambda s: (
          env.terminal(s),
          env.is_chance(s),
          env.actor(s),
          env.legal_mask(s),
      )
  )
  draw = jax.jit(lambda s, k: chance.sample_outcome(env, s, k))
  rng = np.random.default_rng(seed)
  key = jax.random.key(seed)
  s, raw, actors = env.root(), [], []
  while True:
    term, is_c, actor, legal = node(s)
    if bool(term):
      return raw, actors, [float(x) for x in np.asarray(env.returns(s))]
    if bool(is_c):
      key, k = jax.random.split(key)
      a, who = int(draw(s, k)), -1
    else:
      a, who = int(rng.choice(np.flatnonzero(np.asarray(legal)))), int(actor)
    raw.append(a)
    actors.append(who)
    s = apply_chance(s, a) if bool(is_c) else apply(s, a)


@pytest.mark.parametrize("oklahoma_frac", [1.0, None, 0.5])
def test_gin_raw_history_replays_in_pyspiel(oklahoma_frac):
  """Env gin hands replay in pyspiel (gin.raw_history_to_pyspiel).

  Random env hands with either first mover (the env's first-player node) map
  to a pyspiel gin_rummy history with no starting_player parameter: every
  entry is legal for the mapped actor, the hand ends where the env's does,
  and the returns are the env's per mapped seat, except that a hand whose
  knocker and defender tie on deadwood scores 0-0 in released pyspiel
  (2.0.2 and earlier) and an undercut in the env (the games package's
  gin_rummy module doc).

  Args:
    oklahoma_frac: The env's Oklahoma deal fraction.
  """
  env = registry.make("gin_rummy", oklahoma_frac=oklahoma_frac)
  kwargs = registry.spec_of(env).kwargs
  first = set()
  for seed in range(8):
    raw, actors, ret = _env_playout(env, seed)
    params, order, seat_of = gin.raw_history_to_pyspiel(raw, kwargs)
    first.add(seat_of[0])
    name = "gin_rummy(oklahoma=true)" if params.get("oklahoma") else "gin_rummy"
    state = pyspiel.load_game(name).new_initial_state()
    for i in order:
      assert not state.is_terminal(), (seed, i)
      if state.is_chance_node():
        assert actors[i] == -1, (seed, i)
        assert raw[i] in [o for o, _ in state.chance_outcomes()], (seed, i)
      else:
        assert seat_of[state.current_player()] == actors[i], (seed, i)
        assert raw[i] in state.legal_actions(), (seed, i)
      state.apply_action(raw[i])
    assert state.is_terminal(), seed
    want = [ret[q] for q in seat_of]
    got = state.returns()
    if not np.allclose(got, want):
      assert got == [0.0, 0.0] and sorted(want) == [-25.0, 25.0], (got, want)
  assert first == {0, 1}, "both first movers"
  # a rule option off its default has no pyspiel game
  assert (
      gin.raw_history_to_pyspiel(
          raw, dict(kwargs, allow_upcard_throwback=False)
      )
      is None
  )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
