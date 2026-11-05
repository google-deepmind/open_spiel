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

"""encoders.hearts_card_planes: shape, privacy and the waiting-row rule."""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale import deps

pytest.importorskip(deps.GAMES_PACKAGE)

# pylint: disable=g-import-not-at-top,wrong-import-position
from open_spiel.python.jax.games import hearts as game
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.encoders import hearts_card_planes as hcp
from open_spiel.python.scale.encoders import registry
from open_spiel.python.scale.envs import hearts

# pylint: enable=g-import-not-at-top,wrong-import-position

ENV = hearts.HeartsEnv()
_PLANES = jax.jit(jax.vmap(hcp.card_planes))
_WAIT = jax.jit(jax.vmap(hcp.waiting_row))
_INFO4 = jax.jit(
    lambda s: jax.vmap(lambda q: ENV.info_state(s, q))(jnp.arange(4))
)


def _states(seed, n_games=2):
  """Every player node of n_games random hands (uniform legal moves).

  Args:
    seed: the numpy seed.
    n_games: hands to play.

  Yields:
    (state, actor, action) at each player node, then (state, -1, -1) at
    each terminal.
  """
  x64.enable_x64()
  g = ENV.game
  rng = np.random.default_rng(seed)
  apply = jax.jit(g.apply_action)
  legal = jax.jit(g.legal_actions_mask)
  chance = jax.jit(g.chance_outcomes)
  for _ in range(n_games):
    s = g.new_initial_state()
    while not bool(g.is_terminal(s)):
      if bool(g.is_chance_node(s)):
        p = np.asarray(chance(s), np.float64)
        a = int(rng.choice(len(p), p=p / p.sum()))
      else:
        mask = np.asarray(legal(s))
        a = int(rng.choice(np.flatnonzero(mask)))
        yield s, int(g.current_player(s)), a
      s = apply(s, jnp.asarray(a, jnp.int32))
    yield s, -1, -1


def test_registered_shape_and_finite():
  enc = registry.get(hcp.NAME)
  assert enc.shape == (56, 52) and enc.game == "hearts"
  n = 0
  for s, _, _ in _states(0, 1):
    x = np.asarray(_PLANES(_INFO4(s)))
    assert x.shape == (4, 56, 52) and np.isfinite(x).all()
    n += 1
  assert n >= 53


def test_actor_and_phase_planes_on_decision_rows():
  """The acting seat sees itself as the actor, in the right phase."""
  ch = {c: i for i, c in enumerate(hcp.CHANNELS)}
  for s, actor, _ in _states(1, 2):
    if actor < 0:
      x = np.asarray(_PLANES(_INFO4(s)))
      assert (x[:, ch["b_terminal"]] == 1).all()
      continue
    x = np.asarray(_PLANES(_INFO4(s)))[actor]
    assert x[ch["b_actor_rel0"]].min() == 1.0, "the actor is the viewer"
    passing = int(s.phase) == game.PHASE_PASS
    assert x[ch["b_phase_pass"], 0] == float(passing)
    assert x[ch["b_phase_play"], 0] == float(not passing)
    # my_hand is the seat's hand; its legal moves are in it
    hand = np.asarray(ENV.info_state(s, actor)["hand"])
    np.testing.assert_array_equal(x[ch["my_hand"]], hand.astype(np.float32))


def test_privacy_hidden_cards_do_not_change_a_seats_planes():
  """Re-dealing what a seat cannot see leaves its planes; others move."""
  hide = jax.jit(lambda s, q, k: hearts.hide_hidden_cards(ENV, s, q, k))
  changed_other = 0
  for i, (s, actor, _) in enumerate(_states(2, 1)):
    if actor < 0 or i % 3:
      continue
    for q in range(4):
      t = hide(s, jnp.int32(q), jax.random.key(100 * i + q))
      a = np.asarray(hcp.card_planes(ENV.info_state(s, q)))
      b = np.asarray(hcp.card_planes(ENV.info_state(t, q)))
      np.testing.assert_array_equal(a, b)
      # positive control: another seat's view of the re-dealt state
      o = (q + 1) % 4
      if not np.array_equal(
          np.asarray(hcp.card_planes(ENV.info_state(s, o))),
          np.asarray(hcp.card_planes(ENV.info_state(t, o))),
      ):
        changed_other += 1
  assert changed_other > 0


def test_waiting_row_matches_a_round_robin_clock():
  """waiting_row marks exactly the seats a row clock passes while waiting.

  The clock starts on the first leader (the first lead is out of scope);
  each row belongs to seat clock % 4: the seat on move plays, any other seat
  waits (a value-only row) at the state before the next play.
  """
  n_rows = 0
  clock = None
  for s, actor, _ in _states(3, 3):
    if actor < 0:
      clock = None
      continue
    if int(s.phase) != game.PHASE_PLAY:
      continue
    want = np.zeros(4, bool)
    if clock is None:
      clock = actor
    while clock % 4 != actor:
      want[clock % 4] = True
      clock += 1
    clock += 1  # the actor's row
    got = np.asarray(_WAIT(_INFO4(s)))
    n_played = int(s.n_played)
    if n_played == 0:
      want[:] = False
    np.testing.assert_array_equal(got, want, err_msg=f"n_played {n_played}")
    assert not got[actor]
    n_rows += int(want.sum())
  assert n_rows > 10


if __name__ == "__main__":
  sys.exit(pytest.main([__file__, "-p", "no:cacheprovider"]))
