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

"""Tests of encoders.euchre_card_planes: shape, privacy, planes, value rows."""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale import deps

pytest.importorskip(deps.GAMES_PACKAGE)

# pylint: disable=g-import-not-at-top,wrong-import-position
from open_spiel.python.jax.games import euchre as game
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.encoders import euchre_card_planes as ecp
from open_spiel.python.scale.encoders import registry
from open_spiel.python.scale.envs import euchre

# pylint: enable=g-import-not-at-top,wrong-import-position

ENV = euchre.EuchreEnv()
_PLANES = jax.jit(jax.vmap(ecp.card_planes))
_WAIT = jax.jit(jax.vmap(ecp.waiting_row))
_INFO4 = jax.jit(
    lambda s: jax.vmap(lambda q: ENV.info_state(s, q))(jnp.arange(4))
)
CH = {c: i for i, c in enumerate(ecp.CHANNELS)}


def _states(seed, n_games=4, alone_bias=0.0):
  """Every player node of n_games random hands (uniform legal moves).

  Args:
    seed: the numpy seed.
    n_games: hands to play.
    alone_bias: the chance a go-alone decision goes alone (else uniform).

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
        if mask[game.GO_ALONE_ACTION] and rng.random() < alone_bias:
          a = game.GO_ALONE_ACTION
        else:
          a = int(rng.choice(np.flatnonzero(mask)))
        yield s, int(g.current_player(s)), a
      s = apply(s, jnp.asarray(a, jnp.int32))
    yield s, -1, -1


def test_registered_shape_and_finite():
  enc = registry.get(ecp.NAME)
  assert enc.shape == (64, 24) and enc.game == "euchre"
  n = 0
  for s, _, _ in _states(0, 1):
    x = np.asarray(_PLANES(_INFO4(s)))
    assert x.shape == (4, 64, 24) and np.isfinite(x).all()
    n += 1
  assert n >= 18  # one bid, the go-alone decision, 15 plays alone, the terminal


@pytest.mark.parametrize("alone_bias", [0.0, 0.7])
def test_decision_planes(alone_bias):
  """The seat to move sees itself on move, in the engine's phase, its hand.

  Args:
    alone_bias: the random games' go-alone bias.
  """
  phase_of = {
      game.PHASE_BIDDING: None,
      game.PHASE_DISCARD: ecp.PHASE_DISCARD,
      game.PHASE_GO_ALONE: ecp.PHASE_ALONE,
      game.PHASE_PLAY: ecp.PHASE_PLAY,
  }
  for s, actor, _ in _states(1, 6, alone_bias):
    x = np.asarray(_PLANES(_INFO4(s)))
    if actor < 0:
      assert (x[:, CH["b_terminal"]] == 1).all()
      assert (x[:, [CH[f"b_actor_rel{k}"] for k in range(4)]] == 0).all()
      continue
    xa = x[actor]
    assert xa[CH["b_actor_rel0"]].min() == 1.0, "the actor is the viewer"
    want = phase_of[int(s.phase)]
    if want is None:
      want = ecp.PHASE_BID1 if int(s.num_passes) < 4 else ecp.PHASE_BID2
    assert xa[CH[f"b_phase{want}"], 0] == 1.0
    assert sum(xa[CH[f"b_phase{p}"], 0] for p in range(5)) == 1.0
    hand = np.asarray(ENV.info_state(s, actor)["hand"])
    np.testing.assert_array_equal(xa[CH["my_hand"]], hand.astype(np.float32))
    # every seat agrees on the public facts
    for c in (
        "b_trump0",
        "b_trump1",
        "b_trump2",
        "b_trump3",
        "b_alone",
        "b_trick_num_over5",
        "upcard",
    ):
      assert (x[:, CH[c]] == x[0, CH[c]]).all(), c


def test_privacy_hidden_cards_do_not_change_a_seats_planes():
  """Re-dealing what a seat cannot see leaves its planes; others move."""
  hide = jax.jit(lambda s, q, k: euchre.hide_hidden_cards(ENV, s, q, k))
  changed_other = 0
  for i, (s, actor, _) in enumerate(_states(2, 3)):
    if actor < 0 or i % 2:
      continue
    for q in range(4):
      t = hide(s, jnp.int32(q), jax.random.key(100 * i + q))
      a = np.asarray(ecp.card_planes(ENV.info_state(s, q)))
      b = np.asarray(ecp.card_planes(ENV.info_state(t, q)))
      np.testing.assert_array_equal(a, b)
      a_w, b_w = ecp.waiting_row(ENV.info_state(s, q)), ecp.waiting_row(
          ENV.info_state(t, q)
      )
      assert bool(a_w) == bool(b_w)
      o = (
          q + 1
      ) % 4  # positive control: another seat's view of the re-dealt state
      if not np.array_equal(
          np.asarray(ecp.card_planes(ENV.info_state(s, o))),
          np.asarray(ecp.card_planes(ENV.info_state(t, o))),
      ):
        changed_other += 1
  assert changed_other > 0


def test_discard_is_seen_by_the_dealer_only():
  """At the first lead after a discard only the dealer's my_discard is set."""
  seen = 0
  for s, actor, _ in _states(4, 12):
    if actor < 0 or int(s.phase) != game.PHASE_PLAY or int(s.discard) < 0:
      continue
    x = np.asarray(_PLANES(_INFO4(s)))
    d = int(s.dealer)
    assert x[d, CH["my_discard"], int(s.discard)] == 1.0
    for q in range(4):
      if q != d:
        assert x[q, CH["my_discard"]].sum() == 0.0
        assert x[q, CH["unseen"], int(s.discard)] == 1.0
    seen += 1
  assert seen > 0


@pytest.mark.parametrize("alone_bias", [0.0, 0.7])
def test_waiting_row_matches_a_round_robin_clock(alone_bias):
  """waiting_row marks exactly the seats a row clock passes while waiting.

  The clock starts on the first bidder (the first bid is out of scope); each
  row belongs to seat clock % 4: the seat on move acts, any other seat waits
  (a value-only row) at the state before the next move. A seat other than
  the dealer is out of scope between the dealer's taking the up-card and the
  first lead (the discard is hidden from it).

  Args:
    alone_bias: the random games' go-alone bias.
  """
  n_rows = 0
  clock = None
  for s, actor, _ in _states(3, 8, alone_bias):
    if actor < 0:
      clock = None
      continue
    want = np.zeros(4, bool)
    first = clock is None
    if first:
      clock = actor
    while clock % 4 != actor:
      want[clock % 4] = True
      clock += 1
    clock += 1  # the actor's row
    got = np.asarray(_WAIT(_INFO4(s)))
    if first:
      want[:] = False
    takes = (
        (int(s.num_passes) < 4)
        & (int(s.go_alone) >= 0)
        & (int(s.n_played) == 0)
    )
    if takes and int(s.phase) in (game.PHASE_DISCARD, game.PHASE_PLAY):
      dealer_out = (
          int(s.go_alone) == 1 and int(s.dealer) == (int(s.declarer) + 2) % 4
      )
      if not dealer_out:
        mask = np.arange(4) != int(s.dealer)
        assert not got[mask].any()
        want[mask] = False
    np.testing.assert_array_equal(
        got, want, err_msg=f"phase {int(s.phase)} n_played {int(s.n_played)}"
    )
    assert not got[actor]
    n_rows += int(want.sum())
  assert n_rows > 20


if __name__ == "__main__":
  sys.exit(pytest.main([__file__, "-p", "no:cacheprovider"]))
