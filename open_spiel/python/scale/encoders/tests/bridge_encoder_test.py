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

"""Tests for encoders.bridge.

Public_facts replays a seat's view into the board's public
state exactly (checked against the engine state at every ply of random
boards), and bridge_info_state keeps the encoder contract (declared shape and
dtype, batch == per example, jit == eager, finite, blind to hidden cards).
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.jax.games import bridge as game
from open_spiel.python.scale.encoders import bridge as enc
from open_spiel.python.scale.encoders import registry
from open_spiel.python.scale.envs import bridge
from open_spiel.python.scale.testing import infoset_invariance as twins

N = 12


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """A bridge env.

  Returns:
    The env.
  """
  return bridge.BridgeEnv()


@pytest.fixture(name="flat", scope="module")
def _flat_fixture(env):
  """Every ply of N random boards, flattened: states [N * 372].

  Args:
    env: The bridge env.

  Returns:
    The flattened states.
  """
  states, _, _ = twins.playouts(env, N, env.max_len, seed=11)
  return jax.tree_util.tree_map(
      lambda x: x.reshape((-1,) + x.shape[2:]), states
  )


def _infos(env, flat, seat):
  """The InfoStates of `seat` at every flattened state.

  Args:
    env: The env.
    flat: The flattened states.
    seat: The seat.

  Returns:
    The InfoStates.
  """
  return jax.jit(jax.vmap(lambda s: env.info_state(s, jnp.int32(seat))))(flat)


def test_registered():
  """Registered."""
  e = registry.get("bridge_info_state")
  assert e is enc.INFO_STATE and e.game == "bridge" and e.shape == (enc.DIM,)
  assert enc.DIM == 978


def test_public_facts_match_the_engine_state(env, flat):
  """Public facts match the engine state."""
  facts = jax.jit(jax.vmap(enc.public_facts))(_infos(env, flat, 2))
  phase = np.asarray(flat.phase)
  want_phase = np.select(
      [phase == game.PHASE_AUCTION, phase == game.PHASE_PLAY],
      [enc.PHASE_AUCTION, enc.PHASE_PLAY],
      enc.PHASE_OVER,
  )
  np.testing.assert_array_equal(np.asarray(facts.phase), want_phase)
  np.testing.assert_array_equal(
      np.asarray(facts.n_calls), np.asarray(flat.n_calls)
  )
  np.testing.assert_array_equal(
      np.asarray(facts.n_played), np.asarray(flat.n_played)
  )
  np.testing.assert_array_equal(np.asarray(facts.level), np.asarray(flat.level))
  np.testing.assert_array_equal(
      np.asarray(facts.strain), np.asarray(flat.strain)
  )
  np.testing.assert_array_equal(np.asarray(facts.dbl), np.asarray(flat.dbl))
  np.testing.assert_array_equal(
      np.asarray(facts.declarer), np.asarray(flat.declarer)
  )
  np.testing.assert_array_equal(
      np.asarray(facts.decl_tricks), np.asarray(flat.decl_tricks)
  )
  live = want_phase != enc.PHASE_OVER
  np.testing.assert_array_equal(
      np.asarray(facts.on_move)[live], np.asarray(flat.on_move)[live]
  )
  assert (np.asarray(facts.on_move)[~live] == -1).all()
  # Who played each card: a played card's player is the seat it was dealt to.
  player = np.asarray(facts.card_player)
  played = np.asarray(flat.holder) < 0
  np.testing.assert_array_equal(player >= 0, played)
  np.testing.assert_array_equal(player[played], np.asarray(flat.deal)[played])
  tl = np.asarray(facts.trick_leader)
  np.testing.assert_array_equal(tl, np.asarray(flat.trick_leader))
  # The coverage this test relies on: auctions, play and finished boards.
  assert {enc.PHASE_AUCTION, enc.PHASE_PLAY, enc.PHASE_OVER} <= set(
      want_phase.tolist()
  )
  assert (np.asarray(flat.dbl) == 2).any()


def test_encoder_contract(env, flat):
  """Encoder contract."""
  e = enc.INFO_STATE
  info = _infos(env, flat, 1)
  x = jax.jit(e.batch)(info)
  assert x.shape == (flat.phase.shape[0], enc.DIM) and x.dtype == jnp.float32
  assert np.isfinite(np.asarray(x)).all()
  assert ((np.asarray(x) >= 0.0) & (np.asarray(x) <= 1.0)).all()
  for i in (0, 37, 200, 371, 900):
    one = jax.tree_util.tree_map(lambda v, i=i: v[i], info)
    np.testing.assert_array_equal(np.asarray(e(one)), np.asarray(x[i]))
    np.testing.assert_array_equal(np.asarray(e.fn(one)), np.asarray(x[i]))
  # Own remaining cards: the seat's unplayed cards.
  mine = np.asarray(flat.holder) == 1
  np.testing.assert_array_equal(np.asarray(x[:, :52]) > 0, mine)


def test_encoder_is_blind_to_hidden_cards(env, flat):
  """Encoder is blind to hidden cards."""
  keys = jax.random.split(jax.random.key(5), flat.phase.shape[0])
  hidden = jax.jit(
      jax.vmap(lambda s, k: bridge.hide_hidden_cards(env, s, jnp.int32(3), k))
  )(flat, keys)
  a = jax.jit(enc.INFO_STATE.batch)(_infos(env, flat, 3))
  b = jax.jit(enc.INFO_STATE.batch)(_infos(env, hidden, 3))
  np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
  assert (np.asarray(hidden.deal) != np.asarray(flat.deal)).any()


def test_compact_history_encodes_equal_to_the_raw_history(env, flat):
  """Compact history encodes equal to the raw history."""
  cenv = bridge.BridgeEnv(compact_history=True)
  for seat in (0, 3):
    raw = jax.jit(enc.INFO_STATE.batch)(_infos(env, flat, seat))
    compact_info = _infos(cenv, flat, seat)
    assert "actions" not in compact_info and "calls" in compact_info
    compact = jax.jit(enc.INFO_STATE.batch)(compact_info)
    np.testing.assert_array_equal(np.asarray(compact), np.asarray(raw))


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
