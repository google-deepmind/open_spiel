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

"""envs.euchre and encoders.euchre.

The Env contract with privacy, sides and identity, and equality with
OpenSpiel's corrected euchre on random games: every chance distribution,
actor, legal-action set, encoded information state (all four seats) and
return equal, the deal replayed from pyspiel's chance outcomes. Released
OpenSpiel's euchre is the older rules version (a 935-entry tensor), so the
comparison skips unless the installed pyspiel is the corrected build.
EUCHRE_PYSPIEL_GAMES sets the games per option set (default 40; the
acceptance run used 10,000).
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.jax.games import euchre as game_lib
from open_spiel.python.scale.core import sides as sides_lib
from open_spiel.python.scale.encoders import euchre as euchre_enc
from open_spiel.python.scale.encoders import registry as enc_registry
from open_spiel.python.scale.envs import euchre
from open_spiel.python.scale.envs import registry
from open_spiel.python.scale.testing import check_env

pyspiel = pytest.importorskip("pyspiel")

#: Random games per option set of the pyspiel comparison.
PYSPIEL_GAMES = int(os.environ.get("EUCHRE_PYSPIEL_GAMES", "40"))
#: Every option set the comparison plays.
OPTION_SETS = [
    {},
    {"allow_lone_defender": True},
    {"stick_the_dealer": False},
    {"allow_lone_defender": True, "stick_the_dealer": False},
]


def _corrected_reference() -> bool:
  """Whether the installed pyspiel's euchre is the corrected one (f32[959]).

  Returns:
    True for the corrected build.
  """
  shape = pyspiel.load_game("euchre").information_state_tensor_shape()
  return list(shape) == [game_lib.INFO_STATE_SIZE]


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """The default euchre env.

  Returns:
    The EuchreEnv.
  """
  return euchre.EuchreEnv()


def test_check_env_passes_with_privacy(env):
  """Check env passes with privacy."""
  check_env.check_env(env, hide_fn=euchre.hide_hidden_cards, num_playouts=32)


def test_check_env_passes_with_every_option_flipped():
  """Check env passes with every option flipped."""
  flipped = {k: not v for k, v in euchre.OPTIONS.items()}
  check_env.check_env(
      euchre.EuchreEnv(**flipped),
      hide_fn=euchre.hide_hidden_cards,
      num_playouts=32,
  )


def test_sides_are_two_zero_sum_partnerships(env):
  """Sides are North-South and East-West, zero-sum."""
  assert sides_lib.sides_of(env) == sides_lib.Sides.partnerships(
      4, 2, constant_sum=True
  )
  assert env.zero_sum
  assert sides_lib.sides_of(env).seats(0) == (0, 2)


def test_registry_spec_identity_and_encoder(env):
  """Registry spec identity and encoder."""
  assert registry.spec("euchre").kwargs == euchre.OPTIONS
  assert registry.spec_of(env) == registry.spec("euchre")
  made = registry.make("euchre")
  assert isinstance(made, euchre.EuchreEnv)
  sha = registry.rules_sha(made)
  assert sha == euchre.rules_sha(euchre.OPTIONS)
  assert (
      registry.rules_sha(registry.make("euchre", allow_lone_defender=True))
      != sha
  )
  with pytest.raises(TypeError):
    registry.make("euchre", stick_the_dealer=1)
  enc = enc_registry.get(euchre_enc.INFO_STATE_NAME)
  assert enc.shape == (959,) and enc.game == "euchre"


def test_info_state_matches_template(env):
  """Info state matches template."""
  s = env.init(jax.random.key(0))
  info = env.info_state(s, 1)
  tmpl = euchre.info_template()
  assert set(info) == set(tmpl)
  for k, v in tmpl.items():
    assert info[k].shape == v.shape and info[k].dtype == v.dtype, k


def test_encoder_equals_the_games_tensor_on_random_states(env):
  """The encoder equals the game's own tensor on every seat of random hands."""
  enc = jax.jit(lambda s, q: euchre_enc.info_state(env.info_state(s, q)))
  own = jax.jit(env.game.information_state_tensor)
  step = jax.jit(env.step)
  for g in range(8):
    s = env.init(jax.random.key(g))
    rng = np.random.default_rng(g)
    t = 0
    while not bool(env.terminal(s)):
      for q in range(4):
        np.testing.assert_array_equal(
            np.asarray(enc(s, q)), np.asarray(own(s, q))
        )
      legal = np.flatnonzero(np.asarray(env.legal_mask(s)))
      s = step(s, int(rng.choice(legal)), jax.random.key(1000 * g + t))
      t += 1


def test_discard_is_seen_by_the_dealer_only(env):
  """The discard: the dealer sees it, as an action and in the InfoState."""
  s = env.init(jax.random.key(5))
  # The first bidder orders up with partner: the dealer takes the up-card.
  up = int(np.asarray(s.upcard)) % 4
  s = env.apply(s, game_lib.CLUBS_TRUMP_ACTION + up)
  s = env.apply(s, game_lib.PLAY_WITH_PARTNER_ACTION)
  assert int(s.phase) == game_lib.PHASE_DISCARD
  dealer = int(env.actor(s))
  assert dealer == int(s.dealer)
  a = int(np.flatnonzero(np.asarray(env.legal_mask(s)))[0])
  assert int(env.visible_action(s, a, dealer)) == a
  assert int(env.visible_action(s, a, (dealer + 1) % 4)) == -1
  s = env.apply(s, a)
  assert int(env.info_state(s, dealer)["discard"]) == a
  for q in range(1, 4):
    assert int(env.info_state(s, (dealer + q) % 4)["discard"]) == -1


def test_hide_hidden_cards_moves_only_unseen_cards(env):
  """Hide hidden cards keeps every seat's count and the viewer's hand."""
  s = env.init(jax.random.key(3))
  for t in range(6):
    a = int(np.flatnonzero(np.asarray(env.legal_mask(s)))[-1])
    s = env.step(s, a, jax.random.key(100 + t))
  h = euchre.hide_hidden_cards(env, s, 0, jax.random.key(7))
  for q in range(-1, 4):
    assert (np.asarray(h.holder) == q).sum() == (
        np.asarray(s.holder) == q
    ).sum()
  np.testing.assert_array_equal(
      np.asarray(h.holder) == 0, np.asarray(s.holder) == 0
  )
  assert not np.array_equal(np.asarray(h.holder), np.asarray(s.holder))


def _lockstep(env, game_str, games, seed):
  """Plays random games of the env and pyspiel in lockstep and compares.

  Args:
    env: The EuchreEnv.
    game_str: The pyspiel game string.
    games: The games.
    seed: The seed.

  Returns:
    (decision nodes compared, chance nodes compared, terminal returns seen).
  """
  os_game = pyspiel.load_game(game_str)
  enc = jax.jit(
      jax.vmap(
          lambda s, q: euchre_enc.info_state(env.info_state(s, q)),
          in_axes=(None, 0),
      )
  )
  apply = jax.jit(env.apply)
  apply_chance = jax.jit(env.apply_chance)
  legal = jax.jit(env.legal_mask)
  actor = jax.jit(env.actor)
  is_chance = jax.jit(env.is_chance)
  probs = jax.jit(env.chance_probs)
  rets = jax.jit(env.returns)
  seats = jnp.arange(4)
  rng = np.random.default_rng(seed)
  nodes = chance_nodes = 0
  outcomes = set()
  for g in range(games):
    os_s, s = os_game.new_initial_state(), env.root()
    while not os_s.is_terminal():
      ctx = f"{game_str} game {g} history {os_s.history()}"
      assert not bool(env.terminal(s)), ctx
      if os_s.is_chance_node():
        assert bool(is_chance(s)), ctx
        outs, ps = zip(*os_s.chance_outcomes())
        p = np.asarray(probs(s))
        assert set(np.flatnonzero(p).tolist()) == set(outs), ctx
        np.testing.assert_allclose(p[list(outs)], ps, rtol=1e-6, err_msg=ctx)
        a = int(rng.choice(outs, p=ps))
        os_s.apply_action(a)
        s = apply_chance(s, a)
        chance_nodes += 1
        continue
      assert not bool(is_chance(s)), ctx
      assert int(actor(s)) == os_s.current_player(), ctx
      mask = np.asarray(legal(s))
      assert np.flatnonzero(mask).tolist() == sorted(os_s.legal_actions()), ctx
      x = np.asarray(enc(s, seats))
      for q in range(4):
        np.testing.assert_array_equal(
            x[q],
            np.asarray(os_s.information_state_tensor(q)),
            err_msg=f"{ctx} seat {q}",
        )
      np.testing.assert_array_equal(
          np.asarray(rets(s)), os_s.returns(), err_msg=ctx
      )
      nodes += 1
      a = int(rng.choice(os_s.legal_actions()))
      os_s.apply_action(a)
      s = apply(s, a)
    assert bool(env.terminal(s)) and int(actor(s)) == -1
    r = np.asarray(rets(s))
    np.testing.assert_array_equal(
        r, os_s.returns(), err_msg=f"{game_str} game {g} returns"
    )
    outcomes.add(tuple(r.tolist()))
  return nodes, chance_nodes, outcomes


@pytest.mark.parametrize("options", OPTION_SETS)
def test_matches_corrected_pyspiel_on_random_games(options):
  """Matches OpenSpiel's corrected euchre on random games."""
  if not _corrected_reference():
    pytest.skip(
        "needs a pyspiel with the corrected euchre (959-entry tensor); "
        "released OpenSpiel's euchre has 935"
    )
  env = euchre.EuchreEnv(**options)
  game_str = "euchre" + (
      "(" + ",".join(f"{k}={str(v).lower()}" for k, v in options.items()) + ")"
      if options
      else ""
  )
  nodes, chance_nodes, outcomes = _lockstep(
      env, game_str, PYSPIEL_GAMES, seed=len(options) + 7 * len(game_str)
  )
  assert chance_nodes == 22 * PYSPIEL_GAMES
  assert nodes >= PYSPIEL_GAMES * 4
  print(
      f"euchre vs pyspiel {game_str}: {PYSPIEL_GAMES} games, {nodes} decision"
      f" nodes and {chance_nodes} chance nodes equal, returns seen "
      f"{sorted(outcomes)}"
  )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider", "-s"]
      )
  )
