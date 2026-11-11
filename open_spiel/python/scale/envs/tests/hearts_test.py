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

"""envs.hearts and encoders.hearts.

The Env contract with privacy, sides and identity, and equality with pyspiel
hearts on random games (every actor, legal-action set, encoded information state
and return equal).
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import sides as sides_lib
from open_spiel.python.scale.encoders import hearts as hearts_enc
from open_spiel.python.scale.encoders import registry as enc_registry
from open_spiel.python.scale.envs import hearts
from open_spiel.python.scale.envs import registry
from open_spiel.python.scale.testing import check_env

pyspiel = pytest.importorskip("pyspiel")

#: Random games per case of the pyspiel comparison (HEARTS_PYSPIEL_GAMES
#: overrides it; a longer run used 300).
PYSPIEL_GAMES = int(os.environ.get("HEARTS_PYSPIEL_GAMES", "40"))


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """The default hearts env.

  Returns:
    The HeartsEnv.
  """
  return hearts.HeartsEnv()


def test_check_env_passes_with_privacy(env):
  """Check env passes with privacy."""
  check_env.check_env(env, hide_fn=hearts.hide_hidden_cards, num_playouts=16)


def test_check_env_passes_with_every_option_flipped():
  """Check env passes with every option flipped."""
  flipped = {k: not v for k, v in hearts.OPTIONS.items()}
  check_env.check_env(
      hearts.HeartsEnv(**flipped),
      hide_fn=hearts.hide_hidden_cards,
      num_playouts=8,
  )


def test_sides_are_four_individual_general_sum(env):
  """Sides are four individual general sum."""
  assert sides_lib.sides_of(env) == sides_lib.Sides.individual(
      4, constant_sum=False
  )
  assert not env.zero_sum
  assert sides_lib.sides_of(env).num_sides == 4


def test_registry_spec_identity_and_encoder(env):
  """Registry spec identity and encoder."""
  assert registry.spec("hearts").kwargs == hearts.OPTIONS
  assert registry.spec_of(env) == registry.spec("hearts")
  made = registry.make("hearts")
  assert isinstance(made, hearts.HeartsEnv)
  sha = registry.rules_sha(made)
  assert sha == hearts.rules_sha(hearts.OPTIONS)
  assert registry.rules_sha(registry.make("hearts", jd_bonus=True)) != sha
  with pytest.raises(TypeError):
    registry.make("hearts", jd_bonus=1)
  enc = enc_registry.get(hearts_enc.INFO_STATE_NAME)
  assert enc.shape == (5088,) and enc.game == "hearts"


def test_info_state_matches_template(env):
  """Info state matches template."""
  s = env.init(jax.random.key(0))
  info = env.info_state(s, 1)
  tmpl = hearts.info_template()
  assert set(info) == set(tmpl)
  for k, v in tmpl.items():
    assert info[k].shape == v.shape and info[k].dtype == v.dtype, k


def _lockstep(env, game_str, games, seed):
  """Plays random games of the env and pyspiel in lockstep and compares.

  Args:
    env: The HeartsEnv.
    game_str: The pyspiel game string.
    games: The games.
    seed: The seed.

  Returns:
    (nodes compared, games that shot the moon).
  """
  os_game = pyspiel.load_game(game_str)
  enc = jax.jit(
      jax.vmap(
          lambda s, q: hearts_enc.info_state(env.info_state(s, q)),
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
  nodes = moons = 0
  for g in range(games):
    os_s, s = os_game.new_initial_state(), env.root()
    while not os_s.is_terminal():
      ctx = f"game {g} history {os_s.history()}"
      if os_s.is_chance_node():
        assert bool(is_chance(s)), ctx
        outs, ps = zip(*os_s.chance_outcomes())
        p = np.asarray(probs(s))
        assert set(np.flatnonzero(p).tolist()) == set(outs), ctx
        np.testing.assert_allclose(p[list(outs)], ps, rtol=1e-6, err_msg=ctx)
        a = int(rng.choice(outs, p=ps))
        os_s.apply_action(a)
        s = apply_chance(s, a)
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
      nodes += 1
      a = int(rng.choice(os_s.legal_actions()))
      os_s.apply_action(a)
      s = apply(s, a)
    assert bool(env.terminal(s)) and int(actor(s)) == -1
    r = np.asarray(rets(s))
    np.testing.assert_array_equal(
        r, os_s.returns(), err_msg=f"game {g} returns"
    )
    moons += int(r.sum() != 78.0)
  return nodes, moons


@pytest.mark.parametrize(
    "options",
    [
        {},
        {
            "pass_cards": False,
            "no_pts_on_first_trick": False,
            "can_lead_any_club": True,
            "jd_bonus": True,
        },
        {
            "avoid_all_tricks_bonus": True,
            "must_break_hearts": False,
            "qs_breaks_hearts": False,
            "can_lead_hearts_instead_of_qs": True,
        },
    ],
)
def test_matches_pyspiel_on_random_games(options):
  """Matches pyspiel on random games."""
  env = hearts.HeartsEnv(**options)
  game_str = "hearts" + (
      "(" + ",".join(f"{k}={str(v).lower()}" for k, v in options.items()) + ")"
      if options
      else ""
  )
  games = PYSPIEL_GAMES if not options else max(PYSPIEL_GAMES // 4, 5)
  nodes, _ = _lockstep(env, game_str, games, seed=len(options))
  assert nodes >= games * 52
  print(f"hearts vs pyspiel {game_str}: {games} games, {nodes} nodes equal")


def test_hide_hidden_cards_moves_only_others_unplaced_cards(env):
  """Hide hidden cards moves only others unplaced cards."""
  s = env.init(jax.random.key(3))
  for t in range(20):  # into the play phase
    a = int(np.flatnonzero(np.asarray(env.legal_mask(s)))[0])
    s = env.step(s, a, jax.random.key(100 + t))
  h = hearts.hide_hidden_cards(env, s, 0, jax.random.key(7))
  for q in range(4):
    assert (np.asarray(h.holder) == q).sum() == (
        np.asarray(s.holder) == q
    ).sum()
  np.testing.assert_array_equal(
      np.asarray(h.holder) == 0, np.asarray(s.holder) == 0
  )
  assert not np.array_equal(np.asarray(h.holder), np.asarray(s.holder))


def test_visible_action_hides_a_pass_from_the_others(env):
  """Visible action hides a pass from the others."""
  k = 5
  s = env.init(jax.random.key(k))
  while int(s.phase) != 2:  # a hand with a pass (3 in 4 are)
    k += 1
    assert k < 50
    s = env.init(jax.random.key(k))
  a = int(np.flatnonzero(np.asarray(env.legal_mask(s)))[0])
  p = int(env.actor(s))
  assert int(env.visible_action(s, a, p)) == a
  assert int(env.visible_action(s, a, (p + 1) % 4)) == -1


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider", "-s"]
      )
  )
