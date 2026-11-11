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

"""Serving hearts: a four-seat game through the checkpoint seat.

Hearts is the first served game with more than two players. pyspiel's hearts
at any of its eight game parameters is played by the env of the same
parameters (a pyspiel history is the env's raw history); an artifact on
disk (a freshly initialised network, loaded through the public loader)
plays every seat through checkpoint_bot, the GUI's "checkpoint" controller
(build_bot, what the GUI's plugin calls) and the Remote bot, every move legal
in pyspiel; and each seat's move depends on its own view only: re-dealing
cards between two other seats, so that the seat to move has the same
information state in pyspiel, leaves its move and distribution unchanged.
"""

import sys

import jax
import numpy as np
import pytest

from open_spiel.python.scale.envs import registry
from open_spiel.python.scale.serve import checkpoint_bot as checkpoint_bot_lib
from open_spiel.python.scale.serve import controller
from open_spiel.python.scale.serve import games
from open_spiel.python.scale.serve import remote_bot
from open_spiel.python.scale.serve import replay
from open_spiel.python.scale.serve import server as server_lib
from open_spiel.python.scale.serve import wire
from open_spiel.python.scale.serve.tests import checkpoint_bot_helpers
from open_spiel.python.scale.serve.tests import fresh_artifact
from open_spiel.python.scale.serve.tests import remote_bot_helpers
import pyspiel

GAME = "hearts"
ENCODER = "hearts_info_state"
NUM_PLAYERS = 4
STEP = 1
# History index of the first dealt card (index 0 is the pass direction); the
# i-th card dealt goes to seat i % 4.
FIRST_DEAL = 1
NUM_CARDS = 52


@pytest.fixture(name="artifact_dir", scope="module")
def _artifact_dir(tmp_path_factory):
  """A freshly initialised hearts network, written as artifact path@STEP.

  Args:
    tmp_path_factory: pytest's.

  Returns:
    The artifact folder.
  """
  env = registry.make(GAME)
  return fresh_artifact.write(
      tmp_path_factory.mktemp("hearts_art"), env, ENCODER, step=STEP, seed=0
  )


@pytest.fixture(name="bot", scope="module")
def _bot(artifact_dir):
  """The artifact as a greedy CheckpointBot, built as the GUI's seat builds it.

  controller.build_bot: loads, checks the game and warms up every seat.

  Args:
    artifact_dir: the artifact folder.

  Returns:
    The CheckpointBot.
  """
  return controller.build_bot(
      pyspiel.load_game(GAME),
      {"checkpoint": f"{artifact_dir}@{STEP}", "deterministic": True},
      seed=0,
      server=server_lib.PolicyServer(),
  )


def _sample_chance(state, rng):
  """A chance outcome of `state`, sampled.

  Args:
    state: a pyspiel chance state.
    rng: a numpy Generator.

  Returns:
    The action.
  """
  actions, probs = zip(*state.chance_outcomes())
  probs = np.asarray(probs, np.float64)
  return int(rng.choice(actions, p=probs / probs.sum()))


def _random_game(game, seed):
  """Every state of one uniformly random pyspiel game, in order.

  Args:
    game: the pyspiel game.
    seed: the seed.

  Returns:
    The states, from the root to the terminal one.
  """
  rng = np.random.default_rng(seed)
  s = game.new_initial_state()
  out = [s.clone()]
  while not s.is_terminal():
    if s.is_chance_node():
      s.apply_action(_sample_chance(s, rng))
    else:
      s.apply_action(int(rng.choice(s.legal_actions())))
    out.append(s.clone())
  return out


def test_hearts_is_served_at_every_game_parameter():
  """Hearts is served at every game parameter; other parameters refused."""
  assert GAME in games.served_games()
  assert games.env_spec_for(GAME, {}) == registry.spec(GAME)
  game = pyspiel.load_game("hearts(jd_bonus=true,pass_cards=false)")
  assert games.env_spec_for(GAME, wire.game_params(game)) == registry.spec(
      GAME, jd_bonus=True, pass_cards=False
  )
  # every parameter of pyspiel's hearts is one the env plays
  spec = dict(pyspiel.load_game(GAME).get_type().parameter_specification)
  assert set(spec) == set(registry.game(GAME).serving.pyspiel_params)
  env = games.env_for(GAME, {"avoid_all_tricks_bonus": True})
  assert (
      env.num_players == NUM_PLAYERS and env.options["avoid_all_tricks_bonus"]
  )
  with pytest.raises(ValueError, match="not implemented"):
    games.check_params(GAME, {"players": 4})
  assert controller.refusal(pyspiel.load_game(GAME)) is None


@pytest.mark.parametrize("game_string", ["hearts", "hearts(jd_bonus=true)"])
def test_pyspiel_history_is_the_raw_history_and_replays(game_string):
  """A pyspiel history is the env's raw history; it replays to pyspiel's node.

  Checked at nodes of random games (the pass phase, the play phase and the
  terminal), with the terminal returns.

  Args:
    game_string: the pyspiel game.
  """
  game = pyspiel.load_game(game_string)
  env = games.env_for(GAME, wire.game_params(game))
  checked = 0
  for g in range(2):
    states = [s for s in _random_game(game, seed=g) if not s.is_chance_node()]
    for state in states[::9] + [states[-1]]:
      hist = [int(a) for a in state.history()]
      hands = wire.pyspiel_history_to_raw(GAME, hist, wire.game_params(game))
      assert hands == [hist]
      _, _, final = replay.replay_hand(env, hist)
      wire.check_replayed_node(env, final, state)  # raises on a mismatch
      checked += 1
    np.testing.assert_allclose(
        np.asarray(env.returns(final)), np.asarray(states[-1].returns()), rtol=0
    )
  assert checked >= 16


def test_checkpoint_bot_plays_every_seat(bot):
  """The artifact plays all four seats of pyspiel games, every move legal.

  Greedy and stateless: the same deal is played the same way twice; no move
  passes through a rules difference.

  Args:
    bot: the CheckpointBot.
  """
  assert bot.num_players == NUM_PLAYERS
  assert bot.game_string == GAME
  game = pyspiel.load_game(GAME)
  first = checkpoint_bot_helpers.self_play(bot, game, deals=[0, 1])
  for hist in first:
    s = game.new_initial_state()
    for a in hist:
      s.apply_action(a)
    assert s.is_terminal()
    assert sum(s.returns()) in (78.0, 26.0)  # 26 after a moon
  seats_moved = set()

  def choose(state, seat):
    """The bot's move, recording the seat.

    Args:
      state: the state.
      seat: the bot's seat.

    Returns:
      The action.
    """
    seats_moved.add(seat)
    a = bot.choose_action(state, seat)
    assert bot.last_notice is None, bot.last_notice
    assert set(bot.last_probs) <= set(state.legal_actions())
    np.testing.assert_allclose(sum(bot.last_probs.values()), 1.0, atol=1e-9)
    return a

  again = checkpoint_bot_helpers.self_play(
      bot, game, deals=[0, 1], choose=choose
  )
  assert again == first
  assert seats_moved == set(range(NUM_PLAYERS))
  assert bot.stats == {"legal_mismatch": 0, "diverged": 0}


def test_one_seat_against_random_seats(bot):
  """The artifact in one seat, uniform random moves in the other three."""
  game = pyspiel.load_game(GAME)
  for seat in range(NUM_PLAYERS):
    rng = np.random.default_rng(10 + seat)
    s = game.new_initial_state()
    moves = 0
    while not s.is_terminal():
      if s.is_chance_node():
        s.apply_action(_sample_chance(s, rng))
      elif s.current_player() == seat:
        a = bot.choose_action(s, seat)
        assert a in s.legal_actions(seat)
        s.apply_action(a)
        moves += 1
      else:
        with pytest.raises(ValueError, match="not to move"):
          bot.choose_action(s, seat)
        s.apply_action(int(rng.choice(s.legal_actions())))
    assert moves in (13, 16)  # 13 plays, and 3 passes unless "no pass"


def _deal_swaps(history, mover, rng, k):
  """Up to k histories with two cards swapped between two seats != mover.

  Each swaps the cards of two deal entries dealt to two different seats,
  neither of them `mover`.

  Args:
    history: a pyspiel hearts history past the deal.
    mover: the seat to move.
    rng: a numpy Generator.
    k: the most histories.

  Returns:
    The histories.
  """
  others = [q for q in range(NUM_PLAYERS) if q != mover]
  out = []
  for _ in range(k):
    j, l = rng.choice(others, size=2, replace=False)
    ij = FIRST_DEAL + int(rng.choice(np.arange(j, NUM_CARDS, NUM_PLAYERS)))
    il = FIRST_DEAL + int(rng.choice(np.arange(l, NUM_CARDS, NUM_PLAYERS)))
    h = list(history)
    h[ij], h[il] = h[il], h[ij]
    out.append(h)
  return out


def _state_of(game, history):
  """The pyspiel state after `history`, or None when it is not legal.

  Args:
    game: the pyspiel game.
    history: the actions.

  Returns:
    The state, or None.
  """
  s = game.new_initial_state()
  for a in history:
    if s.is_chance_node():
      if a not in dict(s.chance_outcomes()):
        return None
    elif a not in s.legal_actions():
      return None
    s.apply_action(a)
  return s


def test_each_seat_sees_only_its_own_view(bot):
  """Re-dealing other seats' cards leaves the seat's move unchanged.

  At decisions of random games (passes and plays, every seat), cards are
  swapped between two other seats in the deal; when the swapped history is
  legal in pyspiel, the seat to move has the same information state there
  and another seat does not, the bot's move and distribution must be equal.
  Swapping one of the mover's own cards changes its view, and its
  distribution changes at least once: the check is not vacuous.

  Args:
    bot: the CheckpointBot.
  """
  game = pyspiel.load_game(GAME)
  rng = np.random.default_rng(7)
  pairs, own_changed, seats = 0, False, set()
  for g in range(3):
    states = [
        s
        for s in _random_game(game, seed=100 + g)
        if not s.is_chance_node() and not s.is_terminal()
    ]
    for state in states[::5]:
      seat = state.current_player()
      hist = [int(a) for a in state.history()]
      action = bot.choose_action(state, seat)
      probs = dict(bot.last_probs)
      view = state.information_state_string(seat)
      for h in _deal_swaps(hist, seat, rng, k=3):
        alt = _state_of(game, h)
        if alt is None or alt.current_player() != seat:
          continue
        assert alt.information_state_string(seat) == view
        if all(
            alt.information_state_string(q) == state.information_state_string(q)
            for q in range(NUM_PLAYERS)
            if q != seat
        ):
          continue
        assert bot.choose_action(alt, seat) == action, (hist, h)
        assert bot.last_probs.keys() == probs.keys()
        np.testing.assert_allclose(
            [bot.last_probs[a] for a in sorted(probs)],
            [probs[a] for a in sorted(probs)],
            rtol=0,
            atol=0,
        )
        pairs += 1
        seats.add(seat)
      # one of the seat's own dealt cards swapped with another seat's
      if not own_changed:
        other = (seat + 1) % NUM_PLAYERS
        h = list(hist)
        ia = FIRST_DEAL + seat + NUM_PLAYERS * int(rng.integers(13))
        ib = FIRST_DEAL + other + NUM_PLAYERS * int(rng.integers(13))
        h[ia], h[ib] = h[ib], h[ia]
        alt = _state_of(game, h)
        if alt is not None and alt.current_player() == seat:
          assert alt.information_state_string(seat) != view
          bot.choose_action(alt, seat)
          own_changed = bot.last_probs != probs
  assert pairs >= 20 and seats == set(range(NUM_PLAYERS)), (pairs, seats)
  assert own_changed


def test_remote_bot_answers_every_seat(artifact_dir):
  """The Remote bot (full_state) answers each seat with a legal move."""
  srv = remote_bot.RemoteCheckpointBot(f"{artifact_dir}@{STEP}", warmup=False)
  assert srv.games == (GAME,)
  game = pyspiel.load_game(GAME)
  answered = set()
  for state in _random_game(game, seed=21)[::3]:
    if state.is_chance_node() or state.is_terminal():
      continue
    seat = state.current_player()
    reply = srv.step(
        remote_bot.FULL_STATE, remote_bot_helpers.request_of(game, state, seat)
    )
    assert reply["action"] in state.legal_actions(seat)
    assert reply["notice"] is None
    answered.add(seat)
  assert answered == set(range(NUM_PLAYERS))


def test_a_variant_the_artifact_was_not_trained_on_is_refused(bot):
  """An artifact trained on default hearts refuses hearts(jd_bonus=true)."""
  with pytest.raises(ValueError, match="cover"):
    bot.check_serves(pyspiel.load_game("hearts(jd_bonus=true)"))
  # and still plays the default game afterwards
  bot.check_serves(pyspiel.load_game(GAME))


def test_artifact_manifest_records_what_serving_checks(artifact_dir):
  """The artifact's manifest: game, env spec, rules identity, encoder."""
  art = server_lib.PolicyServer().artifact(artifact_dir, STEP)
  m = art.manifest
  env = registry.make(GAME)
  assert m.game == GAME
  assert m.rules_sha == registry.rules_sha(env) == env.declared_rules_sha
  assert registry.EnvSpec.from_json(m.env_spec) == registry.spec(GAME)
  assert m.obs_format == ENCODER
  assert checkpoint_bot_lib.game_string_of(m) == GAME
  assert jax.tree_util.tree_leaves(art.params)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
