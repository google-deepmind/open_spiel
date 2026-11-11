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

"""Serving euchre: a four-seat partnership game through the checkpoint seat.

pyspiel's euchre at either value of its two game parameters is played by the
env of the same parameters (a pyspiel history is the env's raw history) when
the installed pyspiel is the corrected engine (its euchre tensor has 959
entries); released OpenSpiel's euchre (935 entries, other rules) is refused.
An artifact on disk (a freshly initialised network, loaded through the
public loader) plays every seat through checkpoint_bot, the GUI's
"checkpoint" controller (build_bot, what the GUI's plugin calls) and the
Remote bot, every move legal in pyspiel; and each seat's move depends on its
own view only: re-dealing cards between two other seats, so that the seat to
move has the same information state in pyspiel, leaves its move and
distribution unchanged.

Every test but the refusal needs the corrected pyspiel and skips without it
(for example when running against an older pre-built pyspiel release); run them
on a pyspiel built from
OpenSpiel's master branch, which has the corrected engine.
"""

import sys

import jax
import numpy as np
import pytest

from open_spiel.python.scale.envs import registry
from open_spiel.python.scale.serve import build_check
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

GAME = "euchre"
ENCODER = "euchre_info_state"
NUM_PLAYERS = 4
STEP = 1
# History index 0 is the dealer; entries FIRST_DEAL .. FIRST_DEAL + 19 are the
# dealt cards, the i-th to seat (dealer + i) % 4; the next is the up-card.
FIRST_DEAL = 1
NUM_DEALT = 20

CORRECTED = list(pyspiel.load_game(GAME).information_state_tensor_shape()) == [
    registry.EUCHRE_CORRECTED_TENSOR
]
needs_corrected = pytest.mark.skipif(
    not CORRECTED,
    reason="needs a pyspiel with the corrected euchre (959-entry tensor)",
)


@pytest.fixture(name="artifact_dir", scope="module")
def artifact_dir_fixture(tmp_path_factory):
  """A freshly initialised euchre network, written as artifact path@STEP.

  Args:
    tmp_path_factory: pytest's.

  Returns:
    The artifact folder.
  """
  env = registry.make(GAME)
  return fresh_artifact.write(
      tmp_path_factory.mktemp("euchre_art"), env, ENCODER, step=STEP, seed=0
  )


@pytest.fixture(name="bot", scope="module")
def bot_fixture(artifact_dir):
  """The artifact as a greedy CheckpointBot, built as the GUI's seat builds it.

  Args:
    artifact_dir: the artifact folder.

  Returns:
    The CheckpointBot.
  """
  if not CORRECTED:
    pytest.skip("needs a pyspiel with the corrected euchre (959-entry tensor)")
  return controller.build_bot(
      pyspiel.load_game(GAME),
      {"checkpoint": f"{artifact_dir}@{STEP}", "deterministic": True},
      seed=0,
      server=server_lib.PolicyServer(),
  )


def _sample_chance(state, rng):
  """A chance outcome of `state`, sampled.

  Args:
    state: a chance node.
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
    seed: the game's seed.

  Returns:
    The states.
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


class _ReleasedEuchre:
  """The installed euchre, reporting released OpenSpiel's 935-entry tensor."""

  def __init__(self, game):
    self._game = game

  def information_state_tensor_shape(self):
    """The released engine's shape.

    Returns:
      [935].
    """
    return [935]

  def __getattr__(self, name):
    return getattr(self._game, name)


class _TensorLike959:
  """The installed euchre, reporting the corrected engine's 959 entries."""

  def __init__(self, game):
    self._game = game

  def information_state_tensor_shape(self):
    """The corrected engine's shape.

    Returns:
      [959].
    """
    return [registry.EUCHRE_CORRECTED_TENSOR]

  def __getattr__(self, name):
    return getattr(self._game, name)


def test_the_rules_probe_tells_the_engines_apart():
  """The build check reads the rules, not only the tensor size.

  With the tensor size faked to 959, the released engine is still refused
  (the dealer takes up the up-card before the go-alone decision) and the
  corrected engine is accepted.
  """
  problem = build_check.pyspiel_game_problem(
      _TensorLike959(pyspiel.load_game(GAME))
  )
  if CORRECTED:
    assert problem is None, problem
  else:
    assert "takes up the up-card first" in problem, problem


def test_a_released_pyspiel_euchre_is_refused():
  """A euchre build whose tensor is not 959 entries is refused, by name.

  On the installed pyspiel: refused when it is released OpenSpiel, accepted
  when it is the corrected engine; a stand-in reporting 935 is refused
  either way, by the GUI's refusal and by the serving check.
  """
  real = pyspiel.load_game(GAME)
  problem = build_check.pyspiel_game_problem(real)
  assert (problem is None) == CORRECTED, problem
  released = _ReleasedEuchre(real)
  assert "corrected" in build_check.pyspiel_game_problem(released)
  with pytest.raises(ValueError, match="corrected"):
    build_check.check_pyspiel_game(released)
  assert "corrected" in controller.refusal(released)
  # the name and parameters alone are still served (game strings, specs)
  assert games.env_spec_for(GAME, {}) == registry.spec(GAME)
  # another game is not affected
  assert build_check.pyspiel_game_problem(pyspiel.load_game("hearts")) is None


@needs_corrected
def test_euchre_is_served_at_every_game_parameter():
  """Euchre is served at both values of its parameters; others refused."""
  assert GAME in games.served_games()
  assert games.env_spec_for(GAME, {}) == registry.spec(GAME)
  game = pyspiel.load_game(
      "euchre(allow_lone_defender=true,stick_the_dealer=false)"
  )
  assert games.env_spec_for(GAME, wire.game_params(game)) == registry.spec(
      GAME, allow_lone_defender=True, stick_the_dealer=False
  )
  spec = dict(pyspiel.load_game(GAME).get_type().parameter_specification)
  assert set(spec) == set(registry.game(GAME).serving.pyspiel_params)
  with pytest.raises(ValueError, match="not implemented"):
    games.check_params(GAME, {"players": 4})
  assert controller.refusal(pyspiel.load_game(GAME)) is None


@needs_corrected
@pytest.mark.parametrize(
    "game_string",
    ["euchre", "euchre(allow_lone_defender=true,stick_the_dealer=false)"],
)
def test_pyspiel_history_is_the_raw_history_and_replays(game_string):
  """A pyspiel history is the env's raw history; it replays to pyspiel's node.

  Args:
    game_string: the pyspiel game.
  """
  game = pyspiel.load_game(game_string)
  env = games.env_for(GAME, wire.game_params(game))
  checked = 0
  for g in range(4):
    states = [s for s in _random_game(game, seed=g) if not s.is_chance_node()]
    for state in states[::5] + [states[-1]]:
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


@needs_corrected
def test_checkpoint_bot_plays_every_seat(bot):
  """The artifact plays all four seats of pyspiel games, every move legal.

  Args:
    bot: the CheckpointBot.
  """
  assert bot.num_players == NUM_PLAYERS
  assert bot.game_string == GAME
  game = pyspiel.load_game(GAME)
  first = checkpoint_bot_helpers.self_play(bot, game, deals=[0, 1, 2, 3])
  for hist in first:
    s = game.new_initial_state()
    for a in hist:
      s.apply_action(a)
    assert s.is_terminal()
    assert sum(s.returns()) == 0.0
  seats_moved = set()

  def choose(state, seat):
    """The bot's move, recording the seat.

    Args:
      state: the pyspiel state.
      seat: the seat to move.

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
      bot, game, deals=[0, 1, 2, 3], choose=choose
  )
  assert again == first
  assert seats_moved == set(range(NUM_PLAYERS))
  assert bot.stats == {"legal_mismatch": 0, "diverged": 0}


@needs_corrected
def test_one_seat_against_random_seats(bot):
  """The artifact in one seat, uniform random moves in the other three."""
  game = pyspiel.load_game(GAME)
  for seat in range(NUM_PLAYERS):
    for g in range(3):
      rng = np.random.default_rng(10 + 4 * g + seat)
      s = game.new_initial_state()
      while not s.is_terminal():
        if s.is_chance_node():
          s.apply_action(_sample_chance(s, rng))
        elif s.current_player() == seat:
          a = bot.choose_action(s, seat)
          assert a in s.legal_actions(seat)
          s.apply_action(a)
        else:
          with pytest.raises(ValueError, match="not to move"):
            bot.choose_action(s, seat)
          s.apply_action(int(rng.choice(s.legal_actions())))


def _deal_swaps(history, mover, rng, k):
  """Up to k histories with two dealt cards swapped between two seats != mover.

  Args:
    history: a pyspiel euchre history past the deal.
    mover: the seat to move.
    rng: a numpy Generator.
    k: the most histories.

  Returns:
    The histories.
  """
  dealer = history[0]
  holder = [(dealer + i) % NUM_PLAYERS for i in range(NUM_DEALT)]
  others = [q for q in range(NUM_PLAYERS) if q != mover]
  out = []
  for _ in range(k):
    j, l = rng.choice(others, size=2, replace=False)
    ij = FIRST_DEAL + int(
        rng.choice([i for i, q in enumerate(holder) if q == j])
    )
    il = FIRST_DEAL + int(
        rng.choice([i for i, q in enumerate(holder) if q == l])
    )
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


def _twins(game, decide, seed0=100, games_n=8):
  """Twin decisions of random games: (seat, probs, twin probs, own_changed).

  At decisions of random games (bids, go-alone, discards and plays, every
  seat), dealt cards are swapped between two other seats; a swapped history
  that is legal in pyspiel, leaves the mover's information state unchanged
  and changes another seat's is a twin, decided by `decide(state, seat) ->
  {action: probability}`. Separately, one of the mover's own cards is
  swapped with another seat's, once, and whether its distribution changed is
  recorded (the control).

  Args:
    game: the pyspiel game.
    decide: the decider.
    seed0: the first game seed.
    games_n: the games.

  Returns:
    ([(seat, probs, twin probs), ...], own_changed).
  """
  rng = np.random.default_rng(7)
  out, own_changed = [], False
  for g in range(games_n):
    states = [
        s
        for s in _random_game(game, seed=seed0 + g)
        if not s.is_chance_node() and not s.is_terminal()
    ]
    for state in states[::2]:
      seat = state.current_player()
      hist = [int(a) for a in state.history()]
      probs = decide(state, seat)
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
        out.append((seat, probs, decide(alt, seat)))
      if not own_changed:
        dealer = hist[0]
        mine = [
            i for i in range(NUM_DEALT) if (dealer + i) % NUM_PLAYERS == seat
        ]
        theirs = [
            i for i in range(NUM_DEALT) if (dealer + i) % NUM_PLAYERS != seat
        ]
        h = list(hist)
        ia = FIRST_DEAL + int(rng.choice(mine))
        ib = FIRST_DEAL + int(rng.choice(theirs))
        h[ia], h[ib] = h[ib], h[ia]
        alt = _state_of(game, h)
        if alt is not None and alt.current_player() == seat:
          assert alt.information_state_string(seat) != view
          own_changed = decide(alt, seat) != probs
  return out, own_changed


def _bot_decider(bot):
  """The bot as a decider: its distribution over the legal actions.

  Args:
    bot: the CheckpointBot.

  Returns:
    decide(state, seat) -> {action: probability}.
  """

  def decide(state, seat):
    """The bot's distribution at `state` for `seat`.

    Args:
      state: the pyspiel state.
      seat: the seat to move.

    Returns:
      {action: probability}.
    """
    bot.choose_action(state, seat)
    return dict(bot.last_probs)

  return decide


@needs_corrected
def test_each_seat_sees_only_its_own_view(bot):
  """Re-dealing other seats' cards leaves the seat's move unchanged.

  Every twin (_twins) has the same distribution, exactly, so the same
  greedy move; the own-card control changes it at least once, so the check
  is not vacuous.

  Args:
    bot: the CheckpointBot.
  """
  twins, own_changed = _twins(pyspiel.load_game(GAME), _bot_decider(bot))
  seats = {seat for seat, _, _ in twins}
  assert len(twins) >= 20 and seats == set(range(NUM_PLAYERS)), (
      len(twins),
      seats,
  )
  for seat, probs, alt in twins:
    assert alt.keys() == probs.keys(), seat
    np.testing.assert_allclose(
        [alt[a] for a in sorted(probs)],
        [probs[a] for a in sorted(probs)],
        rtol=0,
        atol=0,
    )
  assert own_changed


@needs_corrected
def test_the_twin_check_catches_a_decider_that_sees_other_hands(bot):
  """Negative control: a decider that sees other hands differs on a twin.

  Its distribution also depends on the other seats' cards (the full state
  string), so the twin check above would fail for a leaking seat.

  Args:
    bot: the CheckpointBot.
  """
  honest = _bot_decider(bot)

  def leaky(state, seat):
    """The honest distribution, bumped by a hash of the whole deal.

    Args:
      state: the pyspiel state.
      seat: the seat to move.

    Returns:
      {action: probability}.
    """
    probs = honest(state, seat)
    bump = (hash(str(state)) % 1000 + 1) / 1e5  # the whole deal, not the view
    a = min(probs)
    total = sum(probs.values()) + bump
    return {
        k: (v + (bump if k == a else 0.0)) / total for k, v in probs.items()
    }

  twins, _ = _twins(pyspiel.load_game(GAME), leaky, games_n=3)
  assert twins
  assert any(alt != probs for _, probs, alt in twins)


@needs_corrected
def test_remote_bot_answers_every_seat(artifact_dir):
  """The Remote bot (full_state) answers each seat with a legal move."""
  srv = remote_bot.RemoteCheckpointBot(f"{artifact_dir}@{STEP}", warmup=False)
  assert srv.games == (GAME,)
  game = pyspiel.load_game(GAME)
  answered = set()
  for g in range(6):
    for state in _random_game(game, seed=21 + g)[::2]:
      if state.is_chance_node() or state.is_terminal():
        continue
      seat = state.current_player()
      reply = srv.step(
          remote_bot.FULL_STATE,
          remote_bot_helpers.request_of(game, state, seat),
      )
      assert reply["action"] in state.legal_actions(seat)
      assert reply["notice"] is None
      answered.add(seat)
  assert answered == set(range(NUM_PLAYERS))


@needs_corrected
def test_the_gui_seat_refuses_a_released_build(bot):
  """CheckpointBot.check_serves (the GUI seat) refuses the 935 stand-in."""
  with pytest.raises(ValueError, match="corrected"):
    bot.check_serves(_ReleasedEuchre(pyspiel.load_game(GAME)))
  bot.check_serves(pyspiel.load_game(GAME))


@needs_corrected
def test_a_variant_the_artifact_was_not_trained_on_is_refused(bot):
  """An artifact trained on default euchre refuses another parametrisation."""
  with pytest.raises(ValueError, match="cover"):
    bot.check_serves(pyspiel.load_game("euchre(stick_the_dealer=false)"))
  bot.check_serves(pyspiel.load_game(GAME))


@needs_corrected
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
