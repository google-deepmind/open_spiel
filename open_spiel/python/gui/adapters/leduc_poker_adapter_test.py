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


"""Tests for the leduc_poker adapter."""

from collections.abc import Iterator, Mapping
import json
import random

from absl.testing import absltest

import pyspiel

from open_spiel.python.gui import adapter
from open_spiel.python.gui.adapters import leduc_poker_adapter
from open_spiel.python.gui.adapters import poker_table

_ADAPTER = leduc_poker_adapter.ADAPTER
_ORACLE = adapter.Viewpoint(seat=None, omniscient=True)
_SPECTATOR = adapter.Viewpoint(seat=None, omniscient=False)
# Cards: the jack, queen and king of spades are 0, 2 and 4, of hearts 1, 3
# and 5, in the two-seat game.
_JACK_OF_SPADES = 0
_JACK_OF_HEARTS = 1
_KING_OF_HEARTS = 5
_FOLD = 0
_CALL = 1
_RAISE = 2
# The random hands each property test plays.
_HANDS = 200
_GAMES = ("leduc_poker", "leduc_poker(players=3)")
_HIDDEN = poker_table.HIDDEN_DEAL
# A hand to the public card: seat 0 bets, seat 1 raises, seat 0 calls.
_TURNED = (
    _KING_OF_HEARTS,
    _JACK_OF_SPADES,
    _RAISE,
    _RAISE,
    _CALL,
    _JACK_OF_HEARTS,
)


def _seat(seat: int) -> adapter.Viewpoint:
  """The viewpoint of seat `seat`."""
  return adapter.Viewpoint(seat=seat, omniscient=False)


def _hand(*actions: int, game: str = "leduc_poker") -> pyspiel.State:
  """The state after `actions`, deals included, from the start."""
  state = pyspiel.load_game(game).new_initial_state()
  for action in actions:
    state.apply_action(action)
  return state


def _view(
    state: pyspiel.State, viewpoint: adapter.Viewpoint
) -> Mapping[str, object]:
  """The view of `state` for `viewpoint`, as the client decodes it."""
  view: Mapping[str, object] = json.loads(
      json.dumps(_ADAPTER.present(state, viewpoint))
  )
  return view


def _struct(
    state: pyspiel.State, viewpoint: adapter.Viewpoint
) -> Mapping[str, object]:
  """The struct of `state`'s view for `viewpoint`."""
  struct = _view(state, viewpoint)["struct"]
  assert isinstance(struct, dict)
  return struct


def _random_hands(game: str, seed: int) -> Iterator[pyspiel.State]:
  """Yields every state of `_HANDS` seeded random hands of `game`."""
  rng = random.Random(seed)
  loaded = pyspiel.load_game(game)
  for _ in range(_HANDS):
    state = loaded.new_initial_state()
    while True:
      yield state
      if state.is_terminal():
        break
      if state.is_chance_node():
        outcomes, weights = zip(*state.chance_outcomes())
        state = state.child(rng.choices(outcomes, weights)[0])
      else:
        state = state.child(rng.choice(state.legal_actions()))


class ViewTest(absltest.TestCase):

  def test_the_first_decision(self) -> None:
    view = _view(_hand(_KING_OF_HEARTS, _JACK_OF_SPADES), _seat(1))
    struct = view["struct"]
    assert isinstance(struct, dict)
    self.assertEqual(struct["player_hands"], ["", "Js"])
    self.assertEqual(struct["player_contributions"], [1, 1])
    self.assertEqual(struct["current_player"], 0)
    self.assertEqual(struct["ante"], 1)
    self.assertEqual(struct["best_hand_rank_types"], ["", "High Card"])
    self.assertEqual(
        view["pokerGame"],
        {
            "title": "Leduc Hold'em",
            "betting": "limit",
            "holeCards": 1,
            "boardSizes": [0, 1],
        },
    )

  def test_names_each_round_in_the_boards_words(self) -> None:
    raised = _hand(_KING_OF_HEARTS, _JACK_OF_SPADES, _RAISE, _RAISE)
    self.assertEqual(
        _ADAPTER.describe_actions(raised, [_FOLD, _CALL, _RAISE], _seat(0)),
        {
            _FOLD: {"label": "player=0 move=Fold"},
            _CALL: {"label": "player=0 move=Call"},
            _RAISE: {"label": "player=0 move=Bet7"},
        },
    )
    turned = _hand(*_TURNED)
    view = _view(turned, _SPECTATOR)
    self.assertEqual(
        view["historyActionStrings"],
        [
            _HIDDEN,
            _HIDDEN,
            "player=0 move=Bet3",
            "player=1 move=Bet5",
            "player=0 move=Call",
            "player=-1 move=Deal Jh",
        ],
    )
    struct = _struct(turned, _SPECTATOR)
    self.assertEqual(struct["board_cards"], "Jh")
    self.assertEqual(struct["betting_history"], "r3r5c/")
    self.assertEqual(struct["pot_size"], 10)
    self.assertEqual(
        _ADAPTER.describe_actions(turned, [_CALL, _RAISE], _seat(0)),
        {
            _CALL: {"label": "player=0 move=Call"},
            _RAISE: {"label": "player=0 move=Bet9"},
        },
    )

  def test_a_pair_with_the_public_card_is_ranked(self) -> None:
    struct = _struct(_hand(*_TURNED), _seat(1))
    self.assertEqual(struct["best_hand_rank_types"], ["", "Pair"])
    self.assertEqual(struct["best_five_card_hands"], ["", "JsJh"])

  def test_a_showdown_tables_both_cards(self) -> None:
    state = _hand(*_TURNED, _CALL, _CALL)
    struct = _struct(state, _SPECTATOR)
    self.assertEqual(struct["player_hands"], ["Kh", "Js"])
    self.assertEqual(struct["best_hand_rank_types"], ["High Card", "Pair"])
    self.assertEqual(struct["betting_history"], "r3r5c/cc")
    self.assertEqual(state.returns(), [-5.0, 5.0])

  def test_a_fold_shows_no_cards(self) -> None:
    state = _hand(_KING_OF_HEARTS, _JACK_OF_SPADES, _RAISE, _FOLD)
    self.assertEqual(_struct(state, _SPECTATOR)["player_hands"], ["", ""])
    self.assertEqual(_struct(state, _seat(0))["player_hands"], ["Kh", ""])
    self.assertEqual(_struct(state, _seat(0))["betting_history"], "r3f")

  def test_public_labels_name_the_betting_and_the_public_card(self) -> None:
    self.assertIsNone(_ADAPTER.public_action_label(_hand(), _KING_OF_HEARTS))
    self.assertEqual(
        _ADAPTER.public_action_label(_hand(*_TURNED[:2]), _RAISE), "Raise"
    )
    self.assertEqual(
        _ADAPTER.public_action_label(_hand(*_TURNED[:5]), _JACK_OF_HEARTS),
        "Public card Jh",
    )

  def test_refuses_parameters_it_does_not_read(self) -> None:
    state = _hand(game="leduc_poker(suit_isomorphism=true)")
    with self.assertRaisesRegex(ValueError, "suit_isomorphism"):
      _ADAPTER.present(state, _ORACLE)


class PrivacyTest(absltest.TestCase):

  def test_no_viewer_sees_a_hidden_card_of_another_seat(self) -> None:
    for game in _GAMES:
      for state in _random_hands(game, seed=7):
        oracle = _struct(state, _ORACLE)
        dealt = oracle["player_hands"]
        assert isinstance(dealt, list)
        tabled = poker_table.showdown_seats(oracle)
        viewers = [_SPECTATOR] + [
            _seat(seat) for seat in range(state.num_players())
        ]
        for viewpoint in viewers:
          seen = _view(state, viewpoint)
          struct = _struct(state, viewpoint)
          hands = struct["player_hands"]
          ranks = struct["best_hand_rank_types"]
          assert isinstance(hands, list) and isinstance(ranks, list)
          text = json.dumps(seen)
          for other, card in enumerate(dealt):
            if not card or other == viewpoint.seat:
              continue
            if other in tabled:
              self.assertEqual(hands[other], card)
              continue
            self.assertEqual(hands[other], "")
            self.assertEqual(ranks[other], "")
            self.assertNotIn(card, text)

  def test_the_public_card_is_public(self) -> None:
    for state in _random_hands("leduc_poker", seed=5):
      board = _struct(state, _ORACLE)["board_cards"]
      self.assertEqual(_struct(state, _SPECTATOR)["board_cards"], board)


class ChipsTest(absltest.TestCase):

  def test_the_contributions_settle_as_the_returns_say(self) -> None:
    for game in _GAMES:
      for state in _random_hands(game, seed=11):
        if not state.is_terminal():
          continue
        struct = _struct(state, _ORACLE)
        spent = struct["player_contributions"]
        assert isinstance(spent, list)
        returns = state.returns()
        self.assertAlmostEqual(sum(returns), 0)
        # A seat that loses loses all it put in; the winners share the pot.
        for seat, paid in enumerate(spent):
          if returns[seat] < 0:
            self.assertAlmostEqual(returns[seat], -paid)


if __name__ == "__main__":
  absltest.main()
