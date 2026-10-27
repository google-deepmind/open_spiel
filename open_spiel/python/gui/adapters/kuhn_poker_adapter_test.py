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


"""Tests for the kuhn_poker adapter."""

from collections.abc import Iterator, Mapping
import json
import random

from absl.testing import absltest

import pyspiel

from open_spiel.python.gui import adapter
from open_spiel.python.gui.adapters import kuhn_poker_adapter
from open_spiel.python.gui.adapters import poker_table

_ADAPTER = kuhn_poker_adapter.ADAPTER
_ORACLE = adapter.Viewpoint(seat=None, omniscient=True)
_SPECTATOR = adapter.Viewpoint(seat=None, omniscient=False)
_PASS = 0
_BET = 1
# Cards are numbered by rank: jack, queen, king in the two-seat game.
_JACK = 0
_QUEEN = 1
_KING = 2
# The random hands each property test plays.
_HANDS = 200
_GAMES = ("kuhn_poker", "kuhn_poker(players=3)")
_HIDDEN = poker_table.HIDDEN_DEAL


def _seat(seat: int) -> adapter.Viewpoint:
  """The viewpoint of seat `seat`."""
  return adapter.Viewpoint(seat=seat, omniscient=False)


def _hand(*actions: int, game: str = "kuhn_poker") -> pyspiel.State:
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
    view = _view(_hand(_KING, _JACK), _seat(0))
    self.assertEqual(
        view["struct"],
        {
            "board_cards": "",
            "player_hands": ["Ks", ""],
            "player_contributions": [1, 1],
            "starting_stacks": [],
            "pot_size": 2,
            "current_player": 0,
            "betting_history": "",
            "blinds": [],
            "ante": 1,
            # Kuhn has no hand categories.
            "best_hand_rank_types": ["", ""],
            "best_five_card_hands": ["Ks", ""],
        },
    )
    self.assertEqual(view["historyActionStrings"], [_HIDDEN, _HIDDEN])
    self.assertEqual(
        view["pokerGame"],
        {
            "title": "Kuhn Poker",
            "betting": "limit",
            "holeCards": 1,
            "boardSizes": [0],
        },
    )

  def test_names_the_actions_in_the_boards_words(self) -> None:
    opening = _hand(_KING, _JACK)
    self.assertEqual(
        _ADAPTER.describe_actions(opening, [_PASS, _BET], _seat(0)),
        {
            _PASS: {"label": "player=0 move=Call"},
            _BET: {"label": "player=0 move=Bet2"},
        },
    )
    facing = _hand(_KING, _JACK, _PASS, _BET)
    self.assertEqual(
        _ADAPTER.describe_actions(facing, [_PASS, _BET], _seat(0)),
        {
            _PASS: {"label": "player=0 move=Fold"},
            _BET: {"label": "player=0 move=Call"},
        },
    )
    view = _view(facing, _SPECTATOR)
    self.assertEqual(
        view["historyActionStrings"],
        [_HIDDEN, _HIDDEN, "player=0 move=Call", "player=1 move=Bet2"],
    )
    self.assertEqual(_struct(facing, _SPECTATOR)["betting_history"], "cr2")

  def test_the_oracle_reads_every_deal(self) -> None:
    view = _view(_hand(_KING, _JACK, _BET), _ORACLE)
    self.assertEqual(
        view["historyActionStrings"],
        [
            "player=-1 move=Deal Ks",
            "player=-1 move=Deal Js",
            "player=0 move=Bet2",
        ],
    )

  def test_a_showdown_tables_both_cards(self) -> None:
    state = _hand(_KING, _JACK, _BET, _BET)
    struct = _struct(state, _SPECTATOR)
    self.assertEqual(struct["player_hands"], ["Ks", "Js"])
    self.assertEqual(struct["current_player"], poker_table.TERMINAL_PLAYER)
    self.assertEqual(struct["pot_size"], 4)
    self.assertEqual(struct["betting_history"], "r2c")

  def test_a_fold_shows_no_cards(self) -> None:
    state = _hand(_QUEEN, _KING, _BET, _PASS)
    self.assertEqual(_struct(state, _SPECTATOR)["player_hands"], ["", ""])
    self.assertEqual(_struct(state, _seat(1))["player_hands"], ["", "Ks"])
    self.assertEqual(_struct(state, _seat(1))["betting_history"], "r2f")

  def test_public_labels_name_the_betting_only(self) -> None:
    self.assertIsNone(_ADAPTER.public_action_label(_hand(), _KING))
    self.assertEqual(
        _ADAPTER.public_action_label(_hand(_KING, _JACK), _BET), "Bet"
    )

  def test_three_seats_use_four_ranks(self) -> None:
    state = _hand(3, 0, 1, game="kuhn_poker(players=3)")
    self.assertEqual(
        _struct(state, _ORACLE)["player_hands"], ["As", "Js", "Qs"]
    )


class PrivacyTest(absltest.TestCase):

  def test_no_viewer_sees_a_hidden_card_of_another_seat(self) -> None:
    for game in _GAMES:
      for state in _random_hands(game, seed=7):
        dealt = _struct(state, _ORACLE)["player_hands"]
        assert isinstance(dealt, list)
        tabled = poker_table.showdown_seats(_struct(state, _ORACLE))
        viewers = [_SPECTATOR] + [
            _seat(seat) for seat in range(state.num_players())
        ]
        for viewpoint in viewers:
          seen = _view(state, viewpoint)
          hands = _struct(state, viewpoint)["player_hands"]
          assert isinstance(hands, list)
          text = json.dumps(seen)
          for other, card in enumerate(dealt):
            if not card or other == viewpoint.seat:
              continue
            if other in tabled:
              self.assertEqual(hands[other], card)
              continue
            self.assertEqual(hands[other], "")
            self.assertNotIn(card, text)

  def test_a_seat_sees_its_own_card_and_the_oracle_every_card(self) -> None:
    for state in _random_hands("kuhn_poker(players=3)", seed=5):
      dealt = _struct(state, _ORACLE)["player_hands"]
      assert isinstance(dealt, list)
      self.assertEqual(
          dealt, [str(card) for card in dealt]
      )  # Every entry is a code, or "" before its deal.
      for seat in range(state.num_players()):
        hands = _struct(state, _seat(seat))["player_hands"]
        assert isinstance(hands, list)
        self.assertEqual(hands[seat], dealt[seat])


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
        pot = sum(spent)
        self.assertEqual(struct["pot_size"], pot)
        winners = [seat for seat, value in enumerate(returns) if value > 0]
        self.assertEqual(len(winners), 1)
        for seat, paid in enumerate(spent):
          won = pot if seat in winners else 0
          self.assertAlmostEqual(returns[seat], won - paid)


if __name__ == "__main__":
  absltest.main()
