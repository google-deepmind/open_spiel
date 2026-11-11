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

# STRUCT-GLUE(euchre StateStruct): the board reads a struct-shaped view (the
# four hands by compass, the up-card, trump, the trick in play and the last
# one, the tricks each side has won), filtered per viewer, which the game's
# structs do not carry in this shape; the adapter builds it from the state's
# accessors.
"""The euchre adapter: the table for one viewer, as the euchre board reads it.

The view is the record the `euchre` client renderer reads:

- `struct`: `hands` (the compass points "N", "E", "S" and "W", each a list of
  card codes such as "Js"), `upcard`, `trumpSuit` ("c", "d", "h" or "s"),
  `dealer` and `declarer` (compass points), `phase` ("dealing", "bidding1",
  "bidding2", "alone", "playing" or "terminal"), `currentTrick` and
  `lastTrick` (each card with the seat that played it), `tricksWon` per side
  ("NS", "EW"), `goingAlone`, `currentPlayer` and `discarding` (the dealer
  is choosing its discard; the phase then reads "playing").
- `currentPosition`: the compass point of the seat to act, or None.

A hand the viewer may not see is one `None` per card: the number of cards a
seat holds is public, and nothing about which cards they are is sent, not
even where a public card sits among them. A seat sees its own hand and the
oracle sees every hand. Everything else in the view is public: the up-card,
each call, trump, who plays alone, the cards on the table and the tricks won.

Each legal action is described for the acting seat with `cardCode`,
`actionType` ("card", "pass", "call_suit", "alone" or "partner"),
`calledSuit` and `displayLabel`. In the history every call and card played is
public under its own label; the dealer's discard is face down, so another
viewer reads only that the dealer discarded, and the deal stays hidden.

Seats are numbered as the game numbers them: 0 North, 1 East, 2 South and
3 West, with North and South partners against East and West.
"""

from collections.abc import Mapping, Sequence
from typing import Protocol

import pyspiel

from open_spiel.python.gui import adapter
from open_spiel.python.gui import contract

GAME = "euchre"

Json = contract.JsonValue

_NUM_SEATS = 4
_NUM_CARDS = pyspiel.euchre.NUM_CARDS

# The game's card numbering: card = rank * 4 + suit.
_RANKS = ("9", "T", "J", "Q", "K", "A")
_SUITS = ("c", "d", "h", "s")
_SEATS = ("N", "E", "S", "W")

_PASS = pyspiel.euchre.PASS_ACTION
_CALL_CLUBS = pyspiel.euchre.CLUBS_TRUMP_ACTION
_GO_ALONE = pyspiel.euchre.GO_ALONE_ACTION
_WITH_PARTNER = pyspiel.euchre.PLAY_WITH_PARTNER_ACTION

_Phase = pyspiel.euchre.Phase

# What a viewer other than the dealer reads in place of the dealer's discard.
DISCARD_HIDDEN_LABEL = "Dealer discards (hidden)"


class _Trick(Protocol):
  """A trick as the game binds it: its cards, leader and winner."""

  def cards(self) -> list[int]:
    """The cards played, in order; -1 for a trick not yet led."""
    ...

  def leader(self) -> int:
    """The seat that led."""
    ...

  def winner(self) -> int:
    """The seat that wins the trick as it stands."""
    ...


def card_code(card: int) -> str:
  """The client's code of the card numbered `card`: rank then suit, "Ah"."""
  if not 0 <= card < _NUM_CARDS:
    raise ValueError(f"euchre: {card} is not a card.")
  return f"{_RANKS[card // 4]}{_SUITS[card % 4]}"


def _seat_compass(seat: int) -> str:
  """The compass point of `seat`: "N", "E", "S" or "W"."""
  if not 0 <= seat < _NUM_SEATS:
    raise ValueError(f"euchre: {seat} is not a seat.")
  return _SEATS[seat]


def action_type(action: int) -> str:
  """The kind of `action`: "card", "pass", "call_suit", "alone", "partner"."""
  if 0 <= action < _NUM_CARDS:
    return "card"
  if action == _PASS:
    return "pass"
  if _CALL_CLUBS <= action < _CALL_CLUBS + len(_SUITS):
    return "call_suit"
  if action == _GO_ALONE:
    return "alone"
  if action == _WITH_PARTNER:
    return "partner"
  raise ValueError(f"euchre: {action} is not an action.")


def _called_suit(action: int) -> str | None:
  """The suit a call names, or None for any other action."""
  if _CALL_CLUBS <= action < _CALL_CLUBS + len(_SUITS):
    return _SUITS[action - _CALL_CLUBS]
  return None


def _trick_cards(trick: _Trick, active: Sequence[bool]) -> list[Json]:
  """The cards of one trick, each with the seat that played it.

  Play goes clockwise from the leader and skips the partner who sits out a
  lone hand, as the game formats a trick. A trick not yet led holds the
  game's placeholder card -1 and has no cards.
  """
  cards = [card for card in trick.cards() if card >= 0]
  if not cards:
    return []
  player = trick.leader()
  if not 0 <= player < _NUM_SEATS or not active[player]:
    raise ValueError(f"euchre: a trick led by inactive seat {player}.")
  out: list[Json] = []
  for card in cards:
    while not active[player]:
      player = (player + 1) % _NUM_SEATS
    out.append({
        "card": card_code(card),
        "player": player,
        "position": _SEATS[player],
    })
    player = (player + 1) % _NUM_SEATS
  return out


def _finished_tricks(
    state: pyspiel.State,
) -> tuple[list[bool], int, list[tuple[_Trick, list[Json]]]]:
  """`(active, per_trick, finished)`: the finished tricks with their cards.

  The one place that decides which tricks are finished, so `lastTrick` and
  `tricksWon` cannot disagree.
  """
  active = [bool(a) for a in state.active_players()]
  per_trick = sum(1 for a in active if a)
  played = state.num_cards_played()
  finished: list[tuple[_Trick, list[Json]]] = []
  if played == 0:
    return active, per_trick, finished
  tricks: Sequence[_Trick] = state.tricks()
  for trick in tricks[: played // per_trick]:
    cards = _trick_cards(trick, active)
    if len(cards) != per_trick:
      raise ValueError(f"euchre: a finished trick has {len(cards)} cards.")
    if not 0 <= trick.winner() < _NUM_SEATS:
      raise ValueError(f"euchre: a finished trick has no winner: {cards}.")
    finished.append((trick, cards))
  return active, per_trick, finished


def _trick_fields(state: pyspiel.State) -> tuple[list[Json], Json]:
  """`(currentTrick, lastTrick)` from the game's tricks.

  `currentTrick` holds the cards of the unfinished trick (empty between
  tricks and before play); `lastTrick` is `{"index", "cards", "winner"}` for
  the last finished trick (`index` counts from 1, `winner` is a compass
  point), or None before the first. The game's current trick runs past its
  five tricks once the hand is over, so the tricks are counted from the
  cards played instead.
  """
  played = state.num_cards_played()
  if played == 0:
    return [], None
  active, per_trick, finished = _finished_tricks(state)
  current: list[Json] = []
  if played % per_trick:
    tricks: Sequence[_Trick] = state.tricks()
    current = _trick_cards(tricks[len(finished)], active)
    if len(current) != played % per_trick:
      raise ValueError(f"euchre: the trick in play has {len(current)} cards.")
  last: Json = None
  if finished:
    trick, cards = finished[-1]
    last = {
        "index": len(finished),
        "cards": cards,
        "winner": _SEATS[trick.winner()],
    }
  return current, last


def _tricks_won(state: pyspiel.State) -> dict[str, Json]:
  """The tricks each side has won so far: {"NS": n, "EW": m}."""
  won = {"NS": 0, "EW": 0}
  for trick, _ in _finished_tricks(state)[2]:
    won["NS" if trick.winner() in (0, 2) else "EW"] += 1
  return dict(won)


def _phase(state: pyspiel.State) -> str:
  """The stage of the hand as the board names it.

  The dealer's discard reads "playing": its legal actions are cards, and
  `discarding` says which it is.
  """
  if state.is_terminal():
    return "terminal"
  if state.is_chance_node():
    return "dealing"
  phase = state.current_phase()
  if phase == _Phase.BIDDING:
    return "bidding1" if state.num_passes() < _NUM_SEATS else "bidding2"
  if phase == _Phase.GO_ALONE:
    return "alone"
  if phase in (_Phase.DISCARD, _Phase.PLAY):
    return "playing"
  raise ValueError(f"euchre: no board phase for {phase}.")


def _hands(state: pyspiel.State) -> dict[str, list[int]]:
  """The cards each seat holds, by compass point, in card order."""
  hands: dict[str, list[int]] = {seat: [] for seat in _SEATS}
  for card, holder in enumerate(state.card_holder()):
    if holder is not None and holder >= 0:
      hands[_seat_compass(holder)].append(card)
  return hands


def _viewer_hands(
    state: pyspiel.State, viewpoint: adapter.Viewpoint
) -> dict[str, Json]:
  """The hands as the viewer may see them: another's is one None per card.

  A hidden hand holds no card at all, public or not: a card shown among
  hidden ones would sit where the hidden cards put it.
  """
  hands = _hands(state)
  shown: dict[str, Json] = {}
  for seat, compass in enumerate(_SEATS):
    cards = hands[compass]
    if viewpoint.sees_private(seat):
      shown[compass] = [card_code(card) for card in cards]
    else:
      shown[compass] = [None] * len(cards)
  return shown


def _struct(
    state: pyspiel.State, viewpoint: adapter.Viewpoint
) -> dict[str, Json]:
  """The board's struct of `state`, with the hands the viewer may see."""
  dealer = state.dealer()
  upcard = state.upcard()
  trump = state.trump_suit()
  declarer = state.declarer()
  current_trick, last_trick = _trick_fields(state)
  return {
      "hands": _viewer_hands(state, viewpoint),
      "upcard": card_code(upcard) if upcard >= 0 else None,
      "trumpSuit": _SUITS[trump] if trump >= 0 else None,
      # Before the dealer is drawn the board seats the deal at North.
      "dealer": _seat_compass(dealer) if dealer >= 0 else "N",
      "phase": _phase(state),
      "currentTrick": current_trick,
      "lastTrick": last_trick,
      "tricksWon": _tricks_won(state),
      "declarer": _seat_compass(declarer) if declarer >= 0 else None,
      "goingAlone": bool(state.declarer_go_alone()),
      "currentPlayer": int(state.current_player()),
      "discarding": (
          not state.is_terminal()
          and not state.is_chance_node()
          and state.current_phase() == _Phase.DISCARD
      ),
  }


class EuchreAdapter(adapter.Adapter):
  """Presents a euchre hand to one viewer as the board's struct."""

  renderer = "euchre"

  def present(
      self, state: pyspiel.State, viewpoint: adapter.Viewpoint
  ) -> dict[str, Json]:
    """The struct and the seat to act, holding only what the viewer may see."""
    if state.get_game().get_type().short_name != GAME:
      raise ValueError("The euchre adapter was given another game.")
    player = state.current_player()
    return {
        "struct": _struct(state, viewpoint),
        "currentPosition": (
            _seat_compass(player) if 0 <= player < _NUM_SEATS else None
        ),
    }

  def describe_actions(
      self,
      state: pyspiel.State,
      actions: Sequence[int],
      viewpoint: adapter.Viewpoint,
  ) -> Mapping[int, Json]:
    """Each action's card, kind and called suit, for the acting seat."""
    del viewpoint  # The actions are the acting seat's, whom it may see.
    player = state.current_player()
    hints: dict[int, Json] = {}
    for action in actions:
      kind = action_type(action)
      hints[action] = {
          "cardCode": card_code(action) if kind == "card" else None,
          "actionType": kind,
          "calledSuit": _called_suit(action),
          "displayLabel": str(state.action_to_string(player, action)),
      }
    return hints

  def public_action_label(
      self, state: pyspiel.State, action: int
  ) -> str | None:
    """A call or a card played under its own label; the discard face down.

    Every call, the lone-hand choice and every card played are public. The
    dealer's discard is face down: another viewer reads only that the dealer
    discarded. The deal, the up-card included, is a hidden chance outcome.
    """
    player = state.current_player()
    if player == pyspiel.PlayerId.CHANCE:
      return None
    if state.current_phase() == _Phase.DISCARD:
      return DISCARD_HIDDEN_LABEL
    return str(state.action_to_string(player, action))


ADAPTER = EuchreAdapter()
