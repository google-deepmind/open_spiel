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

"""Tests for the euchre adapter."""

from collections.abc import Iterator, Mapping, Sequence
import dataclasses
import json
import random
import re

from absl.testing import absltest

import pyspiel

from open_spiel.python.gui import adapter
from open_spiel.python.gui import contract
from open_spiel.python.gui import controllers
from open_spiel.python.gui import presenters
from open_spiel.python.gui import session as session_lib
from open_spiel.python.gui.adapters import euchre_adapter

_EUCHRE = pyspiel.euchre
_PHASE = _EUCHRE.Phase

_ORACLE = adapter.Viewpoint(seat=None, omniscient=True)
_SPECTATOR = adapter.Viewpoint(seat=None, omniscient=False)
_SEATS = tuple(
    adapter.Viewpoint(seat=seat, omniscient=False) for seat in range(4)
)
_VIEWERS = (*_SEATS, _SPECTATOR)
_COMPASS = ("N", "E", "S", "W")

# The struct's fields, as the euchre board reads them.
_STRUCT_FIELDS = {
    "hands",
    "upcard",
    "trumpSuit",
    "dealer",
    "phase",
    "currentTrick",
    "lastTrick",
    "tricksWon",
    "declarer",
    "goingAlone",
    "currentPlayer",
    "discarding",
}

# A card code anywhere in a JSON text.
_CARD = re.compile(r'"([9TJQKA][cdhs])"')

# Hands played to the end; enough to reach every phase and a lone hand.
_HANDS = 40


def _code(card: int) -> str:
  """The client's code of a card: "Js"."""
  return "9TJQKA"[card // 4] + "cdhs"[card % 4]


def _history(seed: int, **parameters: bool) -> list[int]:
  """The actions of a seeded random hand, chance outcomes included."""
  game = pyspiel.load_game("euchre", parameters)
  state = game.new_initial_state()
  rng = random.Random(seed)
  while not state.is_terminal():
    state.apply_action(rng.choice(state.legal_actions()))
  return list(state.history())


def _replay(history: Sequence[int], **parameters: bool) -> pyspiel.State:
  """The state after `history`; raises if an action is illegal there."""
  state = pyspiel.load_game("euchre", parameters).new_initial_state()
  for action in history:
    if action not in state.legal_actions():
      raise ValueError(f"{action} is illegal after {state.history()}.")
    state.apply_action(action)
  return state


def _positions(seed: int, **parameters: bool) -> Iterator[pyspiel.State]:
  """Every position of a seeded random hand, from the deal to its end."""
  history = _history(seed, **parameters)
  state = pyspiel.load_game("euchre", parameters).new_initial_state()
  yield state.clone()
  for action in history:
    state.apply_action(action)
    yield state.clone()


def _all_positions() -> Iterator[pyspiel.State]:
  """Every position of the seeded hands, with and without lone defenders."""
  for seed in range(_HANDS):
    yield from _positions(seed)
  for seed in range(_HANDS // 4):
    yield from _positions(seed, allow_lone_defender=True)


def _view(
    state: pyspiel.State, viewpoint: adapter.Viewpoint
) -> Mapping[str, contract.JsonValue]:
  """The view through the presenter, with the actions the viewer is given."""
  acting = state.current_player()
  given = (
      list(state.legal_actions())
      if acting >= 0
      and (viewpoint.omniscient or viewpoint.seat == acting)
      and not state.is_terminal()
      else None
  )
  presentation = presenters.present("euchre", state, viewpoint, given)
  assert presentation is not None
  view = presentation.view
  assert isinstance(view, dict)
  hints = presentation.actions
  return {
      **view,
      "hints": (
          None if hints is None else [dataclasses.asdict(h) for h in hints]
      ),
  }


def _struct(
    state: pyspiel.State, viewpoint: adapter.Viewpoint
) -> Mapping[str, contract.JsonValue]:
  """The view's struct."""
  struct = euchre_adapter.ADAPTER.present(state, viewpoint)["struct"]
  assert isinstance(struct, dict)
  return struct


def _hands(state: pyspiel.State) -> dict[str, list[str]]:
  """Every hand as the game holds it, by compass point, in card order."""
  hands: dict[str, list[str]] = {compass: [] for compass in _COMPASS}
  for card, holder in enumerate(state.card_holder()):
    if holder is not None and holder >= 0:
      hands[_COMPASS[holder]].append(_code(card))
  return hands


def _public_cards(history: Sequence[int], upto: int) -> set[int]:
  """The cards every viewer has seen in the first `upto` actions.

  The up-card, and every card played to a trick; the dealer's discard is
  face down, so it is not public.
  """
  state = pyspiel.load_game("euchre").new_initial_state()
  public: set[int] = set()
  for action in history[:upto]:
    if state.current_phase() == _PHASE.PLAY and not state.is_chance_node():
      public.add(action)
    state.apply_action(action)
    if state.upcard() >= 0:
      public.add(state.upcard())
  return public


def _own_cards(history: Sequence[int], seat: int) -> set[int]:
  """The cards dealt to `seat`, and the up-card if `seat` dealt and took it."""
  state = pyspiel.load_game("euchre").new_initial_state()
  for action in history:
    if not state.is_chance_node():
      break
    state.apply_action(action)
  return {
      card for card, holder in enumerate(state.card_holder()) if holder == seat
  }


def _permuted(
    history: Sequence[int],
    upto: int,
    seat: int | None,
    rng: random.Random,
) -> list[int] | None:
  """`history[:upto]` with the cards hidden from `seat` shuffled among them.

  The cards the viewer may know (its own, the up-card and every card played
  so far) stay where they are; every other card, dealt to another seat or
  left undealt, is mapped to another of them, and the deal and the dealer's
  discard follow the mapping. None when the shuffled deal makes a later move
  illegal (a seat that could not follow suit now can).
  """
  known = _public_cards(history, upto)
  if seat is not None:
    known |= _own_cards(history, seat)
  hidden = sorted(set(range(_EUCHRE.NUM_CARDS)) - known)
  shuffled = list(hidden)
  rng.shuffle(shuffled)
  mapping = dict(zip(hidden, shuffled))
  state = pyspiel.load_game("euchre").new_initial_state()
  out: list[int] = []
  for action in history[:upto]:
    moves_card = (
        state.is_chance_node() and state.current_phase() == _PHASE.DEAL
    ) or (
        not state.is_chance_node() and state.current_phase() == _PHASE.DISCARD
    )
    new = mapping.get(action, action) if moves_card else action
    if new not in state.legal_actions():
      return None
    state.apply_action(new)
    out.append(new)
  return out


class EuchreAdapterTest(absltest.TestCase):

  def test_is_registered_for_its_renderer(self) -> None:
    registered = presenters.adapter_for(euchre_adapter.GAME)
    self.assertIs(registered, euchre_adapter.ADAPTER)
    assert registered is not None
    self.assertEqual(registered.renderer, "euchre")

  def test_every_view_is_json_through_the_presenter(self) -> None:
    for state in _positions(0):
      for viewpoint in (_ORACLE, *_VIEWERS):
        json.dumps(_view(state, viewpoint))

  def test_the_oracle_sees_every_hand_as_the_game_holds_it(self) -> None:
    for state in _positions(1):
      struct = _struct(state, _ORACLE)
      self.assertEqual(set(struct), _STRUCT_FIELDS)
      self.assertEqual(struct["hands"], _hands(state))

  def test_a_seat_sees_its_hand_and_only_the_size_of_the_others(self) -> None:
    for state in _all_positions():
      hands = _hands(state)
      for viewpoint in _VIEWERS:
        shown = _struct(state, viewpoint)["hands"]
        assert isinstance(shown, dict)
        for seat, compass in enumerate(_COMPASS):
          if viewpoint.seat == seat:
            self.assertEqual(shown[compass], hands[compass])
          else:
            self.assertEqual(shown[compass], [None] * len(hands[compass]))

  def test_no_viewer_is_sent_a_card_it_may_not_know(self) -> None:
    for seed in range(_HANDS):
      history = _history(seed)
      for upto in range(len(history) + 1):
        state = _replay(history[:upto])
        public = {_code(card) for card in _public_cards(history, upto)}
        for viewpoint in _VIEWERS:
          known = set(public)
          if viewpoint.seat is not None:
            known |= set(_hands(state)[_COMPASS[viewpoint.seat]])
            known |= {_code(c) for c in _own_cards(history, viewpoint.seat)}
          text = json.dumps(_view(state, viewpoint))
          self.assertLessEqual(set(_CARD.findall(text)), known, f"{upto}")

  def test_the_view_does_not_change_when_hidden_cards_are_shuffled(
      self,
  ) -> None:
    """Permutation invariance: two deals a viewer cannot tell apart.

    The cards hidden from a viewer are shuffled among themselves, the moves
    replayed, and the viewer's view (struct, seat to act, its action hints)
    must not change, at every position of every hand. A public card drawn
    in a place a hidden card decides, a hand sorted with hidden cards in it,
    or a count that depends on them would change it.
    """
    rng = random.Random(0)
    compared = 0
    for seed in range(_HANDS):
      history = _history(seed)
      for upto in range(len(history) + 1):
        state = _replay(history[:upto])
        for viewpoint in _VIEWERS:
          for _ in range(3):
            permuted = _permuted(history, upto, viewpoint.seat, rng)
            if permuted is None:
              continue
            twin = _replay(permuted)
            self.assertEqual(
                _view(twin, viewpoint),
                _view(state, viewpoint),
                f"seed {seed}, after {upto} actions, {viewpoint}",
            )
            compared += 1
    self.assertGreater(compared, 5000)

  def test_the_dealers_hand_holding_the_up_card_stays_face_down(self) -> None:
    checked = 0
    for state in _all_positions():
      if state.is_terminal() or state.current_phase() != _PHASE.DISCARD:
        continue
      dealer = _COMPASS[state.dealer()]
      upcard = _code(state.upcard())
      for viewpoint in _VIEWERS:
        if viewpoint.seat == state.dealer():
          continue
        shown = _struct(state, viewpoint)["hands"]
        assert isinstance(shown, dict)
        self.assertEqual(shown[dealer], [None] * 6)
        self.assertNotIn(upcard, json.dumps(shown))
        checked += 1
    self.assertGreater(checked, 0)

  def test_the_phases_follow_the_hand(self) -> None:
    seen = set()
    for state in _all_positions():
      struct = _struct(state, _ORACLE)
      seen.add(struct["phase"])
      if state.is_terminal():
        self.assertEqual(struct["phase"], "terminal")
      elif state.is_chance_node():
        self.assertEqual(struct["phase"], "dealing")
      elif state.current_phase() == _PHASE.DISCARD:
        self.assertEqual(struct["phase"], "playing")
        self.assertTrue(struct["discarding"])
      if struct["phase"] in ("bidding1", "bidding2"):
        suits = [
            a
            for a in state.legal_actions()
            if euchre_adapter.action_type(a) == "call_suit"
        ]
        self.assertEqual(struct["phase"] == "bidding1", len(suits) == 1)
    self.assertEqual(
        seen,
        {"dealing", "bidding1", "bidding2", "alone", "playing", "terminal"},
    )

  def test_tricks_won_and_the_last_trick_agree(self) -> None:
    for state in _all_positions():
      struct = _struct(state, _ORACLE)
      won = struct["tricksWon"]
      assert isinstance(won, dict)
      last = struct["lastTrick"]
      total = int(str(won["NS"])) + int(str(won["EW"]))
      if last is None:
        self.assertEqual(total, 0)
      else:
        assert isinstance(last, dict)
        self.assertEqual(last["index"], total)
      if state.is_terminal() and state.declarer() >= 0:
        self.assertEqual(total, 5)

  def test_the_actions_carry_their_card_kind_and_suit(self) -> None:
    for state in _positions(2):
      if state.is_terminal() or state.is_chance_node():
        continue
      seat = state.current_player()
      hints = euchre_adapter.ADAPTER.describe_actions(
          state, state.legal_actions(), _SEATS[seat]
      )
      for action, hint in hints.items():
        assert isinstance(hint, dict)
        self.assertEqual(
            set(hint), {"cardCode", "actionType", "calledSuit", "displayLabel"}
        )
        if action < _EUCHRE.NUM_CARDS:
          self.assertEqual(hint["cardCode"], _code(action))
          self.assertEqual(hint["actionType"], "card")
        else:
          self.assertIsNone(hint["cardCode"])
        if hint["actionType"] == "call_suit":
          self.assertEqual(
              hint["calledSuit"],
              "cdhs"[action - _EUCHRE.CLUBS_TRUMP_ACTION],
          )

  def test_history_labels_hide_the_deal_and_the_discard(self) -> None:
    oracle = contract.Viewer(contract.ViewerKind.ORACLE)
    viewers = (
        *(contract.Viewer(contract.ViewerKind.SEAT, s) for s in range(4)),
        contract.Viewer(contract.ViewerKind.SPECTATOR),
    )
    random_spec = controllers.Spec(controllers.RANDOM)
    discards = 0
    for seed in range(8):
      game = session_lib.Session.create(
          "euchre",
          {},
          seed=seed,
          chance_policy=contract.ChancePolicy.AUTO,
          controllers=(random_spec,) * 4,
          autoplay=(True,) * 4,
      )
      while not game.is_terminal:
        game.advance(oracle)
      full = game.frame(oracle).history
      for viewer in viewers:
        for truth, entry in zip(full, game.frame(viewer).history, strict=True):
          if not entry.withheld:
            self.assertEqual(entry.seat, viewer.seat)
            continue
          if entry.seat is None:
            self.assertIsNone(entry.public_label)
          elif entry.public_label == euchre_adapter.DISCARD_HIDDEN_LABEL:
            discards += 1
            self.assertNotIn(truth.label, entry.public_label)
          else:
            self.assertEqual(entry.public_label, truth.label)
    self.assertGreater(discards, 0)


if __name__ == "__main__":
  absltest.main()
