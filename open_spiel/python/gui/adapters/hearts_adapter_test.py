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

"""Tests for the hearts adapter: the struct per viewer, its privacy, the hints
and the public labels."""

from collections.abc import Iterator
import json
import random
import re
from typing import Any

from absl.testing import absltest

import pyspiel

from open_spiel.python.gui import adapter
from open_spiel.python.gui import contract
from open_spiel.python.gui import controllers
from open_spiel.python.gui import presenters
from open_spiel.python.gui import session as session_lib
from open_spiel.python.gui.adapters import hearts_adapter

_ORACLE = adapter.Viewpoint(seat=None, omniscient=True)
_SPECTATOR = adapter.Viewpoint(seat=None, omniscient=False)
_SEATS = [adapter.Viewpoint(seat=seat, omniscient=False) for seat in range(4)]
_CODE = re.compile(r'"([2-9TJQKA][cdhs])"')
_COMPASS = "NESW"
_NO_PASS = 0
_PASS_LEFT = 1
_HANDS = 6

# The struct's fields, as the board's state hook reads them.
_STRUCT_FIELDS = {
    "hands",
    "currentTrick",
    "completedTricks",
    "lastTrick",
    "points",
    "passDirection",
    "phase",
    "currentPlayer",
}


def _view(state: pyspiel.State, viewpoint: adapter.Viewpoint) -> Any:
  """The adapter's view for `viewpoint`, untyped so the tests can index it."""
  return hearts_adapter.ADAPTER.present(state, viewpoint)


def _dealt(direction: int, seed: int) -> pyspiel.State:
  """A hand dealt at random, after the pass direction `direction`."""
  rng = random.Random(seed)
  state = pyspiel.load_game("hearts").new_initial_state()
  state.apply_action(direction)
  while state.is_chance_node():
    state.apply_action(rng.choice(state.legal_actions()))
  return state


def _positions(seed: int) -> Iterator[pyspiel.State]:
  """Every position of a hand played at random, the deal included."""
  rng = random.Random(seed)
  state = pyspiel.load_game("hearts").new_initial_state()
  while True:
    yield state
    if state.is_terminal():
      return
    state = state.child(rng.choice(state.legal_actions()))


def _first_play(state: pyspiel.State) -> int:
  """The history index of the first card played to a trick."""
  return 1 + 52 + (0 if state.history()[0] == _NO_PASS else 12)


def _played(state: pyspiel.State) -> set[str]:
  """The cards played to tricks, which are public."""
  return {
      hearts_adapter.action_to_card_code(action)
      for action in state.history()[_first_play(state) :]
  }


def _hand(state: pyspiel.State, seat: int) -> set[str]:
  """`seat`'s hand, as the oracle's struct holds it."""
  return set(_view(state, _ORACLE)["struct"]["hands"][_COMPASS[seat]])


def _passed_by(state: pyspiel.State, seat: int) -> set[str]:
  """The cards `seat` has passed so far (its own knowledge)."""
  history = state.history()
  if not history or history[0] == _NO_PASS:
    return set()
  passes = history[1 + 52 : _first_play(state)]
  return {
      hearts_adapter.action_to_card_code(a)
      for i, a in enumerate(passes)
      if i // 3 == seat
  }


class HeartsAdapterTest(absltest.TestCase):

  def test_registered(self) -> None:
    self.assertIs(presenters.adapter_for("hearts"), hearts_adapter.ADAPTER)
    self.assertEqual(hearts_adapter.ADAPTER.renderer, "hearts")

  def test_the_view_is_the_boards_struct_and_json(self) -> None:
    state = _dealt(_PASS_LEFT, seed=1)
    for viewpoint in (_ORACLE, _SPECTATOR, *_SEATS):
      view = _view(state, viewpoint)
      self.assertEqual(set(view), {"struct"})
      self.assertEqual(set(view["struct"]), _STRUCT_FIELDS)
      self.assertEqual(json.loads(json.dumps(view)), view)

  def test_the_oracle_sees_every_hand(self) -> None:
    struct = _view(_dealt(_PASS_LEFT, seed=2), _ORACLE)["struct"]
    cards = [c for hand in struct["hands"].values() for c in hand]
    self.assertEqual(len(cards), 52)
    self.assertEqual(len(set(cards)), 52)
    self.assertEqual(struct["phase"], "passing")
    self.assertEqual(struct["passDirection"], "left")

  def test_a_seat_sees_its_own_hand_and_backs_for_the_others(self) -> None:
    state = _dealt(_PASS_LEFT, seed=3)
    for seat, viewpoint in enumerate(_SEATS):
      hands = _view(state, viewpoint)["struct"]["hands"]
      for other, compass in enumerate(_COMPASS):
        if other == seat:
          self.assertEqual(set(hands[compass]), _hand(state, seat))
        else:
          self.assertEqual(hands[compass], [None] * 13)

  def test_a_spectator_sees_no_hand(self) -> None:
    hands = _view(_dealt(_NO_PASS, seed=4), _SPECTATOR)["struct"]["hands"]
    self.assertEqual(hands, {c: [None] * 13 for c in _COMPASS})

  def test_no_viewer_gets_a_card_it_may_not_see(self) -> None:
    for seed in range(_HANDS):
      for state in _positions(seed):
        if state.is_chance_node() and state.history():
          continue
        played = _played(state) if state.history() else set()
        for seat, viewpoint in enumerate(_SEATS):
          shown = set(_CODE.findall(json.dumps(_view(state, viewpoint))))
          allowed = played | _hand(state, seat) | _passed_by(state, seat)
          self.assertLessEqual(shown, allowed, f"seed {seed} seat {seat}")
        shown = set(_CODE.findall(json.dumps(_view(state, _SPECTATOR))))
        self.assertLessEqual(shown, played, f"seed {seed} spectator")

  def test_the_trick_in_play_names_its_players(self) -> None:
    rng = random.Random(7)
    state = _dealt(_NO_PASS, seed=7)
    state.apply_action(rng.choice(state.legal_actions()))
    state.apply_action(rng.choice(state.legal_actions()))
    trick = _view(state, _SPECTATOR)["struct"]["currentTrick"]
    self.assertEqual(len(trick), 2)
    for entry in trick:
      self.assertEqual(entry["position"], _COMPASS[entry["player"]])
    self.assertEqual(
        {entry["card"] for entry in trick}, _played(state), "the cards played"
    )

  def test_the_last_trick_has_a_winner(self) -> None:
    for state in _positions(8):
      struct = _view(state, _SPECTATOR)["struct"]
      last = struct["lastTrick"]
      if struct["completedTricks"] == 0:
        self.assertIsNone(last)
        continue
      self.assertEqual(last["index"], struct["completedTricks"])
      self.assertEqual(len(last["cards"]), 4)
      led = last["cards"][0]["card"][1]
      best = max(
          (c for c in last["cards"] if c["card"][1] == led),
          key=lambda c: hearts_adapter.RANK_IDX[c["card"][0]],
      )
      self.assertEqual(last["winner"], best["position"])

  def test_every_action_is_hinted_with_its_card(self) -> None:
    state = _dealt(_PASS_LEFT, seed=9)
    hints = hearts_adapter.ADAPTER.describe_actions(
        state, state.legal_actions(), _SEATS[state.current_player()]
    )
    for action in state.legal_actions():
      label = state.action_to_string(action)
      self.assertEqual(
          hints[action],
          {"cardCode": hearts_adapter.action_label_to_card_code(label)},
      )

  def test_plays_are_public_and_passes_and_the_deal_are_not(self) -> None:
    rng = random.Random(10)
    state = pyspiel.load_game("hearts").new_initial_state()
    state.apply_action(_PASS_LEFT)
    while not state.is_terminal():
      action = rng.choice(state.legal_actions())
      label = hearts_adapter.ADAPTER.public_action_label(state, action)
      index = len(state.history())
      if state.is_chance_node():
        self.assertIsNone(label)
      elif index < _first_play(state):
        self.assertEqual(label, hearts_adapter.PASS_HIDDEN_LABEL)
      else:
        self.assertEqual(label, state.action_to_string(action))
      state.apply_action(action)

  def test_a_frame_holds_no_hidden_card_for_any_viewer(self) -> None:
    oracle = contract.Viewer(contract.ViewerKind.ORACLE)
    viewers = [
        contract.Viewer(contract.ViewerKind.SEAT, seat) for seat in range(4)
    ]
    viewers.append(contract.Viewer(contract.ViewerKind.SPECTATOR))
    random_spec = controllers.Spec(controllers.RANDOM)
    for seed in range(2):
      game = session_lib.Session.create(
          "hearts",
          {},
          seed=seed,
          chance_policy=contract.ChancePolicy.AUTO,
          controllers=(random_spec,) * 4,
          autoplay=(True,) * 4,
      )
      while not game.is_terminal:
        game.advance(oracle)
        for viewer in viewers:
          frame = game.frame(viewer)
          if frame.presentation is None:
            self.fail("hearts frames carry a presentation")
          view: Any = frame.presentation.view
          for compass, cards in view["struct"]["hands"].items():
            if viewer.seat != _COMPASS.index(compass):
              self.assertLessEqual(set(cards), {None}, f"{viewer} {compass}")


if __name__ == "__main__":
  absltest.main()
