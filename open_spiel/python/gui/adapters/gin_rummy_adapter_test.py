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

"""Tests for the gin_rummy adapter."""

from collections.abc import Callable, Iterator, Mapping
import random

from absl.testing import absltest

import pyspiel

from open_spiel.python.gui import adapter
from open_spiel.python.gui import contract
from open_spiel.python.gui import controllers
from open_spiel.python.gui import presenters
from open_spiel.python.gui import session as session_lib
from open_spiel.python.gui.adapters import gin_rummy_adapter

_GIN = pyspiel.gin_rummy
_UTILS = _GIN.GinRummyUtils(
    _GIN.DEFAULT_NUM_RANKS, _GIN.DEFAULT_NUM_SUITS, _GIN.DEFAULT_HAND_SIZE
)

_ORACLE = adapter.Viewpoint(seat=None, omniscient=True)
_SPECTATOR = adapter.Viewpoint(seat=None, omniscient=False)
_SEATS = (
    adapter.Viewpoint(seat=0, omniscient=False),
    adapter.Viewpoint(seat=1, omniscient=False),
)
_VIEWERS = (*_SEATS, _SPECTATOR)

# Hands played to the end with each policy; enough to reach every phase.
_HANDS = 12

# The struct's fields, as the engine's state struct names them.
_STRUCT_FIELDS = {
    "phase",
    "current_player",
    "knock_card",
    "upcard",
    "prev_upcard",
    "stock_size",
    "hands",
    "discard_pile",
    "deadwood",
    "knocked",
    "pass_on_first_upcard",
    "layed_melds",
    "layoffs",
    "finished_layoffs",
}

Policy = Callable[[pyspiel.State, random.Random], int]


def _code(card: int) -> str:
  """The code of the default deck's card numbered `card`."""
  return str(_UTILS.card_string(card))


def _greedy(state: pyspiel.State, rng: random.Random) -> int:
  """Knocks when it can, else discards to the least deadwood; lays first."""
  legal: list[int] = [int(action) for action in state.legal_actions()]
  phase = state.current_phase()
  if _GIN.KNOCK_ACTION in legal:
    return int(_GIN.KNOCK_ACTION)
  if phase == _GIN.DISCARD:
    hand = state.hands()[state.current_player()]
    return min(
        (a for a in legal if a < _GIN.DRAW_UPCARD_ACTION),
        key=lambda c: _UTILS.min_deadwood([x for x in hand if x != c]),
    )
  if phase in (_GIN.KNOCK, _GIN.LAYOFF):
    laying = [a for a in legal if a != _GIN.PASS_ACTION]
    return laying[0] if laying else int(_GIN.PASS_ACTION)
  if phase == _GIN.DRAW:
    return int(_GIN.DRAW_STOCK_ACTION)
  return rng.choice(legal)


def _positions(seed: int, policy: Policy = _greedy) -> Iterator[pyspiel.State]:
  """Every decision and the end of one seeded hand."""
  rng = random.Random(seed)
  state = pyspiel.load_game("gin_rummy").new_initial_state()
  while not state.is_terminal():
    if state.is_chance_node():
      state.apply_action(rng.choice([a for a, _ in state.chance_outcomes()]))
      continue
    yield state.clone()
    state.apply_action(policy(state, rng))
  yield state


def _steps(
    seed: int, policy: Policy = _greedy
) -> Iterator[tuple[pyspiel.State, int, str | None]]:
  """Every action and chance outcome of one seeded hand, with its state
  before the action and the action's public label."""
  rng = random.Random(seed)
  state = pyspiel.load_game("gin_rummy").new_initial_state()
  while not state.is_terminal():
    if state.is_chance_node():
      action = rng.choice([a for a, _ in state.chance_outcomes()])
    else:
      action = policy(state, rng)
    public = gin_rummy_adapter.ADAPTER.public_action_label(state, action)
    yield state.clone(), action, public
    state.apply_action(action)


def _all_positions() -> Iterator[pyspiel.State]:
  """The positions of seeded hands of the greedy and the random policy."""
  for seed in range(_HANDS):
    yield from _positions(seed)
    yield from _positions(
        1000 + seed, lambda s, rng: rng.choice(s.legal_actions())
    )


def _hidden_cards(state: pyspiel.State, seat: int) -> set[str]:
  """The cards in `seat`'s hand that no one else has seen."""
  known = set(state.known_cards()[seat])
  return {_code(card) for card in state.hands()[seat] if card not in known}


def _strings(value: object) -> set[str]:
  """Every string anywhere in a JSON value."""
  found: set[str] = set()
  if isinstance(value, str):
    found.add(value)
  elif isinstance(value, list):
    for item in value:
      found |= _strings(item)
  elif isinstance(value, dict):
    for item in value.values():
      found |= _strings(item)
  return found


def _dict(value: contract.JsonValue) -> Mapping[str, contract.JsonValue]:
  """`value`, checked to be an object."""
  assert isinstance(value, dict)
  return value


def _list(value: contract.JsonValue) -> list[contract.JsonValue]:
  """`value`, checked to be a list."""
  assert isinstance(value, list)
  return value


def _struct(
    state: pyspiel.State, viewpoint: adapter.Viewpoint
) -> Mapping[str, contract.JsonValue]:
  """The struct in the view of `state` for `viewpoint`."""
  view = gin_rummy_adapter.ADAPTER.present(state, viewpoint)
  return _dict(view["struct"])


def _first(phase: int) -> pyspiel.State:
  """The first position of the seeded hands in `phase`."""
  for state in _all_positions():
    if not state.is_terminal() and state.current_phase() == phase:
      return state
  raise AssertionError(f"No position in phase {phase} was reached.")


class GinRummyAdapterTest(absltest.TestCase):

  def test_is_registered_for_its_renderer(self) -> None:
    registered = presenters.adapter_for(gin_rummy_adapter.GAME)
    self.assertIs(registered, gin_rummy_adapter.ADAPTER)
    assert registered is not None
    self.assertEqual(registered.renderer, "gin_rummy")

  def test_the_view_is_json_through_the_presenter(self) -> None:
    for state in _positions(0):
      presentation = presenters.present(
          "gin_rummy", state, _ORACLE, None if state.is_terminal() else []
      )
      assert presentation is not None
      self.assertEqual(presentation.renderer, "gin_rummy")

  def test_the_oracle_sees_the_engine_struct(self) -> None:
    state = _first(_GIN.DISCARD)
    struct = _struct(state, _ORACLE)
    self.assertEqual(set(struct), _STRUCT_FIELDS)
    self.assertEqual(struct["phase"], "Discard")
    self.assertEqual(
        struct["current_player"], f"Player_{state.current_player()}"
    )
    self.assertEqual(struct["stock_size"], state.stock_size())
    self.assertEqual(struct["knock_card"], state.knock_card())
    for seat in range(2):
      self.assertEqual(
          _list(struct["hands"])[seat],
          [_code(c) for c in sorted(state.hands()[seat])],
      )
    self.assertEqual(
        struct["discard_pile"], [_code(c) for c in state.discard_pile()]
    )

  def test_a_seat_sees_its_hand_and_the_other_face_down(self) -> None:
    state = _first(_GIN.DRAW)
    for seat in range(2):
      struct = _struct(state, _SEATS[seat])
      hands = _list(struct["hands"])
      deadwood = _list(struct["deadwood"])
      self.assertEqual(
          hands[seat], [_code(c) for c in sorted(state.hands()[seat])]
      )
      self.assertEqual(hands[1 - seat], ["XX"] * len(state.hands()[1 - seat]))
      self.assertEqual(deadwood[1 - seat], -1)
      self.assertEqual(deadwood[seat], state.deadwood()[seat])

  def test_no_viewer_gets_a_card_another_seat_hides(self) -> None:
    checked = 0
    for state in _all_positions():
      open_phase = state.current_phase() in (_GIN.LAYOFF, _GIN.GAME_OVER)
      for viewpoint in _VIEWERS:
        sent = _strings(gin_rummy_adapter.ADAPTER.present(state, viewpoint))
        for seat in range(2):
          if seat == viewpoint.seat or open_phase or state.knocked()[seat]:
            continue
          leaked = _hidden_cards(state, seat) & sent
          self.assertEqual(
              leaked,
              set(),
              f"viewer {viewpoint} sees seat {seat}'s hidden cards in"
              f" phase {state.current_phase()}",
          )
          checked += 1
    self.assertGreater(checked, 500)

  def test_the_knock_lays_the_knockers_hand_open(self) -> None:
    state = _first(_GIN.KNOCK)
    knocker = state.current_player()
    view = gin_rummy_adapter.ADAPTER.present(state, _SEATS[1 - knocker])
    self.assertEqual(view["ginVisibleHands"], {str(knocker): True})
    self.assertFalse(view["ginRevealHands"])
    self.assertTrue(view["ginKnocked"])
    hands = _list(_dict(view["struct"])["hands"])
    self.assertEqual(
        hands[knocker], [_code(c) for c in sorted(state.hands()[knocker])]
    )
    knocker_view = gin_rummy_adapter.ADAPTER.present(state, _SEATS[knocker])
    defender = _list(_dict(knocker_view["struct"])["hands"])[1 - knocker]
    self.assertEqual(set(_list(defender)), {"XX"})

  def test_analysis_is_given_for_the_hands_a_viewer_sees(self) -> None:
    state = _first(_GIN.DISCARD)
    seat = state.current_player()
    view = gin_rummy_adapter.ADAPTER.present(state, _SEATS[seat])
    melds = _dict(view["ginAnalysisMeldsBySeat"])
    best = _dict(view["ginAnalysisBestIndicesBySeat"])
    self.assertLessEqual(set(melds), {str(seat)})
    self.assertEqual(set(melds), set(best))
    for key, indices in best.items():
      candidates = _list(melds[key])
      for index in _list(indices):
        assert isinstance(index, int)
        self.assertLess(index, len(candidates))
    spectator = gin_rummy_adapter.ADAPTER.present(state, _SPECTATOR)
    self.assertEqual(spectator["ginAnalysisMeldsBySeat"], {})

  def test_the_laying_seat_is_given_its_plans(self) -> None:
    state = _first(_GIN.KNOCK)
    seat = state.current_player()
    # The knocker discards, then lays melds until it may pass.
    while _GIN.PASS_ACTION not in state.legal_actions():
      state.apply_action(state.legal_actions()[0])
    own = gin_rummy_adapter.ADAPTER.present(state, _SEATS[seat])
    self.assertEqual(own["ginLaydownSeat"], seat)
    self.assertEqual(own["ginLaydownStage"], "melds")
    self.assertEqual(own["ginLaydownPassAction"], _GIN.PASS_ACTION)
    legal = state.legal_actions()
    for plan in _list(own.get("ginLaydownPlans", [])):
      first = _dict(_list(plan)[0])
      self.assertIn(first["action"], legal)
      for step in _list(plan):
        self.assertTrue(_list(_dict(step)["cards"]))
    other = gin_rummy_adapter.ADAPTER.present(state, _SEATS[1 - seat])
    self.assertEqual(other["ginLaydownStage"], "melds")
    self.assertNotIn("ginLaydownPlans", other)
    self.assertNotIn("ginLaydownPassAction", other)

  def test_the_knockers_discard_is_its_own_stage(self) -> None:
    state = _first(_GIN.KNOCK)
    view = gin_rummy_adapter.ADAPTER.present(state, _ORACLE)
    self.assertEqual(view["ginLaydownStage"], "discard")
    self.assertNotIn("ginLaydownPassAction", view)

  def test_every_hand_is_open_at_the_end(self) -> None:
    for state in _all_positions():
      if not state.is_terminal():
        continue
      view = gin_rummy_adapter.ADAPTER.present(state, _SPECTATOR)
      self.assertTrue(view["ginRevealHands"])
      self.assertEqual(view["ginVisibleHands"], {"0": True, "1": True})
      self.assertNotIn("XX", _strings(view["struct"]))
      return
    self.fail("No hand reached its end.")

  def test_known_cards_are_the_upcards_taken_and_kept(self) -> None:
    for state in _all_positions():
      # Laid melds leave the hand but stay known, so check before a knock.
      if state.is_terminal() or state.current_phase() not in (
          _GIN.DRAW,
          _GIN.DISCARD,
      ):
        continue
      view = gin_rummy_adapter.ADAPTER.present(state, _SPECTATOR)
      known = _dict(view["knownCardsBySeat"])
      for seat in range(2):
        hand = {_code(c) for c in state.hands()[seat]}
        for card in _list(known.get(str(seat), [])):
          self.assertIn(card, hand)

  def test_every_seat_action_is_public_with_its_own_label(self) -> None:
    for seed in range(_HANDS):
      for state, action, public in _steps(seed):
        if state.is_chance_node():
          continue
        self.assertEqual(
            public, state.action_to_string(state.current_player(), action)
        )

  def test_of_the_chance_outcomes_only_the_first_upcard_is_public(
      self,
  ) -> None:
    for seed in range(_HANDS):
      public_outcomes = []
      for state, action, public in _steps(seed):
        if not state.is_chance_node():
          continue
        if public is not None:
          public_outcomes.append((state, action, public))
      self.assertEqual(len(public_outcomes), 1)
      state, action, public = public_outcomes[0]
      self.assertEqual(public, state.action_to_string(-1, action))
      state.apply_action(action)
      self.assertEqual(state.current_phase(), _GIN.FIRST_UPCARD)
      self.assertEqual(state.upcard(), action)

  def test_a_draw_from_the_stock_names_no_card(self) -> None:
    drawn = 0
    for seed in range(_HANDS):
      previous: int | None = None
      for state, action, public in _steps(seed):
        if state.is_chance_node() and previous == _GIN.DRAW_STOCK_ACTION:
          self.assertIsNone(public)
          drawn += 1
        if not state.is_chance_node():
          previous = action
          if action == _GIN.DRAW_STOCK_ACTION:
            self.assertEqual(
                public, f"Player: {state.current_player()} Action: Draw stock"
            )
    self.assertGreater(drawn, 50)

  def test_a_frame_shows_other_seats_public_labels_and_no_hidden_card(
      self,
  ) -> None:
    oracle = contract.Viewer(contract.ViewerKind.ORACLE)
    viewers = (
        contract.Viewer(contract.ViewerKind.SEAT, 0),
        contract.Viewer(contract.ViewerKind.SEAT, 1),
        contract.Viewer(contract.ViewerKind.SPECTATOR),
    )
    random_spec = controllers.Spec(controllers.RANDOM)
    # The deal is 20 hidden cards; the 21st outcome is the first upcard.
    first_upcard = 2 * _GIN.DEFAULT_HAND_SIZE
    for seed in range(3):
      game = session_lib.Session.create(
          "gin_rummy",
          {},
          seed=seed,
          chance_policy=contract.ChancePolicy.AUTO,
          controllers=(random_spec, random_spec),
          autoplay=(True, True),
      )
      while not game.is_terminal:
        game.advance(oracle)
      full = game.frame(oracle).history
      self.assertFalse(any(entry.public_label for entry in full))
      for viewer in viewers:
        history = game.frame(viewer).history
        public_chance = []
        for index, (truth, entry) in enumerate(zip(full, history)):
          if not entry.withheld:
            self.assertEqual(entry.seat, viewer.seat)
            self.assertIsNone(entry.public_label)
          elif entry.seat is not None:
            # Another seat's draw, discard, knock or meld: its own label.
            self.assertEqual(entry.public_label, truth.label)
          elif entry.public_label is not None:
            self.assertEqual(entry.public_label, truth.label)
            public_chance.append(index)
        # Dealt cards and cards drawn from the stock stay hidden.
        self.assertEqual(public_chance, [first_upcard], f"{viewer}")


if __name__ == "__main__":
  absltest.main()
