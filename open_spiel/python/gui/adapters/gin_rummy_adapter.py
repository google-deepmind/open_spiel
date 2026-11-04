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

# STRUCT-GLUE(gin_rummy ObservationStruct): the board needs more than the
# observation struct carries (a spectator's view, the meld analysis, the
# lay-down plans, the cards known from the discard pile), so the adapter
# builds the struct-shaped view from the state's accessors and adds them.
"""The gin_rummy adapter: the game's struct for one viewer, and its analysis.

The view is the record the `gin_rummy` client renderer reads:

- `struct`: the fields of the game's state struct (phase, current player,
  knock card, upcard, previous upcard, stock size, hands, discard pile,
  deadwood, knocked, passes on the first upcard, laid melds, lay-offs,
  finished lay-offs), filtered for the viewer as the game's observation
  struct filters them: a hand the viewer may not see is a list of "XX", and
  its deadwood is -1. A seat sees its own hand; every viewer sees a hand once
  its seat has knocked, and both hands in the lay-off phase and at the end.
- `ginVisibleHands`: the seats whose hands the rules have laid open.
- `ginRevealHands`: whether every hand is open.
- `ginLayedMelds`: each seat's laid melds.
- `ginAnalysisMeldsBySeat`, `ginAnalysisBestIndicesBySeat`: every meld in a
  hand the viewer can see, and which of them make its best arrangement.
- `ginLaydownSeat`, `ginLaydownStage`, `ginLaydownPassAction`,
  `ginLaydownPlans`: while a seat lays down its melds or lays off cards, the
  seat, the sub-stage ("discard", "layoffs" or "melds"), and, for the seat
  itself, its pass action and every set of melds and lay-offs it can lay
  before passing.
- `ginPhase`, `ginKnocked`: the phase name and whether anyone has knocked.
- `knownCardsBySeat`: the cards each seat is known to hold because it took
  them from the discard pile, which every viewer saw.

`public_action_label` names the other seat's draws, discards, knocks and
melds in the history, and the first upcard; the cards dealt and drawn from
the stock stay hidden.
"""

from collections.abc import Sequence
import functools

import pyspiel

from open_spiel.python.gui import adapter
from open_spiel.python.gui import contract

GAME = "gin_rummy"

_GIN = pyspiel.gin_rummy
_DRAW_UPCARD = _GIN.DRAW_UPCARD_ACTION
_PASS = _GIN.PASS_ACTION
_MELD_BASE = _GIN.MELD_ACTION_BASE

_NUM_SEATS = 2
_HIDDEN_CARD = "XX"

# The struct's phase names, in the engine's phase order.
_PHASE_NAMES = {
    _GIN.DEAL: "Deal",
    _GIN.FIRST_UPCARD: "FirstUpcard",
    _GIN.DRAW: "Draw",
    _GIN.DISCARD: "Discard",
    _GIN.KNOCK: "Knock",
    _GIN.LAYOFF: "Layoff",
    _GIN.WALL: "Wall",
    _GIN.GAME_OVER: "GameOver",
}

# The phases in which every hand is laid open.
_OPEN_PHASES = (_GIN.LAYOFF, _GIN.GAME_OVER)

# The phases in which a seat lays melds or lays off cards.
_LAY_PHASES = (_GIN.KNOCK, _GIN.LAYOFF)

# A bound on the laid sets the lay-down enumeration expands. A real turn
# reaches a few dozen; passing it means the enumeration is wrong.
_PLAN_BUDGET = 10_000

Json = contract.JsonValue


@functools.lru_cache(maxsize=8)
def _utils_for(
    num_ranks: int, num_suits: int, hand_size: int
) -> "pyspiel.gin_rummy.GinRummyUtils":
  """The card and meld tables of a deck; they depend on its size."""
  return _GIN.GinRummyUtils(num_ranks, num_suits, hand_size)


def _utils(state: pyspiel.State) -> "pyspiel.gin_rummy.GinRummyUtils":
  """The card and meld tables of `state`'s game."""
  parameters = state.get_game().get_parameters()
  return _utils_for(
      int(parameters["num_ranks"]),
      int(parameters["num_suits"]),
      int(parameters["hand_size"]),
  )


def _code(utils: "pyspiel.gin_rummy.GinRummyUtils", card: int) -> str:
  """The code of the card numbered `card`: "As", "Td"."""
  code = utils.card_string(card)
  if not isinstance(code, str) or len(code) != 2:
    raise ValueError(f"Card {card} has no gin_rummy code: {code!r}.")
  return code


def _codes(
    utils: "pyspiel.gin_rummy.GinRummyUtils", cards: Sequence[int]
) -> list[Json]:
  """The codes of `cards`, in their order."""
  return [_code(utils, card) for card in cards]


def _player_string(player: int) -> str:
  """The struct's name for `player`, as the engine writes it."""
  if player >= 0:
    return f"Player_{player}"
  if player == pyspiel.PlayerId.CHANCE:
    return "Chance"
  if player == pyspiel.PlayerId.TERMINAL:
    return "Terminal"
  raise ValueError(f"gin_rummy has no player {player}.")


def _struct(
    state: pyspiel.State, utils: "pyspiel.gin_rummy.GinRummyUtils"
) -> dict[str, Json]:
  """The state struct, every hand shown, as the engine fills it."""
  upcard = state.upcard()
  prev_upcard = state.prev_upcard()
  return {
      "phase": _PHASE_NAMES[state.current_phase()],
      "current_player": _player_string(state.current_player()),
      "knock_card": int(state.knock_card()),
      "upcard": None if upcard is None else _code(utils, upcard),
      "prev_upcard": None if prev_upcard is None else _code(utils, prev_upcard),
      "stock_size": int(state.stock_size()),
      "hands": [_codes(utils, sorted(hand)) for hand in state.hands()],
      "discard_pile": _codes(utils, state.discard_pile()),
      "deadwood": [int(d) for d in state.deadwood()],
      "knocked": [bool(k) for k in state.knocked()],
      "pass_on_first_upcard": [bool(p) for p in state.pass_on_first_upcard()],
      "layed_melds": [
          [_codes(utils, utils.int_to_meld[meld]) for meld in seat_melds]
          for seat_melds in state.layed_melds()
      ],
      "layoffs": _codes(utils, state.layoffs()),
      "finished_layoffs": bool(state.finished_layoffs()),
  }


def _open_seats(state: pyspiel.State) -> set[int]:
  """The seats whose hands the rules have laid open to every viewer."""
  if state.current_phase() in _OPEN_PHASES:
    return set(range(_NUM_SEATS))
  return {seat for seat, k in enumerate(state.knocked()) if k}


def _visible_seats(
    state: pyspiel.State, viewpoint: adapter.Viewpoint
) -> set[int]:
  """The seats whose hands the viewer may see."""
  own = {s for s in range(_NUM_SEATS) if viewpoint.sees_private(s)}
  return own | _open_seats(state)


def _viewer_struct(
    struct: dict[str, Json], visible: set[int]
) -> dict[str, Json]:
  """`struct` with every hand the viewer may not see face down.

  As the engine's observation struct hides a hand: each card is "XX" and the
  hand's deadwood is -1.
  """
  hands = struct["hands"]
  deadwood = struct["deadwood"]
  assert isinstance(hands, list) and isinstance(deadwood, list)
  shown_hands: list[Json] = []
  shown_deadwood: list[Json] = []
  for seat, (hand, wood) in enumerate(zip(hands, deadwood, strict=True)):
    assert isinstance(hand, list)
    shown = seat in visible
    shown_hands.append(list(hand) if shown else [_HIDDEN_CARD] * len(hand))
    shown_deadwood.append(wood if shown else -1)
  return {**struct, "hands": shown_hands, "deadwood": shown_deadwood}


def _analysis(
    state: pyspiel.State,
    utils: "pyspiel.gin_rummy.GinRummyUtils",
    visible: set[int],
) -> tuple[dict[str, Json], dict[str, Json]]:
  """Every meld of each visible hand, and which make its best arrangement.

  A seat is in both maps or in neither: its candidate melds without the
  best arrangement would let the client pick overlapping melds.
  """
  melds_by_seat: dict[str, Json] = {}
  best_by_seat: dict[str, Json] = {}
  for seat, hand in enumerate(state.hands()):
    if seat not in visible or not hand:
      continue
    candidates = [sorted(meld) for meld in utils.all_melds(list(hand))]
    best = [sorted(meld) for meld in utils.best_meld_group(list(hand))]
    if not candidates:
      continue
    indices: list[Json] = []
    for meld in best:
      if meld not in candidates:
        raise AssertionError(
            f"Best meld {meld} of seat {seat} is not among its melds."
        )
      indices.append(candidates.index(meld))
    if not indices:
      raise AssertionError(f"Seat {seat} has melds but no best arrangement.")
    melds_by_seat[str(seat)] = [_codes(utils, meld) for meld in candidates]
    best_by_seat[str(seat)] = indices
  return melds_by_seat, best_by_seat


def _lay_cards(
    utils: "pyspiel.gin_rummy.GinRummyUtils", action: int, pass_legal: bool
) -> list[int] | None:
  """The cards `action` lays during a lay-down, or None for no lay action.

  A meld action lays its meld. A card action is a lay-off only when passing
  is also legal; otherwise it is the knocker's discard.
  """
  if action >= _MELD_BASE:
    return sorted(utils.int_to_meld[action - _MELD_BASE])
  if pass_legal and 0 <= action < _DRAW_UPCARD:
    return [action]
  return None


def _laydown_plans(
    state: pyspiel.State, utils: "pyspiel.gin_rummy.GinRummyUtils"
) -> tuple[list[Json], int | None]:
  """Every set of melds and lay-offs the seat can lay, then pass.

  A plan is a list of steps, each `{"cards": codes, "action": id}`. The
  state after a set of lay actions does not depend on their order, so each
  set is expanded once.
  """
  plans: dict[frozenset[tuple[int, ...]], list[Json]] = {}
  expanded: set[frozenset[tuple[int, ...]]] = set()
  pass_action: int | None = None

  def expand(
      current: pyspiel.State,
      steps: tuple[tuple[int, ...], ...],
      actions: tuple[int, ...],
  ) -> None:
    nonlocal pass_action
    laid = frozenset(steps)
    if laid in expanded:
      return
    expanded.add(laid)
    if len(expanded) > _PLAN_BUDGET:
      raise RuntimeError(
          f"The lay-down enumeration passed {_PLAN_BUDGET} laid sets."
      )
    legal = current.legal_actions()
    pass_legal = _PASS in legal
    for action in legal:
      if action == _PASS:
        pass_action = action
        if steps and laid not in plans:
          plans[laid] = [
              {"cards": _codes(utils, cards), "action": act}
              for cards, act in zip(steps, actions, strict=True)
          ]
        continue
      cards = _lay_cards(utils, action, pass_legal)
      if cards is None or tuple(cards) in steps:
        continue
      child = current.clone()
      child.apply_action(action)
      expand(child, steps + (tuple(cards),), actions + (action,))

  expand(state.clone(), (), ())
  return list(plans.values()), pass_action


def _laydown(
    state: pyspiel.State,
    utils: "pyspiel.gin_rummy.GinRummyUtils",
    viewpoint: adapter.Viewpoint,
) -> dict[str, Json]:
  """The lay-down fields: the seat, its sub-stage, and for it, its plans."""
  if state.is_terminal() or state.current_phase() not in _LAY_PHASES:
    return {}
  seat = state.current_player()
  plans, pass_action = _laydown_plans(state, utils)
  if state.current_phase() == _GIN.KNOCK:
    stage = "melds" if pass_action is not None else "discard"
  elif state.finished_layoffs():
    stage = "melds"
  else:
    stage = "layoffs"
  fields: dict[str, Json] = {"ginLaydownSeat": seat, "ginLaydownStage": stage}
  # Only the seat itself (or the oracle) is told what it can lay: its plans
  # are made of the cards in its hand.
  if viewpoint.sees_private(seat):
    if plans:
      fields["ginLaydownPlans"] = plans
    if pass_action is not None:
      fields["ginLaydownPassAction"] = pass_action
  return fields


def _known_cards(
    state: pyspiel.State, utils: "pyspiel.gin_rummy.GinRummyUtils"
) -> dict[str, Json]:
  """The cards each seat took from the discard pile and still holds.

  Read from the history: the last card dealt or discarded is the upcard; a
  seat that draws it is known to hold it until it discards it.
  """
  known: dict[int, set[str]] = {seat: set() for seat in range(_NUM_SEATS)}
  upcard: str | None = None
  for item in state.full_history():
    player, action = item.player, item.action
    if player < 0:
      upcard = _code(utils, action)
    elif action == _DRAW_UPCARD and upcard is not None:
      known[player].add(upcard)
      upcard = None
    elif action < _DRAW_UPCARD:
      card = _code(utils, action)
      known[player].discard(card)
      upcard = card
  return {str(seat): sorted(cards) for seat, cards in known.items() if cards}


class GinRummyAdapter(adapter.Adapter):
  """Presents a gin rummy hand to one viewer as its struct and analysis."""

  renderer = "gin_rummy"

  def present(
      self, state: pyspiel.State, viewpoint: adapter.Viewpoint
  ) -> dict[str, Json]:
    """The struct and analysis, holding only what the viewer may see."""
    utils = _utils(state)
    struct = _struct(state, utils)
    visible = _visible_seats(state, viewpoint)
    open_seats = _open_seats(state)
    melds_by_seat, best_by_seat = _analysis(state, utils, visible)
    layed: dict[str, Json] = {
        str(seat): [_codes(utils, utils.int_to_meld[m]) for m in melds]
        for seat, melds in enumerate(state.layed_melds())
        if melds
    }
    view: dict[str, Json] = {
        "struct": _viewer_struct(struct, visible),
        "ginPhase": struct["phase"],
        "ginKnocked": any(state.knocked()),
        "ginRevealHands": state.current_phase() in _OPEN_PHASES,
        "ginVisibleHands": {str(seat): True for seat in sorted(open_seats)},
        "ginLayedMelds": layed,
        "ginAnalysisMeldsBySeat": melds_by_seat,
        "ginAnalysisBestIndicesBySeat": best_by_seat,
        "knownCardsBySeat": _known_cards(state, utils),
    }
    view.update(_laydown(state, utils, viewpoint))
    return view

  def public_action_label(
      self, state: pyspiel.State, action: int
  ) -> str | None:
    """The action's own label when the table sees it; None when it is hidden.

    Every seat action is public: a discard is laid face up, a draw names the
    pile it takes from (the stock's card is a chance outcome of its own), and
    knocks, passes and laid melds are on the table. Of the chance outcomes
    only the first upcard, dealt face up, is public; a dealt card and the card
    a draw from the stock takes are hidden.
    """
    player = state.current_player()
    if player != pyspiel.PlayerId.CHANCE:
      return str(state.action_to_string(player, action))
    parameters = state.get_game().get_parameters()
    num_cards = int(parameters["num_ranks"]) * int(parameters["num_suits"])
    first_upcard = num_cards - _NUM_SEATS * int(parameters["hand_size"])
    if int(state.stock_size()) == first_upcard:
      return str(state.action_to_string(player, action))
    return None


ADAPTER = GinRummyAdapter()
