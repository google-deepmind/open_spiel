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


"""Tests for the registry of game adapters and the presentations they make."""

from collections.abc import Iterator, Mapping, Sequence
import contextlib
import dataclasses
import random
from unittest import mock

from absl.testing import absltest

import pyspiel

from open_spiel.python.gui import contract
from open_spiel.python.gui import controllers
from open_spiel.python.gui import loader
from open_spiel.python.gui import presenters
from open_spiel.python.gui import views

_ORACLE = contract.Viewer(contract.ViewerKind.ORACLE)
_SPECTATOR = contract.Viewer(contract.ViewerKind.SPECTATOR)

# A game that hides each player's dice, for the privacy tests.
_HIDDEN_GAME = "liars_dice"

# The most plies of each adapter's game played.
_PLIES = 200


class _DiceAdapter(presenters.Adapter):
  """Shows each seat's information state to whoever may see it."""

  renderer = "dice"

  def present(
      self, state: pyspiel.State, viewpoint: presenters.Viewpoint
  ) -> dict[str, contract.JsonValue]:
    return {
        "hands": [
            (
                state.information_state_string(seat)
                if viewpoint.sees_private(seat)
                else None
            )
            for seat in range(state.num_players())
        ]
    }

  def describe_actions(
      self,
      state: pyspiel.State,
      actions: Sequence[int],
      viewpoint: presenters.Viewpoint,
  ) -> Mapping[int, contract.JsonValue]:
    del state, viewpoint
    return {action: {"bid": action} for action in actions[:2]}


class _NotJsonAdapter(presenters.Adapter):
  """Returns a view that is not JSON."""

  renderer = "broken"

  def present(
      self, state: pyspiel.State, viewpoint: presenters.Viewpoint
  ) -> dict[str, contract.JsonValue]:
    del state, viewpoint
    return {"value": float("nan")}


@contextlib.contextmanager
def _registered(game: str, adapter: presenters.Adapter) -> Iterator[None]:
  """Registers `adapter` for `game`, setting aside any it already has."""
  previous = presenters.adapter_for(game)
  if previous is not None:
    presenters.unregister(game)
  presenters.register(game, adapter)
  try:
    yield
  finally:
    presenters.unregister(game)
    if previous is not None:
      presenters.register(game, previous)


def _dealt(name: str) -> views.Position:
  """Returns the first position of `name` after its chance outcomes."""
  loaded = loader.load(name, {})
  state = loaded.game.new_initial_state()
  while state.is_chance_node():
    state.apply_action(state.chance_outcomes()[0][0])
  seats = tuple(
      contract.SeatView(seat, contract.HUMAN, {}, False, None)
      for seat in range(loaded.num_seats)
  )
  return views.Position(
      info=contract.SessionInfo(
          id="s",
          game=loaded.name,
          parameters=dict(loaded.parameters),
          num_seats=loaded.num_seats,
          chance_policy=contract.ChancePolicy.AUTO,
          turn_based=loaded.turn_based,
          hides_information=loaded.hides_information,
          seed=1,
          controllers=controllers.available(loaded.game),
      ),
      loaded=loaded,
      status=contract.SessionStatus.PLAYING,
      seats=seats,
      state=state,
      ply=0,
      history=(),
  )


def _seat(seat: int) -> contract.Viewer:
  """The viewer of seat `seat`."""
  return contract.Viewer(contract.ViewerKind.SEAT, seat)


class RegistryTest(absltest.TestCase):

  def test_registers_the_adapters_package(self) -> None:
    adapter = presenters.adapter_for("tic_tac_toe")
    assert adapter is not None
    self.assertEqual(adapter.renderer, "tic_tac_toe")

  def test_a_game_without_an_adapter_has_no_presentation(self) -> None:
    self.assertIsNone(presenters.adapter_for("no_such_game"))
    position = _dealt("blotto")
    self.assertIsNone(views.build(position, _ORACLE).presentation)

  def test_an_adapter_keeps_other_seats_actions_hidden_by_default(
      self,
  ) -> None:
    for game in ("tic_tac_toe", "nim"):
      state = pyspiel.load_game(game).new_initial_state()
      action = state.legal_actions()[0]
      self.assertIsNone(presenters.public_action_label(state, action))

  def test_refuses_a_public_label_that_is_not_text(self) -> None:
    grid = presenters.adapter_for("tic_tac_toe")
    state = pyspiel.load_game("tic_tac_toe").new_initial_state()
    with mock.patch.object(grid, "public_action_label", lambda s, a: a):
      with self.assertRaisesRegex(TypeError, "public action label"):
        presenters.public_action_label(state, 4)

  def test_refuses_a_second_adapter_for_a_game(self) -> None:
    with _registered(_HIDDEN_GAME, _DiceAdapter()):
      with self.assertRaisesRegex(ValueError, "already registered"):
        presenters.register(_HIDDEN_GAME, _DiceAdapter())

  def test_refuses_to_unregister_a_game_without_an_adapter(self) -> None:
    with self.assertRaisesRegex(ValueError, "No adapter"):
      presenters.unregister("no_such_game")

  def test_refuses_an_adapter_without_a_renderer(self) -> None:
    adapter = _DiceAdapter()
    adapter.renderer = " "
    with self.assertRaisesRegex(ValueError, "names no renderer"):
      presenters.register("no_such_game", adapter)

  def test_refuses_a_view_that_is_not_json(self) -> None:
    with _registered(_HIDDEN_GAME, _NotJsonAdapter()):
      with self.assertRaisesRegex(TypeError, "not JSON"):
        views.build(_dealt(_HIDDEN_GAME), _ORACLE)


class PrivacyTest(absltest.TestCase):

  def test_each_viewer_sees_only_what_it_may(self) -> None:
    position = _dealt(_HIDDEN_GAME)
    with _registered(_HIDDEN_GAME, _DiceAdapter()):
      seat0 = views.build(position, _seat(0)).presentation
      seat1 = views.build(position, _seat(1)).presentation
      spectator = views.build(position, _SPECTATOR).presentation
      oracle = views.build(position, _ORACLE).presentation
    assert seat0 and seat1 and spectator and oracle
    self.assertEqual(seat0.renderer, "dice")
    hands = {
        name: presentation.view["hands"]  # type: ignore[index]
        for name, presentation in (
            ("seat0", seat0),
            ("seat1", seat1),
            ("spectator", spectator),
            ("oracle", oracle),
        )
    }
    self.assertIsNotNone(hands["seat0"][0])
    self.assertIsNone(hands["seat0"][1])
    self.assertIsNone(hands["seat1"][0])
    self.assertIsNotNone(hands["seat1"][1])
    self.assertEqual(hands["spectator"], [None, None])
    self.assertNotIn(None, hands["oracle"])

  def test_hints_come_only_with_legal_actions(self) -> None:
    position = _dealt(_HIDDEN_GAME)
    with _registered(_HIDDEN_GAME, _DiceAdapter()):
      mover = views.build(position, _seat(0)).presentation
      waiting = views.build(position, _seat(1)).presentation
    assert mover and waiting and mover.actions is not None
    legal = position.state.legal_actions()
    self.assertEqual(
        mover.actions,
        tuple(contract.ActionHint(a, {"bid": a}) for a in legal[:2]),
    )
    self.assertIsNone(waiting.actions)


def _viewers(num_seats: int) -> list[contract.Viewer]:
  """Every viewer of a game with `num_seats` seats."""
  return [_ORACLE, _SPECTATOR, *(_seat(seat) for seat in range(num_seats))]


class EveryAdapterTest(absltest.TestCase):
  """Plays seeded random games of every registered adapter's game."""

  def test_presents_every_position_to_every_viewer(self) -> None:
    for game, registered in presenters.adapters().items():
      with self.subTest(game=game):
        self.assertTrue(registered.renderer.strip())
        position = _dealt(game)
        rng = random.Random(game)
        state = position.loaded.game.new_initial_state()
        for _ in range(_PLIES):
          seats = _viewers(position.loaded.num_seats)
          shown = dataclasses.replace(
              position,
              state=state,
              status=views.status(state, position.seats),
          )
          for viewer in seats:
            # Frame validation checks each hint names a legal action.
            built = views.build(shown, viewer)
            self.assertIsNotNone(built.presentation)
          if state.is_terminal():
            break
          if state.is_chance_node():
            outcomes, weights = zip(*state.chance_outcomes())
            state.apply_action(rng.choices(outcomes, weights)[0])
          else:
            state.apply_action(rng.choice(state.legal_actions()))


class TicTacToeTest(absltest.TestCase):

  def test_presents_the_position_the_panel_reads(self) -> None:
    position = _dealt("tic_tac_toe")
    state = position.state.clone()
    state.apply_action(4)
    built = views.build(
        dataclasses.replace(position, state=state), _SPECTATOR
    ).presentation
    assert built is not None and built.actions is not None
    view = built.view
    assert isinstance(view, dict)
    self.assertEqual(view["history"], [4])
    self.assertEqual(view["currentPlayerLabel"], "player 1")
    # The panel reads the legal actions from the frame, so none is hinted.
    self.assertEqual(built.actions, ())


if __name__ == "__main__":
  absltest.main()
