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

"""Tests for the tic_tac_toe adapter, its privacy among them."""

import random

from absl.testing import absltest

import pyspiel

from open_spiel.python.gui import adapter
from open_spiel.python.gui import contract
from open_spiel.python.gui import controllers
from open_spiel.python.gui import loader
from open_spiel.python.gui import plies
from open_spiel.python.gui import presenters
from open_spiel.python.gui import views
from open_spiel.python.gui.adapters import tic_tac_toe_adapter

_ADAPTER = tic_tac_toe_adapter.ADAPTER
_OMNISCIENT = adapter.Viewpoint(seat=None, omniscient=True)
_FIELDS = {
    "history",
    "historyStr",
    "currentPlayer",
    "currentPlayerLabel",
    "isTerminal",
    "returns",
    "turnNumber",
    "gameParams",
}


def _played(actions: list[int], name: str = "tic_tac_toe") -> pyspiel.State:
  """The state of `name` after `actions`."""
  state = pyspiel.load_game(name).new_initial_state()
  for action in actions:
    state.apply_action(action)
  return state


def _position(actions: list[int]) -> views.Position:
  """The live session position after `actions`, with two human seats."""
  loaded = loader.load("tic_tac_toe", {})
  state = loaded.game.new_initial_state()
  history: list[plies.Ply] = []
  for action in actions:
    seat = state.current_player()
    label = state.action_to_string(seat, action)
    state.apply_action(action)
    history.append(plies.Ply(seat, action, label, tuple(state.rewards())))
  seats = tuple(
      contract.SeatView(seat, contract.HUMAN, {}, False, None)
      for seat in range(2)
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
      status=views.status(state, seats),
      seats=seats,
      state=state,
      ply=len(history),
      history=tuple(history),
  )


def _viewers() -> list[contract.Viewer]:
  """Every kind of viewer: both seats, a spectator and the oracle."""
  return [
      contract.Viewer(contract.ViewerKind.SEAT, 0),
      contract.Viewer(contract.ViewerKind.SEAT, 1),
      contract.Viewer(contract.ViewerKind.SPECTATOR),
      contract.Viewer(contract.ViewerKind.ORACLE),
  ]


class PresentTest(absltest.TestCase):

  def test_is_registered_for_its_renderer(self) -> None:
    self.assertIs(presenters.adapter_for("tic_tac_toe"), _ADAPTER)
    self.assertEqual(_ADAPTER.renderer, "tic_tac_toe")

  def test_the_first_position(self) -> None:
    view = _ADAPTER.present(_played([]), _OMNISCIENT)
    self.assertEqual(set(view), _FIELDS)
    self.assertEqual(view["history"], [])
    self.assertEqual(view["historyStr"], "")
    self.assertEqual(view["currentPlayer"], 0)
    self.assertEqual(view["currentPlayerLabel"], "player 0")
    self.assertIs(view["isTerminal"], False)
    self.assertIsNone(view["returns"])
    self.assertEqual(view["turnNumber"], 0)

  def test_after_moves(self) -> None:
    state = _played([4, 0, 8])
    view = _ADAPTER.present(state, _OMNISCIENT)
    self.assertEqual(view["history"], [4, 0, 8])
    self.assertEqual(view["historyStr"], state.history_str())
    self.assertEqual(view["currentPlayer"], 1)
    self.assertEqual(view["currentPlayerLabel"], "player 1")
    self.assertEqual(view["turnNumber"], 3)

  def test_a_finished_game(self) -> None:
    state = _played([0, 3, 1, 4, 2])
    self.assertTrue(state.is_terminal())
    view = _ADAPTER.present(state, _OMNISCIENT)
    self.assertIs(view["isTerminal"], True)
    self.assertEqual(view["currentPlayerLabel"], "terminal")
    self.assertEqual(view["returns"], [1.0, -1.0])

  def test_game_params(self) -> None:
    params = _ADAPTER.present(_played([]), _OMNISCIENT)["gameParams"]
    game = pyspiel.load_game("tic_tac_toe")
    self.assertEqual(params, dict(game.get_parameters()))


class PrivacyTest(absltest.TestCase):

  def test_every_viewer_gets_the_same_view(self) -> None:
    # The game hides nothing, so no viewer may be told more, or less, than
    # another; every viewer of such a game is omniscient.
    rng = random.Random(4)
    for _ in range(5):
      actions: list[int] = []
      while True:
        position = _position(actions)
        shown = []
        for viewer in _viewers():
          frame = views.build(position, viewer)
          self.assertTrue(frame.omniscient)
          assert frame.presentation is not None
          view = frame.presentation.view
          assert isinstance(view, dict)
          self.assertEqual(set(view), _FIELDS)
          self.assertEqual(view["history"], actions)
          shown.append(view)
        for view in shown:
          self.assertEqual(view, shown[0])
        if position.state.is_terminal():
          break
        actions.append(rng.choice(position.state.legal_actions()))

  def test_refuses_a_viewer_that_is_not_omniscient(self) -> None:
    for seat in (None, 0, 1):
      with self.assertRaisesRegex(ValueError, "omniscient"):
        _ADAPTER.present(
            _played([]), adapter.Viewpoint(seat=seat, omniscient=False)
        )

  def test_refuses_a_game_that_hides_information(self) -> None:
    state = pyspiel.load_game("kuhn_poker").new_initial_state()
    state.apply_action(0)
    state.apply_action(1)
    with self.assertRaisesRegex(ValueError, "hides information"):
      _ADAPTER.present(state, _OMNISCIENT)


if __name__ == "__main__":
  absltest.main()
