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

"""Tests for the JSON API, over the in-memory store and real games."""

import http
import json
from typing import Any
from unittest import mock

from absl.testing import absltest

from open_spiel.python.gui import api
from open_spiel.python.gui import catalogue as catalogue_lib
from open_spiel.python.gui import contract
from open_spiel.python.gui import controllers
from open_spiel.python.gui import health as health_lib
from open_spiel.python.gui import request
from open_spiel.python.gui import store as store_lib

_GAME = "tic_tac_toe"
_HUMAN_SEAT = {"controller": "human", "config": {}, "autoplay": False}
_CREATE: dict[str, object] = {
    "game": _GAME,
    "parameters": {},
    "seats": [_HUMAN_SEAT, _HUMAN_SEAT],
    "chance_policy": "auto",
    "seed": 5,
}


def _catalogue() -> catalogue_lib.Catalogue:
  """Returns a catalogue of tic-tac-toe alone."""
  return catalogue_lib.Catalogue(
      games=(
          catalogue_lib.CatalogueEntry(
              name=_GAME,
              long_name="Tic Tac Toe",
              dynamics=catalogue_lib.Dynamics.SEQUENTIAL,
              chance_mode=catalogue_lib.ChanceMode.DETERMINISTIC,
              information=catalogue_lib.Information.PERFECT_INFORMATION,
              min_seats=2,
              max_seats=2,
              num_players=2,
              parameters=(),
              loadable=True,
              unavailable_reason=None,
              controllers=(contract.HUMAN,),
              renderer="tic_tac_toe",
          ),
      )
  )


def _health() -> health_lib.Health:
  """Returns a health report naming every registered controller."""
  return health_lib.Health(
      version="0.0.0",
      revision=None,
      capabilities=health_lib.Capabilities(
          controllers=controllers.registered(),
          chance_policies=tuple(contract.ChancePolicy),
          cpp_mcts_limits=controllers.mcts_limits(),
      ),
  )


class ApiTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.store = store_lib.SessionStore()
    self.api = api.Api(self.store, catalogue=_catalogue, health=_health)

  def call(
      self, method: str, target: str, body: object = None
  ) -> tuple[http.HTTPStatus, Any]:
    """Calls the API, returning the status and the decoded body."""
    raw = b"" if body is None else json.dumps(body).encode()
    response = self.api.handle(method, target, raw)
    return response.status, response.body

  def create(self) -> str:
    """Creates a session as the oracle, returning its id."""
    status, frame = self.call("POST", "/api/sessions?viewer=oracle", _CREATE)
    self.assertEqual(status, http.HTTPStatus.OK)
    return str(frame["session"]["id"])

  def assert_refused(
      self,
      method: str,
      target: str,
      body: object,
      code: str,
      status: http.HTTPStatus,
  ) -> Any:
    """Asserts the request is refused, returning the error body."""
    actual_status, error = self.call(method, target, body)
    self.assertEqual((actual_status, error["code"]), (status, code))
    return error

  def test_health_answers_with_the_report(self) -> None:
    status, health = self.call("GET", "/api/health")
    self.assertEqual(status, http.HTTPStatus.OK)
    self.assertEqual(health["capabilities"]["controllers"][0]["name"], "human")

  def test_games_lists_the_catalogue(self) -> None:
    status, catalogue = self.call("GET", "/api/games")
    self.assertEqual(status, http.HTTPStatus.OK)
    self.assertEqual([game["name"] for game in catalogue["games"]], [_GAME])

  def test_create_returns_the_initial_frame_for_the_viewer(self) -> None:
    status, frame = self.call("POST", "/api/sessions?viewer=1", _CREATE)
    self.assertEqual(status, http.HTTPStatus.OK)
    self.assertEqual(frame["viewer"], {"kind": "seat", "seat": 1})
    self.assertEqual((frame["ply"], frame["num_plies"]), (0, 0))
    self.assertIsNone(frame["session"]["seed"])

  def test_only_the_oracle_sees_the_seed(self) -> None:
    session = self.create()
    _, frame = self.call("GET", f"/api/sessions/{session}?viewer=oracle")
    self.assertEqual(frame["session"]["seed"], 5)

  def test_create_refuses_an_unknown_game(self) -> None:
    self.assert_refused(
        "POST",
        "/api/sessions?viewer=oracle",
        _CREATE | {"game": "no_such_game"},
        "not_found",
        http.HTTPStatus.NOT_FOUND,
    )
    self.assertEmpty(self.store)

  def test_create_refuses_a_malformed_body_before_creating(self) -> None:
    error = self.assert_refused(
        "POST",
        "/api/sessions?viewer=oracle",
        _CREATE | {"seed": "five"},
        "bad_request",
        http.HTTPStatus.BAD_REQUEST,
    )
    self.assertIn("body.seed", error["message"])
    self.assertEmpty(self.store)

  def test_create_refuses_a_bad_viewer_before_creating(self) -> None:
    self.assert_refused(
        "POST",
        "/api/sessions?viewer=everyone",
        _CREATE,
        "bad_request",
        http.HTTPStatus.BAD_REQUEST,
    )
    self.assertEmpty(self.store)

  def test_create_keeps_no_session_for_a_viewer_it_refuses(self) -> None:
    self.assert_refused(
        "POST",
        "/api/sessions?viewer=5",
        _CREATE,
        "not_found",
        http.HTTPStatus.NOT_FOUND,
    )
    self.assertEmpty(self.store)

  def test_every_session_route_requires_exactly_one_viewer(self) -> None:
    session = self.create()
    for query in (
        "",
        "?viewer=",
        "?viewer=0&viewer=1",
        "?viewer=-1",
        "?viewer=01",
        "?viewer=Oracle",
    ):
      with self.subTest(query=query):
        self.assert_refused(
            "GET",
            f"/api/sessions/{session}{query}",
            None,
            "bad_request",
            http.HTTPStatus.BAD_REQUEST,
        )

  def test_a_viewer_seat_the_game_lacks_is_not_found(self) -> None:
    session = self.create()
    self.assert_refused(
        "GET",
        f"/api/sessions/{session}?viewer=2",
        None,
        "not_found",
        http.HTTPStatus.NOT_FOUND,
    )

  def test_an_unknown_session_is_not_found(self) -> None:
    self.assert_refused(
        "GET",
        "/api/sessions/missing?viewer=oracle",
        None,
        "not_found",
        http.HTTPStatus.NOT_FOUND,
    )

  def test_actions_play_for_the_seat_to_move(self) -> None:
    session = self.create()
    status, frame = self.call(
        "POST",
        f"/api/sessions/{session}/actions?viewer=0",
        {"action": 4, "ply": None},
    )
    self.assertEqual(status, http.HTTPStatus.OK)
    self.assertEqual(frame["history"][0]["action"], 4)
    self.assertEqual(frame["to_move"], {"kind": "seat", "seat": 1})

  def test_actions_play_from_the_ply_they_name(self) -> None:
    session = self.create()
    for viewer, action in ((0, 4), (1, 0)):
      self.call(
          "POST",
          f"/api/sessions/{session}/actions?viewer={viewer}",
          {"action": action, "ply": None},
      )
    status, frame = self.call(
        "POST",
        f"/api/sessions/{session}/actions?viewer=1",
        {"action": 2, "ply": 1},
    )
    self.assertEqual(status, http.HTTPStatus.OK)
    self.assertEqual([entry["action"] for entry in frame["history"]], [4, 2])

  def test_actions_refuse_a_ply_past_the_live_game(self) -> None:
    session = self.create()
    self.assert_refused(
        "POST",
        f"/api/sessions/{session}/actions?viewer=0",
        {"action": 4, "ply": 3},
        "not_found",
        http.HTTPStatus.NOT_FOUND,
    )

  def test_actions_refuse_a_seat_out_of_turn(self) -> None:
    session = self.create()
    self.assert_refused(
        "POST",
        f"/api/sessions/{session}/actions?viewer=1",
        {"action": 4, "ply": None},
        "not_your_turn",
        http.HTTPStatus.CONFLICT,
    )

  def test_actions_refuse_an_illegal_action(self) -> None:
    session = self.create()
    self.assert_refused(
        "POST",
        f"/api/sessions/{session}/actions?viewer=0",
        {"action": 9, "ply": None},
        "illegal_action",
        http.HTTPStatus.CONFLICT,
    )

  def test_chance_is_passed_to_the_session(self) -> None:
    session = self.create()
    self.assert_refused(
        "POST",
        f"/api/sessions/{session}/chance?viewer=oracle",
        {"outcome": 0, "ply": None},
        "not_your_turn",
        http.HTTPStatus.CONFLICT,
    )

  def test_advance_returns_the_live_frame(self) -> None:
    session = self.create()
    status, frame = self.call(
        "POST",
        f"/api/sessions/{session}/advance?viewer=oracle",
        {"max_plies": None},
    )
    self.assertEqual((status, frame["ply"]), (http.HTTPStatus.OK, 0))

  def test_advance_is_bounded(self) -> None:
    session = self.create()
    play = _HUMAN_SEAT | {"controller": controllers.RANDOM}
    for seat in (0, 1):
      self.call(
          "PUT", f"/api/sessions/{session}/seats/{seat}?viewer=oracle", play
      )
    target = f"/api/sessions/{session}/advance?viewer=oracle"
    status, frame = self.call("POST", target, {"max_plies": 2})
    self.assertEqual((status, frame["ply"]), (http.HTTPStatus.OK, 2))
    self.assert_refused(
        "POST",
        target,
        {"max_plies": request.MAX_ADVANCE_PLIES + 1},
        "bad_request",
        http.HTTPStatus.BAD_REQUEST,
    )
    live = self.store.get(session)
    with mock.patch.object(live, "advance", wraps=live.advance) as advance:
      self.call("POST", target, {"max_plies": None})
    self.assertEqual(
        advance.call_args.kwargs, {"max_plies": request.MAX_ADVANCE_PLIES}
    )

  def test_plies_replays_an_earlier_position(self) -> None:
    session = self.create()
    for action in (4, 0, 8):
      self.call(
          "POST",
          f"/api/sessions/{session}/actions?viewer=oracle",
          {"action": action, "ply": None},
      )
    status, frame = self.call(
        "GET", f"/api/sessions/{session}/plies/1?viewer=oracle"
    )
    self.assertEqual(status, http.HTTPStatus.OK)
    self.assertEqual((frame["ply"], frame["num_plies"]), (1, 3))
    self.assert_refused(
        "GET",
        f"/api/sessions/{session}/plies/4?viewer=oracle",
        None,
        "not_found",
        http.HTTPStatus.NOT_FOUND,
    )
    self.assert_refused(
        "GET",
        f"/api/sessions/{session}/plies/last?viewer=oracle",
        None,
        "bad_request",
        http.HTTPStatus.BAD_REQUEST,
    )

  def test_undo_removes_the_latest_action(self) -> None:
    session = self.create()
    self.assert_refused(
        "POST",
        f"/api/sessions/{session}/undo?viewer=oracle",
        None,
        "nothing_to_undo",
        http.HTTPStatus.CONFLICT,
    )
    self.call(
        "POST",
        f"/api/sessions/{session}/actions?viewer=0",
        {"action": 4, "ply": None},
    )
    _, frame = self.call("POST", f"/api/sessions/{session}/undo?viewer=0")
    self.assertEqual(frame["num_plies"], 0)

  def test_reset_replays_a_history(self) -> None:
    session = self.create()
    status, frame = self.call(
        "POST",
        f"/api/sessions/{session}/reset?viewer=oracle",
        {"history": [4, 0]},
    )
    self.assertEqual(status, http.HTTPStatus.OK)
    self.assertEqual([entry["action"] for entry in frame["history"]], [4, 0])

  def test_reset_names_the_illegal_entry(self) -> None:
    session = self.create()
    error = self.assert_refused(
        "POST",
        f"/api/sessions/{session}/reset?viewer=oracle",
        {"history": [4, 4]},
        "illegal_action",
        http.HTTPStatus.CONFLICT,
    )
    self.assertEqual(error["ply"], 1)

  def test_reset_refuses_a_bad_viewer_without_resetting(self) -> None:
    session = self.create()
    self.call(
        "POST",
        f"/api/sessions/{session}/actions?viewer=0",
        {"action": 4, "ply": None},
    )
    self.assert_refused(
        "POST",
        f"/api/sessions/{session}/reset?viewer=9",
        {"history": None},
        "not_found",
        http.HTTPStatus.NOT_FOUND,
    )
    self.assertEqual(self.store.get(session).history, (4,))

  def test_any_viewer_but_a_spectator_resets_or_reseats(self) -> None:
    status, frame = self.call(
        "POST", "/api/sessions?viewer=oracle", _CREATE | {"game": "kuhn_poker"}
    )
    self.assertEqual(status, http.HTTPStatus.OK)
    session = frame["session"]["id"]
    self.assert_refused(
        "POST",
        f"/api/sessions/{session}/reset?viewer=spectator",
        {"history": [0, 1]},
        "not_your_turn",
        http.HTTPStatus.CONFLICT,
    )
    self.assert_refused(
        "PUT",
        f"/api/sessions/{session}/seats/1?viewer=spectator",
        _HUMAN_SEAT,
        "not_your_turn",
        http.HTTPStatus.CONFLICT,
    )
    for viewer in ("0", "oracle"):
      status, answer = self.call(
          "POST",
          f"/api/sessions/{session}/reset?viewer={viewer}",
          {"history": [0, 1]},
      )
      self.assertEqual(status, http.HTTPStatus.OK)
      self.assertEqual(answer["viewer"]["seat"], 0 if viewer == "0" else None)
    status, _ = self.call(
        "PUT", f"/api/sessions/{session}/seats/1?viewer=0", _HUMAN_SEAT
    )
    self.assertEqual(status, http.HTTPStatus.OK)

  def test_a_seat_never_sees_another_seat_s_card(self) -> None:
    status, created = self.call(
        "POST", "/api/sessions?viewer=0", _CREATE | {"game": "kuhn_poker"}
    )
    self.assertEqual(status, http.HTTPStatus.OK)
    session = created["session"]["id"]
    self.call(
        "POST",
        f"/api/sessions/{session}/actions?viewer=0",
        {"action": 0, "ply": None},
    )
    _, oracle = self.call("GET", f"/api/sessions/{session}?viewer=oracle")
    # The first two plies deal seat 0's card and then seat 1's.
    deal = [entry["action"] for entry in oracle["history"][:2]]
    self.assertNotIn(None, deal)
    self.assertIsNotNone(oracle["state_text"])
    self.assertEqual(oracle["session"]["seed"], 5)
    replay = f"/api/sessions/{session}/plies/2?viewer=0"
    frames = {
        "create": created,
        "live": self.call("GET", f"/api/sessions/{session}?viewer=0")[1],
        "replay": self.call("GET", replay)[1],
    }
    for name, frame in frames.items():
      with self.subTest(frame=name):
        self.assertFalse(frame["omniscient"])
        self.assertIsNone(frame["state_text"])
        self.assertIsNone(frame["struct"])
        self.assertIsNone(frame["session"]["seed"])
        for entry in frame["history"][:2]:
          self.assertIsNone(entry["action"])
          self.assertIsNone(entry["label"])

  def test_seats_hands_a_seat_to_a_controller(self) -> None:
    session = self.create()
    status, _ = self.call(
        "PUT", f"/api/sessions/{session}/seats/1?viewer=oracle", _HUMAN_SEAT
    )
    self.assertEqual(status, http.HTTPStatus.OK)
    self.assert_refused(
        "PUT",
        f"/api/sessions/{session}/seats/1?viewer=oracle",
        _HUMAN_SEAT | {"controller": "no_such_controller"},
        "controller_unavailable",
        http.HTTPStatus.CONFLICT,
    )

  def test_an_unknown_route_is_not_found(self) -> None:
    self.assert_refused(
        "GET", "/api/players", None, "not_found", http.HTTPStatus.NOT_FOUND
    )

  def test_a_known_route_refuses_another_method(self) -> None:
    error = self.assert_refused(
        "DELETE",
        "/api/games",
        None,
        "method_not_allowed",
        http.HTTPStatus.METHOD_NOT_ALLOWED,
    )
    self.assertIn("accepts GET", error["message"])

  def test_close_forgets_the_session(self) -> None:
    session = self.create()
    status, closed = self.call("DELETE", f"/api/sessions/{session}?viewer=0")
    self.assertEqual(status, http.HTTPStatus.OK)
    self.assertEqual(
        closed, {"contract_version": contract.CONTRACT_VERSION, "id": session}
    )
    self.assert_refused(
        "GET",
        f"/api/sessions/{session}?viewer=oracle",
        None,
        "not_found",
        http.HTTPStatus.NOT_FOUND,
    )
    self.assert_refused(
        "DELETE",
        f"/api/sessions/{session}?viewer=oracle",
        None,
        "not_found",
        http.HTTPStatus.NOT_FOUND,
    )

  def test_a_spectator_never_closes_a_session(self) -> None:
    session = self.create()
    self.assert_refused(
        "DELETE",
        f"/api/sessions/{session}?viewer=spectator",
        None,
        "not_your_turn",
        http.HTTPStatus.CONFLICT,
    )
    self.assertLen(self.store, 1)

  def test_close_requires_a_viewer(self) -> None:
    session = self.create()
    self.assert_refused(
        "DELETE",
        f"/api/sessions/{session}",
        None,
        "bad_request",
        http.HTTPStatus.BAD_REQUEST,
    )

  def test_a_session_fault_propagates(self) -> None:
    session = self.create()
    with mock.patch.object(
        self.store.get(session), "frame", side_effect=RuntimeError("broken")
    ):
      with self.assertRaisesRegex(RuntimeError, "broken"):
        self.api.handle("GET", f"/api/sessions/{session}?viewer=oracle")


if __name__ == "__main__":
  absltest.main()
