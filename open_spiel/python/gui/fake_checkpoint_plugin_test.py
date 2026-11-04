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


"""A plugin's required string setting, end to end through the API."""

import http
import json
from typing import Any

from absl.testing import absltest

from open_spiel.python.gui import api
from open_spiel.python.gui import catalogue as catalogue_lib
from open_spiel.python.gui import controllers
from open_spiel.python.gui import fake_checkpoint_plugin
from open_spiel.python.gui import health_report
from open_spiel.python.gui import plugins
from open_spiel.python.gui import store as store_lib

_PLUGIN = "open_spiel.python.gui.fake_checkpoint_plugin"
_HUMAN_SEAT = {"controller": "human", "config": {}, "autoplay": False}
_CREATE: dict[str, object] = {
    "game": "tic_tac_toe",
    "parameters": {},
    "seats": [_HUMAN_SEAT, _HUMAN_SEAT],
    "chance_policy": "auto",
    "seed": 5,
}
_SEAT_1 = "/api/sessions/{session}/seats/1?viewer=oracle"
_CHECKPOINT = "runs/run_1/checkpoint_100"


class FakeCheckpointPluginTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.assertEqual(plugins.load([_PLUGIN]), (fake_checkpoint_plugin.NAME,))
    self.addCleanup(controllers.unregister, fake_checkpoint_plugin.NAME)
    fake_checkpoint_plugin.built.clear()
    self.api = api.Api(
        store_lib.SessionStore(),
        catalogue=lambda: catalogue_lib.Catalogue(games=()),
        health=health_report.build,
    )

  def call(
      self, method: str, target: str, body: object = None
  ) -> tuple[http.HTTPStatus, Any]:
    """Calls the API, returning the status and the decoded body."""
    raw = b"" if body is None else json.dumps(body).encode()
    response = self.api.handle(method, target, raw)
    return response.status, response.body

  def create(self) -> str:
    """Creates a session of two human seats, returning its id."""
    status, frame = self.call("POST", "/api/sessions?viewer=oracle", _CREATE)
    self.assertEqual(status, http.HTTPStatus.OK)
    return str(frame["session"]["id"])

  def test_health_gives_the_required_setting_in_the_schema(self) -> None:
    status, health = self.call("GET", "/api/health")
    self.assertEqual(status, http.HTTPStatus.OK)
    listed = {
        controller["name"]: controller
        for controller in health["capabilities"]["controllers"]
    }
    schema = listed[fake_checkpoint_plugin.NAME]["config_schema"]
    self.assertEqual(schema["required"], ["checkpoint"])
    self.assertEqual(
        schema["properties"]["checkpoint"],
        {
            "type": "string",
            "description": "The checkpoint to play.",
            "minLength": 1,
        },
    )
    self.assertIs(schema["properties"]["deterministic"]["default"], True)
    self.assertEqual(schema["properties"]["temperature"]["default"], 1.0)

  def test_a_seat_without_the_required_setting_is_refused(self) -> None:
    session = self.create()
    for config in ({}, {"temperature": 0.5}):
      with self.subTest(config=config):
        status, error = self.call(
            "PUT",
            _SEAT_1.format(session=session),
            {
                "controller": fake_checkpoint_plugin.NAME,
                "config": config,
                "autoplay": False,
            },
        )
        self.assertEqual(
            (status, error["code"]),
            (http.HTTPStatus.CONFLICT, "controller_unavailable"),
        )
        self.assertIn("needs the setting 'checkpoint'", error["message"])
    self.assertEqual(fake_checkpoint_plugin.built, [])

  def test_an_empty_checkpoint_is_refused(self) -> None:
    session = self.create()
    status, error = self.call(
        "PUT",
        _SEAT_1.format(session=session),
        {
            "controller": fake_checkpoint_plugin.NAME,
            "config": {"checkpoint": ""},
            "autoplay": False,
        },
    )
    self.assertEqual(status, http.HTTPStatus.CONFLICT)
    self.assertIn("at least 1 character long", error["message"])

  def test_a_seat_with_the_checkpoint_plays_with_the_defaults(self) -> None:
    session = self.create()
    seat = {
        "controller": fake_checkpoint_plugin.NAME,
        "config": {"checkpoint": _CHECKPOINT},
        "autoplay": False,
    }
    status, frame = self.call("PUT", _SEAT_1.format(session=session), seat)
    self.assertEqual(status, http.HTTPStatus.OK)
    self.assertEqual(frame["seats"][1]["controller"], "fake_checkpoint")
    self.assertEqual(frame["seats"][1]["config"], {"checkpoint": _CHECKPOINT})
    self.assertIn(
        {"checkpoint": _CHECKPOINT, "deterministic": True, "temperature": 1.0},
        fake_checkpoint_plugin.built,
    )
    status, frame = self.call(
        "POST",
        f"/api/sessions/{session}/actions?viewer=0",
        {"action": 4, "ply": None},
    )
    self.assertEqual(status, http.HTTPStatus.OK, frame)
    status, frame = self.call(
        "POST",
        f"/api/sessions/{session}/advance?viewer=oracle",
        {"max_plies": 1},
    )
    self.assertEqual((status, frame["ply"]), (http.HTTPStatus.OK, 2))
    self.assertEqual(frame["history"][1]["action"], 0)  # Its lowest legal.


if __name__ == "__main__":
  absltest.main()
