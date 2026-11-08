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


"""The Remote bot seat's checkpoint setting, sent as `config.checkpoint`."""

from typing import Any

from absl.testing import absltest
import pyspiel

from open_spiel.python.gui import controllers
from open_spiel.python.gui import remote_bot
from open_spiel.python.gui import remote_bot_config


def _settings(**changes: Any) -> dict[str, Any]:
  """A Remote bot seat's config and checkpoint settings, with `changes`."""
  values: dict[str, Any] = {"config": "{}", "checkpoint": ""}
  values.update(changes)
  return values


class SeatBotConfigTest(absltest.TestCase):

  def test_the_checkpoint_setting_is_sent_in_the_config(self) -> None:
    self.assertEqual(
        remote_bot_config.seat_bot_config(
            _settings(
                checkpoint=" /models/run@30000 ", config='{"temperature": 0}'
            )
        ),
        {"checkpoint": "/models/run@30000", "temperature": 0},
    )

  def test_an_empty_checkpoint_setting_sends_none(self) -> None:
    self.assertEqual(remote_bot_config.seat_bot_config(_settings()), {})

  def test_the_same_checkpoint_in_both_places_is_accepted(self) -> None:
    self.assertEqual(
        remote_bot_config.seat_bot_config(
            _settings(
                checkpoint="/models/a@3", config='{"checkpoint": "/models/a@3"}'
            )
        ),
        {"checkpoint": "/models/a@3"},
    )

  def test_a_config_checkpoint_alone_is_sent(self) -> None:
    self.assertEqual(
        remote_bot_config.seat_bot_config(
            _settings(config='{"checkpoint": "/models/b@1"}')
        ),
        {"checkpoint": "/models/b@1"},
    )

  def test_a_checkpoint_that_disagrees_with_the_config_is_refused(
      self,
  ) -> None:
    with self.assertRaisesRegex(ValueError, "checkpoint"):
      remote_bot_config.seat_bot_config(
          _settings(
              checkpoint="/models/a@3", config='{"checkpoint": "/models/b@3"}'
          )
      )

  def test_the_seat_refuses_a_disagreeing_checkpoint(self) -> None:
    with self.assertRaisesRegex(controllers.UnavailableError, "checkpoint"):
      controllers.builder(
          controllers.Spec(
              remote_bot.NAME,
              {
                  "url": "http://localhost:1",
                  "checkpoint": "/models/a@3",
                  "config": '{"checkpoint": "/models/b@3"}',
              },
          ),
          pyspiel.load_game("kuhn_poker"),
      )

  def test_the_seat_offers_a_checkpoint_setting(self) -> None:
    names = [setting.name for setting in remote_bot.kind().settings]
    self.assertIn("checkpoint", names)
    self.assertLess(names.index("checkpoint"), names.index("config"))


if __name__ == "__main__":
  absltest.main()
