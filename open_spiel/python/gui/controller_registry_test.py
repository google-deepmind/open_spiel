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

"""Tests for the registry of controllers."""

from collections.abc import Mapping

from absl.testing import absltest

import pyspiel

from open_spiel.python.gui import contract
from open_spiel.python.gui import controller_settings
from open_spiel.python.gui import controllers


class _FirstLegal:
  """A controller that plays its seat's first legal action."""

  def choose_action(self, state: pyspiel.State, seat: int) -> int:
    """Returns the lowest of `seat`'s legal actions."""
    return int(state.legal_actions(seat)[0])


def _build_first_legal(
    game: pyspiel.Game, config: Mapping[str, contract.ParameterValue], seed: int
) -> controllers.Controller:
  """Builds a controller that plays its seat's first legal action."""
  del game, config, seed
  return _FirstLegal()


_FIRST_LEGAL = controllers.Kind(
    "First legal", (), reads_full_state=False, build=_build_first_legal
)


class RegistryTest(absltest.TestCase):

  def register(self, name: str, kind: controllers.Kind) -> None:
    """Registers `kind` for the length of the test."""
    controllers.register(name, kind)
    self.addCleanup(controllers.unregister, name)

  def test_the_builtin_controllers_are_registered_human_first(self) -> None:
    self.assertEqual(
        tuple(controllers.kinds())[:3],
        (controllers.HUMAN, controllers.RANDOM, controllers.MCTS),
    )

  def test_a_registered_controller_is_listed_and_plays(self) -> None:
    self.register("first_legal", _FIRST_LEGAL)
    self.assertEqual(controllers.registered()[-1].name, "first_legal")
    game = pyspiel.load_game("tic_tac_toe")
    self.assertIn(
        "first_legal",
        {controller.name for controller in controllers.available(game)},
    )
    controller = controllers.create(
        controllers.Spec("first_legal"), game, seed=0
    )
    state = game.new_initial_state()
    self.assertEqual(controllers.choose(controller, state, 0), 0)

  def test_an_unregistered_controller_is_gone(self) -> None:
    controllers.register("first_legal", _FIRST_LEGAL)
    controllers.unregister("first_legal")
    self.assertNotIn("first_legal", controllers.kinds())
    with self.assertRaisesRegex(controllers.UnavailableError, "no controller"):
      controllers.create(
          controllers.Spec("first_legal"),
          pyspiel.load_game("tic_tac_toe"),
          seed=0,
      )

  def test_refuses_a_name_registered_twice(self) -> None:
    with self.assertRaisesRegex(ValueError, "already registered"):
      controllers.register(controllers.RANDOM, _FIRST_LEGAL)

  def test_refuses_a_malformed_name(self) -> None:
    for name in ("", "First", "1st", "first-legal", "first legal"):
      with self.subTest(name):
        with self.assertRaisesRegex(ValueError, "lower case"):
          controllers.register(name, _FIRST_LEGAL)

  def test_refuses_to_unregister_human_or_an_unknown_name(self) -> None:
    with self.assertRaisesRegex(ValueError, "cannot be unregistered"):
      controllers.unregister(controllers.HUMAN)
    with self.assertRaisesRegex(ValueError, "No controller"):
      controllers.unregister("nobody")

  def test_refuses_a_blank_label(self) -> None:
    with self.assertRaisesRegex(ValueError, "blank"):
      controllers.Kind(
          " ", (), reads_full_state=False, build=_build_first_legal
      )

  def test_a_kind_refuses_what_its_refusal_refuses(self) -> None:
    def refuse_all(game: pyspiel.Game) -> str | None:
      return f"Never {game.get_type().short_name}."

    self.register(
        "refuses",
        controllers.Kind(
            "Refuses",
            (),
            reads_full_state=False,
            build=_build_first_legal,
            refusal=refuse_all,
        ),
    )
    game = pyspiel.load_game("tic_tac_toe")
    with self.assertRaisesRegex(controllers.UnavailableError, "Never"):
      controllers.create(controllers.Spec("refuses"), game, seed=0)

  def test_a_kind_refuses_a_missing_required_setting(self) -> None:
    kind = controllers.Kind(
        "Checkpoint",
        (
            controllers.Setting(
                "checkpoint", str, controllers.REQUIRED, "The checkpoint."
            ),
        ),
        reads_full_state=False,
        build=_build_first_legal,
    )
    self.assertEqual(kind.config_schema()["required"], ["checkpoint"])
    self.assertEqual(kind.parse({"checkpoint": "c"}), {"checkpoint": "c"})
    with self.assertRaisesRegex(
        controllers.UnavailableError,
        "^Checkpoint needs the setting 'checkpoint'.$",
    ) as raised:
      kind.parse({})
    self.assertIsInstance(
        raised.exception.__cause__, controller_settings.InvalidSettingError
    )


if __name__ == "__main__":
  absltest.main()
