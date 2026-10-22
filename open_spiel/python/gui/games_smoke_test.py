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

"""Sweeps a few representative games through real sessions.

Each game is played to its end by seeded random controllers, and every
viewer's frame is checked at every position (`sweep.py`), privacy included
in a game that hides information. The games cover each kind the server
treats differently: perfect information, imperfect information, chance,
simultaneous moves, and parameters other than the defaults.

The sweep of every registered game takes minutes, so it is a separate
command: `python -m open_spiel.python.gui.sweep_all`.
"""

from collections.abc import Callable, Mapping
from typing import cast

from absl.testing import absltest
from absl.testing import parameterized

from open_spiel.python.gui import contract
from open_spiel.python.gui import controllers
from open_spiel.python.gui import frame_checks
from open_spiel.python.gui import session
from open_spiel.python.gui import sweep
import pyspiel

# absl's decorator is not annotated; this gives it a type for strict mypy.
named_parameters = cast(
    Callable[..., Callable[[Callable[..., None]], Callable[..., None]]],
    parameterized.named_parameters,
)

# Each case: its name, the game, its parameters, and whether it hides
# information, so that the sweep checks what each viewer may see.
_CASES: tuple[tuple[str, str, Mapping[str, int], bool], ...] = (
    ("perfect_information", "tic_tac_toe", {}, False),
    ("imperfect_information_and_chance", "kuhn_poker", {}, True),
    ("chance", "pig", {"winscore": 10}, False),
    ("simultaneous", "goofspiel", {"num_cards": 4}, True),
    ("parameters", "leduc_poker", {"players": 3}, True),
)


def _game_string(game_name: str, parameters: Mapping[str, int]) -> str:
  """Returns the string that loads `game_name` with `parameters`."""
  if not parameters:
    return game_name
  settings = ",".join(f"{key}={value}" for key, value in parameters.items())
  return f"{game_name}({settings})"


def _open_random_session(
    game_name: str, parameters: Mapping[str, int]
) -> session.Session:
  """Opens a seeded session of the game with every seat played at random."""
  num_seats = pyspiel.load_game(game_name, dict(parameters)).num_players()
  return session.Session.create(
      game_name,
      parameters,
      seed=sweep.SEED,
      chance_policy=contract.ChancePolicy.AUTO,
      controllers=[controllers.Spec(controllers.RANDOM)] * num_seats,
  )


class GamesSmokeTest(parameterized.TestCase):

  @named_parameters(*_CASES)
  def test_game_passes_the_sweep(
      self,
      game_name: str,
      parameters: Mapping[str, int],
      hides_information: bool,
  ) -> None:
    game_string = _game_string(game_name, parameters)
    self.assertEqual(
        frame_checks.Replay(game_string).hides_information, hides_information
    )
    result = sweep.sweep_game(
        game_string, lambda _: _open_random_session(game_name, parameters)
    )
    self.assertEqual(result.outcome, sweep.Outcome.PASSED, str(result))
    self.assertGreater(result.plies, 0)

  def test_each_seat_sees_its_own_card_and_not_the_other(self) -> None:
    live = _open_random_session("kuhn_poker", {})
    oracle = live.frame(contract.Viewer(contract.ViewerKind.ORACLE))
    self.assertTrue(oracle.omniscient)
    self.assertIsNotNone(oracle.state_text)
    # The two chance outcomes deal seat 0's card, then seat 1's.
    state = pyspiel.load_game("kuhn_poker").new_initial_state()
    for entry in oracle.history:
      self.assertIsNotNone(entry.action)
      state.apply_action(entry.action)
    own = [state.information_state_string(seat) for seat in range(2)]
    self.assertNotEqual(own[0], own[1])
    for seat in range(2):
      with self.subTest(seat=seat):
        frame = live.frame(contract.Viewer(contract.ViewerKind.SEAT, seat))
        self.assertFalse(frame.omniscient)
        self.assertIsNone(frame.state_text)
        self.assertIsNone(frame.struct)
        self.assertIsNone(frame.session.seed)
        self.assertEqual(frame.information_state_text, own[seat])
        self.assertTrue(frame.history[1 - seat].withheld)


if __name__ == "__main__":
  absltest.main()
