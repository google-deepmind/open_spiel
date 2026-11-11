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


"""A controller plugin with a required string setting, for testing.

`--controllers open_spiel.python.gui.fake_checkpoint_plugin` registers one
controller, `fake_checkpoint`, with the settings of a bot that plays a
trained checkpoint: `checkpoint`, a string a client must send, and
`deterministic` and `temperature`, which have defaults. It loads nothing and
plays its seat's lowest legal action; `built` keeps the settings it was
built with, so that a test can see what reached it.
"""

from collections.abc import Mapping
import types

import pyspiel

from open_spiel.python.gui import contract

# The name the plugin registers its controller under.
NAME = "fake_checkpoint"

# The settings each controller was built with, in the order built.
built: list[Mapping[str, contract.ParameterValue]] = []


class FakeCheckpoint:
  """Plays its seat's lowest legal action, whatever its checkpoint."""

  def choose_action(self, state: pyspiel.State, seat: int) -> int:
    """Returns the lowest of `seat`'s legal actions."""
    return int(state.legal_actions(seat)[0])


def _build(
    game: pyspiel.Game, config: Mapping[str, contract.ParameterValue], seed: int
) -> FakeCheckpoint:
  """Builds the controller, keeping the settings it is built with."""
  del game, seed
  built.append(dict(config))
  return FakeCheckpoint()


def register_into(controllers: types.ModuleType) -> None:
  """Registers `fake_checkpoint` with the GUI's `controllers` module."""
  settings = (
      controllers.Setting(
          "checkpoint",
          str,
          controllers.REQUIRED,
          "The checkpoint to play.",
          min_length=1,
      ),
      controllers.Setting(
          "deterministic",
          bool,
          True,
          "Whether it plays its most likely move rather than sampling.",
      ),
      controllers.Setting(
          "temperature",
          float,
          1.0,
          "The temperature it samples its moves at.",
          minimum=0.0,
      ),
  )
  controllers.register(
      NAME,
      controllers.Kind(
          "Fake checkpoint", settings, reads_full_state=False, build=_build
      ),
  )
