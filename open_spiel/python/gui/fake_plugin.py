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


"""A controller plugin, for testing the plugin loader and showing its shape.

`--controllers open_spiel.python.gui.fake_plugin` registers one controller,
`first_legal`, which plays its seat's lowest legal action. A third-party
plugin has the same shape: a module with a `register_into(controllers)`.
"""

from collections.abc import Mapping
import types

import pyspiel

from open_spiel.python.gui import contract

# The name the plugin registers its controller under.
NAME = "first_legal"


class FirstLegal:
  """Plays its seat's lowest legal action."""

  def choose_action(self, state: pyspiel.State, seat: int) -> int:
    """Returns the lowest of `seat`'s legal actions."""
    return int(state.legal_actions(seat)[0])


def _build(
    game: pyspiel.Game, config: Mapping[str, contract.ParameterValue], seed: int
) -> FirstLegal:
  """Builds the controller, which takes no settings and no seed."""
  del game, config, seed
  return FirstLegal()


def register_into(controllers: types.ModuleType) -> None:
  """Registers `first_legal` with the GUI's `controllers` module."""
  controllers.register(
      NAME,
      controllers.Kind("First legal", (), reads_full_state=False, build=_build),
  )
