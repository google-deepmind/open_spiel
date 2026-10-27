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


"""Game adapters: one module per game, each named `<game>_adapter.py`.

Each module defines `GAME`, the game's short name, and `ADAPTER`, an
`adapter.Adapter`, and is listed once below; `presenters` registers every
adapter listed. An adapter stands alone, so a game can replace its adapter
without touching another.
"""

from collections.abc import Mapping
import types

from open_spiel.python.gui import adapter
from open_spiel.python.gui.adapters import checkers_adapter
from open_spiel.python.gui.adapters import chess_adapter
from open_spiel.python.gui.adapters import connect_four_adapter
from open_spiel.python.gui.adapters import euchre_adapter
from open_spiel.python.gui.adapters import gin_rummy_adapter
from open_spiel.python.gui.adapters import hearts_adapter
from open_spiel.python.gui.adapters import kuhn_poker_adapter
from open_spiel.python.gui.adapters import leduc_poker_adapter
from open_spiel.python.gui.adapters import tic_tac_toe_adapter

# Every adapter module. Add a game's module with one line, in name order.
_MODULES = [
    checkers_adapter,
    chess_adapter,
    connect_four_adapter,
    euchre_adapter,
    gin_rummy_adapter,
    hearts_adapter,
    kuhn_poker_adapter,
    leduc_poker_adapter,
    tic_tac_toe_adapter,
]

# Every adapter, by game short name.
ADAPTERS: Mapping[str, adapter.Adapter] = types.MappingProxyType(
    {module.GAME: module.ADAPTER for module in _MODULES}
)
