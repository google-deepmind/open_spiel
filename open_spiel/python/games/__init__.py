# Copyright 2019 DeepMind Technologies Limited
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

"""Games implemented in Python.

These games are registered as they are imported. It's perfectly possible to
import just a single game if you prefer. There is no need to add new games here,
so long as they register themselves and you import them when wanting to use
them. However, adding them here will make them available for playthroughs and
for automated API testing.

Registration looks like this:
```
pyspiel.register_game(_GAME_TYPE, KuhnPokerGame)
```
"""

from open_spiel.python.games import dynamic_routing
from open_spiel.python.games import iterated_prisoners_dilemma
from open_spiel.python.games import kuhn_poker
from open_spiel.python.games import tic_tac_toe
