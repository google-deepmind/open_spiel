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

"""A test tagged "manual": CMake adds it only with OPEN_SPIEL_RUN_MANUAL_TESTS.

No other OpenSpiel test is tagged "manual"; GitHub CI checks the CMake switch
with this one, so it only needs to exist.
"""

import pyspiel


def test_fixture_from_conftest(tic_tac_toe: pyspiel.Game) -> None:
  assert tic_tac_toe.num_players() == 2
