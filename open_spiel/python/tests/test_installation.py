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

"""OpenSpiel Installation Test Script.

This script is used to verify whether OpenSpiel is installed correctly.
"""

import sys

# pylint: disable=g-interpreter-mismatch
# pylint: disable=g-import-not-at-top
# pylint: disable=broad-exception-caught


def test_openspiel():
  """Test OpenSpiel installation and basic functionality."""
  print(" Testing OpenSpiel Installation")
  print("=" * 50)

  try:
    import pyspiel

    print(" pyspiel imported successfully")
    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()
    print(f" Game loaded: {game.get_type().short_name}")
    print(f" Players: {game.num_players()}")
    print(f" Legal actions: {len(state.legal_actions())}")
    print("\n OpenSpiel is working correctly!")
    return True
  except Exception as e:  # pragma: no cover - diagnostic script
    print(f"❌ Error: {e}")
    return False


if __name__ == "__main__":
  success = test_openspiel()
  sys.exit(0 if success else 1)
