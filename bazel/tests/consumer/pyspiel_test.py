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

"""Checks that the pyspiel extension module loads and plays a game."""

import pyspiel


def main() -> None:
  game = pyspiel.load_game("kuhn_poker")
  assert game.num_players() == 2, game.num_players()
  state = game.new_initial_state()
  while not state.is_terminal():
    state.apply_action(state.legal_actions()[0])
  assert sum(state.returns()) == 0, state.returns()


if __name__ == "__main__":
  main()
