// Copyright 2026 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Plays one game of tic-tac-toe, always taking the first legal action.

#include <iostream>
#include <memory>

#include "open_spiel/spiel.h"

int main() {
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame("tic_tac_toe");
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();
  while (!state->IsTerminal()) {
    state->ApplyAction(state->LegalActions()[0]);
  }
  std::cout << state->ToString() << "\nReturns: " << state->Returns()[0]
            << " " << state->Returns()[1] << std::endl;
  return state->Returns()[0] == 1 ? 0 : 1;
}
