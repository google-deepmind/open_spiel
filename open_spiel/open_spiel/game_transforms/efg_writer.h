// Copyright 2021 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_GAME_TRANSFORMS_EFG_WRITER_H_
#define OPEN_SPIEL_GAME_TRANSFORMS_EFG_WRITER_H_

#include <map>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

namespace open_spiel {

// Takes an OpenSpiel game and converts it to the .efg format used by Gambit:
// http://www.gambit-project.org/gambit14/formats.html
//
// USE WITH CAUTION! For small games only. This could easily fill up disk
// space for large games.
//
// Note: Currently only supports sequential games and terminal rewards.

class EFGWriter {
 public:
  EFGWriter(const Game& game, const std::string filename,
            bool action_names = true, bool separate_infostate_numbers = true);
  void Write();

 private:
  const Game& game_;
  const std::string filename_;
  // Use descriptive action names. If false, action ints are used.
  bool action_names_;
  // Keep track of infostate numbers for each player separately. In general,
  // the same integer will specify different information sets for different
  // players.
  bool separate_infostate_numbers_;
  int chance_node_counter_;
  int terminal_node_counter_;
  std::vector<std::map<std::string, int>> infostate_numbers_;

  void Write(std::ostream& f, const State& state);
};

}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAME_TRANSFORMS_EFG_WRITER_H_
