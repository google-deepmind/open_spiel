// Copyright 2019 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_GAMES_GO_SGF_GAME_LOADER_H_
#define OPEN_SPIEL_GAMES_GO_SGF_GAME_LOADER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/spiel.h"

// A simple SGF reader for Go. Supports a subset of SGF properties and makes
// some assumptions. Please see sgf_reader.cc for more details.

namespace open_spiel {
namespace go {

using GameAndState =
    std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>;
using VectorOfGamesAndStates = std::vector<GameAndState>;

// Loads games and states from an SGF file. Note: if setup moves are used at
// any point in the game, e.g. AB[] or AW[], the resulting state will not be
// serializable because the board is modified directly and the history does not
// contain the setup moves.
VectorOfGamesAndStates LoadGamesFromSGFFile(const std::string& sgf_filename);
VectorOfGamesAndStates LoadGamesFromSGFString(const std::string& sgf_string);

}  // namespace go
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_GO_SGF_READER_H_
