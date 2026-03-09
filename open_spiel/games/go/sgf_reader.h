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

#ifndef OPEN_SPIEL_GAMES_GO_SGF_READER_H_
#define OPEN_SPIEL_GAMES_GO_SGF_READER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/games/go/go.h"
#include "open_spiel/games/go/go_board.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// A simple SGF reader for Go. Supports a subset of SGF properties and makes
// some assumptions. Please see sgf_reader.cc for more details.

namespace open_spiel {
namespace go {

using PropertyValuesPair = std::pair<std::string, std::vector<std::string>>;
using GameAndState =
    std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>;
using VectorOfGamesAndStates = std::vector<GameAndState>;

class SGFReader {
 public:
  explicit SGFReader(const std::string& sgf_string)
      : sgf_string_(sgf_string), index_(0), current_game_index_(0) {}

  // Reads the next game or returns and empty vector if none found.
  std::vector<GameAndState> ReadNextGames();
  absl::flat_hash_map<std::string, std::string> ReadRootNode();
  std::vector<PropertyValuesPair> ReadNextNode();

 private:
  // Check a character and output error with context if it doesn't match.
  void CheckCharAtIndex(int index, char expected) const;
  std::string GetContext(int index) const;
  std::string GetCurrentContext() const { return GetContext(index_); }
  std::string GetCurrentGame() const;
  bool CheckCaptureKoPassConnect(
      GoColor color, const std::vector<Player>& player_history) const;

  void SkipWhitespace();
  std::string ReadPropertyName();
  std::vector<std::string> ReadPropertyValues();

  void AddStones(GoState* go_state, GoColor color,
                 const std::vector<std::string>& property_values) const;
  void ProcessRootNodeSetup(
      absl::flat_hash_map<std::string, std::string>* root_node,
      GoState* go_state) const;

  std::string sgf_string_;
  int index_;
  int current_game_index_;
};

VectorOfGamesAndStates LoadGamesFromSGFFile(const std::string& sgf_filename);

VectorOfGamesAndStates LoadGamesFromSGFString(const std::string& sgf_string);

}  // namespace go
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_GO_SGF_READER_H_
