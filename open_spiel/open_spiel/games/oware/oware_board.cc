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

#include "open_spiel/games/oware/oware_board.h"

namespace open_spiel {
namespace oware {

OwareBoard::OwareBoard(int num_houses_per_player, int num_seeds_per_house)
    : current_player(Player{0}),
      score(kNumPlayers, 0),
      seeds(kNumPlayers * num_houses_per_player, num_seeds_per_house) {}

OwareBoard::OwareBoard(Player current_player, const std::vector<int>& score,
                       const std::vector<int>& seeds)
    : current_player(current_player), score(score), seeds(seeds) {
  SPIEL_CHECK_EQ(score.size(), kNumPlayers);
}

bool OwareBoard::operator==(const OwareBoard& other) const {
  return current_player == other.current_player && score == other.score &&
         seeds == other.seeds;
}

bool OwareBoard::operator!=(const OwareBoard& other) const {
  return !(*this == other);
}

std::string OwareBoard::ToString() const {
  return absl::StrCat(current_player, " | ", absl::StrJoin(score, " "), " | ",
                      absl::StrJoin(seeds, " "));
}

size_t OwareBoard::HashValue() const {
  // Hashing similar to boost::hash_combine.
  size_t hash = current_player;
  for (int player_score : score) {
    hash ^= (size_t)player_score + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  }
  for (int house_seeds : seeds) {
    hash ^= (size_t)house_seeds + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  }
  return hash;
}
int OwareBoard::TotalSeeds() const {
  int total = 0;
  for (int house_seeds : seeds) {
    total += house_seeds;
  }
  for (int score_seeds : score) {
    total += score_seeds;
  }
  return total;
}

std::ostream& operator<<(std::ostream& os, const OwareBoard& board) {
  return os << board.ToString();
}

}  // namespace oware
}  // namespace open_spiel
