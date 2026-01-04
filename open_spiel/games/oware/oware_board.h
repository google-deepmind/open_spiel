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

#ifndef OPEN_SPIEL_GAMES_OWARE_OWARE_BOARD_H_
#define OPEN_SPIEL_GAMES_OWARE_OWARE_BOARD_H_

#include <string>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
namespace open_spiel {
namespace oware {

inline constexpr int kNumPlayers = 2;

// Simple Oware board struct storing the current player, scores and seeds.
struct OwareBoard {
 public:
  OwareBoard(int num_houses_per_player, int num_seeds_per_house);
  // Custom board setup to support testing.
  OwareBoard(Player current_player, const std::vector<int>& score,
             const std::vector<int>& seeds);
  OwareBoard(const OwareBoard&) = default;
  OwareBoard& operator=(const OwareBoard&) = default;
  bool operator==(const OwareBoard& other) const;
  bool operator!=(const OwareBoard& other) const;
  std::string ToString() const;
  size_t HashValue() const;

  // Returns total number of seeds, both those
  // captured and the ones still in play.
  int TotalSeeds() const;

  Player current_player;
  // The number of seeds each player has in their score house, one entry
  // for each player.
  std::vector<int> score;
  // The number of seeds in each house. First the (kNumHousesPerPlayer) houses
  // for player 0, then for player 1, in counterclockwise order (i.e. the order
  // in which seeds are sown).
  std::vector<int> seeds;
};

std::ostream& operator<<(std::ostream& os, const OwareBoard& board);

}  // namespace oware
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_OWARE_OWARE_BOARD_H_
