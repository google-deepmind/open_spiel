// Copyright 2025 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_GAMES_AZUL_H_
#define OPEN_SPIEL_GAMES_AZUL_H_

#include <array>
#include <memory>
#include <string>
#include <vector>
#include <random>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// Game of Azul:
// https://en.wikipedia.org/wiki/Azul_(board_game)
//
// Parameters:
//       "num_players"    int      number of players (2-4, default: 2)
//

namespace open_spiel {
namespace azul {

// Forward declarations
class AzulGame;

// Constants.
inline constexpr int kDefaultNumPlayers = 2;
inline constexpr int kMinNumPlayers = 2;
inline constexpr int kMaxNumPlayers = 4;
inline constexpr int kDefaultSeed = 0;
inline constexpr int kNumTileColors = 5;  // Blue, Yellow, Red, Black, White
inline constexpr int kNumFactories = 5;   // For 2 players (2*num_players + 1)
inline constexpr int kTilesPerFactory = 4;
inline constexpr int kWallSize = 5;       // 5x5 wall
inline constexpr int kNumPatternLines = 5;
inline constexpr int kTotalTilesPerColor = 20;
inline constexpr int kFirstPlayerTileValue = -1;  // Special tile value

// Maximum game length estimate (each player takes ~60-80 turns in worst case)
inline constexpr int kMaxGameLength = 400;

// Total distinct actions: Factory selection (factory_id * color) + Center selection (colors)
inline constexpr int kNumDistinctActions = 
    (kMaxNumPlayers * 2 + 1) * kNumTileColors * kNumPatternLines + // Factory to pattern line
    kNumTileColors * kNumPatternLines +                           // Center to pattern line  
    (kMaxNumPlayers * 2 + 1) * kNumTileColors +                  // Factory to floor
    kNumTileColors;                                              // Center to floor

// Tile colors
enum class TileColor {
  kBlue = 0,
  kYellow = 1,
  kRed = 2,
  kBlack = 3,
  kWhite = 4,
  kFirstPlayer = 5  // Special tile for first player
};

// Wall pattern for Azul (each row is shifted by one position)
inline constexpr std::array<std::array<TileColor, kWallSize>, kWallSize> kWallPattern = {{
  {{TileColor::kBlue, TileColor::kYellow, TileColor::kRed, TileColor::kBlack, TileColor::kWhite}},
  {{TileColor::kWhite, TileColor::kBlue, TileColor::kYellow, TileColor::kRed, TileColor::kBlack}},
  {{TileColor::kBlack, TileColor::kWhite, TileColor::kBlue, TileColor::kYellow, TileColor::kRed}},
  {{TileColor::kRed, TileColor::kBlack, TileColor::kWhite, TileColor::kBlue, TileColor::kYellow}},
  {{TileColor::kYellow, TileColor::kRed, TileColor::kBlack, TileColor::kWhite, TileColor::kBlue}}
}};

// Represents a factory display
struct Factory {
  std::array<int, kNumTileColors> tiles;
  
  Factory() {
    tiles.fill(0);
  }
  
  bool IsEmpty() const {
    for (int count : tiles) {
      if (count > 0) return false;
    }
    return true;
  }
  
  int TotalTiles() const {
    int total = 0;
    for (int count : tiles) {
      total += count;
    }
    return total;
  }
};

// Player board state
struct PlayerBoard {
  // Pattern lines (1 to 5 tiles) - each line can only contain one color
  struct PatternLine {
    TileColor color;
    int count;
    
    PatternLine() : color(TileColor::kBlue), count(0) {}  // color is irrelevant when count is 0
    
    bool IsEmpty() const { return count == 0; }
    bool IsFull(int line_index) const { return count == line_index + 1; }
    bool CanAccept(TileColor tile_color, int line_index) const {
      return IsEmpty() || (color == tile_color && !IsFull(line_index));
    }
  };
  
  std::array<PatternLine, kNumPatternLines> pattern_lines;
  // Wall (5x5 grid)
  std::array<std::array<bool, kWallSize>, kWallSize> wall;
  // Floor line (penalty tiles)
  std::vector<TileColor> floor_line;
  // Score
  int score;
  
  PlayerBoard() : score(0) {
    for (int i = 0; i < kWallSize; ++i) {
      wall[i].fill(false);
    }
  }
};

// State of an in-play game.
class AzulState : public State {
 public:
  AzulState(std::shared_ptr<const Game> game);
  AzulState(std::shared_ptr<const Game> game, int num_players);

  AzulState(const AzulState&) = default;
  AzulState& operator=(const AzulState&) = default;

  Player CurrentPlayer() const override;
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action action) override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;

  // Game-specific methods
  const std::vector<Factory>& Factories() const { return factories_; }
  const Factory& CenterPile() const { return center_pile_; }
  const std::vector<PlayerBoard>& PlayerBoards() const { return player_boards_; }
  const PlayerBoard& GetPlayerBoard(Player player) const { return player_boards_[player]; }
  bool HasFirstPlayerTile() const { return first_player_tile_available_; }
  
  // Public for testing
  std::vector<Factory> factories_;
  Factory center_pile_;
  std::vector<PlayerBoard> player_boards_;
  bool game_ended_;
  void EndRoundScoring();
  int CalculateScore(Player player) const;

 protected:
  void DoApplyAction(Action action) override;

 private:
  void SetupNewRound();
  void FillFactories();
  bool IsWallComplete(Player player) const;
  int GetNumFactories() const { return 2 * num_players_ + 1; }
  
  // Decode action
  struct DecodedAction {
    bool from_center;
    int factory_id;
    TileColor color;
    int destination;  // Pattern line (0-4) or -1 for floor
  };
  
  DecodedAction DecodeAction(Action action) const;
  Action EncodeAction(bool from_center, int factory_id, TileColor color, int destination) const;
  
  // Game state
  int num_players_;
  Player current_player_;
  std::vector<TileColor> bag_;
  std::vector<TileColor> discard_pile_;
  bool first_player_tile_available_;
  Player first_player_next_round_;
  int round_number_;
  // Chance node state
  bool needs_bag_shuffle_;
};

// Game object.
class AzulGame : public Game {
 public:
  explicit AzulGame(const GameParameters& params);
  
  int NumDistinctActions() const override { return kNumDistinctActions; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::make_unique<AzulState>(shared_from_this(), num_players_);
  }
  int NumPlayers() const override { return num_players_; }
  double MinUtility() const override { return -1; }  // Losing player utility
  absl::optional<double> UtilitySum() const override { return 0.0; }  // Zero-sum
  double MaxUtility() const override { return 1; }   // Winning player utility
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override { return kMaxGameLength; }
  int MaxChanceOutcomes() const override { return kTotalTilesPerColor * kNumTileColors; }  // Max tiles that can be shuffled
  
  // RNG support for deterministic shuffling
  std::mt19937& GetRNG() const { return rng_; }
  int GetOriginalSeed() const { return original_seed_; }

 private:
  int num_players_;
  mutable std::mt19937 rng_;
  int original_seed_;
};

// Utility functions
std::string TileColorToString(TileColor color);
TileColor StringToTileColor(const std::string& str);
std::string ActionToString(Action action, int num_players);

// Stream operator for TileColor
inline std::ostream& operator<<(std::ostream& os, TileColor color) {
  return os << TileColorToString(color);
}

}  // namespace azul
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_AZUL_H_ 