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

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

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

namespace open_spiel::azul {

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
inline constexpr int kWallSize = 5;  // 5x5 wall
inline constexpr int kNumPatternLines = 5;
inline constexpr int kTotalTilesPerColor = 20;
inline constexpr int kFirstPlayerTileValue = -1;  // Special tile value

// Maximum game length estimate (each player takes ~60-80 turns in worst case)
inline constexpr int kMaxGameLength = 400;

// Total distinct actions: Factory selection (factory_id * color) + Center
// selection (colors)
inline constexpr int kNumDistinctActions =
    ((kMaxNumPlayers * 2 + 1) * kNumTileColors *
     kNumPatternLines) +                           // Factory to pattern line
    (kNumTileColors * kNumPatternLines) +          // Center to pattern line
    ((kMaxNumPlayers * 2 + 1) * kNumTileColors) +  // Factory to floor
    kNumTileColors;                                // Center to floor

// Tile colors
enum class TileColor : std::uint8_t {
  kBlue = 0,
  kYellow = 1,
  kRed = 2,
  kBlack = 3,
  kWhite = 4,
  kFirstPlayer = 5  // Special tile for first player
};

// Wall pattern for Azul (each row is shifted by one position)
inline constexpr std::array<std::array<TileColor, kWallSize>, kWallSize>
    kWallPattern = {{{{TileColor::kBlue, TileColor::kYellow, TileColor::kRed,
                       TileColor::kBlack, TileColor::kWhite}},
                     {{TileColor::kWhite, TileColor::kBlue, TileColor::kYellow,
                       TileColor::kRed, TileColor::kBlack}},
                     {{TileColor::kBlack, TileColor::kWhite, TileColor::kBlue,
                       TileColor::kYellow, TileColor::kRed}},
                     {{TileColor::kRed, TileColor::kBlack, TileColor::kWhite,
                       TileColor::kBlue, TileColor::kYellow}},
                     {{TileColor::kYellow, TileColor::kRed, TileColor::kBlack,
                       TileColor::kWhite, TileColor::kBlue}}}};

// Represents a factory display
struct Factory {
  std::array<int, kNumTileColors> tiles;

  Factory() { tiles.fill(0); }
  [[nodiscard]] auto IsEmpty() const -> bool {
    return std::all_of(tiles.begin(), tiles.end(),
                       [](int count) { return count == 0; });
  }

  [[nodiscard]] auto TotalTiles() const -> int {
    return std::accumulate(tiles.begin(), tiles.end(), 0);
  }
};

// Player board state
struct PlayerBoard {
  // Pattern lines (1 to 5 tiles) - each line can only contain one color
  struct PatternLine {
    TileColor color{};
    int count{};

    PatternLine() = default;  // color is irrelevant when count is 0

    [[nodiscard]] auto IsEmpty() const -> bool { return count == 0; }
    [[nodiscard]] auto IsFull(int line_index) const -> bool {
      return count == line_index + 1;
    }
    [[nodiscard]] auto CanAccept(TileColor tile_color, int line_index) const
        -> bool {
      return IsEmpty() || (color == tile_color && !IsFull(line_index));
    }
  };

  std::array<PatternLine, kNumPatternLines> pattern_lines;
  // Wall (5x5 grid)
  std::array<std::array<bool, kWallSize>, kWallSize> wall;
  // Floor line (penalty tiles)
  std::vector<TileColor> floor_line;
  // Score
  int score{0};

  PlayerBoard() {
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
  auto operator=(const AzulState&) -> AzulState& = delete;

  [[nodiscard]] auto CurrentPlayer() const -> Player override;
  [[nodiscard]] auto LegalActions() const -> std::vector<Action> override;
  [[nodiscard]] auto ActionToString(Player player, Action action) const
      -> std::string override;
  [[nodiscard]] auto ToString() const -> std::string override;
  [[nodiscard]] auto IsTerminal() const -> bool override;
  [[nodiscard]] auto Returns() const -> std::vector<double> override;
  [[nodiscard]] auto InformationStateString(Player player) const
      -> std::string override;
  [[nodiscard]] auto ObservationString(Player player) const
      -> std::string override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  [[nodiscard]] auto Clone() const -> std::unique_ptr<State> override;
  void UndoAction(Player player, Action action) override;
  [[nodiscard]] auto ChanceOutcomes() const
      -> std::vector<std::pair<Action, double>> override;

  // Game-specific methods
  [[nodiscard]] auto Factories() const -> const std::vector<Factory>& {
    return factories_;
  }
  [[nodiscard]] auto CenterPile() const -> const Factory& {
    return center_pile_;
  }
  [[nodiscard]] auto PlayerBoards() const -> const std::vector<PlayerBoard>& {
    return player_boards_;
  }
  [[nodiscard]] auto GetPlayerBoard(Player player) const -> const PlayerBoard& {
    return player_boards_[player];
  }
  [[nodiscard]] auto HasFirstPlayerTile() const -> bool {
    return first_player_tile_available_;
  }

  // Public for testing and JSON reconstruction
  std::vector<Factory> factories_;
  Factory center_pile_;
  int num_players_;
  std::vector<PlayerBoard> player_boards_;
  bool game_ended_{false};

  // Game state - made public for JSON reconstruction
  Player current_player_;
  std::vector<TileColor> bag_;
  std::vector<TileColor> discard_pile_;
  bool first_player_tile_available_;
  Player first_player_next_round_;
  int round_number_;
  bool needs_bag_shuffle_;

  void EndRoundScoring();
  [[nodiscard]] auto CalculateScore(Player player) const -> int;

  // Decode action - made public for JSON bridge
  struct DecodedAction {
    bool from_center;
    int factory_id;
    TileColor color;
    int destination;  // Pattern line (0-4) or -1 for floor
  };

  [[nodiscard]] auto DecodeAction(Action action) const -> DecodedAction;
  [[nodiscard]] auto EncodeAction(bool from_center, int factory_id,
                                  TileColor color, int destination) const
      -> Action;

 protected:
  void DoApplyAction(Action action) override;

 private:
  void SetupNewRound();
  void FillFactories();
  [[nodiscard]] auto IsWallComplete(Player player) const -> bool;
  [[nodiscard]] auto GetNumFactories() const -> int {
    return (2 * num_players_) + 1;
  }
};

// Game object.
class AzulGame : public Game {
 public:
  explicit AzulGame(const GameParameters& params);

  [[nodiscard]] auto NumDistinctActions() const -> int override {
    return kNumDistinctActions;
  }
  [[nodiscard]] auto NewInitialState() const
      -> std::unique_ptr<State> override {
    return std::make_unique<AzulState>(shared_from_this(), num_players_);
  }
  [[nodiscard]] auto NumPlayers() const -> int override { return num_players_; }
  [[nodiscard]] auto MinUtility() const -> double override {
    return -1;
  }  // Losing player utility
  [[nodiscard]] auto UtilitySum() const -> absl::optional<double> override {
    return 0.0;
  }  // Zero-sum
  [[nodiscard]] auto MaxUtility() const -> double override {
    return 1;
  }  // Winning player utility
  [[nodiscard]] auto ObservationTensorShape() const
      -> std::vector<int> override;
  [[nodiscard]] auto MaxGameLength() const -> int override {
    return kMaxGameLength;
  }
  [[nodiscard]] auto MaxChanceOutcomes() const -> int override {
    return kTotalTilesPerColor * kNumTileColors;
  }  // Max tiles that can be shuffled

  // RNG support for deterministic shuffling
  [[nodiscard]] auto GetRNG() const -> std::mt19937& { return rng_; }
  [[nodiscard]] auto GetOriginalSeed() const -> int { return original_seed_; }

 private:
  int num_players_;
  mutable std::mt19937 rng_;
  int original_seed_;
};

// Utility functions
[[nodiscard]] auto TileColorToString(TileColor color) -> std::string;
[[nodiscard]] auto StringToTileColor(const std::string& str) -> TileColor;
[[nodiscard]] auto ActionToString(Action action, int num_players)
    -> std::string;

// Stream operator for TileColor
inline auto operator<<(std::ostream& os, TileColor color) -> std::ostream& {
  return os << TileColorToString(color);
}

}  // namespace open_spiel::azul

#endif  // OPEN_SPIEL_GAMES_AZUL_H_