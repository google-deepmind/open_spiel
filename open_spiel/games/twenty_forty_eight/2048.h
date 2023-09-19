// Copyright 2022 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_GAMES_2048_H_
#define OPEN_SPIEL_GAMES_2048_H_

// Implementation of the popular game 2048.
// https://en.wikipedia.org/wiki/2048_(video_game)
// https://github.com/gabrielecirulli/2048
//
// The objective of the game is to slide numbered tiles on a grid to combine
// them to create bigger tiles.
//
// Some notes about this implementation:
// - End condition:
//     The game ends when a player has no more valid actions, or a maximum tile
//     value is reached (default: 2048).
//
// Parameters:
//   max_tile            int     End the game when max_tile is reached?
//                               (default: 2048)

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

namespace open_spiel {
namespace twenty_forty_eight {

enum Move { kMoveUp = 0, kMoveRight = 1, kMoveDown = 2, kMoveLeft = 3 };

constexpr int kNumPlayers = 1;
constexpr int kRows = 4;
constexpr int kColumns = 4;

constexpr int kDefaultMaxTile = 2048;

// The chance tiles that randomly appear on the board after each move
constexpr std::array<int, 2> kChanceTiles = {2, 4};
const int kNoCellAvailableAction = kRows * kColumns * kChanceTiles.size();

struct Coordinate {
  int row, column;
  constexpr Coordinate(int _row, int _column) : row(_row), column(_column) {}
};

struct ChanceAction {
  int row;
  int column;
  bool is_four;
  ChanceAction(int _row, int _column, bool _is_four)
      : row(_row), column(_column), is_four(_is_four) {}
};

struct Tile {
  int value;
  bool is_merged;
  Tile() : value(0), is_merged(false) {}
  Tile(int _value, bool _is_merged) : value(_value), is_merged(_is_merged) {}
};

class TwentyFortyEightGame;  // Needed for back-pointer to parent game.

// State of an in-play game.
class TwentyFortyEightState : public State {
 public:
  explicit TwentyFortyEightState(std::shared_ptr<const Game> game);
  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : current_player_;
  }
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override {
    return std::unique_ptr<State>(new TwentyFortyEightState(*this));
  }
  std::vector<double> Rewards() const override;
  std::vector<Action> LegalActions() const override;
  ActionsAndProbs ChanceOutcomes() const override;

  // Game-specific methods outside the core API:
  Tile BoardAt(int row, int column) const {
    return board_[row * kColumns + column];
  }
  Tile BoardAt(Coordinate coordinate) const {
    return board_[coordinate.row * kColumns + coordinate.column];
  }
  void SetCustomBoard(const std::vector<int>& board_seq);

 protected:
  void DoApplyAction(Action action) override;

 private:
  ChanceAction SpielActionToChanceAction(Action action) const;
  Action ChanceActionToSpielAction(ChanceAction move) const;
  void SetBoard(int row, int column, Tile tile) {
    board_[row * kColumns + column] = tile;
  }
  void SetTileIsMerged(int row, int column, bool is_merged) {
    board_[row * kColumns + column].is_merged = is_merged;
  }
  int AvailableCellCount() const;
  bool CellAvailable(int r, int c) const;
  std::array<Coordinate, 2> FindFarthestPosition(int r, int c,
                                                 int direction) const;
  bool TileMatchAvailable(int r, int c) const;
  bool TileMatchesAvailable() const;
  void PrepareTiles();
  int GetCellContent(int r, int c) const;
  bool DoesActionChangeBoard(Action action) const;

  const TwentyFortyEightGame& parent_game_;
  Player current_player_ = kChancePlayerId;
  std::vector<Tile> board_;
  bool extra_chance_turn_ = true;
  int total_score_ = 0;
  int action_score_ = 0;
  int total_actions_ = 0;
};

// Game object.
class TwentyFortyEightGame : public Game {
 public:
  explicit TwentyFortyEightGame(const GameParameters& params);
  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override {
    return absl::make_unique<TwentyFortyEightState>(shared_from_this());
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return 0; }

  std::vector<int> ObservationTensorShape() const override {
    return {kRows, kColumns};
  }
  int MaxChanceOutcomes() const override {
    return kRows * kColumns * kChanceTiles.size() + 1;
  }

  // Using analysis here to derive these bounds:
  // https://www.reddit.com/r/2048/comments/214njx/highest_possible_score_for_2048_warning_math/
  double MaxUtility() const override {
    return (std::log2(max_tile_) - 1) * max_tile_;
  }
  // First 2 is for the chance actions, second 2 for all the action required
  // to get the max tile.
  int MaxGameLength() const override { return 2 * 2 * max_tile_; }

  const int max_tile() const { return max_tile_; }

 private:
  const int max_tile_;
};

}  // namespace twenty_forty_eight
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_2048_H_
