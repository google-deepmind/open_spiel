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
//     The original game gives an option for the player to end the game once the
//     2048 tile is created. But this implementation goes on till no more moves
//     are available for the player or kMaxGameLength number of moves is reached

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

namespace open_spiel {
namespace twenty_forty_eight {

constexpr int kNumPlayers = 1;
constexpr int kRows = 4;
constexpr int kColumns = 4;

constexpr int kMaxGameLength = INT_MAX;
constexpr int kMaxScore = INT_MAX;

// The chance tiles that randomly appear on the board after each move
constexpr std::array<int, 2> kChanceTiles = {2, 4};
const int kNoCellAvailableAction = kRows * kColumns
    * kChanceTiles.size();

struct Coordinate {
  int row, column;
  Coordinate(int _row, int _column)
      : row(_row), column(_column) {}
};

struct ChanceAction {
  int row;
  int column;
  bool is_four;
  ChanceAction(int _row, int _column, bool _is_four)
      : row(_row),
        column(_column),
        is_four(_is_four) {}
};

struct Tile {
  int value;
  bool is_merged;
  Tile(int _value, bool _is_merged)
      : value(_value),
        is_merged(_is_merged) {}
};

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
  void UndoAction(Player player, Action action) override;
  std::vector<double> Rewards() const override;
  bool InBounds(int row, int column) const;
  std::vector<Action> LegalActions() const override;
  ActionsAndProbs ChanceOutcomes() const override;  
  
  // Methods below are outside core API
  void SetCustomBoard(const std::vector<int>& board_seq);
  ChanceAction SpielActionToChanceAction(Action action) const;
  Action ChanceActionToSpielAction(ChanceAction move) const;
  void SetBoard(int row, int column, Tile tile) {
    board_[row * kColumns + column] = tile;
  }
  void SetTileIsMerged(int row, int column, bool is_merged) {
    board_[row * kColumns + column].is_merged = is_merged;
  }
  Tile BoardAt(int row, int column) const {
    return board_[row * kColumns + column];
  }
  Tile BoardAt(Coordinate coordinate) const {
    return board_[coordinate.row * kColumns + coordinate.column];
  }
  int AvailableCellCount() const;
  std::array<std::vector<int>, 2> BuildTraversals(int direction) const;
  bool WithinBounds(int x, int y) const;
  bool CellAvailable(int x, int y) const;
  std::array<Coordinate, 2> 
      FindFarthestPosition(int x, int y, int direction) const;
  bool TileMatchesAvailable() const;
  bool Reached2048() const;
  void PrepareTiles();
  int GetCellContent(int x, int y) const;

 protected:
  void DoApplyAction(Action action) override;

 private:
  Player current_player_ = kChancePlayerId;
  std::vector<Tile> board_;
  bool extra_chance_turn_ = true;
  int total_score_ = 0;
  int action_score_ = 0;
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
  double MaxUtility() const override { return max_score_; }
  std::vector<int> ObservationTensorShape() const override {
    return {kRows, kColumns};
  }
  // There is arbitrarily chosen number to ensure the game is finite.
  int MaxGameLength() const override { return max_game_length_; }
  int MaxChanceOutcomes() const override { 
    return kRows * kColumns * kChanceTiles.size() + 1;
  }
 private:
  int max_game_length_;
  long max_score_;
};

}  // namespace twenty_forty_eight
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_2048_H_
