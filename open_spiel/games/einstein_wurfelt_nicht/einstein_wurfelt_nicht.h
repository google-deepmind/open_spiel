// Copyright 2024 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_GAMES_EINSTEIN_WURFELT_NICHT_H_
#define OPEN_SPIEL_GAMES_EINSTEIN_WURFELT_NICHT_H_

#include <time.h>

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// An implementation of the game EinStein würfelt nicht!
// This is the implementation of the basic game with a 5x5 board and 6 cubes
// per player.
// https://en.wikipedia.org/wiki/EinStein_w%C3%BCrfelt_nicht!

namespace open_spiel {
namespace einstein_wurfelt_nicht {

enum class Color : int8_t { kBlack = 0, kWhite = 1, kEmpty = 2 };

struct Cube {
  Color color;
  int value;  // player's die value
};

inline constexpr int kNumPlayers = 2;
inline constexpr int kBlackPlayerId = 0;
inline constexpr int kWhitePlayerId = 1;
inline constexpr int kNumPlayerCubes = 6;
// 720 possible permutations of 6 cubes on the board
inline constexpr int kNumCubesPermutations = 720;
inline constexpr int kDefaultRows = 5;
inline constexpr int kDefaultColumns = 5;
inline constexpr int k2dMaxBoardSize = kDefaultRows * kDefaultColumns;
inline constexpr const int kStateEncodingSize =
    kNumPlayers * kNumPlayerCubes * kDefaultRows * kDefaultColumns;

// This is a small helper to track historical turn info not stored in the moves.
// It is only needed for proper implementation of Undo.
struct TurnHistoryInfo {
  int player;
  int prev_player;
  int die_roll_;
  Action action;
  Cube captured_cube;
  TurnHistoryInfo(int _player, int _prev_player, int _die_roll, int _action,
                  Cube _captured_cube)
      : player(_player),
        prev_player(_prev_player),
        die_roll_(_die_roll),
        action(_action),
        captured_cube(_captured_cube) {}
};

class EinsteinWurfeltNichtState : public State {
 public:
  explicit EinsteinWurfeltNichtState(std::shared_ptr<const Game> game, int rows,
                                     int cols);
  Player CurrentPlayer() const override;
  // Returns the opponent of the specified player.
  int Opponent(int player) const;
  std::vector<std::vector<int>> AvailableCubesPosition(Color color) const;
  std::string ActionToString(Player player, Action action) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action action) override;

  bool InBounds(int r, int c) const;
  void SetBoard(int r, int c, Cube cube) { board_[r * cols_ + c] = cube; }
  Cube board(int row, int col) const { return board_[row * cols_ + col]; }
  std::vector<Action> LegalActions() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  void SetState(int cur_player, int die_roll,
                const std::array<Cube, k2dMaxBoardSize> board, int cubes_black,
                int cubes_white);

 protected:
  void DoApplyAction(Action action) override;

 private:
  void SetupInitialBoard(Player player, Action action);

  Player cur_player_ = kInvalidPlayer;
  Player prev_player_ = kInvalidPlayer;
  int winner_ = kInvalidPlayer;
  int total_moves_ = -1;
  int turns_ = -1;
  std::array<int, 2> cubes_;
  int rows_ = -1;
  int cols_ = -1;
  int die_roll_ = 0;
  std::array<Cube,
             k2dMaxBoardSize>
      board_;  // for (row,col) we use row*cols_+col
  std::vector<TurnHistoryInfo> turn_history_info_;
};

class EinsteinWurfeltNichtGame : public Game {
 public:
  explicit EinsteinWurfeltNichtGame(const GameParameters& params);
  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(
        new EinsteinWurfeltNichtState(shared_from_this(), rows_, cols_));
  }

  int MaxChanceOutcomes() const override { return kNumCubesPermutations; }

  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  absl::optional<double> UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override {
    return {kStateEncodingSize};
  }

  // Assuming that each cube is moved first along the horizontal axis and then
  // along the vertical axis, which is the maximum number of moves for a cube
  // (only the cubes in the corners). This accounts for (row-1) * (cols-1)
  // moves. If we assume that each player makes all these moves we get
  // (row-1) * (cols-1) * num_players. If we consider the chance player as
  // the third player which makes the same number of moves, the upper bound
  // for the number of moves is (row-1) * (cols-1) * (num_players + 1).
  int MaxGameLength() const override {
    return (kDefaultRows - 1) * (kDefaultColumns - 1) * (kNumPlayerCubes + 1);
  }

 private:
  int rows_ = -1;
  int cols_ = -1;
};

}  // namespace einstein_wurfelt_nicht
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_EINSTEIN_WURFELT_NICHT_H_
