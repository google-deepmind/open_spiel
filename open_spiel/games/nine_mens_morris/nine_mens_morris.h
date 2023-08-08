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

#ifndef OPEN_SPIEL_NINE_MENS_MORRIS_H_
#define OPEN_SPIEL_NINE_MENS_MORRIS_H_

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// Nine men's morris:
// https://en.m.wikipedia.org/wiki/Nine_men%27s_morris
//
// Parameters: none

namespace open_spiel {
namespace nine_mens_morris {

// Constants.
inline constexpr int kNumPlayers = 2;
inline constexpr int kNumMen = 9;
inline constexpr int kNumPoints = 24;  // A point is a place on the board.
inline constexpr int kCellStates = 1 + kNumPlayers;  // empty, 'x', and 'o'.
inline constexpr int kMaxNumTurns = 200;
inline constexpr int kObservationSize = 7;

// State of a cell.
enum class CellState {
  kEmpty,
  kWhite,  // W
  kBlack,  // B
};

using Mill = std::array<int, 3>;

// State of an in-play game.
class NineMensMorrisState : public State {
 public:
  NineMensMorrisState(std::shared_ptr<const Game> game);

  NineMensMorrisState(const NineMensMorrisState&) = default;
  NineMensMorrisState& operator=(const NineMensMorrisState&) = default;

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
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions() const override;

  // Extra methods not part of the core API.
  CellState BoardAt(int cell) const { return board_[cell]; }
  Player outcome() const { return outcome_; }

 protected:
  std::array<CellState, kNumPoints> board_;
  void DoApplyAction(Action move) override;

 private:
  Player current_player_ = 0;  // Player zero goes first
  Player outcome_ = kInvalidPlayer;
  int num_turns_ = 0;
  bool capture_ = false;
  std::array<int, 2> men_to_deploy_ = {kNumMen, kNumMen};
  std::array<int, 2> num_men_ = {kNumMen, kNumMen};
  std::vector<Action> cur_legal_actions_;

  void GetCurrentLegalActions();
  bool CheckInMill(int pos) const;
  bool CheckAllMills(Player player) const;
};

// Game object.
class NineMensMorrisGame : public Game {
 public:
  explicit NineMensMorrisGame(const GameParameters& params);
  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new NineMensMorrisState(shared_from_this()));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  absl::optional<double> UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override {
    return {kCellStates + 2, kObservationSize, kObservationSize};
  }
  int MaxGameLength() const override { return kMaxNumTurns + 2 * kNumMen - 4; }
  std::string ActionToString(Player player, Action action_id) const override;
};

CellState PlayerToState(Player player);
char StateToChar(CellState state);
const char* PlayerToStr(Player player);
Player StateToPlayer(CellState state);
Action ToMoveAction(int source, int dest);
void FromMoveAction(Action action, int* source, int* dest);

}  // namespace nine_mens_morris
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_NINE_MENS_MORRIS_H_
