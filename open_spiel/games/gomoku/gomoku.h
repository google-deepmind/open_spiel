// Copyright 2026 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_GAMES_GOMOKU_H_
#define OPEN_SPIEL_GAMES_GOMOKU_H_

#include <array>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/spiel.h"
#include "open_spiel/games/gomoku/gomoku_grid.h"

// Simple game of Noughts and Crosses:
// https://en.wikipedia.org/wiki/Tic-tac-toe
//
// Parameters: none

namespace open_spiel {
namespace gomoku {

inline constexpr int kDefaultSize = 15;
inline constexpr int kDefaultDims = 2;
inline constexpr int kDefaultConnect = 5;
inline constexpr bool kDefaultWrap= false;
inline constexpr int kNumPlayers = 2;
inline constexpr int kBlackPlayer = 0;
inline constexpr int kWhitePlayer = 1;


// 
enum class Stone {
	kEmpty,
	kBlack,
	kWhite,
};


// State of an in-play game.
class GomokuState : public State {
 public:
	explicit GomokuState(std::shared_ptr<const Game> game,
                     const std::string& state_str = "");

  GomokuState(const GomokuState&) = default;
  GomokuState& operator=(const GomokuState&) = default;
  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : current_player_;
  }

	static Player Opponent(Player player) {
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, kNumPlayers);
		return player - 1;
	}

  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
	bool IsTerminal() const override {
    return terminal_;
  }
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action move) override;
  std::vector<Action> LegalActions() const override;

 protected:
  void DoApplyAction(Action move) override;

 private:
  void CheckWinFromLastMove(Action move);
  Player current_player_ = 0;   // Player zero goes first by default
  int move_count_ = 0;
  Grid<Stone> board_;
	int size_;
	int dims_;
	int connect_;
	bool wrap_;
	int initial_stones_;
	float black_score_;
	float white_score_;
	bool terminal_ =  false;
};

// Game object.
class GomokuGame : public Game {
 public:
  explicit GomokuGame(const GameParameters& params);
  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new GomokuState(shared_from_this()));
  }
  std::unique_ptr<State> NewInitialState(
      const nlohmann::json& json) const {
    return std::unique_ptr<State>(new GomokuState(shared_from_this(), json));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  absl::optional<double> UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override;
  std::string ActionToString(Player player, Action action_id) const override;
	std::vector<int> ActionToMove(Action action) const;
  Action MoveToAction(const std::vector<int>& move) const;

	int Size() const { return size_; }
	int Dims() const { return dims_; }
	int Connect() const { return connect_; }
	bool Wrap() const { return wrap_; }

 private:
	int size_;
	int dims_;
	int connect_;
	int wrap_;
	int total_size_;
	std::vector<std::size_t> strides_;
	std::vector<int> UnflattenAction(Action action_id) const;
};

}  // namespace gomoku
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_GOMOKU_H_
