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

#ifndef OPEN_SPIEL_GAMES_QUARTO_H_
#define OPEN_SPIEL_GAMES_QUARTO_H_

#include <array>
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/spiel.h"

// Quarto is a two-player board game played with 16 unique pieces. Each piece
// has four binary attributes. On a turn, a player places the piece chosen by
// their opponent, then chooses the piece that the opponent must place. The
// player who completes a row, column, or diagonal whose pieces share any
// attribute wins.
//
// Parameters: none.

namespace open_spiel {
namespace quarto {

inline constexpr int kNumPlayers = 2;
inline constexpr int kNumRows = 4;
inline constexpr int kNumCols = 4;
inline constexpr int kNumCells = kNumRows * kNumCols;
inline constexpr int kNumPieces = 16;
inline constexpr int kNumAttributes = 4;
inline constexpr int kEmptyCell = -1;
inline constexpr int kNoPiece = -1;

enum class Phase {
  kSelect,
  kPlace,
  kTerminal,
};

inline std::ostream& operator<<(std::ostream& stream, Phase phase) {
  return stream << static_cast<int>(phase);
}

struct QuartoStructContents {
  std::string current_player;
  std::string phase;
  std::vector<int> board;
  int selected_piece;
  std::string outcome;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(QuartoStructContents, current_player, phase,
                                 board, selected_piece, outcome);
};

SPIEL_DEFINE_STRUCT(QuartoStateStruct, StateStruct, QuartoStructContents);
SPIEL_DEFINE_STRUCT(QuartoObservationStruct, ObservationStruct,
                    QuartoStructContents);

struct QuartoActionStruct : public ActionStruct {
  std::string action_type;
  int piece;
  int row;
  int col;
  SPIEL_STRUCT_BOILERPLATE(QuartoActionStruct, action_type, piece, row, col);
};

class QuartoState : public State {
 public:
  explicit QuartoState(std::shared_ptr<const Game> game);
  QuartoState(std::shared_ptr<const Game> game,
              const QuartoStateStruct& state_struct);

  QuartoState(const QuartoState&) = default;
  QuartoState& operator=(const QuartoState&) = default;

  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : current_player_;
  }
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override { return phase_ == Phase::kTerminal; }
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action action) override;
  std::vector<Action> LegalActions() const override;

  std::unique_ptr<StateStruct> ToStruct() const override;
  std::unique_ptr<ObservationStruct> ToObservationStruct(
      Player player) const override;
  std::unique_ptr<ActionStruct> ActionToStruct(Player player,
                                               Action action_id) const override;
  std::vector<Action> StructToActions(
      const ActionStruct& action_struct) const override;

  int BoardAt(int row, int col) const { return board_[row * kNumCols + col]; }
  int SelectedPiece() const { return selected_piece_; }
  Phase CurrentPhase() const { return phase_; }
  Player Outcome() const { return outcome_; }

 protected:
  void DoApplyAction(Action action) override;

 private:
  bool HasQuarto() const;
  bool IsPieceUsed(int piece) const {
    return (used_pieces_ & (uint16_t{1} << piece)) != 0;
  }

  std::array<int, kNumCells> board_;
  uint16_t used_pieces_ = 0;
  int selected_piece_ = kNoPiece;
  int num_placements_ = 0;
  Player current_player_ = 0;
  Player outcome_ = kInvalidPlayer;
  Phase phase_ = Phase::kSelect;
};

class QuartoGame : public Game {
 public:
  explicit QuartoGame(const GameParameters& params);
  int NumDistinctActions() const override { return kNumPieces; }
  using Game::NewInitialState;
  std::unique_ptr<State> NewInitialState() const override {
    return std::make_unique<QuartoState>(shared_from_this());
  }
  std::unique_ptr<State> NewInitialState(
      const QuartoStateStruct& state_struct) const {
    return std::make_unique<QuartoState>(shared_from_this(), state_struct);
  }
  std::unique_ptr<State> NewInitialState(
      const nlohmann::json& json) const override {
    return NewInitialState(QuartoStateStruct(json));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  absl::optional<double> UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override {
    return {kNumPieces, kNumCells + 1};
  }
  int MaxGameLength() const override { return 2 * kNumPieces; }
  std::string ActionToString(Player player, Action action_id) const override;
};

bool LineHasQuarto(const std::array<int, 4>& pieces);
bool BoardHasQuarto(const std::array<int, kNumCells>& board);

}  // namespace quarto
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_QUARTO_H_
