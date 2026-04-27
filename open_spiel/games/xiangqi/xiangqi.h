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
// Minor change to trigger CI rerun

#ifndef OPEN_SPIEL_GAMES_XIANGQI_H_
#define OPEN_SPIEL_GAMES_XIANGQI_H_

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

// Chinese Chess (Xiangqi):
// https://en.wikipedia.org/wiki/Xiangqi
//
// Parameters: none

namespace open_spiel {
namespace xiangqi {

inline constexpr int kNumPlayers = 2;
inline constexpr int kNumRows = 10;
inline constexpr int kNumCols = 9;
inline constexpr int kNumCells = kNumRows * kNumCols;  // 90
inline constexpr int kNumPieceTypes = 7;
inline constexpr int kNumDistinctActions = kNumCells * kNumCells;  // 8100
inline constexpr int kMaxGameLength = 500;

// Observation tensor: 7 planes per player (one per piece type) + 1 current
// player plane = 15 planes total.
inline constexpr int kNumObservationPlanes = kNumPieceTypes * 2 + 1;  // 15

enum PieceType {
  kEmpty = 0,
  kGeneral,
  kAdvisor,
  kElephant,
  kHorse,
  kChariot,
  kCannon,
  kSoldier,
};

struct Piece {
  PieceType type = kEmpty;
  int player = -1;  // 0 = Red, 1 = Black, -1 = no piece

  bool IsEmpty() const { return type == kEmpty; }
  bool operator==(const Piece& other) const {
    return type == other.type && player == other.player;
  }
  bool operator!=(const Piece& other) const { return !(*this == other); }
};

inline constexpr Piece kEmptyPiece = {kEmpty, -1};

inline Action EncodeMove(int from, int to) {
  return static_cast<Action>(from) * kNumCells + to;
}

inline std::pair<int, int> DecodeMove(Action action) {
  return {static_cast<int>(action / kNumCells),
          static_cast<int>(action % kNumCells)};
}

inline int SquareIndex(int row, int col) { return row * kNumCols + col; }
inline int SquareRow(int sq) { return sq / kNumCols; }
inline int SquareCol(int sq) { return sq % kNumCols; }

bool IsInPalace(int row, int col, int player);
bool IsOnOwnSide(int row, int player);

struct MoveHistoryEntry {
  int from;
  int to;
  Piece captured;
};

class XiangqiState : public State {
 public:
  explicit XiangqiState(std::shared_ptr<const Game> game);
  XiangqiState(const XiangqiState&) = default;
  XiangqiState& operator=(const XiangqiState&) = default;

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
  void UndoAction(Player player, Action move) override;
  std::vector<Action> LegalActions() const override;

  const std::array<Piece, kNumCells>& Board() const { return board_; }
  Piece BoardAt(int sq) const { return board_[sq]; }
  Piece BoardAt(int row, int col) const {
    return board_[SquareIndex(row, col)];
  }

 protected:
  void DoApplyAction(Action action) override;

 private:
  void SetupInitialBoard();
  void GeneratePseudoLegalMoves(std::vector<Action>* moves) const;
  void GenerateGeneralMoves(int sq, std::vector<Action>* moves) const;
  void GenerateAdvisorMoves(int sq, std::vector<Action>* moves) const;
  void GenerateElephantMoves(int sq, std::vector<Action>* moves) const;
  void GenerateHorseMoves(int sq, std::vector<Action>* moves) const;
  void GenerateChariotMoves(int sq, std::vector<Action>* moves) const;
  void GenerateCannonMoves(int sq, std::vector<Action>* moves) const;
  void GenerateSoldierMoves(int sq, std::vector<Action>* moves) const;

  bool IsInCheck(int player) const;
  bool WouldLeaveInCheck(int from, int to) const;
  bool ViolatesFlyingGeneral() const;
  int FindGeneral(int player) const;
  bool IsAttackedBy(int sq, int attacker) const;

  Player outcome_ = kInvalidPlayer;
  bool no_legal_moves_ = false;

  std::array<Piece, kNumCells> board_;
  Player current_player_ = 0;  // Red goes first
  std::vector<MoveHistoryEntry> undo_stack_;
};

class XiangqiGame : public Game {
 public:
  explicit XiangqiGame(const GameParameters& params);
  int NumDistinctActions() const override { return kNumDistinctActions; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new XiangqiState(shared_from_this()));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  absl::optional<double> UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override {
    return {kNumObservationPlanes, kNumRows, kNumCols};
  }
  int MaxGameLength() const override { return kMaxGameLength; }
};

char PieceTypeToChar(PieceType type);

}  // namespace xiangqi
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_XIANGQI_H_
