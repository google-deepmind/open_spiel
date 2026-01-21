
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

#ifndef OPEN_SPIEL_GAMES_CRAZYHOUSE_H_
#define OPEN_SPIEL_GAMES_CRAZYHOUSE_H_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/games/chess/chess.h"
#include "open_spiel/games/crazyhouse/crazyhouse_board.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace crazyhouse {

// Constants.
// Standard chess count (4674) + Drop moves (5 types * 64 squares = 320)
// 5 types: P, N, B, R, Q (King cannot be dropped)
inline constexpr int kNumDropActions = 5 * 64;
inline constexpr int NumDistinctActions() {
  return open_spiel::chess::NumDistinctActions() + kNumDropActions;
}

class CrazyhouseGame;

class CrazyhouseState : public State {
public:
  CrazyhouseState(std::shared_ptr<const Game> game);
  CrazyhouseState(std::shared_ptr<const Game> game, const std::string &fen);
  CrazyhouseState(const CrazyhouseState &) = default;
  CrazyhouseState &operator=(const CrazyhouseState &) = default;

  Player CurrentPlayer() const override;
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action) const override;
  std::string ToString() const override;

  bool IsTerminal() const override {
    return static_cast<bool>(MaybeFinalReturns());
  }

  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action action) override;

  CrazyhouseBoard &Board() { return current_board_; }
  const CrazyhouseBoard &Board() const { return current_board_; }

  std::string Serialize() const override;

protected:
  void DoApplyAction(Action action) override;

private:
  void MaybeGenerateLegalActions() const;
  absl::optional<std::vector<double>> MaybeFinalReturns() const;

  // We copy structure from ChessState since we can't inherit private members
  std::vector<chess::Move> moves_history_;
  CrazyhouseBoard start_board_;
  CrazyhouseBoard current_board_;
  std::string specific_initial_fen_;

  // Repetition handling
  class PassthroughHash {
  public:
    std::size_t operator()(uint64_t x) const {
      return static_cast<std::size_t>(x);
    }
  };
  using RepetitionTable = absl::flat_hash_map<uint64_t, int, PassthroughHash>;
  RepetitionTable repetitions_;
  mutable absl::optional<std::vector<Action>> cached_legal_actions_;
};

class CrazyhouseGame : public Game {
public:
  explicit CrazyhouseGame(const GameParameters &params);
  int NumDistinctActions() const override {
    return crazyhouse::NumDistinctActions();
  }
  std::unique_ptr<State>
  NewInitialState(const std::string &fen) const override {
    return std::make_unique<CrazyhouseState>(shared_from_this(), fen);
  }
  std::unique_ptr<State> NewInitialState() const override {
    return std::make_unique<CrazyhouseState>(shared_from_this());
  }

  int NumPlayers() const override { return 2; }
  double MinUtility() const override { return -1; }
  absl::optional<double> UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }

  // Observation Tensor needs to include pockets!
  std::vector<int> ObservationTensorShape() const override;

  int MaxGameLength() const override { return chess::MaxGameLength(); } // Same?

  std::unique_ptr<State>
  DeserializeState(const std::string &str) const override;
};

// Valid drop types: P, N, B, R, Q
const std::array<chess::PieceType, 5> &GetDropTypes();

} // namespace crazyhouse
} // namespace open_spiel

#endif // OPEN_SPIEL_GAMES_CRAZYHOUSE_H_
