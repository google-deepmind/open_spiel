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

#ifndef OPEN_SPIEL_GAMES_KAYLES_KAYLES_H_
#define OPEN_SPIEL_GAMES_KAYLES_KAYLES_H_

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace kayles {

inline constexpr int kNumPlayers = 2;
inline constexpr int kDefaultRowLength = 10;

class KaylesState : public State {
 public:
  KaylesState(std::shared_ptr<const Game> game, int row_length);
  KaylesState(const KaylesState&) = default;
  KaylesState& operator=(const KaylesState&) = default;

  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : current_player_;
  }
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
  std::vector<Action> LegalActions() const override;

 protected:
  void DoApplyAction(Action action) override;

 private:
  int row_length_;
  std::vector<bool> pins_;
  Player current_player_ = 0;
};

class KaylesGame : public Game {
 public:
  explicit KaylesGame(const GameParameters& params);
  int NumDistinctActions() const override { return 2 * row_length_ - 1; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(
        new KaylesState(shared_from_this(), row_length_));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  absl::optional<double> UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override {
    return {2, row_length_};
  }
  int MaxGameLength() const override { return row_length_; }

 private:
  int row_length_;
};

}  // namespace kayles
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_KAYLES_KAYLES_H_
