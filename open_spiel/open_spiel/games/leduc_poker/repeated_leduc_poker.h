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

#ifndef OPEN_SPIEL_GAMES_LEDUC_POKER_REPEATED_LEDUC_POKER_H_
#define OPEN_SPIEL_GAMES_LEDUC_POKER_REPEATED_LEDUC_POKER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/leduc_poker/leduc_poker.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace leduc_poker {

class RepeatedLeducPokerGame;

inline constexpr int kDefaultNumHands = 2;
inline constexpr Action kContinueAction = 3;

class RepeatedLeducPokerState : public State {
 public:
  RepeatedLeducPokerState(std::shared_ptr<const Game> game,
                          GameParameters leduc_poker_game_params,
                          int num_hands);
  RepeatedLeducPokerState(const RepeatedLeducPokerState& other);

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::vector<double> Rewards() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;

  int HandNumber() const { return hand_number_; }
  const LeducState* GetLeducState() const { return leduc_state_.get(); }

 protected:
  void DoApplyAction(Action action) override;

 private:
  void UpdateStacks();
  void UpdateLeducPoker();
  void GoToBetweenHandsState();
  void StartNewHand();

  GameParameters leduc_poker_game_params_;
  std::unique_ptr<LeducState> leduc_state_;

  int hand_number_ = 0;
  const int num_hands_;
  bool is_terminal_ = false;
  std::vector<double> stacks_;
  std::vector<std::vector<double>> hand_returns_;
  bool between_hands_ = false;
  int num_players_acted_this_turn_ = 0;
};

class RepeatedLeducPokerGame : public Game {
 public:
  RepeatedLeducPokerGame(const GameParameters& params);

  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override;
  int NumPlayers() const override;
  double MinUtility() const override;
  double MaxUtility() const override;
  absl::optional<double> UtilitySum() const override { return 0; }
  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  int MaxGameLength() const override;
  int MaxChanceOutcomes() const override;

 private:
  GameParameters leduc_poker_game_params_;
  std::shared_ptr<const Game> base_game_;
  const int num_hands_;
  const int num_players_;
};

}  // namespace leduc_poker
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_LEDUC_POKER_REPEATED_LEDUC_POKER_H_
