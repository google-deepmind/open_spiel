// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_OPEN_SPIEL_GAMES_UNIVERSAL_POKER_H_
#define THIRD_PARTY_OPEN_SPIEL_GAMES_UNIVERSAL_POKER_H_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/games/universal_poker/acpc_cpp/acpc_game.h"
#include "open_spiel/games/universal_poker/logic/card_set.h"
#include "open_spiel/spiel.h"

// This is a wrapper around the Annual Computer Poker Competition bot (ACPC)
// environment. See http://www.computerpokercompetition.org/. The code is
// initially available at https://github.com/ethansbrown/acpc
// It is an optional dependency (see install.md for documentation and
// open_spiel/scripts/global_variables.sh to enable this).
//
// It has not been extensively reviewed/tested by the DeepMind OpenSpiel team.
namespace open_spiel {
namespace universal_poker {

class UniversalPokerGame;

constexpr uint8_t kMaxUniversalPokerPlayers = 10;

enum ActionType { kFold = 0, kCall = 1, kRaise = 2 };

class UniversalPokerState : public State {
 public:
  explicit UniversalPokerState(std::shared_ptr<const Game> game);

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action move) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void InformationStateTensor(Player player,
                              std::vector<double> *values) const override;
  void ObservationTensor(Player player,
                         std::vector<double> *values) const override;
  std::unique_ptr<State> Clone() const override;

  // The probability of taking each possible action in a particular info state.
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::vector<Action> LegalActions() const override;

 protected:
  void DoApplyAction(Action action_id) override;

  enum NodeType {
    NODE_TYPE_CHANCE,
    NODE_TYPE_CHOICE,
    NODE_TYPE_TERMINAL_FOLD,
    NODE_TYPE_TERMINAL_SHOWDOWN
  };
  enum ActionType {
    ACTION_DEAL = 1,
    ACTION_FOLD = 2,
    ACTION_CHECK_CALL = 4,
    ACTION_BET_POT = 8,
    ACTION_ALL_IN = 16
  };
  static constexpr ActionType ALL_ACTIONS[5] = {ACTION_DEAL, ACTION_FOLD,
                                                ACTION_CHECK_CALL,
                                                ACTION_BET_POT, ACTION_ALL_IN};

 public:
  const acpc_cpp::ACPCGame *acpc_game_;
  acpc_cpp::ACPCState acpc_state_;
  logic::CardSet deck_;  // The remaining cards to deal.
  // The cards already owned by each player
  std::vector<logic::CardSet> hole_cards_;
  logic::CardSet board_cards_;

  NodeType nodeType_;
  uint32_t possibleActions_;
  int32_t potSize_ = 0;
  int32_t allInSize_ = 0;
  std::string actionSequence_;

  uint8_t nbHoleCardsDealtPerPlayer_[kMaxUniversalPokerPlayers] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  uint8_t nbBoardCardsDealt_ = 0;

  void _CalculateActionsAndNodeType();

  double GetTotalReward(Player player) const;

  const uint32_t &GetPossibleActionsMask() const { return possibleActions_; }
  const int GetPossibleActionCount() const;

  void ApplyChoiceAction(ActionType action_type);
  virtual void ApplyDealCards();
  int GetDepth();
  std::string GetActionSequence() const { return actionSequence_; }
};

class UniversalPokerGame : public Game {
 public:
  explicit UniversalPokerGame(const GameParameters &params);

  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override;
  int NumPlayers() const override;
  double MinUtility() const override;
  double MaxUtility() const override;
  int MaxChanceOutcomes() const override;
  double UtilitySum() const override { return 0; }
  std::shared_ptr<const Game> Clone() const override;
  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override;

 private:
  std::string gameDesc_;
  const acpc_cpp::ACPCGame acpc_game_;

 public:
  const acpc_cpp::ACPCGame *GetACPCGame() const { return &acpc_game_; }

  std::string parseParameters(const GameParameters &map);
};

}  // namespace universal_poker
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_GAMES_UNIVERSAL_POKER_H_
