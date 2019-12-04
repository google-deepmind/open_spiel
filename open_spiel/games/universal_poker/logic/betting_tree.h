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

#ifndef OPEN_SPIEL_BETTING_TREE_H
#define OPEN_SPIEL_BETTING_TREE_H

#include <cstdint>
#include <vector>

#include "open_spiel/games/universal_poker/acpc_cpp/acpc_game.h"

namespace open_spiel {
namespace universal_poker {
namespace logic {

constexpr uint8_t MAX_PLAYERS = 10;

class BettingTree : public acpc_cpp::ACPCGame {
 public:
  BettingTree(const std::string& gameDef);
  uint32_t GetMaxBettingActions() const;
};

class BettingNode : public acpc_cpp::ACPCState {
  friend BettingTree;

 public:
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

  BettingNode(BettingTree* bettingTree);

  NodeType GetNodeType() const;

  const uint32_t& GetPossibleActionsMask() const;
  const int GetPossibleActionCount() const;

  void ApplyChoiceAction(ActionType actionType);
  virtual void ApplyDealCards();
  std::string ToString() const;
  int GetDepth();
  std::string GetActionSequence() const;
  bool IsFinished() const;

 private:
  const BettingTree* bettingTree_;
  NodeType nodeType_;
  uint32_t possibleActions_;
  int32_t potSize_;
  int32_t allInSize_;
  std::string actionSequence_;

  uint8_t nbHoleCardsDealtPerPlayer_[MAX_PLAYERS];
  uint8_t nbBoardCardsDealt_;

  void _CalculateActionsAndNodeType();
};

}  // namespace logic
}  // namespace universal_poker
}  // namespace open_spiel

#endif  // OPEN_SPIEL_BETTING_TREE_H
