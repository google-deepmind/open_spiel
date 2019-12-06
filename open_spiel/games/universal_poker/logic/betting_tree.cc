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

#include "open_spiel/games/universal_poker/logic/betting_tree.h"

#include <assert.h>

#include <sstream>

namespace open_spiel {
namespace universal_poker {
namespace logic {

const char* actions = "0df0c000p0000000a";

BettingNode::BettingNode(const acpc_cpp::ACPCGame* acpc_game)
    : ACPCState(acpc_game),
      acpc_game_(acpc_game),
      nodeType_(NODE_TYPE_CHANCE),
      possibleActions_(ACTION_DEAL),
      potSize_(0),
      allInSize_(0),
      nbHoleCardsDealtPerPlayer_{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      nbBoardCardsDealt_(0) {}

void BettingNode::ApplyChoiceAction(ActionType action_type) {
  assert(nodeType_ == NODE_TYPE_CHOICE);
  assert((possibleActions_ & action_type) > 0);

  actionSequence_ += (char)actions[action_type];
  switch (action_type) {
    case ACTION_FOLD:
      DoAction(ACPC_FOLD, 0);
      break;
    case ACTION_CHECK_CALL:
      DoAction(ACPC_CALL, 0);
      break;
    case ACTION_BET_POT:
      DoAction(ACPC_RAISE, potSize_);
      break;
    case ACTION_ALL_IN:
      DoAction(ACPC_RAISE, allInSize_);
      break;
    case ACTION_DEAL:
    default:
      assert(false);
      break;
  }

  _CalculateActionsAndNodeType();
}

void BettingNode::ApplyDealCards() {
  assert(nodeType_ == NODE_TYPE_CHANCE);
  actionSequence_ += 'd';

  for (uint8_t p = 0; p < acpc_game_->GetNbPlayers(); p++) {
    if (nbHoleCardsDealtPerPlayer_[p] < acpc_game_->GetNbHoleCardsRequired()) {
      nbHoleCardsDealtPerPlayer_[p]++;
      _CalculateActionsAndNodeType();
      return;
    }
  }

  if (nbBoardCardsDealt_ < acpc_game_->GetNbBoardCardsRequired(GetRound())) {
    nbBoardCardsDealt_++;
    _CalculateActionsAndNodeType();
    return;
  }

  assert(false);
}

void BettingNode::_CalculateActionsAndNodeType() {
  possibleActions_ = 0;

  if (ACPCState::IsFinished()) {
    if (NumFolded() >= acpc_game_->GetNbPlayers() - 1) {
      nodeType_ = NODE_TYPE_TERMINAL_FOLD;
    } else {
      if (nbBoardCardsDealt_ <
          acpc_game_->GetNbBoardCardsRequired(GetRound())) {
        nodeType_ = NODE_TYPE_CHANCE;
        possibleActions_ = ACTION_DEAL;
        return;
      }
      nodeType_ = NODE_TYPE_TERMINAL_SHOWDOWN;
    }

  } else {
    // Check for sth to deal
    for (int p = 0; p < acpc_game_->GetNbPlayers(); ++p) {
      if (nbHoleCardsDealtPerPlayer_[p] <
          acpc_game_->GetNbHoleCardsRequired()) {
        nodeType_ = NODE_TYPE_CHANCE;
        possibleActions_ = ACTION_DEAL;
        return;
      }
    }
    if (nbBoardCardsDealt_ < acpc_game_->GetNbBoardCardsRequired(GetRound())) {
      nodeType_ = NODE_TYPE_CHANCE;
      possibleActions_ = ACTION_DEAL;
      return;
    }

    // Check for CHOICE Actions
    nodeType_ = NODE_TYPE_CHOICE;
    if (IsValidAction(ACPC_FOLD, 0)) {
      possibleActions_ |= ACTION_FOLD;
    }
    if (IsValidAction(ACPC_CALL, 0)) {
      possibleActions_ |= ACTION_CHECK_CALL;
    }

    potSize_ = 0;
    allInSize_ = 0;

    if (RaiseIsValid(&potSize_, &allInSize_)) {
      if (acpc_game_->IsLimitGame()) {
        potSize_ = 0;
        possibleActions_ |= ACTION_BET_POT;
      } else {
        int32_t currentPot =
            MaxSpend() * (acpc_game_->GetNbPlayers() - NumFolded());
        potSize_ = currentPot > potSize_ ? currentPot : potSize_;
        potSize_ = allInSize_ < potSize_ ? allInSize_ : potSize_;

        possibleActions_ |= ACTION_BET_POT;
        if (allInSize_ > potSize_) {
          possibleActions_ |= ACTION_ALL_IN;
        }
      }
    }
  }
}

std::string BettingNode::ToString() const {
  std::ostringstream buf;
  buf << "NodeType: ";
  buf << (nodeType_ == NODE_TYPE_CHANCE ? "NODE_TYPE_CHANCE" : "");
  buf << (nodeType_ == NODE_TYPE_CHOICE ? "NODE_TYPE_CHOICE" : "");
  buf << (nodeType_ == NODE_TYPE_TERMINAL_SHOWDOWN
              ? "NODE_TYPE_TERMINAL_SHOWDOWN"
              : "");
  buf << (nodeType_ == NODE_TYPE_TERMINAL_FOLD ? "NODE_TYPE_TERMINAL_FOLD"
                                               : "");
  buf << std::endl;

  buf << "PossibleActions (" << GetPossibleActionCount() << "): [";
  for (auto action : ALL_ACTIONS) {
    if (action & possibleActions_) {
      buf << ((action == ACTION_ALL_IN) ? " ACTION_ALL_IN " : "");
      buf << ((action == ACTION_BET_POT) ? " ACTION_BET_POT " : "");
      buf << ((action == ACTION_CHECK_CALL) ? " ACTION_CHECK_CALL " : "");
      buf << ((action == ACTION_FOLD) ? " ACTION_FOLD " : "");
      buf << ((action == ACTION_DEAL) ? " ACTION_DEAL " : "");
    }
  }
  buf << "]" << std::endl;
  buf << "Round: " << GetRound() << std::endl;
  buf << "ACPC State: " << ACPCState::ToString() << std::endl;
  buf << "Action Sequence: " << actionSequence_ << std::endl;
  return buf.str();
}

int BettingNode::GetDepth() {
  int maxDepth = 0;
  for (auto action : ALL_ACTIONS) {
    if (action & possibleActions_) {
      BettingNode child(*this);
      if (child.GetNodeType() == NODE_TYPE_CHANCE) {
        child.ApplyDealCards();
      } else if (child.GetNodeType() == NODE_TYPE_CHOICE) {
        child.ApplyChoiceAction(action);
      }
      int depth = child.GetDepth();
      maxDepth = depth > maxDepth ? depth : maxDepth;
    }
  }

  return 1 + maxDepth;
}

const int BettingNode::GetPossibleActionCount() const {
  // _builtin_popcount(int) function is used to count the number of one's
  return __builtin_popcount(possibleActions_);
}

}  // namespace logic
}  // namespace universal_poker
}  // namespace open_spiel
