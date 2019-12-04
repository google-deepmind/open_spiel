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

#include "open_spiel/games/universal_poker/logic/game_tree.h"

#include <assert.h>

#include <sstream>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace universal_poker {
namespace logic {

GameNode::GameNode(acpc_cpp::ACPCGame* acpc_game)
    : BettingNode(acpc_game),
      deck_(/*num_suits=*/acpc_game->NumSuitsDeck(),
            /*num_ranks=*/acpc_game->NumRanksDeck()),
      action_count_(GetPossibleActionCount()),
      hole_cards_(acpc_game_->GetNbPlayers()) {
  SPIEL_CHECK_EQ(GetNodeType(), NODE_TYPE_CHANCE);
  action_count_ = deck_.ToCardArray().size();
}

void GameNode::ApplyAction(uint32_t actionIdx) {
  if (GetNodeType() == NODE_TYPE_CHANCE) {
    BettingNode::ApplyDealCards();
    uint8_t card = deck_.ToCardArray()[actionIdx];
    deck_.RemoveCard(card);

    // Check where to add this card
    for (uint8_t p = 0; p < acpc_game_->GetNbPlayers(); p++) {
      if (hole_cards_[p].CountCards() < acpc_game_->GetNbHoleCardsRequired()) {
        hole_cards_[p].AddCard(card);
        break;
      }
    }

    if (board_cards_.CountCards() <
        acpc_game_->GetNbBoardCardsRequired(GetRound())) {
      board_cards_.AddCard(card);
    }
  } else {
    uint32_t idx = 0;
    for (auto action : ALL_ACTIONS) {
      if (action & GetPossibleActionsMask()) {
        if (idx == actionIdx) {
          BettingNode::ApplyChoiceAction(action);
          break;
        }
        idx++;
      }
    }
  }

  if (GetNodeType() == NODE_TYPE_CHANCE) {
    action_count_ = deck_.CountCards();
  } else {
    action_count_ = GetPossibleActionCount();
  }
}

std::string GameNode::ToString() const {
  std::ostringstream buf;

  for (uint8_t p = 0; p < acpc_game_->GetNbPlayers(); p++) {
    buf << "P" << (int)p << " Cards: " << hole_cards_[p].ToString()
        << std::endl;
  }
  buf << "BoardCards " << board_cards_.ToString() << std::endl;

  if (GetNodeType() == NODE_TYPE_CHANCE) {
    buf << "PossibleCardsToDeal " << deck_.ToString();
  }
  if (GetNodeType() == NODE_TYPE_TERMINAL_FOLD ||
      GetNodeType() == NODE_TYPE_TERMINAL_SHOWDOWN) {
    for (uint8_t p = 0; p < acpc_game_->GetNbPlayers(); p++) {
      buf << "P" << (int)p << " Reward: " << GetTotalReward(p) << std::endl;
    }
  }
  buf << BettingNode::ToString();

  return buf.str();
}

const CardSet& GameNode::GetHoleCardsOfPlayer(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, hole_cards_.size());
  return hole_cards_[player];
}

double GameNode::GetTotalReward(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, acpc_game_->GetNbPlayers());
  // Copy Board Cards and Hole Cards
  uint8_t holeCards[10][3], boardCards[7], nbHoleCards[10];

  for (size_t p = 0; p < hole_cards_.size(); ++p) {
    auto cards = hole_cards_[p].ToCardArray();
    for (size_t c = 0; c < cards.size(); ++c) {
      holeCards[p][c] = cards[c];
    }
    nbHoleCards[p] = cards.size();
  }

  auto bc = board_cards_.ToCardArray();
  for (size_t c = 0; c < bc.size(); ++c) {
    boardCards[c] = bc[c];
  }

  SetHoleAndBoardCards(holeCards, boardCards, nbHoleCards,
                       /*nbBoardCards=*/bc.size());

  return ValueOfState(player);
}

}  // namespace logic
}  // namespace universal_poker
}  // namespace open_spiel
