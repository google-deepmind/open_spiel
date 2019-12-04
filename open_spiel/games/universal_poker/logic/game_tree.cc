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

namespace open_spiel::universal_poker::logic {
GameTree::GameNode::GameNode(logic::GameTree* gameTree)
    : BettingNode(gameTree),
      gameTree_(gameTree),
      deck_(gameTree->NumSuitsDeck(), gameTree->NumRanksDeck()),
      actionCount_(GetPossibleActionCount()) {
  for (uint8_t p = 0; p < gameTree_->GetNbPlayers(); p++) {
    holeCards_.emplace_back();
  }

  if (GetNodeType() == NODE_TYPE_CHANCE) {
    actionCount_ = deck_.ToCardArray().size();
  }
}

void GameTree::GameNode::ApplyAction(uint32_t actionIdx) {
  if (GetNodeType() == NODE_TYPE_CHANCE) {
    BettingNode::ApplyDealCards();
    uint8_t card = deck_.ToCardArray()[actionIdx];
    deck_.RemoveCard(card);

    // Check where to add this card
    for (uint8_t p = 0; p < gameTree_->GetNbPlayers(); p++) {
      if (holeCards_[p].CountCards() < gameTree_->GetNbHoleCardsRequired()) {
        holeCards_[p].AddCard(card);
        break;
      }
    }

    if (boardCards_.CountCards() <
        gameTree_->GetNbBoardCardsRequired(GetRound())) {
      boardCards_.AddCard(card);
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
    actionCount_ = deck_.CountCards();
  } else {
    actionCount_ = GetPossibleActionCount();
  }
}

uint32_t GameTree::GameNode::GetActionCount() const { return actionCount_; }

std::string GameTree::GameNode::ToString() const {
  std::ostringstream buf;

  for (uint8_t p = 0; p < gameTree_->GetNbPlayers(); p++) {
    buf << "P" << (int)p << " Cards: " << holeCards_[p].ToString() << std::endl;
  }
  buf << "BoardCards " << boardCards_.ToString() << std::endl;

  if (GetNodeType() == NODE_TYPE_CHANCE) {
    buf << "PossibleCardsToDeal " << deck_.ToString();
  }
  if (GetNodeType() == NODE_TYPE_TERMINAL_FOLD ||
      GetNodeType() == NODE_TYPE_TERMINAL_SHOWDOWN) {
    for (uint8_t p = 0; p < gameTree_->GetNbPlayers(); p++) {
      buf << "P" << (int)p << " Reward: " << GetTotalReward(p) << std::endl;
    }
  }
  buf << BettingNode::ToString();

  return buf.str();
}

const CardSet& GameTree::GameNode::GetBoardCards() const { return boardCards_; }

const CardSet& GameTree::GameNode::GetHoleCardsOfPlayer(uint8_t player) const {
  assert(player < holeCards_.size());
  return holeCards_[player];
}

double GameTree::GameNode::GetTotalReward(uint8_t player) const {
  assert(player < gameTree_->GetNbPlayers());
  // Copy Board Cards and Hole Cards
  uint8_t holeCards[10][3], boardCards[7], nbHoleCards[10], nbBoardCards;

  for (size_t p = 0; p < holeCards_.size(); p++) {
    auto cards = holeCards_[p].ToCardArray();
    for (size_t c = 0; c < cards.size(); c++) {
      holeCards[p][c] = cards[c];
    }
    nbHoleCards[p] = cards.size();
  }

  auto bc = boardCards_.ToCardArray();
  for (size_t c = 0; c < bc.size(); c++) {
    boardCards[c] = bc[c];
  }
  nbBoardCards = bc.size();

  SetHoleAndBoardCards(holeCards, boardCards, nbHoleCards, nbBoardCards);

  return ValueOfState(player);
}

GameTree::GameTree(const std::string& gameDef) : BettingTree(gameDef) {}

}  // namespace open_spiel::universal_poker::logic
