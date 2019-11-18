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

#ifndef OPEN_SPIEL_GAME_TREE_H
#define OPEN_SPIEL_GAME_TREE_H

#include "open_spiel/games/universal_poker/logic/betting_tree.h"
#include "open_spiel/games/universal_poker/logic/card_set.h"

namespace open_spiel {
namespace universal_poker {
namespace logic {

class GameTree : public BettingTree {
 public:
  class GameNode : public BettingNode {
   public:
    GameNode(GameTree* gameTree);

    uint32_t GetActionCount() const;
    void ApplyAction(uint32_t actionIdx);
    std::string ToString() const;

    const CardSet& GetBoardCards() const;
    const CardSet& GetHoleCardsOfPlayer(uint8_t player) const;
    double GetTotalReward(uint8_t player) const;

   private:
    GameTree* gameTree_;
    CardSet deck_;
    uint32_t actionCount_;
    std::vector<CardSet> holeCards_;
    CardSet boardCards_;
  };

 public:
  GameTree(const std::string& gameDef);
};
}  // namespace logic
}  // namespace universal_poker
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAME_TREE_H
