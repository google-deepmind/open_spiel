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
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace universal_poker {
namespace logic {

class GameNode : public BettingNode {
 public:
  GameNode(const acpc_cpp::ACPCGame* acpc_game);

  int GetActionCount() const { return action_count_; }
  void ApplyAction(uint32_t actionIdx);
  std::string ToString() const;

  const CardSet& GetBoardCards() const { return board_cards_; }
  const CardSet& GetHoleCardsOfPlayer(Player player) const;
  double GetTotalReward(Player player) const;

 private:
  CardSet deck_;  // The remaining cards to deal.
  // Memoize the number of available actions at this node (recomputed after
  // each action).
  int action_count_;
  std::vector<CardSet> hole_cards_;  // The cards already owned by each player
  CardSet board_cards_;
};

}  // namespace logic
}  // namespace universal_poker
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAME_TREE_H
