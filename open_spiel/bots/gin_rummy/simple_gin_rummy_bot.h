// Copyright 2021 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_BOTS_GIN_RUMMY_SIMPLE_GIN_RUMMY_BOT_H_
#define OPEN_SPIEL_BOTS_GIN_RUMMY_SIMPLE_GIN_RUMMY_BOT_H_

// This bot plays about the simplest possible strategy that performs reasonably
// well and effectively explores the game tree. It's useful both as a test
// of the game implementation and as a benchmark of playing strength.
//
// The strategy can be summarized as follows:
//
// If phase == kDraw:
//   Draw the upcard under either of the following two conditions, otherwise
//   draw from the stock:
//     1) If doing so allows for an immediate knock.
//     2) If the upcard belongs to a meld, and that meld lowers the deadwood
//        count of the hand. The second part of this condition in relevant in
//        the following example where we would not want to pick up the Js even
//        though it makes three jacks, because it breaks up a better meld
//        thereby increasing the total deadwood count.
//
//        Upcard: Js
//        +--------------------------+
//        |                          |
//        |Ac2c3c4c                  |
//        |                9dTdJdQd  |
//        |    3h              Jh    |
//        +--------------------------+
//
// If phase == kDiscard:
//  Always knock if legal, otherwise throw the deadwood card worth the most
//  points, with ties being broken arbitrarily.
//
// If phase == kKnock:
//   When laying the hand, the meld arrangement is chosen that minimizes the
//   total deadwood count. If two different meld arrangements are equal in this
//   regard, one is chosen arbitrarily. No layoffs are made if opponent knocks.

#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/games/gin_rummy.h"
#include "open_spiel/games/gin_rummy/gin_rummy_utils.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"

namespace open_spiel {
namespace gin_rummy {

class SimpleGinRummyBot : public Bot {
 public:
  SimpleGinRummyBot(GameParameters params, const Player player_id)
      : params_(params),
        player_id_(player_id),
        hand_size_(params["hand_size"].int_value()),
        utils_(GinRummyUtils(params["num_ranks"].int_value(),
                             params["num_suits"].int_value(),
                             params["hand_size"].int_value())) {}

  void Restart() override;
  Action Step(const State& state) override;
  bool ProvidesPolicy() override { return true; }
  ActionsAndProbs GetPolicy(const State& state) override;

 private:
  GameParameters params_;
  const Player player_id_;
  const int hand_size_;
  const GinRummyUtils utils_;

  bool knocked_ = false;
  std::vector<Action> next_actions_;

  std::vector<int> GetBestDeadwood(const std::vector<int> hand,
      const absl::optional<int> card = absl::nullopt) const;
  int GetDiscard(const std::vector<int> &hand) const;
  std::vector<int> GetMelds(std::vector<int> hand) const;
};

}  // namespace gin_rummy
}  // namespace open_spiel

#endif  // OPEN_SPIEL_BOTS_GIN_RUMMY_SIMPLE_GIN_RUMMY_BOT_H_

