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

#include <algorithm>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/spiel_bots.h"

#include "open_spiel/bots/gin_rummy/simple_gin_rummy_bot.h"
#include "open_spiel/games/gin_rummy.h"
#include "open_spiel/games/gin_rummy/gin_rummy_utils.h"

namespace open_spiel {
namespace gin_rummy {

void SimpleGinRummyBot::Restart() {
  knocked_ = false;
  next_actions_ = {};
}

ActionsAndProbs SimpleGinRummyBot::GetPolicy(const State& state) {
  ActionsAndProbs policy;
  auto legal_actions = state.LegalActions(player_id_);
  auto chosen_action = Step(state);
  for (auto action : legal_actions)
    policy.emplace_back(action, action == chosen_action ? 1.0 : 0.0);
  return policy;
}

Action SimpleGinRummyBot::Step(const State& state) {
  std::vector<float> observation;
  state.ObservationTensor(player_id_, &observation);

  std::vector<int> hand;
  std::vector<int> layed_melds;
  std::vector<int> discard_pile;
  absl::optional<int> upcard = absl::nullopt;
  int knock_card = 0;
  int stock_size = 0;

  // Decode observation tensor.
  int offset = 0;
  SPIEL_CHECK_TRUE(observation[player_id_] == 1);
  offset += kNumPlayers;
  // Player hand.
  if (player_id_ == 1) offset += kDefaultNumCards;
  for (int i = 0; i < kDefaultNumCards; ++i) {
    if (observation[offset + i] == 1) hand.push_back(i);
  }
  offset += kDefaultNumCards;
  if (player_id_ == 0) offset += kDefaultNumCards;
  // Current player.
  SPIEL_CHECK_EQ(observation[offset + player_id_], 1);
  offset += kNumPlayers;
  // Knock card.
  for (int i = 0; i < kDefaultKnockCard; ++i) {
    if (observation[offset + i] == 1) knock_card += 1;
  }
  offset += kDefaultKnockCard;
  // Upcard.
  for (int i = 0; i < kDefaultNumCards; ++i) {
    if (observation[offset + i] == 1) upcard = i;
  }
  offset += kDefaultNumCards;
  // Discard pile.
  for (int i = 0; i < kDefaultNumCards; ++i) {
    if (observation[offset + i] == 1) discard_pile.push_back(i);
  }
  offset += kDefaultNumCards;
  // Stock size.
  for (int i = 0; i < kDefaultNumCards; ++i) {
    if (observation[offset + i] == 1) stock_size += 1;
  }
  offset += kDefaultNumCards;
  // Layed melds. Player 0 looks at player 1's layed melds and vice versa.
  if (player_id_ == 0) offset += kNumMeldActions;
  for (int i = 0; i < kNumMeldActions; ++i) {
    if (observation[offset + i] == 1) {
      layed_melds.push_back(i);
      knocked_ = true;
    }
  }  // Completed decoding observation.

  auto legal_actions = state.LegalActions(player_id_);
  // Next actions must be legal, in order from back to front.
  if (!next_actions_.empty()) {
    Action action = next_actions_.back();
    if (std::find(legal_actions.begin(), legal_actions.end(), action) ==
        legal_actions.end()) {
      std::cerr << "Game state:" << std::endl;
      std::cerr << state.ToString() << std::endl;
      std::cerr << "Legal actions: " << legal_actions << std::endl;
      std::cerr << "Bot next actions: " << next_actions_ << std::endl;
      SpielFatalError("Previously determined next action is illegal.");
    }
    next_actions_.pop_back();
    return action;
  }

  // When knocking, bot decides how to lay the hand all at once and saves the
  // corresponding meld actions in next_actions_.
  if (knocked_) {
    if (!layed_melds.empty()) {
      // Opponent knocked.
      next_actions_.push_back(kPassAction);  // Bot never lays off.
      for (int meld_id : GetMelds(hand)) {
        next_actions_.push_back(kMeldActionBase + meld_id);
      }
      next_actions_.push_back(kPassAction);
    } else {
      next_actions_.push_back(kPassAction);
      std::vector<int> melds_to_lay = GetMelds(hand);
      for (int meld_id : melds_to_lay) {
        next_actions_.push_back(kMeldActionBase + meld_id);
      }
      int best_discard = GetDiscard(hand);
      next_actions_.push_back(best_discard);
    }
    Action action = next_actions_.back();
    SPIEL_CHECK_TRUE(std::find(legal_actions.begin(),
        legal_actions.end(), action) != legal_actions.end());
    next_actions_.pop_back();
    return action;
  } else if (!upcard.has_value()) {
    // MoveType kDiscard
    if (hand.size() != hand_size_ + 1) {
      std::cerr << "Game state:" << std::endl;
      std::cerr << state.ToString() << std::endl;
      std::cerr << "Bot hand:" << std::endl;
      std::cerr << utils_.HandToString(hand);
      SpielFatalError("Discarding with an insufficient number of cards.");
    }
    int deadwood = utils_.MinDeadwood(hand);
    if (deadwood <= knock_card && !knocked_) {
      knocked_ = true;
      return kKnockAction;
    } else {
      int best_discard = GetDiscard(hand);
      if (best_discard >= 0) {
        return best_discard;
      } else {
        return legal_actions[0];
      }
    }
  } else {
    // MoveType kDraw
    if (stock_size == kWallStockSize) {
      // Special rules apply when we've reached the wall.
      if (legal_actions.back() == kKnockAction) {
        knocked_ = true;
        return kKnockAction;
      } else {
        return kPassAction;
      }
    } else if (utils_.MinDeadwood(hand, upcard) <= knock_card ||
        !absl::c_linear_search(GetBestDeadwood(hand, upcard), upcard)) {
      // Draw upcard if doing so permits a knock, or if the upcard would not be
      // in the "best" deadwood (=> upcard would be in a "best" meld).
      return kDrawUpcardAction;
    } else {
      return legal_actions.back();  // Draw from stock or pass.
    }
  }
}

// Returns the "best" deadwood, i.e. the cards that do not belong to one of the
// "best" melds. Here "best" means any meld group that achieves the lowest
// possible deadwood count for the given hand. In general this is non-unique.
std::vector<int> SimpleGinRummyBot::GetBestDeadwood(std::vector<int> hand,
    const absl::optional<int> card) const {
  if (card.has_value()) hand.push_back(card.value());
  for (const auto& meld : utils_.BestMeldGroup(hand)) {
    for (auto card : meld) {
      hand.erase(remove(hand.begin(), hand.end(), card), hand.end());
    }
  }
  return hand;
}

int SimpleGinRummyBot::GetDiscard(const std::vector<int> &hand) const {
  std::vector<int> deadwood = GetBestDeadwood(hand);
  if (!deadwood.empty()) {
    std::sort(deadwood.begin(), deadwood.end(),
              RankComparator(kDefaultNumRanks));
    return deadwood.back();
  } else {
    // 11 card gin. All cards are melded so there is no deadwood to throw from.
    // Must be careful to throw a card from a meld that does not break up that
    // meld. E.g. consider an 11 card gin containing the meld As2s3s4s. With a
    // knock card of 10, all of these cards are legal discards following a
    // knock, but only the As and 4s preserve gin.
    for (int i = 0; i < hand.size(); ++i) {
      std::vector<int> hand_copy = hand;
      hand_copy.erase(hand_copy.begin() + i);
      if (utils_.MinDeadwood(hand_copy) == 0)
        return hand[i];
    }
    SpielFatalError("11 card gin error.");
  }
}

std::vector<int> SimpleGinRummyBot::GetMelds(std::vector<int> hand) const {
  if (hand.size() == hand_size_ + 1 && utils_.MinDeadwood(hand) == 0) {
    // 11 card gin. Must select discard that preserves gin. See GetDiscard().
    hand.erase(remove(hand.begin(), hand.end(), GetDiscard(hand)), hand.end());
  }
  std::vector<int> rv;
  for (const auto& meld : utils_.BestMeldGroup(hand)) {
    rv.push_back(utils_.meld_to_int.at(meld));
  }
  return rv;
}

}  // namespace gin_rummy
}  // namespace open_spiel

