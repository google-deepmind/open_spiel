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

#include "open_spiel/games/cheat.h"

#include <algorithm>
#include <map>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace cheat {
namespace {

const GameType kGameType{
    /*short_name=*/"cheat",
    /*long_name=*/"Cheat",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/kNumPlayers,
    /*min_num_players=*/kNumPlayers,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/false,
    /*provides_observation_tensor=*/false,
    /*parameter_specification=*/
    {}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new CheatGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

CheatGame::CheatGame(const GameParameters& params)
    : Game(kGameType, params) {}

CheatState::CheatState(std::shared_ptr<const Game> game)
    : State(game) {}

std::string CheatState::ActionToString(Player player, Action action) const {
  // convert the action to cards -> 
  return CardString(action);
}

std::string CheatState::ToString() const {
  absl::StrAppend(&rv, FormatDeal());
  if (!passed_cards_[0].empty()) absl::StrAppend(&rv, FormatPass());
  if (num_cards_played_ > 0) absl::StrAppend(&rv, FormatPlay(), FormatPoints());
  return rv;
}

std::string CheatState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  if (IsTerminal()) return ToString();
  std::string rv = "Hand: \n";
  auto cards = FormatHand(player, /*mark_voids=*/true);
  for (int suit = kNumSuits - 1; suit >= 0; --suit)
    absl::StrAppend(&rv, cards[suit], "\n");
  if (!passed_cards_[player].empty()) absl::StrAppend(&rv, FormatPass(player));
  if (num_cards_played_ > 0) absl::StrAppend(&rv, FormatPlay(), FormatPoints());
  return rv;
}

void CheatState::InformationStateTensor(Player player,
                                         absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  std::fill(values.begin(), values.end(), 0.0);
  SPIEL_CHECK_EQ(values.size(), kInformationStateTensorSize);
  auto ptr = values.begin();
  // Current hand
  for (int i = 0; i < kNumCards; ++i)
    if (player_hand_[i] == player) ptr[i] = 1;
  ptr += kNumCards;
  // Cards claimed
  for (int i = 0; i < kNumCards; ++i)
    if (card_claimed_[i] == player) ptr[i] = 1;
  ptr += kNumCards;
  // Cards seen
  for (int i = 0; i < kNumCards; ++i)
    if (card_seen_[i] == player) ptr[i] = 1;
  ptr += kNumCards;
  // Action history
  for (int i = 0; i < kNumPlayers; ++i) {
    for (int j = 0; j < kNumCards; ++j) {
      ptr[i] = action_history_[i][j];
    }
    ptr += kNumCards;
  }
  SPIEL_CHECK_EQ(ptr, values.end());
}

std::vector<Action> CheatState::LegalActions() const {
  std::vector<Action> legal_actions;
  // If initial move, deal cards - chance node
  if (num_cards_dealt_ == 0) {
    // Deal 7 random cards to each player
    for (int i = 0; i < kNumPlayers; ++i) {
      for (int j = 0; j < kNumInitCardsPerPlayer; ++j) {
        legal_actions.push_back(kDealAction);
      }
    }
    legal_actions.reserve(kNumCards - num_cards_dealt_);
    for (int i = 0; i < kNumCards; ++i) {
      if (!player_hand_[i].has_value()) legal_actions.push_back(i);
    }
    SPIEL_CHECK_GT(legal_actions.size(), 0);
    return legal_actions;
  }
  legal_actions.reserve(kNumTricks - num_cards_played_ / kNumPlayers);

  // Check if we can follow suit.
  if (num_cards_played_ % kNumPlayers != 0) {
    auto suit = CurrentTrick().LedSuit();
    for (int rank = 0; rank < kNumCardsPerSuit; ++rank) {
      if (player_hand_[Card(suit, rank)] == current_player_) {
        legal_actions.push_back(Card(suit, rank));
      }
    }
  }
  if (!legal_actions.empty()) return legal_actions;


  // Otherwise, we can play any of our cards. 
  for (int card = 0; card < kNumCards; ++card) {
    if (player_hand_[card] == current_player_) legal_actions.push_back(card);
  }
  return legal_actions;
}

std::vector<std::pair<Action, double>> CheatState::ChanceOutcomes() const {
  std::vector<std::pair<Action, double>> outcomes;
  if (history_.empty()) {
    outcomes.reserve(kNumPlayers);
    const double p = 1.0 / kNumPlayers;
    for (int dir = 0; dir < kNumPlayers; ++dir) {
      outcomes.emplace_back(dir, p);
    }
    return outcomes;
  }
  int num_cards_remaining = kNumCards - num_cards_dealt_;
  outcomes.reserve(num_cards_remaining);
  const double p = 1.0 / num_cards_remaining;
  for (int card = 0; card < kNumCards; ++card) {
    if (!player_hand_[card].has_value()) outcomes.emplace_back(card, p);
  }
  return outcomes;
}

void CheatState::DoApplyAction(Action action) {
  switch (phase_) {
    case Phase::kDeal:
      return ApplyDealAction(action);
    case Phase::kPass:
      return ApplyPassAction(action);
    case Phase::kPlay:
      return ApplyPlayAction(action);
    case Phase::kGameOver:
      SpielFatalError("Cannot act in terminal states");
  }
}

void CheatState::ApplyDealAction(int card) {
  player_hand_[card] = num_cards_dealt_ % kNumPlayers;
  ++num_cards_dealt_;
  if (num_cards_dealt_ == kNumCards) {
    // Preserve the initial deal for easy retrieval
    initial_deal_ = player_hand_;
    if (pass_dir_ == PassDir::kNoPass) {
      phase_ = Phase::kPlay;
      // Play starts with the holder of the 2C
      current_player_ = player_hand_[Card(Suit::kClubs, 0)].value();
    } else {
      phase_ = Phase::kPass;
      current_player_ = 0;
    }
  }
}

void CheatState::ApplyPassAction(int card) {
  passed_cards_[current_player_].push_back(card);
  player_hand_[card] = absl::nullopt;
  if (passed_cards_[current_player_].size() % kNumCardsInPass == 0)
    ++current_player_;
  if (current_player_ == kNumPlayers) {
    // Players have completed passing. Now let's distribute the passed cards.
    for (int player = 0; player < kNumPlayers; ++player) {
      for (int card : passed_cards_[player]) {
        player_hand_[card] = (player + static_cast<int>(pass_dir_)) % kNumPlayers;
      }
    }
    phase_ = Phase::kPlay;
    // Play starts with the holder of the 2C
    current_player_ = player_hand_[Card(Suit::kClubs, 0)].value();
  }
}

void CheatState::ApplyPlayAction(int card) {
  SPIEL_CHECK_TRUE(player_hand_[card] == current_player_);
  player_hand_[card] = absl::nullopt;
  if (num_cards_played_ % kNumPlayers == 0) {
    CurrentTrick() = Trick(current_player_, card, jd_bonus_);
  } else {
    CurrentTrick().Play(current_player_, card);
  }
  // Check if action breaks hearts.
  if (CardSuit(card) == Suit::kCheat) hearts_broken_ = true;
  if (qs_breaks_hearts_ && card == Card(Suit::kSpades, 10))
    hearts_broken_ = true;
  // Update player and point totals.
  Trick current_trick = CurrentTrick();
  ++num_cards_played_;
  if (num_cards_played_ % kNumPlayers == 0) {
    current_player_ = current_trick.Winner();
    points_[current_player_] += current_trick.Points();
  } else {
    current_player_ = (current_player_ + 1) % kNumPlayers;
  }
  if (num_cards_played_ == kNumCards) {
    phase_ = Phase::kGameOver;
    current_player_ = kTerminalPlayerId;
    ComputeScore();
  }
}

Player CheatState::CurrentPlayer() const {
  if (phase_ == Phase::kDeal) return kChancePlayerId;
  return current_player_;
}

// Does not account for void suit information exposed by other players during
// the play phase
std::unique_ptr<State> CheatState::ResampleFromInfostate(
    int player_id, std::function<double()> rng) const {
  std::unique_ptr<State> clone = game_->NewInitialState();
  Action pass_dir = static_cast<int>(pass_dir_);
  clone->ApplyAction(pass_dir);

  // start by gathering all public and private info known to player_id to
  // simplify the logic for applying deal / pass actions
  // first thing we know is the player's entire hand
  std::vector<int> initial_hand;
  for (int card = 0; card < kNumCards; card++) {
    if (initial_deal_[card] == player_id) initial_hand.push_back(card);
  }

  // collect cards that have been revealed through the play phase
  std::vector<std::vector<int>> play_known(kNumPlayers);
  if (phase_ == Phase::kPlay) {
    for (int card = 0; card < kNumCards; card++) {
      absl::optional<Player> p = Played(card);
      if (p && *p != player_id) {
        play_known[*p].push_back(card);
      }
    }
  }

  // given that we should now have a state consistent with the public actions
  // and player_id's private cards, we can just copy the action sequence in
  // the play phase
  int play_start_index = kNumCards + 1;
  if (pass_dir_ != PassDir::kNoPass)
    play_start_index += kNumPlayers * kNumCardsInPass;
  for (size_t i = play_start_index; i < history_.size(); i++) {
    clone->ApplyAction(history_.at(i).action);
  }

  SPIEL_CHECK_EQ(FullHistory().size(), clone->FullHistory().size());
  SPIEL_CHECK_EQ(InformationStateString(player_id),
                 clone->InformationStateString(player_id));
  return clone;
}

}  // namespace hearts
}  // namespace open_spiel
