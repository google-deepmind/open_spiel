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

#include "open_spiel/games/hearts.h"

#include <map>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace hearts {
namespace {

const GameType kGameType{
    /*short_name=*/"hearts",
    /*long_name=*/"Hearts",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/kNumPlayers,
    /*min_num_players=*/kNumPlayers,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/false,
    /*provides_observation_tensor=*/false,
    /*parameter_specification=*/
    {
        // Pass cards at the beginning of the hand.
        {"pass_cards", GameParameter(true)},
        // Cannot play hearts or QS on first trick.
        {"no_pts_on_first_trick", GameParameter(true)},
        // First player to play can lead any club.
        {"can_lead_any_club", GameParameter(false)},
        // -10 for taking JD.
        {"jd_bonus", GameParameter(false)},
        // -5 for taking no tricks.
        {"avoid_all_tricks_bonus", GameParameter(false)},
        // Must break hearts.
        {"must_break_hearts", GameParameter(true)},
        // QS breaks hearts.
        {"qs_breaks_hearts", GameParameter(true)},
        // If aside from QS only hearts remain, player is
        // permitted to lead hearts even if hearts are
        // not broken.
        {"can_lead_hearts_instead_of_qs", GameParameter(false)},
    }};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new HeartsGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

HeartsGame::HeartsGame(const GameParameters& params)
    : Game(kGameType, params),
      pass_cards_(ParameterValue<bool>("pass_cards")),
      no_pts_on_first_trick_(ParameterValue<bool>("no_pts_on_first_trick")),
      can_lead_any_club_(ParameterValue<bool>("can_lead_any_club")),
      jd_bonus_(ParameterValue<bool>("jd_bonus")),
      avoid_all_tricks_bonus_(ParameterValue<bool>("avoid_all_tricks_bonus")),
      qs_breaks_hearts_(ParameterValue<bool>("qs_breaks_hearts")),
      must_break_hearts_(ParameterValue<bool>("must_break_hearts")),
      can_lead_hearts_instead_of_qs_(
          ParameterValue<bool>("can_lead_hearts_instead_of_qs")) {}

HeartsState::HeartsState(std::shared_ptr<const Game> game, bool pass_cards,
                         bool no_pts_on_first_trick, bool can_lead_any_club,
                         bool jd_bonus, bool avoid_all_tricks_bonus,
                         bool must_break_hearts, bool qs_breaks_hearts,
                         bool can_lead_hearts_instead_of_qs)
    : State(game),
      pass_cards_(pass_cards),
      no_pts_on_first_trick_(no_pts_on_first_trick),
      can_lead_any_club_(can_lead_any_club),
      jd_bonus_(jd_bonus),
      avoid_all_tricks_bonus_(avoid_all_tricks_bonus),
      qs_breaks_hearts_(qs_breaks_hearts),
      must_break_hearts_(must_break_hearts),
      can_lead_hearts_instead_of_qs_(can_lead_hearts_instead_of_qs),
      hearts_broken_(!must_break_hearts) {}

std::string HeartsState::ActionToString(Player player, Action action) const {
  if (history_.empty()) return pass_dir_str[action];
  return CardString(action);
}

std::string HeartsState::ToString() const {
  std::string rv = "Pass Direction: ";
  absl::StrAppend(&rv, pass_dir_str[static_cast<int>(pass_dir_)], "\n\n");
  absl::StrAppend(&rv, FormatDeal());
  if (!passed_cards_[0].empty()) absl::StrAppend(&rv, FormatPass());
  if (num_cards_played_ > 0) absl::StrAppend(&rv, FormatPlay(), FormatPoints());
  return rv;
}

std::string HeartsState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  if (IsTerminal()) return ToString();
  std::string rv = "Pass Direction: ";
  absl::StrAppend(&rv, pass_dir_str[static_cast<int>(pass_dir_)], "\n\n");
  absl::StrAppend(&rv, "Hand: \n");
  auto cards = FormatHand(player, /*mark_voids=*/true);
  for (int suit = kNumSuits - 1; suit >= 0; --suit)
    absl::StrAppend(&rv, cards[suit], "\n");
  if (!passed_cards_[player].empty()) absl::StrAppend(&rv, FormatPass(player));
  if (num_cards_played_ > 0) absl::StrAppend(&rv, FormatPlay(), FormatPoints());
  return rv;
}

std::array<std::string, kNumSuits> HeartsState::FormatHand(
    int player, bool mark_voids) const {
  // Current hand, except in the terminal state when we use the original hand
  // to enable an easy review of the whole deal.
  auto deal = IsTerminal() ? initial_deal_ : holder_;
  std::array<std::string, kNumSuits> cards;
  for (int suit = 0; suit < kNumSuits; ++suit) {
    cards[suit].push_back(kSuitChar[suit]);
    cards[suit].push_back(' ');
    bool is_void = true;
    for (int rank = kNumCardsPerSuit - 1; rank >= 0; --rank) {
      if (player == deal[Card(Suit(suit), rank)]) {
        cards[suit].push_back(kRankChar[rank]);
        is_void = false;
      }
    }
    if (is_void && mark_voids) absl::StrAppend(&cards[suit], "none");
  }
  return cards;
}

std::string HeartsState::FormatDeal() const {
  std::string rv;
  std::array<std::array<std::string, kNumSuits>, kNumPlayers> cards;
  for (auto player : {kNorth, kEast, kSouth, kWest})
    cards[player] = FormatHand(player, /*mark_voids=*/false);
  constexpr int kColumnWidth = 8;
  std::string padding(kColumnWidth, ' ');
  for (int suit = kNumSuits - 1; suit >= 0; --suit)
    absl::StrAppend(&rv, padding, cards[kNorth][suit], "\n");
  for (int suit = kNumSuits - 1; suit >= 0; --suit)
    absl::StrAppend(&rv, absl::StrFormat("%-8s", cards[kWest][suit]), padding,
                    cards[kEast][suit], "\n");
  for (int suit = kNumSuits - 1; suit >= 0; --suit)
    absl::StrAppend(&rv, padding, cards[kSouth][suit], "\n");
  return rv;
}

std::string HeartsState::FormatPass() const {
  std::string rv = "\n\nPassed Cards:";
  for (int player = 0; player < kNumPlayers; ++player) {
    absl::StrAppend(&rv, "\n", DirString(player), ": ");
    for (int card : passed_cards_[player]) {
      absl::StrAppend(&rv, CardString(card), " ");
    }
  }
  // Cards are not received until all players have completed passing.
  // West is the last player to pass.
  if (passed_cards_[static_cast<int>(kWest)].size() == kNumCardsInPass) {
    absl::StrAppend(&rv, "\n\nReceived Cards:");
    for (int receiver = 0; receiver < kNumPlayers; ++receiver) {
      absl::StrAppend(&rv, "\n", DirString(receiver), ": ");
      int passer =
          (receiver + kNumPlayers - static_cast<int>(pass_dir_)) % kNumPlayers;
      for (int card : passed_cards_[passer]) {
        absl::StrAppend(&rv, CardString(card), " ");
      }
    }
  }
  absl::StrAppend(&rv, "\n");
  return rv;
}

std::string HeartsState::FormatPass(Player player) const {
  std::string rv = "\nPassed Cards: ";
  std::vector<int> passed_cards = passed_cards_[player];
  // Sort cards because players don't have access to the order in which the
  // cards were selected to be passed. Knowing the order could allow for
  // information leakage.
  absl::c_sort(passed_cards);
  for (int card : passed_cards) absl::StrAppend(&rv, CardString(card), " ");
  // Cards are not received until all players have completed passing.
  // West is the last player to pass.
  if (passed_cards_[static_cast<int>(kWest)].size() == kNumCardsInPass) {
    absl::StrAppend(&rv, "\n\nReceived Cards: ");
    int passing_player =
        (player + kNumPlayers - static_cast<int>(pass_dir_)) % kNumPlayers;
    std::vector<int> received_cards = passed_cards_[passing_player];
    absl::c_sort(received_cards);
    for (int card : received_cards) absl::StrAppend(&rv, CardString(card), " ");
  }
  absl::StrAppend(&rv, "\n");
  return rv;
}

std::string HeartsState::FormatPlay() const {
  SPIEL_CHECK_GT(num_cards_played_, 0);
  std::string rv = "\nTricks:";
  absl::StrAppend(&rv, "\nN  E  S  W  N  E  S");
  for (int i = 0; i <= (num_cards_played_ - 1) / kNumPlayers; ++i) {
    absl::StrAppend(&rv, "\n", std::string(3 * tricks_[i].Leader(), ' '));
    for (auto card : tricks_[i].Cards()) {
      absl::StrAppend(&rv, CardString(card), " ");
    }
  }
  return rv;
}

std::string HeartsState::FormatPoints() const {
  std::string rv;
  absl::StrAppend(&rv, "\n\nPoints:");
  for (int i = 0; i < kNumPlayers; ++i)
    absl::StrAppend(&rv, "\n", DirString(i), ": ", points_[i]);
  return rv;
}

void HeartsState::InformationStateTensor(Player player,
                                         std::vector<double>* values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  std::fill(values->begin(), values->end(), 0.0);
  values->resize(game_->InformationStateTensorSize());
  if (phase_ != Phase::kPass && phase_ != Phase::kPlay) return;
  auto ptr = values->begin();
  // Pass direction
  ptr[static_cast<int>(pass_dir_)] = 1;
  ptr += kNumPlayers;
  // Dealt hand
  for (int i = 0; i < kNumCards; ++i)
    if (initial_deal_[i] == player) ptr[i] = 1;
  ptr += kNumCards;
  // Passed cards
  for (int card : passed_cards_[player]) ptr[card] = 1;
  ptr += kNumCards;
  // Received cards
  // Cards are not received until all players have completed passing.
  // West is the last player to pass.
  if (passed_cards_[static_cast<int>(kWest)].size() == kNumCardsInPass) {
    int passer =
        (player - static_cast<int>(pass_dir_) + kNumPlayers) % kNumPlayers;
    for (int card : passed_cards_[passer]) ptr[card] = 1;
  }
  ptr += kNumCards;
  // Current hand
  for (int i = 0; i < kNumCards; ++i)
    if (holder_[i] == player) ptr[i] = 1;
  ptr += kNumCards;
  // Point totals
  for (int i = 0; i < kNumPlayers; ++i) {
    // Use thermometer representation instead of one-hot for point totals.
    // Players can have negative points so we need to offset
    for (int j = 0; j < points_[i] + std::abs(kPointsForJD); ++j) ptr[j] = 1;
    ptr += kMaxScore;
  }
  // History of tricks, presented in the format: N E S W N E S
  int current_trick = num_cards_played_ / kNumPlayers;
  for (int i = 0; i < current_trick; ++i) {
    Player leader = tricks_[i].Leader();
    ptr += leader * kNumCards;
    for (auto card : tricks_[i].Cards()) {
      ptr[card] = 1;
      ptr += kNumCards;
    }
    ptr += (kNumPlayers - leader - 1) * kNumCards;
  }
  Player leader = tricks_[current_trick].Leader();
  if (leader != kInvalidPlayer) {
    auto cards = tricks_[current_trick].Cards();
    ptr += leader * kNumCards;
    for (auto card : cards) {
      ptr[card] = 1;
      ptr += kNumCards;
    }
  }
  // Current trick may contain less than four cards.
  ptr += (kNumPlayers - (num_cards_played_ % kNumPlayers)) * kNumCards;
  // Move to the end of current trick.
  ptr += (kNumPlayers - std::max(leader, 0) - 1) * kNumCards;
  // Skip over unplayed tricks.
  ptr += (kNumTricks - current_trick - 1) * kTrickTensorSize;
  SPIEL_CHECK_EQ(std::distance(values->begin(), ptr),
                 kInformationStateTensorSize);
}

std::vector<Action> HeartsState::LegalActions() const {
  switch (phase_) {
    case Phase::kPassDir:
      return PassDirLegalActions();
    case Phase::kDeal:
      return DealLegalActions();
    case Phase::kPass:
      return PassLegalActions();
    case Phase::kPlay:
      return PlayLegalActions();
    default:
      return {};
  }
}

std::vector<Action> HeartsState::PassDirLegalActions() const {
  SPIEL_CHECK_EQ(history_.size(), 0);
  std::vector<Action> legal_actions;
  if (!pass_cards_) {
    legal_actions.push_back(static_cast<int>(PassDir::kNoPass));
  } else {
    legal_actions.reserve(kNumPlayers);
    for (int i = 0; i < kNumPlayers; ++i) legal_actions.push_back(i);
  }
  return legal_actions;
}

std::vector<Action> HeartsState::DealLegalActions() const {
  std::vector<Action> legal_actions;
  legal_actions.reserve(kNumCards - num_cards_dealt_);
  for (int i = 0; i < kNumCards; ++i) {
    if (!holder_[i].has_value()) legal_actions.push_back(i);
  }
  SPIEL_CHECK_GT(legal_actions.size(), 0);
  return legal_actions;
}

std::vector<Action> HeartsState::PassLegalActions() const {
  std::vector<Action> legal_actions;
  legal_actions.reserve(kNumCards / kNumPlayers);
  for (int card = 0; card < kNumCards; ++card) {
    if (holder_[card] == current_player_) legal_actions.push_back(card);
  }
  return legal_actions;
}

std::vector<Action> HeartsState::PlayLegalActions() const {
  std::vector<Action> legal_actions;
  legal_actions.reserve(kNumTricks - num_cards_played_ / kNumPlayers);

  // Check if we can follow suit.
  if (num_cards_played_ % kNumPlayers != 0) {
    auto suit = CurrentTrick().LedSuit();
    for (int rank = 0; rank < kNumCardsPerSuit; ++rank) {
      if (holder_[Card(suit, rank)] == current_player_) {
        legal_actions.push_back(Card(suit, rank));
      }
    }
  }
  if (!legal_actions.empty()) return legal_actions;

  // Special rules apply to the first card played.
  // Must play 2C unless option to play any club is enabled.
  if (num_cards_played_ == 0) {
    SPIEL_CHECK_TRUE(holder_[Card(Suit::kClubs, 0)] == current_player_);
    legal_actions.push_back(Card(Suit::kClubs, 0));
    if (can_lead_any_club_) {
      for (int rank = 1; rank < kNumCardsPerSuit; ++rank) {
        if (holder_[Card(Suit::kClubs, rank)] == current_player_) {
          legal_actions.push_back(Card(Suit::kClubs, rank));
        }
      }
    }
    return legal_actions;
  }

  // Special rules apply to the first trick.
  // By default, cannot play hearts or QS on first trick.
  if (no_pts_on_first_trick_ && num_cards_played_ < kNumPlayers) {
    for (int card = 0; card < kNumCards; ++card) {
      if (holder_[card] == current_player_ && card != Card(Suit::kSpades, 10) &&
          CardSuit(card) != Suit::kHearts) {
        legal_actions.push_back(card);
      }
    }
  }
  if (!legal_actions.empty()) return legal_actions;

  // Player must lead. By default, cannot lead hearts until broken.
  if (num_cards_played_ % kNumPlayers == 0) {
    for (int card = 0; card < kNumCards; ++card) {
      if (holder_[card] == current_player_) {
        if (CardSuit(card) != Suit::kHearts || hearts_broken_) {
          legal_actions.push_back(card);
        }
      }
    }
    // Don't force player into leading the QS when hearts have not
    // been broken.
    if (can_lead_hearts_instead_of_qs_ && legal_actions.size() == 1 &&
        legal_actions[0] == Card(Suit::kSpades, 10)) {
      legal_actions.pop_back();
    }
  }
  if (!legal_actions.empty()) return legal_actions;

  // Otherwise, we can play any of our cards.
  for (int card = 0; card < kNumCards; ++card) {
    if (holder_[card] == current_player_) legal_actions.push_back(card);
  }
  return legal_actions;
}

std::vector<std::pair<Action, double>> HeartsState::ChanceOutcomes() const {
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
    if (!holder_[card].has_value()) outcomes.emplace_back(card, p);
  }
  return outcomes;
}

void HeartsState::DoApplyAction(Action action) {
  switch (phase_) {
    case Phase::kPassDir:
      return ApplyPassDirAction(action);
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

// See overview in hearts.h for more information on setting the pass direction.
void HeartsState::ApplyPassDirAction(int pass_dir) {
  SPIEL_CHECK_EQ(history_.size(), 0);
  pass_dir_ = static_cast<PassDir>(pass_dir);
  phase_ = Phase::kDeal;
}

void HeartsState::ApplyDealAction(int card) {
  holder_[card] = num_cards_dealt_ % kNumPlayers;
  ++num_cards_dealt_;
  if (num_cards_dealt_ == kNumCards) {
    // Preserve the initial deal for easy retrieval
    initial_deal_ = holder_;
    if (pass_dir_ == PassDir::kNoPass) {
      phase_ = Phase::kPlay;
      // Play starts with the holder of the 2C
      current_player_ = holder_[Card(Suit::kClubs, 0)].value();
    } else {
      phase_ = Phase::kPass;
      current_player_ = 0;
    }
  }
}

void HeartsState::ApplyPassAction(int card) {
  passed_cards_[current_player_].push_back(card);
  holder_[card] = std::nullopt;
  if (passed_cards_[current_player_].size() % kNumCardsInPass == 0)
    ++current_player_;
  if (current_player_ == kNumPlayers) {
    // Players have completed passing. Now let's distribute the passed cards.
    for (int player = 0; player < kNumPlayers; ++player) {
      for (int card : passed_cards_[player]) {
        holder_[card] = (player + static_cast<int>(pass_dir_)) % kNumPlayers;
      }
    }
    phase_ = Phase::kPlay;
    // Play starts with the holder of the 2C
    current_player_ = holder_[Card(Suit::kClubs, 0)].value();
  }
}

void HeartsState::ApplyPlayAction(int card) {
  SPIEL_CHECK_TRUE(holder_[card] == current_player_);
  holder_[card] = std::nullopt;
  if (num_cards_played_ % kNumPlayers == 0) {
    CurrentTrick() = Trick(current_player_, card, jd_bonus_);
  } else {
    CurrentTrick().Play(current_player_, card);
  }
  // Check if action breaks hearts.
  if (CardSuit(card) == Suit::kHearts) hearts_broken_ = true;
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

Player HeartsState::CurrentPlayer() const {
  if (phase_ == Phase::kDeal) return kChancePlayerId;
  return current_player_;
}

void HeartsState::ComputeScore() {
  SPIEL_CHECK_TRUE(IsTerminal());
  // Did anyone shoot the moon?
  Player potential_shooter = kInvalidPlayer;
  bool moon_shot = true;
  for (int i = 0; i < kNumTricks; ++i) {
    int points = tricks_[i].Points();
    // JD not required to shoot the moon.
    if (points != 0 && points != kPointsForJD) {
      // This trick must be taken by the shooter.
      if (potential_shooter == kInvalidPlayer) {
        potential_shooter = tricks_[i].Winner();
      } else if (potential_shooter != tricks_[i].Winner()) {
        moon_shot = false;
        break;
      }
    }
  }
  // Shooting the moon sets the shooter's points to 0, and adds 26 pts to each
  // opponent's score.
  if (moon_shot) {
    for (int i = 0; i < kNumPlayers; ++i) {
      points_[i] += (i == potential_shooter) ? -kTotalPositivePoints
                                             : kTotalPositivePoints;
    }
  }
  // Did anyone avoid taking any tricks?
  if (avoid_all_tricks_bonus_ && !moon_shot) {
    std::vector<int> tricks_taken(kNumPlayers, 0);
    for (int i = 0; i < kNumTricks; ++i) {
      tricks_taken[tricks_[i].Winner()] += 1;
    }
    for (int i = 0; i < kNumPlayers; ++i) {
      if (tricks_taken[i] == 0) points_[i] += kAvoidAllTricksBonus;
    }
  }
}

// Hearts is a trick-avoidance game in which the goal is to accumulate the
// fewest number of points. Because RL algorithms are designed to maximize
// reward, returns are calculated by subtracting the in-game points from an
// upper bound.
std::vector<double> HeartsState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>(kNumPlayers, 0.0);
  }
  std::vector<double> returns = points_;
  for (int i = 0; i < returns.size(); ++i)
    returns[i] = kTotalPositivePoints - returns[i];
  return returns;
}

Trick::Trick(Player leader, int card, bool jd_bonus)
    : jd_bonus_(jd_bonus),
      winning_rank_(CardRank(card)),
      points_(CardPoints(card, jd_bonus)),
      led_suit_(CardSuit(card)),
      leader_(leader),
      winning_player_(leader),
      cards_{card} {}

void Trick::Play(Player player, int card) {
  cards_.push_back(card);
  points_ += CardPoints(card, jd_bonus_);
  if (CardSuit(card) == led_suit_ && CardRank(card) > winning_rank_) {
    winning_rank_ = CardRank(card);
    winning_player_ = player;
  }
}

}  // namespace hearts
}  // namespace open_spiel
