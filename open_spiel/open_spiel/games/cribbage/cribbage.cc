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

#include "open_spiel/games/cribbage/cribbage.h"

#include <sys/types.h>

#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace cribbage {

constexpr int kDefaultNumPlayers = 2;
constexpr int kWinScore = 121;
constexpr double kDefaultWinnerBonus = 1000;


constexpr const std::array<Card, kDeckSize> kAllCards = {
    // Clubs
    Card{0, 0, 0},
    Card{1, 1, 0},
    Card{2, 2, 0},
    Card{3, 3, 0},
    Card{4, 4, 0},
    Card{5, 5, 0},
    Card{6, 6, 0},
    Card{7, 7, 0},
    Card{8, 8, 0},
    Card{9, 9, 0},
    Card{10, 10, 0},
    Card{11, 11, 0},
    Card{12, 12, 0},
    // Diamonds
    Card{13, 0, 1},
    Card{14, 1, 1},
    Card{15, 2, 1},
    Card{16, 3, 1},
    Card{17, 4, 1},
    Card{18, 5, 1},
    Card{19, 6, 1},
    Card{20, 7, 1},
    Card{21, 8, 1},
    Card{22, 9, 1},
    Card{23, 10, 1},
    Card{24, 11, 1},
    Card{25, 12, 1},
    // Hearts
    Card{26, 0, 2},
    Card{27, 1, 2},
    Card{28, 2, 2},
    Card{29, 3, 2},
    Card{30, 4, 2},
    Card{31, 5, 2},
    Card{32, 6, 2},
    Card{33, 7, 2},
    Card{34, 8, 2},
    Card{35, 9, 2},
    Card{36, 10, 2},
    Card{37, 11, 2},
    Card{38, 12, 2},
    // Spades
    Card{39, 0, 3},
    Card{40, 1, 3},
    Card{41, 2, 3},
    Card{42, 3, 3},
    Card{43, 4, 3},
    Card{44, 5, 3},
    Card{45, 6, 3},
    Card{46, 7, 3},
    Card{47, 8, 3},
    Card{48, 9, 3},
    Card{49, 10, 3},
    Card{50, 11, 3},
    Card{51, 12, 3},
};

// Scoring.
constexpr int kNum5Combos = 1;
constexpr int kNum4Combos = 5;
constexpr int kNum3Combos = 10;
constexpr int kNum2Combos = 10;

// Bitmasks used to choose card combinations.
constexpr const std::array<int, kNum5Combos> k5CardMasks = {31};
constexpr const std::array<int, kNum4Combos> k4CardMasks = {30, 29, 27, 23, 15};
constexpr const std::array<int, kNum3Combos> k3CardMasks = {7,  11, 13, 14, 19,
                                                            21, 22, 25, 26, 28};
constexpr const std::array<int, kNum2Combos> k2CardMasks = {3,  5,  6,  9,  10,
                                                            12, 17, 18, 20, 24};

namespace {

// Facts about the game
const GameType kGameType{/*short_name=*/"cribbage",
                         /*long_name=*/"Cribbage",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kImperfectInformation,
                         GameType::Utility::kGeneralSum,
                         GameType::RewardModel::kRewards,
                         /*max_num_players=*/4,
                         /*min_num_players=*/2,
                         /*provides_information_state_string=*/false,
                         /*provides_information_state_tensor=*/false,
                         /*provides_observation_string=*/false,
                         /*provides_observation_tensor=*/false,
                         /*parameter_specification=*/
                         {{"players", GameParameter(kDefaultNumPlayers)},
                          {"winner_bonus_reward",
                           GameParameter(kDefaultWinnerBonus)}}};

static std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new CribbageGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);
}  // namespace

bool operator==(const Card& lhs, const Card& rhs) { return lhs.id == rhs.id; }

// Sort by rank first. This is needed for the proper scoring of runs.
bool operator<(const Card& lhs, const Card& rhs) {
  return (lhs.rank < rhs.rank || (lhs.rank == rhs.rank && lhs.suit < rhs.suit));
}

int CardsPerPlayer(int num_players) {
  switch (num_players) {
    case 2:
      return 6;
    case 3:
      return 5;
    case 4:
      return 5;
    default:
      SpielFatalError(absl::StrCat("Unknown number of players: ", num_players));
  }
}

int CardsToCrib(int num_players) {
  switch (num_players) {
    case 2:
      return 0;
    case 3:
      return 1;
    case 4:
      return 0;
    default:
      SpielFatalError(absl::StrCat("Unknown number of players: ", num_players));
  }
}

Card GetCard(int id) {
  SPIEL_CHECK_GE(id, 0);
  SPIEL_CHECK_LT(id, kDeckSize);
  return kAllCards[id];
}

Card GetCardByString(const std::string& str) {
  for (int i = 0; i < kDeckSize; ++i) {
    if (kAllCards[i].to_string() == str) {
      return kAllCards[i];
    }
  }
  SpielFatalError(absl::StrCat("Unknown card: ", str));
}

std::vector<Card> GetHandFromStrings(
    const std::vector<std::string>& card_strings) {
  std::vector<Card> hand;
  hand.reserve(card_strings.size());
  for (const std::string& cstr : card_strings) {
    hand.push_back(GetCardByString(cstr));
  }
  return hand;
}

int Card::value() const {
  if (rank >= kTen) {
    return 10;
  } else {
    return rank + 1;
  }
}

std::string Card::to_string() const {
  std::string str("XX");
  str[0] = kRanks[rank];
  str[1] = kSuitNames[suit];
  return str;
}

Action ToAction(const Card& c1, const Card& c2) {
  return kDeckSize + (kDeckSize * c1.id + c2.id);
}

std::pair<int, int> FromAction(Action action) {
  action -= kDeckSize;
  return {action / kDeckSize, action % kDeckSize};
}

int CardsSum(const std::vector<Card>& hand, int combo_mask) {
  int sum = 0;
  int bit = 1;
  for (int pos = 0; pos < hand.size(); ++pos) {
    if ((combo_mask & bit) > 0) {
      sum += hand[pos].value();
    }
    bit <<= 1;
  }
  return sum;
}

bool IsPair(const std::vector<Card>& hand, int combo_mask) {
  int bit = 1;
  int rank = -1;

  for (int pos = 0; pos < hand.size(); ++pos) {
    if ((combo_mask & bit) > 0) {
      if (rank == -1) {
        rank = hand[pos].rank;
      } else {
        return (rank == hand[pos].rank);
      }
    }
    bit <<= 1;
  }
  return false;
}

int ScoreHand15(const std::vector<Card>& hand) {
  int score = 0;
  for (int mask : k5CardMasks) {
    if (CardsSum(hand, mask) == 15) {
      score += 2;
    }
  }
  for (int mask : k4CardMasks) {
    if (CardsSum(hand, mask) == 15) {
      score += 2;
    }
  }
  for (int mask : k3CardMasks) {
    if (CardsSum(hand, mask) == 15) {
      score += 2;
    }
  }
  for (int mask : k2CardMasks) {
    if (CardsSum(hand, mask) == 15) {
      score += 2;
    }
  }
  return score;
}

int ScoreHandPairs(const std::vector<Card>& hand) {
  int score = 0;
  for (int mask : k2CardMasks) {
    if (IsPair(hand, mask)) {
      score += 2;
    }
  }
  return score;
}

int ScoreHandFlush(const std::vector<Card>& hand) {
  SPIEL_CHECK_TRUE(hand.size() == 4 || hand.size() == 5);
  int suit = hand[0].suit;
  for (int i = 1; i < hand.size(); ++i) {
    if (hand[i].suit != suit) {
      return 0;
    }
  }
  return hand.size();
}

int ScoreHandRun(const std::vector<Card>& hand, int combo_mask) {
  int rank = -1;
  int bit = 1;
  int length = 0;

  for (int pos = 0; pos < hand.size(); ++pos) {
    if ((combo_mask & bit) > 0) {
      if (rank == -1) {
        // First rank in the run.
        rank = hand[pos].rank;
      } else {
        // Check that the next rank is one up, then move the rank up.
        if (hand[pos].rank != (rank + 1)) {
          return 0;
        } else {
          rank++;
        }
      }
      length++;
    }
    bit <<= 1;
  }
  SPIEL_CHECK_GE(length, 3);
  return length;
}

bool IsSubsetMask(const std::vector<int>& masks, int test_mask) {
  if (masks.empty()) {
    return false;
  }
  for (int mask : masks) {
    if ((mask & test_mask) == test_mask) {
      return true;
    }
  }
  return false;
}

int ScoreHand(const std::vector<Card>& hand) {
  SPIEL_CHECK_EQ(hand.size(), 5);
  int score = 0;

  // 15s.
  score += ScoreHand15(hand);

  // Pairs (and 3-of-a-kind and 4-of-a-kind).
  score += ScoreHandPairs(hand);

  // Score the runs. When doing subsets of size 3, must check that the subset
  // is not a smaller proper subset of a combination that has already been
  // counted. So we keep a set of all the 4-card subsets that were counted
  // for this purpose.
  int score_run_5 = ScoreHandRun(hand, 31);
  if (score_run_5 > 0) {
    return score + score_run_5;
  }

  std::vector<int> combo_masks_scored;
  for (int mask : k4CardMasks) {
    int score_run_4 = ScoreHandRun(hand, mask);
    if (score_run_4 > 0) {
      score += score_run_4;
      combo_masks_scored.push_back(mask);
    }
  }

  for (int mask : k3CardMasks) {
    if (!IsSubsetMask(combo_masks_scored, mask)) {
      score += ScoreHandRun(hand, mask);
    }
  }

  return score;
}

int ScoreHand(const std::vector<Card>& hand, const Card& starter) {
  SPIEL_CHECK_EQ(hand.size(), 4);

  int score = 0;
  // Check for jack of the same suit as the starter
  for (int i = 0; i < hand.size(); ++i) {
    if (hand[i].rank == Rank::kJack && hand[i].suit == starter.suit) {
      score += 1;
      break;
    }
  }

  // Make the 5-card hand which includes the starter.
  std::vector five_card_hand = hand;
  five_card_hand.push_back(starter);
  std::sort(five_card_hand.begin(), five_card_hand.end());

  // Check for flush.
  int flush5 = ScoreHandFlush(five_card_hand);
  if (flush5 != 0) {
    score += flush5;
  } else {
    score += ScoreHandFlush(hand);
  }

  return score + ScoreHand(five_card_hand);
}

std::string CribbageState::ActionToString(Player player, Action move_id) const {
  if (player == kChancePlayerId) {
    return absl::StrCat("Deal ", kAllCards[move_id].to_string());
  } else {
    if (move_id < kDeckSize) {
      return absl::StrCat("Choose ", kAllCards[move_id].to_string());
    } else if (move_id == kPassAction) {
      return "Pass";
    } else {
      std::pair<int, int> card_ids = FromAction(move_id);
      return absl::StrCat("Choose ", kAllCards[card_ids.first].to_string(), " ",
                          kAllCards[card_ids.second].to_string());
    }
  }
}

bool CribbageState::IsTerminal() const {
  return (round_ >= kMaxNumRounds ||
          *std::max_element(scores_.begin(), scores_.end()) >= kWinScore);
}

int CribbageState::DetermineWinner() const {
  for (int p = 0; p < num_players_; ++p) {
    if (scores_[p] >= kWinScore) {
      return p;
    }
  }
  return kInvalidPlayer;
}

void AddWinnerBonusLoserPenalty(std::vector<double>* values, int winner,
                                int num_players, double winner_bonus) {
  if (winner == kInvalidPlayer) {
    return;
  }

  // For 2 and 3 player games, the loss penalty is -win_bonus / (n-1) and
  // win_bonus is given only to one player. For a 4-player game, it's a team
  // game so both the win bonus and loss penalty is shared across losers.
  double win_bonus_per_player =
      num_players <= 3 ? winner_bonus : (winner_bonus / 2.0);

  double loss_penalty_per_player =
      num_players <= 3
          ? (-winner_bonus / (static_cast<double>(values->size()) - 1.0))
          : (-winner_bonus / 2.0);

  for (Player p = 0; p < values->size(); ++p) {
    // In the 4-player games, the score is identical for players {0,2} and {1,3}
    if (p == winner || (num_players == 4 && p == (winner + 2))) {
      (*values)[p] += win_bonus_per_player;
    } else {
      (*values)[p] += loss_penalty_per_player;
    }
  }
}

std::vector<double> CribbageState::Rewards() const {
  int winner = DetermineWinner();
  std::vector<double> ret = rewards_;
  SPIEL_CHECK_EQ(ret.size(), num_players_);
  AddWinnerBonusLoserPenalty(&ret, winner, num_players_,
                             parent_game_.winner_bonus_reward());
  return ret;
}

std::vector<double> CribbageState::Returns() const {
  int winner = DetermineWinner();
  std::vector<double> ret = scores_;
  SPIEL_CHECK_EQ(ret.size(), num_players_);
  AddWinnerBonusLoserPenalty(&ret, winner, num_players_,
                             parent_game_.winner_bonus_reward());
  return ret;
}

std::string CribbageState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, game_->NumPlayers());
  return "";
}

bool CribbageState::AllHandsAreEmpty() const {
  for (Player p = 0; p < num_players_; ++p) {
    if (!hands_[p].empty()) {
      return false;
    }
  }
  return true;
}

bool CribbageState::AllPlayersHavePassed() const {
  for (Player p = 0; p < num_players_; ++p) {
    if (!passed_[p]) {
      return false;
    }
  }
  return true;
}

void CribbageState::NextRound() {
  round_++;
  dealer_ = NextPlayerRoundRobin(dealer_, num_players_);
  start_player_ = NextPlayerRoundRobin(start_player_, num_players_);
  cur_player_ = kChancePlayerId;

  deck_.clear();
  deck_.resize(kDeckSize);
  for (int i = 0; i < kDeckSize; ++i) {
    deck_[i] = kAllCards[i];
  }

  for (int p = 0; p < num_players_; ++p) {
    hands_[p].clear();
    discards_[p].clear();
  }
  std::fill(passed_.begin(), passed_.end(), false);
  crib_.clear();
  played_cards_.clear();

  phase_ = Phase::kCardPhase;
  starter_ = std::nullopt;
  last_played_player_ = -1;
  current_sum_ = 0;
}

void CribbageState::ObservationTensor(Player player,
                                      absl::Span<float> values) const {}

CribbageState::CribbageState(std::shared_ptr<const Game> game)
    : State(game),
      parent_game_(static_cast<const CribbageGame&>(*game)),
      phase_(kCardPhase),
      rewards_(num_players_, 0),
      scores_(num_players_, 0),
      starter_(std::nullopt),
      hands_(num_players_),
      discards_(num_players_),
      passed_(num_players_) {
  NextRound();
}

int CribbageState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  } else {
    return cur_player_;
  }
}

bool SameRank(const std::vector<Card>& played_cards, int start_index) {
  int rank = played_cards[start_index].rank;
  for (int i = start_index + 1; i < played_cards.size(); ++i) {
    if (played_cards[i].rank != rank) {
      return false;
    }
  }
  return true;
}

bool IsUnsortedRun(const std::vector<Card>& played_cards, int start_index) {
  std::vector<Card> played_cards_copy = played_cards;
  std::sort(played_cards_copy.begin() + start_index, played_cards_copy.end());
  for (int i = start_index + 1; i < played_cards_copy.size(); ++i) {
    if (played_cards_copy[i].rank != (played_cards_copy[i - 1].rank + 1)) {
      return false;
    }
  }
  return true;
}

void CribbageState::CheckAndApplyPlayScoring() {
  if (current_sum_ == 15) {
    Score(cur_player_, 2);
  }

  // Check 4ofk, 3ofk, pair.
  if (played_cards_.size() >= 4 &&
      SameRank(played_cards_, played_cards_.size() - 4)) {
    Score(cur_player_, 12);
  } else if (played_cards_.size() >= 3 &&
             SameRank(played_cards_, played_cards_.size() - 3)) {
    Score(cur_player_, 6);
  } else if (played_cards_.size() >= 2 &&
             SameRank(played_cards_, played_cards_.size() - 2)) {
    Score(cur_player_, 2);
  }

  for (int num_cards = std::min<int>(played_cards_.size(), 7); num_cards >= 3;
       --num_cards) {
    if (IsUnsortedRun(played_cards_, played_cards_.size() - num_cards)) {
      Score(cur_player_, num_cards);
      break;
    }
  }
}

void CribbageState::DoEndOfPlayRound() {
  // Apply end-of-play round scoring.
  int end_of_round_points = current_sum_ == 31 ? 2 : 1;
  Score(last_played_player_, end_of_round_points);

  played_cards_.clear();
  current_sum_ = 0;
  std::fill(passed_.begin(), passed_.end(), false);
  SPIEL_CHECK_GE(last_played_player_, 0);
  SPIEL_CHECK_LT(last_played_player_, num_players_);
  cur_player_ = NextPlayerRoundRobin(last_played_player_, num_players_);

  // Check for end of play phase.
  if (AllHandsAreEmpty()) {
    // First, reset the hands to be the discards.
    for (Player p = 0; p < num_players_; ++p) {
      hands_[p] = discards_[p];
      SPIEL_CHECK_EQ(hands_[p].size(), 4);
    }
    ScoreHands();
    ScoreCrib();
    NextRound();
  }
}

void CribbageState::DoApplyAction(Action move) {
  SPIEL_CHECK_EQ(IsTerminal(), false);

  if (IsChanceNode()) {
    SPIEL_CHECK_GE(move, 0);
    SPIEL_CHECK_LT(move, kDeckSize);
    if (phase_ == Phase::kCardPhase) {
      // In the card phase, the chance nodes correspond to the card deals to
      // each player and to the crib.
      auto iter = std::find(deck_.begin(), deck_.end(), kAllCards[move]);
      SPIEL_CHECK_TRUE(iter != deck_.end());
      Card card = *iter;
      deck_.erase(iter);
      bool card_dealt = false;
      bool crib_dealt = false;

      // Deal to players first
      int p = 0;
      for (p = 0; p < num_players_; ++p) {
        if (hands_[p].size() < parent_game_.cards_per_player()) {
          hands_[p].push_back(card);
          card_dealt = true;
          break;
        }
      }

      // Deal to crib if necessary
      if (!card_dealt && crib_.size() < parent_game_.cards_to_crib()) {
        crib_.push_back(card);
        crib_dealt = true;
      }

      // Check if we're ready to start choosing cards.
      if (crib_dealt || (p == num_players_ - 1 &&
                         hands_[p].size() == parent_game_.cards_per_player() &&
                         crib_.size() == parent_game_.cards_to_crib())) {
        SortHands();
        cur_player_ = 0;
      } else {
        cur_player_ = kChancePlayerId;
      }
    } else {
      // A chance node in the play phase corresponds to choosing the starter.
      SPIEL_CHECK_FALSE(starter_.has_value());
      auto iter = std::find(deck_.begin(), deck_.end(), kAllCards[move]);
      SPIEL_CHECK_FALSE(iter == deck_.end());
      starter_ = *iter;
      deck_.erase(iter);
      if ((*starter_).rank == Rank::kJack) {
        // His Nobs
        Score(dealer_, 2);
      }
      // Player left of the dealer starts.
      cur_player_ = NextPlayerRoundRobin(dealer_, num_players_);
    }
  } else {
    // Decision node.
    SPIEL_CHECK_GE(cur_player_, 0);
    SPIEL_CHECK_LT(cur_player_, num_players_);
    // Applying action at decision node: First, clear the intermediate rewards.
    std::fill(rewards_.begin(), rewards_.end(), 0);
    if (phase_ == Phase::kCardPhase) {
      // Move the chose card(s) into the crib.
      if (num_players_ == 3 || num_players_ == 4) {
        SPIEL_CHECK_GE(move, 0);
        SPIEL_CHECK_LT(move, kDeckSize);
        MoveCardToCrib(cur_player_, kAllCards[move]);
      } else {
        std::pair<int, int> card_ids = FromAction(move);
        for (int card_id : {card_ids.first, card_ids.second}) {
          SPIEL_CHECK_GE(card_id, 0);
          SPIEL_CHECK_LT(card_id, kDeckSize);
          MoveCardToCrib(cur_player_, kAllCards[card_id]);
        }
      }

      cur_player_ += 1;
      if (cur_player_ >= num_players_) {
        SortCrib();
        phase_ = Phase::kPlayPhase;
        cur_player_ = kChancePlayerId;  // starter
      }
    } else {
      if (move == kPassAction) {
        passed_[cur_player_] = true;
        // Check for end of current play sequence (or round).
        if (AllPlayersHavePassed()) {
          DoEndOfPlayRound();
        } else {
          cur_player_ = NextPlayerRoundRobin(cur_player_, num_players_);
        }
      } else {
        // Play the chosen card.
        auto iter = std::find(hands_[cur_player_].begin(),
                              hands_[cur_player_].end(), kAllCards[move]);
        SPIEL_CHECK_TRUE(iter != hands_[cur_player_].end());
        Card card = *iter;
        current_sum_ += card.value();
        hands_[cur_player_].erase(iter);
        played_cards_.push_back(card);
        discards_[cur_player_].push_back(card);
        last_played_player_ = cur_player_;
        CheckAndApplyPlayScoring();
        // If the sum is 31 then no need for the passes, we can end the round
        // round right away.
        if (current_sum_ == 31) {
          DoEndOfPlayRound();
        } else {
          cur_player_ = NextPlayerRoundRobin(cur_player_, num_players_);
        }
      }
    }
  }
}

void CribbageState::Score(Player player, int points) {
  rewards_[player] += points;
  scores_[player] += points;

  // 4-player is a team game. Any scoring for p also counts for either (p+2)
  // or (p-2).
  if (num_players_ == 4) {
    Player teammate = (player + 2) % num_players_;
    SPIEL_CHECK_GE(teammate, 0);
    SPIEL_CHECK_LT(teammate, num_players_);
    rewards_[teammate] += points;
    scores_[teammate] += points;
  }
}

void CribbageState::ScoreHands() {
  for (Player p = 0; p < num_players_; ++p) {
    int points = ScoreHand(hands_[p], *starter_);
    Score(p, points);
  }
}

void CribbageState::ScoreCrib() {
  int points = ScoreHand(crib_, *starter_);
  Score(dealer_, points);
}

void CribbageState::MoveCardToCrib(Player player, const Card& card) {
  auto iter = std::find(hands_[player].begin(), hands_[player].end(), card);
  SPIEL_CHECK_TRUE(iter != hands_[player].end());
  Card found_card = *iter;
  hands_[player].erase(iter);
  crib_.push_back(found_card);
}

void CribbageState::SortHands() {
  for (int p = 0; p < num_players_; ++p) {
    std::sort(hands_[p].begin(), hands_[p].end());
  }
}

void CribbageState::SortCrib() { std::sort(crib_.begin(), crib_.end()); }

std::vector<Action> CribbageState::LegalActions() const {
  if (IsChanceNode()) {
    return LegalChanceOutcomes();
  } else if (IsTerminal()) {
    return {};
  } else {
    if (phase_ == Phase::kCardPhase) {
      switch (num_players_) {
        case 2:
          return LegalTwoCardCribActions();
        case 3:
        case 4:
          return LegalOneCardCribActions();
        default:
          SpielFatalError("Unknown number of players");
      }
    } else if (phase_ == Phase::kPlayPhase) {
      // The current player can play anything in their hand that does not bring
      // the current sum over 31, or pass if they have no legal actions.
      SPIEL_CHECK_GE(cur_player_, 0);
      SPIEL_CHECK_LT(cur_player_, num_players_);
      std::vector<Action> legal_actions;
      for (const Card& card : hands_[cur_player_]) {
        if ((current_sum_ + card.value()) <= 31) {
          legal_actions.push_back(card.id);
        }
      }
      if (legal_actions.empty()) {
        legal_actions = {kPassAction};
      }
      std::sort(legal_actions.begin(), legal_actions.end());
      return legal_actions;
    } else {
      SpielFatalError("Unknown phase in LegalActions()");
    }
  }
}

std::vector<Action> CribbageState::LegalOneCardCribActions() const {
  std::vector<Action> legal_actions;
  legal_actions.reserve(hands_[cur_player_].size());
  for (int i = 0; i < hands_[cur_player_].size(); ++i) {
    legal_actions.push_back(hands_[cur_player_][i].id);
  }
  std::sort(legal_actions.begin(), legal_actions.end());
  return legal_actions;
}

std::vector<Action> CribbageState::LegalTwoCardCribActions() const {
  std::vector<Action> legal_actions;
  for (int i = 0; i < hands_[cur_player_].size(); ++i) {
    for (int j = i + 1; j < hands_[cur_player_].size(); ++j) {
      Action action = ToAction(hands_[cur_player_][i], hands_[cur_player_][j]);
      legal_actions.push_back(action);
    }
  }
  std::sort(legal_actions.begin(), legal_actions.end());
  return legal_actions;
}

ActionsAndProbs CribbageState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  ActionsAndProbs outcomes;
  outcomes.reserve(deck_.size());
  double prob = 1.0 / deck_.size();
  for (int o = 0; o < deck_.size(); ++o) {
    outcomes.push_back({deck_[o].id, prob});
  }
  return outcomes;
}

std::string CribbageState::ToString() const {
  std::string str;
  absl::StrAppend(&str, "---------------------------------\n");
  absl::StrAppend(&str, "Num players: ", num_players_, "\n");
  absl::StrAppend(&str, "Round: ", round_, "\n");
  absl::StrAppend(
      &str, "Phase: ", phase_ == Phase::kCardPhase ? "Card" : "Play", "\n");
  absl::StrAppend(&str, "Dealer: ", dealer_, "\n");
  absl::StrAppend(&str, "Cur player: ", cur_player_, "\n");
  absl::StrAppend(&str, "Scores:");
  for (int p = 0; p < num_players_; ++p) {
    absl::StrAppend(&str, " ", scores_[p]);
  }
  absl::StrAppend(&str, "\n");
  absl::StrAppend(&str, "---------------------------------\n");
  absl::StrAppend(&str, "Crib:");
  for (int i = 0; i < crib_.size(); ++i) {
    absl::StrAppend(&str, " ", crib_[i].to_string());
  }
  absl::StrAppend(&str, "\n");
  if (starter_.has_value()) {
    absl::StrAppend(&str, "Starter: ", (*starter_).to_string(), "\n");
  }
  for (int p = 0; p < num_players_; ++p) {
    absl::StrAppend(&str, "P", p, " Hand:");
    for (int i = 0; i < hands_[p].size(); ++i) {
      absl::StrAppend(&str, " ", hands_[p][i].to_string());
    }
    absl::StrAppend(&str, "\n");
  }
  absl::StrAppend(&str, "---------------------------------\n");
  absl::StrAppend(&str, "Running total: ", current_sum_, "\n");
  absl::StrAppend(&str, "Played cards: ");
  for (int i = 0; i < played_cards_.size(); ++i) {
    absl::StrAppend(&str, " ", played_cards_[i].to_string());
  }
  absl::StrAppend(&str, "\n");
  absl::StrAppend(&str, "---------------------------------\n");

  return str;
}

std::unique_ptr<State> CribbageState::Clone() const {
  return std::unique_ptr<State>(new CribbageState(*this));
}

CribbageGame::CribbageGame(const GameParameters& params)
    : Game(kGameType, params),
      num_players_(ParameterValue<int>("players", kDefaultNumPlayers)),
      cards_per_player_(CardsPerPlayer(num_players_)),
      cards_to_crib_(CardsToCrib(num_players_)),
      winner_bonus_reward_(
          ParameterValue<double>("winner_bonus_reward", kDefaultWinnerBonus)) {}

}  // namespace cribbage
}  // namespace open_spiel
