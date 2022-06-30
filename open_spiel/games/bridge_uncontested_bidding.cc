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

#include "open_spiel/games/bridge_uncontested_bidding.h"

#include <cstring>
#include <memory>

#include "open_spiel/games/bridge/double_dummy_solver/include/dll.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/bridge/bridge_scoring.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// For compatibility with versions of the double dummy solver code which
// don't amend exported names.
#ifndef DDS_EXTERNAL
#define DDS_EXTERNAL(x) x
#endif

namespace open_spiel {
namespace bridge_uncontested_bidding {
namespace {

using open_spiel::bridge::kClubs;
using open_spiel::bridge::kDenominationChar;
using open_spiel::bridge::kDiamonds;
using open_spiel::bridge::kHearts;
using open_spiel::bridge::kNoTrump;
using open_spiel::bridge::kSpades;
using open_spiel::bridge::kUndoubled;

constexpr int kDefaultNumRedeals = 10;  // how many possible layouts to analyse

const GameType kGameType{
    /*short_name=*/"bridge_uncontested_bidding",
    /*long_name=*/"Bridge: Uncontested Bidding",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kSampledStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kIdentical,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/kNumPlayers,
    /*min_num_players=*/kNumPlayers,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/false,
    /*provides_observation_tensor=*/false,
    /*parameter_specification=*/
    {
        {"subgame", GameParameter(static_cast<std::string>(""))},
        {"rng_seed", GameParameter(0)},
        {"relative_scoring", GameParameter(false)},
        {"num_redeals", GameParameter(kDefaultNumRedeals)},
    },
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new UncontestedBiddingGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

constexpr Action kPass = 0;
constexpr Action k2NT = 10;

bool Is2NTDeal(const Deal& deal) {
  int lengths[kNumSuits] = {0, 0, 0, 0};
  int hcp = 0;
  for (int i = 0; i < kNumCardsPerHand; ++i) {
    int suit = deal.Suit(i);
    int rank = deal.Rank(i);
    lengths[suit]++;
    if (rank > 8) hcp += (rank - 8);
  }
  // Balanced means 4333, 4432 or 5332
  bool is_balanced = (lengths[0] * lengths[1] * lengths[2] * lengths[3] >= 90);
  return is_balanced && (20 <= hcp) && (hcp <= 21);
}
bool NoFilter(const Deal& deal) { return true; }

}  // namespace

int UncontestedBiddingState::CurrentPlayer() const {
  if (!dealt_) return kChancePlayerId;
  if (IsTerminal()) return kTerminalPlayerId;
  return actions_.size() % 2;
}

constexpr bridge::Denomination Denomination(Action bid) {
  return bridge::Denomination((bid - 1) % kNumDenominations);
}

constexpr int Level(Action bid) { return 1 + (bid - 1) / kNumDenominations; }

std::string UncontestedBiddingState::ActionToString(Player player,
                                                    Action action_id) const {
  if (player == kChancePlayerId) return "Deal";
  if (action_id == kPass) return "Pass";
  return absl::StrCat(
      Level(action_id),
      std::string(1, kDenominationChar[Denomination(action_id)]));
}

Action ActionFromString(const std::string& str) {
  if (str == "Pass") return kPass;
  SPIEL_CHECK_EQ(str.length(), 2);
  auto level = str[0] - '0';
  auto denomination = std::string(kDenominationChar).find(str[1]);
  SPIEL_CHECK_NE(denomination, std::string::npos);
  return (level - 1) * kNumDenominations + denomination + 1;
}

std::string Deal::HandString(int begin, int end) const {
  bool cards[kNumSuits][kNumCardsPerSuit] = {{false}};
  for (int i = begin; i < end; ++i) {
    cards[Suit(i)][Rank(i)] = true;
  }
  std::string hand;
  for (int s = 3; s >= 0; --s) {
    for (int r = 12; r >= 0; --r) {
      if (cards[s][r]) {
        hand.push_back(kRankChar[r]);
      }
    }
    if (s) hand.push_back('.');
  }
  return hand;
}

std::string UncontestedBiddingState::ToString() const {
  if (!dealt_) return "";
  std::string rv = absl::StrCat(deal_.HandString(0, 13), " ",
                                deal_.HandString(13, 26), " ", AuctionString());
  if (IsTerminal()) {
    absl::StrAppend(&rv, " Score:", score_);
    for (int i = 0; i < reference_contracts_.size(); ++i) {
      absl::StrAppend(&rv, " ", reference_contracts_[i].ToString(), ":",
                      reference_scores_[i]);
    }
  }
  return rv;
}

bool UncontestedBiddingState::IsTerminal() const {
  return dealt_ && actions_.size() >= 2 && actions_.back() == kPass;
}

std::vector<double> UncontestedBiddingState::Returns() const {
  if (!IsTerminal()) return {0, 0};
  double v = score_;
  if (reference_scores_.empty()) {
    return {v, v};
  } else {
    const double datum =
        *std::max_element(reference_scores_.begin(), reference_scores_.end());
    return {v, v - datum};
  }
}

std::string UncontestedBiddingState::AuctionString() const {
  std::string actions;
  for (const auto action : actions_) {
    if (!actions.empty()) actions.push_back('-');
    actions.append(ActionToString(0, action));
  }
  return actions;
}

std::string UncontestedBiddingState::InformationStateString(
    Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  if (!dealt_) return "";
  return absl::StrCat(deal_.HandString(player * 13, (player + 1) * 13), " ",
                      AuctionString());
}

void UncontestedBiddingState::InformationStateTensor(
    Player player, absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  SPIEL_CHECK_EQ(values.size(), kStateSize);
  std::fill(values.begin(), values.end(), 0.);
  auto ptr = values.begin();

  for (int i = kNumCardsPerHand * player; i < kNumCardsPerHand * (1 + player);
       ++i) {
    ptr[deal_.Card(i)] = 1.;
  }
  ptr += kNumCards;

  // What actions have been taken, and by whom
  for (int i = 0; i < actions_.size(); ++i) {
    ptr[actions_[i] * kNumPlayers + (i % kNumPlayers)] = 1;
  }
  ptr += kNumActions * kNumPlayers;

  // Which player we are
  ptr[player] = 1;
  ptr += kNumPlayers;
}

std::unique_ptr<State> UncontestedBiddingState::Clone() const {
  return std::unique_ptr<State>(new UncontestedBiddingState(*this));
}

std::vector<Action> UncontestedBiddingState::LegalActions() const {
  if (IsTerminal()) {
    return {};
  } else if (dealt_) {
    std::vector<Action> actions{kPass};
    const Action prev = actions_.empty() ? kPass : actions_.back();
    for (Action a = prev + 1; a < kNumActions; ++a) actions.push_back(a);
    return actions;
  } else {
    return {0};
  }
}

void UncontestedBiddingState::ScoreDeal() {
  // If both Pass, the score is zero.
  const bool passed_out = (actions_.size() == 2);
  if (passed_out && reference_contracts_.empty()) {
    score_ = 0;
    return;
  }

  // Determine the final contract and declarer
  const Action bid = actions_[actions_.size() - 2];
  Contract contract{passed_out ? 0 : Level(bid),
                    passed_out ? kNoTrump : Denomination(bid), kUndoubled};
  for (int i = 0; i < actions_.size(); ++i) {
    if (actions_[i] > 0 && Denomination(actions_[i]) == contract.trumps) {
      contract.declarer = i % 2;
      break;
    }
  }

  // Populate East-West cards
  ddTableDeal dd_table_deal{};
  for (Player player = 0; player < kNumPlayers; ++player) {
    for (int i = kNumCardsPerHand * player; i < kNumCardsPerHand * (1 + player);
         ++i) {
      dd_table_deal.cards[player * 2][deal_.Suit(i)] += 1
                                                        << (2 + deal_.Rank(i));
    }
  }

  // Initialize scores to zero
  score_ = 0;
  reference_scores_.resize(reference_contracts_.size());
  std::fill(reference_scores_.begin(), reference_scores_.end(), 0);

  // For each redeal
  for (int ideal = 0; ideal < num_redeals_; ++ideal) {
    if (ideal > 0) deal_.Shuffle(&rng_, kNumCardsPerHand * 2, kNumCards);

    // Populate (reshuffled) North-South cards
    for (int opponent = 0; opponent < kNumPlayers; ++opponent) {
      std::fill(dd_table_deal.cards[1 + opponent * 2],
                dd_table_deal.cards[1 + opponent * 2] + 4, 0);
      for (int i = kNumCardsPerHand * (2 + opponent);
           i < kNumCardsPerHand * (3 + opponent); ++i) {
        dd_table_deal.cards[1 + opponent * 2][deal_.Suit(i)] +=
            1 << (2 + deal_.Rank(i));
      }
    }

    // Analyze the deal.
    DDS_EXTERNAL(SetMaxThreads)(0);
    struct ddTableResults results;
    const int return_code = DDS_EXTERNAL(CalcDDtable)(dd_table_deal, &results);

    // Check for errors.
    if (return_code != RETURN_NO_FAULT) {
      char error_message[80];
      DDS_EXTERNAL(ErrorMessage)(return_code, error_message);
      SpielFatalError(absl::StrCat("double_dummy_solver:", error_message));
    }

    // Compute the score and update the total.
    if (!passed_out) {
      const int declarer_tricks =
          results.resTable[contract.trumps][2 * contract.declarer];
      const int declarer_score =
          Score(contract, declarer_tricks, /*is_vulnerable=*/false);
      score_ += static_cast<double>(declarer_score) / num_redeals_;
    }

    // Compute the scores for reference contracts.
    for (int i = 0; i < reference_contracts_.size(); ++i) {
      const int declarer_tricks =
          results.resTable[reference_contracts_[i].trumps]
                          [2 * reference_contracts_[i].declarer];
      const int declarer_score = Score(reference_contracts_[i], declarer_tricks,
                                       /*is_vulnerable=*/false);
      reference_scores_[i] +=
          static_cast<double>(declarer_score) / num_redeals_;
    }
  }
}

void UncontestedBiddingState::DoApplyAction(Action action_id) {
  if (dealt_) {
    actions_.push_back(action_id);
    if (IsTerminal()) ScoreDeal();
  } else {
    do {
      deal_.Shuffle(&rng_);
    } while (!deal_filter_(deal_));
    dealt_ = true;
  }
}

std::vector<std::pair<Action, double>> UncontestedBiddingState::ChanceOutcomes()
    const {
  return {{0, 1.0}};
}

UncontestedBiddingGame::UncontestedBiddingGame(const GameParameters& params)
    : Game(kGameType, params),
      forced_actions_{},
      deal_filter_{NoFilter},
      rng_seed_(ParameterValue<int>("rng_seed")),
      num_redeals_(ParameterValue<int>("num_redeals")) {
  std::string subgame = ParameterValue<std::string>("subgame");
  if (subgame == "2NT") {
    deal_filter_ = Is2NTDeal;
    forced_actions_ = {k2NT};
    if (ParameterValue<bool>("relative_scoring")) {
      reference_contracts_ = {
          {2, kNoTrump, kUndoubled, 0},  {3, kClubs, kUndoubled, 1},
          {3, kDiamonds, kUndoubled, 0}, {3, kDiamonds, kUndoubled, 1},
          {3, kHearts, kUndoubled, 0},   {3, kHearts, kUndoubled, 1},
          {3, kSpades, kUndoubled, 0},   {3, kSpades, kUndoubled, 1},
          {3, kNoTrump, kUndoubled, 0},  {4, kClubs, kUndoubled, 0},
          {4, kHearts, kUndoubled, 0},   {4, kHearts, kUndoubled, 1},
          {4, kSpades, kUndoubled, 0},   {4, kSpades, kUndoubled, 1},
          {5, kClubs, kUndoubled, 0},    {5, kClubs, kUndoubled, 1},
          {5, kDiamonds, kUndoubled, 0}, {5, kDiamonds, kUndoubled, 1},
          {6, kClubs, kUndoubled, 0},    {6, kClubs, kUndoubled, 1},
          {6, kDiamonds, kUndoubled, 0}, {6, kDiamonds, kUndoubled, 1},
          {6, kHearts, kUndoubled, 0},   {6, kHearts, kUndoubled, 1},
          {6, kSpades, kUndoubled, 0},   {6, kSpades, kUndoubled, 1},
          {6, kNoTrump, kUndoubled, 0},  {7, kClubs, kUndoubled, 0},
          {7, kClubs, kUndoubled, 1},    {7, kDiamonds, kUndoubled, 0},
          {7, kDiamonds, kUndoubled, 1}, {7, kHearts, kUndoubled, 0},
          {7, kHearts, kUndoubled, 1},   {7, kSpades, kUndoubled, 0},
          {7, kSpades, kUndoubled, 1},   {7, kNoTrump, kUndoubled, 0}};
    }
  } else {
    SPIEL_CHECK_EQ(subgame, "");
    if (ParameterValue<bool>("relative_scoring")) {
      reference_contracts_ = {
          {0, kNoTrump, kUndoubled, 0},  {1, kClubs, kUndoubled, 0},
          {1, kClubs, kUndoubled, 1},    {1, kDiamonds, kUndoubled, 0},
          {1, kDiamonds, kUndoubled, 1}, {1, kHearts, kUndoubled, 0},
          {1, kHearts, kUndoubled, 1},   {1, kSpades, kUndoubled, 0},
          {1, kSpades, kUndoubled, 1},   {1, kNoTrump, kUndoubled, 0},
          {1, kNoTrump, kUndoubled, 1},  {3, kNoTrump, kUndoubled, 0},
          {3, kNoTrump, kUndoubled, 1},  {4, kHearts, kUndoubled, 0},
          {4, kHearts, kUndoubled, 1},   {4, kSpades, kUndoubled, 0},
          {4, kSpades, kUndoubled, 1},   {5, kClubs, kUndoubled, 0},
          {5, kClubs, kUndoubled, 1},    {5, kDiamonds, kUndoubled, 0},
          {5, kDiamonds, kUndoubled, 1}, {6, kClubs, kUndoubled, 0},
          {6, kClubs, kUndoubled, 1},    {6, kDiamonds, kUndoubled, 0},
          {6, kDiamonds, kUndoubled, 1}, {6, kHearts, kUndoubled, 0},
          {6, kHearts, kUndoubled, 1},   {6, kSpades, kUndoubled, 0},
          {6, kSpades, kUndoubled, 1},   {6, kNoTrump, kUndoubled, 0},
          {6, kNoTrump, kUndoubled, 1},  {7, kClubs, kUndoubled, 0},
          {7, kClubs, kUndoubled, 1},    {7, kDiamonds, kUndoubled, 0},
          {7, kDiamonds, kUndoubled, 1}, {7, kHearts, kUndoubled, 0},
          {7, kHearts, kUndoubled, 1},   {7, kSpades, kUndoubled, 0},
          {7, kSpades, kUndoubled, 1},   {7, kNoTrump, kUndoubled, 0},
          {7, kNoTrump, kUndoubled, 1}};
    }
  }
}

// Deserialize the deal and auction
// e.g. "AKQJ.543.QJ8.T92 97532.A2.9.QJ853 2N-3C"
std::unique_ptr<State> UncontestedBiddingGame::DeserializeState(
    const std::string& str) const {
  if (str.empty()) {
    return absl::make_unique<UncontestedBiddingState>(
        shared_from_this(), reference_contracts_, deal_filter_, forced_actions_,
        rng_seed_, num_redeals_);
  }
  SPIEL_CHECK_GE(str.length(),
                 kNumPlayers * (kNumCardsPerHand + kNumSuits) - 1);
  std::array<int, kNumCards> cards{};
  std::array<int, kNumCards> cards_dealt{};
  for (Player player = 0; player < kNumPlayers; ++player) {
    int suit = 0;
    int start = player * (kNumCardsPerHand + kNumSuits);
    for (int i = 0; i < kNumCardsPerHand; ++i) {
      char ch = str[start + i + suit];
      while (ch == '.') {
        ++suit;
        ch = str[start + i + suit];
      }
      const int rank = (std::strchr(kRankChar, ch) - kRankChar);
      const int card = rank * 4 + (3 - suit);
      SPIEL_CHECK_FALSE(cards_dealt[card]);
      cards[player * kNumCardsPerHand + i] = card;
      cards_dealt[card] = true;
    }
  }
  int i = kNumPlayers * kNumCardsPerHand;
  for (int c = 0; c < 52; ++c) {
    if (!cards_dealt[c]) cards[i++] = c;
  }

  // Get any actions there may be.
  std::vector<Action> actions;
  int start = kNumPlayers * (kNumCardsPerHand + kNumSuits);
  while (start < str.length()) {
    auto end = str.find('-', start);
    if (end == std::string::npos) end = str.length();
    actions.push_back(ActionFromString(str.substr(start, end - start)));
    start = end + 1;
  }

  // Check that early actions agree with the forced actions in this game.
  SPIEL_CHECK_GE(actions.size(), forced_actions_.size());
  for (int i = 0; i < forced_actions_.size(); ++i) {
    SPIEL_CHECK_EQ(actions[i], forced_actions_[i]);
  }

  return absl::make_unique<UncontestedBiddingState>(
      shared_from_this(), reference_contracts_, Deal(cards), actions, rng_seed_,
      num_redeals_);
}

std::string UncontestedBiddingGame::GetRNGState() const {
  return std::to_string(rng_seed_);
}

void UncontestedBiddingGame::SetRNGState(const std::string& rng_state) const {
  if (rng_state.empty()) return;
  SPIEL_CHECK_TRUE(absl::SimpleAtoi(rng_state, &rng_seed_));
}

}  // namespace bridge_uncontested_bidding
}  // namespace open_spiel
