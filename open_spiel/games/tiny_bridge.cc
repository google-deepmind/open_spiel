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

#include "open_spiel/games/tiny_bridge.h"

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/algorithms/minimax.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace tiny_bridge {
namespace {

constexpr std::array<const char*, kNumActions> kActionStr{
    "Pass", "1H", "1S", "1NT", "2H", "2S", "2NT", "Dbl", "RDbl"};
constexpr std::array<char, kNumRanks> kRankChar{'J', 'Q', 'K', 'A'};
constexpr std::array<char, 1 + kNumSuits> kSuitChar{'H', 'S', 'N'};
constexpr std::array<char, kNumHands> kHandChar{'W', 'N', 'E', 'S'};

int Suit(int card) { return card / kNumRanks; }
int Rank(int card) { return card % kNumRanks; }

int CharToRank(char c) {
  switch (c) {
    case 'J':
      return 0;
    case 'Q':
      return 1;
    case 'K':
      return 2;
    case 'A':
      return 3;
  }
  SpielFatalError(absl::StrCat("Unknown rank '", std::string(1, c), "'"));
}

int CharToTrumps(char c) {
  switch (c) {
    case 'H':
      return 0;
    case 'S':
      return 1;
    case 'N':  // No-trump
      return 2;
  }
  SpielFatalError(absl::StrCat("Unknown trump suit '", std::string(1, c), "'"));
}

int CharToHand(char c) {
  switch (c) {
    case 'W':
      return 0;
    case 'N':
      return 1;
    case 'E':
      return 2;
    case 'S':
      return 3;
  }
  SpielFatalError(absl::StrCat("Unknown hand '", std::string(1, c), "'"));
}

int StringToCard(const std::string& s) {
  return CharToRank(s[1]) + kNumRanks * CharToTrumps(s[0]);
}

std::string CardString(int card) {
  return absl::StrCat(std::string(1, kSuitChar[Suit(card)]),
                      std::string(1, kRankChar[Rank(card)]));
}

// Facts about the game
const GameType kGameType2p{
    /*short_name=*/"tiny_bridge_2p",
    /*long_name=*/"Tiny Bridge (Uncontested)",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kIdentical,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state=*/true,
    /*provides_information_state_as_normalized_vector=*/true,
    /*provides_observation=*/true,
    /*provides_observation_as_normalized_vector=*/true,
    /*parameter_specification=*/{}  // no parameters
};

const GameType kGameType4p{
    /*short_name=*/"tiny_bridge_4p",
    /*long_name=*/"Tiny Bridge (Contested)",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/4,
    /*min_num_players=*/4,
    /*provides_information_state=*/true,
    /*provides_information_state_as_normalized_vector=*/false,
    /*provides_observation=*/true,
    /*provides_observation_as_normalized_vector=*/false,
    /*parameter_specification=*/{}  // no parameters
};

// Game for the play of the cards. We don't register this - it is for internal
// use only, computing the payoff of a tiny bridge auction.
const GameType kGameTypePlay{
    /*short_name=*/"tiny_bridge_play",
    /*long_name=*/"Tiny Bridge (Play Phase)",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state=*/false,
    /*provides_information_state_as_normalized_vector=*/false,
    /*provides_observation=*/false,
    /*provides_observation_as_normalized_vector=*/false,
    /*parameter_specification=*/
    {
        {"trumps", {GameParameter::Type::kString, true}},
        {"leader", {GameParameter::Type::kString, true}},
        {"hand_W", {GameParameter::Type::kString, true}},
        {"hand_N", {GameParameter::Type::kString, true}},
        {"hand_E", {GameParameter::Type::kString, true}},
        {"hand_S", {GameParameter::Type::kString, true}},
    }};

std::unique_ptr<Game> Factory2p(const GameParameters& params) {
  return std::unique_ptr<Game>(new TinyBridgeGame2p(params));
}

std::unique_ptr<Game> Factory4p(const GameParameters& params) {
  return std::unique_ptr<Game>(new TinyBridgeGame4p(params));
}

REGISTER_SPIEL_GAME(kGameType2p, Factory2p);
REGISTER_SPIEL_GAME(kGameType4p, Factory4p);

// Score a played-out hand.
// Score is 1 for the second trick, 2 for slam (bidding and making two).
// -1 per undertrick. Doubling and redoubling each multiply the score by 2.
int Score(int contract, int tricks, bool doubled, bool redoubled) {
  const int contract_tricks = 1 + (contract - 1) / 3;
  const int contract_result = tricks - contract_tricks;
  const int double_factor = (1 + doubled) * (1 + redoubled);
  if (contract_result < 0) return double_factor * contract_result;
  int score = contract_result;
  if (contract_tricks == 2) score += 3;
  return score * double_factor;
}

}  // namespace

TinyBridgeGame2p::TinyBridgeGame2p(const GameParameters& params)
    : Game(kGameType2p, params) {}

std::unique_ptr<State> TinyBridgeGame2p::NewInitialState() const {
  return std::unique_ptr<State>(
      new TinyBridgeAuctionState(NumDistinctActions(), NumPlayers()));
}

TinyBridgeGame4p::TinyBridgeGame4p(const GameParameters& params)
    : Game(kGameType4p, params) {}

std::unique_ptr<State> TinyBridgeGame4p::NewInitialState() const {
  return std::unique_ptr<State>(
      new TinyBridgeAuctionState(NumDistinctActions(), NumPlayers()));
}

TinyBridgePlayGame::TinyBridgePlayGame(const GameParameters& params)
    : Game(kGameTypePlay, params) {}

std::unique_ptr<State> TinyBridgePlayGame::NewInitialState() const {
  int trumps = CharToTrumps(ParameterValue<std::string>("trumps")[0]);
  int leader = CharToHand(ParameterValue<std::string>("leader")[0]);
  std::array<int, kNumCards> holder;
  for (int i = 0; i < kNumHands; ++i) {
    std::string hand = ParameterValue<std::string>(
        absl::StrCat("hand_", std::string(1, kHandChar[i])));
    for (int j = 0; j < kNumTricks; ++j) {
      int c = StringToCard(hand.substr(j * 2, 2));
      holder[c] = i;
    }
  }
  return std::unique_ptr<State>(new TinyBridgePlayState(
      NumDistinctActions(), NumPlayers(), trumps, leader, holder));
}

std::string TinyBridgeAuctionState::HandString(Player player) const {
  if (!IsDealt(player)) return "??";
  return ActionToString(kChancePlayerId, actions_[player]);
}

std::string TinyBridgeAuctionState::DealString() const {
  std::string deal;
  for (auto player = Player{0}; player < num_players_; ++player) {
    if (player != 0) deal.push_back(' ');
    absl::StrAppend(&deal, HandName(player), ":", HandString(player));
  }
  return deal;
}

TinyBridgeAuctionState::AuctionState TinyBridgeAuctionState::AnalyzeAuction()
    const {
  AuctionState rv;
  rv.last_bid = Call::kPass;
  rv.last_bidder = kInvalidPlayer;
  rv.doubler = kInvalidPlayer;
  rv.redoubler = kInvalidPlayer;
  for (int i = num_players_; i < actions_.size(); ++i) {
    if (actions_[i] == Call::kDouble) {
      rv.doubler = i % num_players_;
    } else if (actions_[i] == Call::kRedouble) {
      rv.redoubler = i % num_players_;
    } else if (actions_[i] != Call::kPass) {
      rv.last_bid = actions_[i];
      rv.last_bidder = i % num_players_;
      rv.doubler = kInvalidPlayer;
      rv.redoubler = kInvalidPlayer;
    }
  }
  return rv;
}

int TinyBridgeAuctionState::Score_p0(std::array<int, kNumCards> holder) const {
  auto state = AnalyzeAuction();
  TinyBridgePlayGame game{{}};
  int trumps = (state.last_bid - 1) % 3;
  int leader = num_players_ == 2 ? (state.last_bidder * 2 + 1)
                                 : (state.last_bidder + 1) % 4;
  int decl = num_players_ == 2 ? 0 : state.last_bidder % 2;
  TinyBridgePlayState play{num_distinct_actions_, num_players_, trumps, leader,
                           holder};
  const int tricks =
      algorithms::AlphaBetaSearch(game, &play, nullptr, -1, decl).first;
  const int declarer_score =
      Score(state.last_bid, tricks, state.doubler != kInvalidPlayer,
            state.redoubler != kInvalidPlayer);
  if (num_players_ == 2)
    return declarer_score;
  else
    return (state.last_bidder % 2 == 0) ? declarer_score : -declarer_score;
}

std::array<int, kNumCards> TinyBridgeAuctionState::CardHolders() const {
  std::array<int, kNumCards> holder;
  std::fill(holder.begin(), holder.end(), kInvalidPlayer);
  for (int i = 0; i < actions_.size() && i < num_players_; ++i) {
    int action_id = actions_[i];
    const int card1 = action_id % kNumCards;
    const int card2 = action_id / kNumCards;
    holder[card1] = i;
    holder[card2] = i;
  }
  return holder;
}

void TinyBridgeAuctionState::DoApplyAction(Action action) {
  actions_.push_back(action);
  if (num_players_ == 2) {
    if (actions_.size() >= 2 * num_players_ &&
        actions_[actions_.size() - 1] == Call::kPass) {
      is_terminal_ = true;
      if (actions_[actions_.size() - 2] == Call::kPass) {
        utility_p0 = 0;
      } else {
        // Evaluate on every possible distribution of the hidden cards and take
        // the mean.
        const double freq = 1. / 6;
        utility_p0 = 0;
        std::array<int, kNumCards> holders_2p = CardHolders();
        std::array<int, kNumCards> holders_4p;
        for (int n0 = 0; n0 < 3; ++n0) {
          for (int n1 = n0 + 1; n1 < 4; ++n1) {
            int n = 0;
            for (int i = 0; i < kNumCards; ++i) {
              if (holders_2p[i] == kInvalidPlayer) {
                holders_4p[i] = (n == n0 || n == n1) ? 1 : 3;
                ++n;
              } else {
                holders_4p[i] = 2 * holders_2p[i];
              }
            }
            utility_p0 += Score_p0(holders_4p) * freq;
          }
        }
      }
    }
  } else {
    if (actions_.size() >= 2 * num_players_ &&
        actions_[actions_.size() - 1] == Call::kPass &&
        actions_[actions_.size() - 2] == Call::kPass &&
        actions_[actions_.size() - 3] == Call::kPass) {
      is_terminal_ = true;
      if (actions_.size() == 2 * num_players_) {
        utility_p0 = 0;
      } else {
        utility_p0 = Score_p0(CardHolders());
      }
    }
  }
}

std::vector<Action> TinyBridgeAuctionState::LegalActions() const {
  std::vector<Action> actions;
  if (IsChanceNode()) {
    return LegalChanceOutcomes();
  } else if (IsTerminal()) {
    return {};
  } else {
    auto state = AnalyzeAuction();
    actions.push_back(Call::kPass);
    for (int bid = state.last_bid + 1; bid <= Call::k2NT; ++bid) {
      actions.push_back(bid);
    }
    if (num_players_ == 4 && state.last_bidder != kInvalidPlayer) {
      if (state.last_bidder % 2 != CurrentPlayer() % 2) {
        if (state.doubler == kInvalidPlayer) actions.push_back(Call::kDouble);
      } else {
        if (state.doubler != kInvalidPlayer &&
            state.redoubler == kInvalidPlayer)
          actions.push_back(Call::kRedouble);
      }
    }
  }
  return actions;
}

std::vector<std::pair<Action, double>> TinyBridgeAuctionState::ChanceOutcomes()
    const {
  std::vector<Action> actions;
  auto holder = CardHolders();
  for (int card1 = 0; card1 < kNumCards; ++card1) {
    if (holder[card1] != kInvalidPlayer) continue;
    for (int card2 = card1 + 1; card2 < kNumCards; ++card2) {
      if (holder[card2] != kInvalidPlayer) continue;
      actions.push_back(card2 * kNumCards + card1);
    }
  }
  const int num_actions = actions.size();
  std::vector<std::pair<Action, double>> outcomes;
  outcomes.reserve(num_actions);
  for (auto action : actions) {
    outcomes.emplace_back(action, 1.0 / num_actions);
  }
  return outcomes;
}

std::string TinyBridgeAuctionState::ActionToString(Player player,
                                                   Action action_id) const {
  if (player == kChancePlayerId) {
    const int card1 = action_id % kNumCards;
    const int card2 = action_id / kNumCards;
    return absl::StrCat(CardString(card1), CardString(card2));
  } else {
    return kActionStr[action_id];
  }
}

int TinyBridgeAuctionState::CurrentPlayer() const {
  if (IsTerminal()) return kTerminalPlayerId;
  return actions_.size() < num_players_ ? kChancePlayerId
                                        : actions_.size() % num_players_;
}

std::string TinyBridgeAuctionState::HandName(Player player) const {
  if (num_players_ == 2)
    return std::string(1, kHandChar[player * 2]);
  else if (num_players_ == 4)
    return std::string(1, kHandChar[player]);
  else
    SpielFatalError("Invalid number of players");
}

std::string TinyBridgeAuctionState::AuctionString() const {
  std::string auction{};
  for (int i = num_players_; i < actions_.size(); ++i) {
    if (!auction.empty()) auction.push_back('-');
    auction.append(ActionToString(i % num_players_, actions_[i]));
  }
  return auction;
}

std::string TinyBridgeAuctionState::ToString() const {
  std::string deal = DealString();
  std::string auction = AuctionString();
  if (!auction.empty())
    return absl::StrCat(deal, " ", auction);
  else
    return deal;
}

bool TinyBridgeAuctionState::IsTerminal() const { return is_terminal_; }

std::vector<double> TinyBridgeAuctionState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>(num_players_, 0.0);
  }

  if (num_players_ == 2) {
    return {utility_p0, utility_p0};
  } else {
    // 4 player version.
    return {utility_p0, -utility_p0, utility_p0, -utility_p0};
  }
}

std::string TinyBridgeAuctionState::InformationState(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  std::string hand = absl::StrCat(HandName(player), ":", HandString(player));
  std::string auction = AuctionString();
  if (!auction.empty())
    return absl::StrCat(hand, " ", auction);
  else
    return hand;
}

// Observation string is the player's cards plus the most recent bid,
// plus any doubles or redoubles. E.g. "W:HJSA 2NT:E Dbl:S RDbl:W"
// This is an observation for West, who holds HJ and SA.
// The most recent bid is 2NT by East, which has been doubled by South
// and redoubled by West.
std::string TinyBridgeAuctionState::Observation(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  std::string observation =
      absl::StrCat(HandName(player), ":", HandString(player));
  if (HasAuctionStarted()) {
    auto state = AnalyzeAuction();
    absl::StrAppend(&observation, " ",
                    ActionToString(state.last_bidder, state.last_bid), ":",
                    HandName(state.last_bidder));
    if (state.doubler != kInvalidPlayer)
      absl::StrAppend(&observation, " ", "Dbl:", HandName(state.doubler));
    if (state.redoubler != kInvalidPlayer)
      absl::StrAppend(&observation, " ", "RDbl:", HandName(state.redoubler));
  }
  return observation;
}

// Information state vector consists of:
//   kNumCards bits showing which cards the observing player holds
//   kNumActions2p*2 bits showing which actions have been taken in the game.
//     For each action, the bits are [1, 0] if we took the action,
//     [0, 1] if our partner took the action, and otherwise [0, 0].
void TinyBridgeAuctionState::InformationStateAsNormalizedVector(
    Player player, std::vector<double>* values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  SPIEL_CHECK_EQ(num_players_, 2);
  values->resize(kNumCards + kNumActions2p * 2);
  std::fill(values->begin(), values->end(), 0);
  if (IsDealt(player)) {
    // The chance action is card_0 * kNumCards + card_1
    values->at(actions_[player] % kNumCards) = 1;
    values->at(actions_[player] / kNumCards) = 1;
  }
  for (int i = num_players_; i < actions_.size(); ++i) {
    values->at(kNumCards + actions_[i] * 2 + (i - player) % num_players_) = 1;
  }
}

// Information state vector consists of:
//   kNumCards bits showing which cards the observing player holds
//   kNumActions2p bits showing the most recent action (one-hot)
void TinyBridgeAuctionState::ObservationAsNormalizedVector(
    Player player, std::vector<double>* values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  SPIEL_CHECK_EQ(num_players_, 2);
  values->resize(kNumCards + kNumActions2p);
  std::fill(values->begin(), values->end(), 0);
  if (IsDealt(player)) {
    // The chance action is card_0 * kNumCards + card_1
    values->at(actions_[player] % kNumCards) = 1;
    values->at(actions_[player] / kNumCards) = 1;
  }
  if (HasAuctionStarted()) {
    values->at(kNumCards + actions_.back()) = 1;
  }
}

std::unique_ptr<State> TinyBridgeAuctionState::Clone() const {
  return std::unique_ptr<State>{new TinyBridgeAuctionState(*this)};
}

void TinyBridgeAuctionState::UndoAction(Player player, Action action) {
  actions_.pop_back();
  is_terminal_ = false;
}

void TinyBridgePlayState::DoApplyAction(Action action) {
  actions_.push_back(std::make_pair(CurrentHand(), action));
  if (actions_.size() % 4 == 0) {
    int win_hand = actions_[actions_.size() - 4].first;
    int win_card = actions_[actions_.size() - 4].second;
    for (int i = actions_.size() - 3; i < actions_.size(); ++i) {
      int hand = actions_[i].first;
      int card = actions_[i].second;
      if (Suit(card) == Suit(win_card)) {
        if (Rank(card) > Rank(win_card)) {
          win_card = card;
          win_hand = hand;
        }
      } else if (Suit(card) == trumps_) {
        win_card = card;
        win_hand = hand;
      }
    }
    winner_[actions_.size() / 4 - 1] = win_hand;
  }
}

std::vector<Action> TinyBridgePlayState::LegalActions() const {
  std::vector<Action> actions;
  const int hand = CurrentHand();
  for (int i = 0; i < kNumCards; ++i) {
    if (holder_[i] == hand &&
        (actions_.size() < 4 ||
         actions_[(4 + hand - leader_) % 4].second != i)) {
      actions.push_back(i);
    }
  }
  // Have to follow suit if we have two cards of different suits.
  if (!actions_.empty() && actions.size() == 2 &&
      Suit(actions[0]) != Suit(actions[1])) {
    return {Suit(actions[0]) == Suit(actions_[0].second) ? actions[0]
                                                         : actions[1]};
  } else {
    return actions;
  }
}

int TinyBridgePlayState::CurrentHand() const {
  return ((actions_.size() < 4 ? leader_ : winner_[0]) + actions_.size()) % 4;
}

std::string TinyBridgePlayState::ActionToString(Player player,
                                                Action action_id) const {
  return CardString(action_id);
}

bool TinyBridgePlayState::IsTerminal() const {
  return actions_.size() == kNumCards;
}

std::vector<double> TinyBridgePlayState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>(num_players_, 0.0);
  }

  std::vector<double> returns(num_players_);
  for (const int winner : winner_) {
    returns[winner & 1] += 1.0;
  }
  return returns;
}

std::unique_ptr<State> TinyBridgePlayState::Clone() const {
  return std::unique_ptr<State>{new TinyBridgePlayState(*this)};
}

void TinyBridgePlayState::UndoAction(Player player, Action action) {
  actions_.pop_back();
  history_.pop_back();
}

std::string TinyBridgePlayState::ToString() const {
  std::array<std::string, kNumHands> hands;
  for (int i = 0; i < kNumCards; ++i) {
    hands[holder_[i]].append(CardString(i));
  }
  std::string s;
  for (int i = 0; i < kNumHands; ++i) {
    if (i > 0) s.push_back(' ');
    s.append(absl::StrCat(std::string(1, kHandChar[i]), ":", hands[i]));
  }
  s.append(absl::StrCat(" Trumps: ", std::string(1, kSuitChar[trumps_]),
                        " Leader:", std::string(1, kHandChar[leader_])));
  for (const auto& action : actions_) {
    s.append(absl::StrCat(" ", std::string(1, kHandChar[action.first]), ":",
                          CardString(action.second)));
  }
  return s;
}

}  // namespace tiny_bridge
}  // namespace open_spiel
