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

#include "open_spiel/games/tiny_bridge.h"

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/algorithms/minimax.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace tiny_bridge {
namespace {

constexpr std::array<char, kNumRanks> kRankChar{'J', 'Q', 'K', 'A'};
constexpr std::array<char, 1 + kNumSuits> kSuitChar{'H', 'S', 'N'};
constexpr std::array<char, kNumSeats> kSeatChar{'W', 'N', 'E', 'S'};

int RelativeSeatIndex(Seat player, Seat observer) {
  return (kNumSeats + static_cast<int>(player) - static_cast<int>(observer)) %
         kNumSeats;
}

std::string RelativeSeatString(Seat player, Seat observer) {
  constexpr std::array<absl::string_view, 4> relative_player{"Us", "LH", "Pd",
                                                             "RH"};
  return std::string(relative_player[RelativeSeatIndex(player, observer)]);
}

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

Seat CharToSeat(char c) {
  switch (c) {
    case 'W':
      return kWest;
    case 'N':
      return kNorth;
    case 'E':
      return kEast;
    case 'S':
      return kSouth;
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

// Requires card0 > card1
int CardsToChanceOutcome(int card0, int card1) {
  return (card0 * (card0 - 1)) / 2 + card1;
}

// Returns first > second
std::pair<int, int> ChanceOutcomeToCards(int outcome) {
  int card0 = 1;
  while (CardsToChanceOutcome(card0 + 1, 0) <= outcome) ++card0;
  return {card0, outcome - CardsToChanceOutcome(card0, 0)};
}

// Hand abstraction. Each line is a bucket of hands that are
// indistinguishable.
inline constexpr const char* kAbstraction[kNumAbstractHands] = {
    // Mixed suits.
    "SAHA",
    "SJHA SKHA SQHA",
    "SAHJ SAHK SAHQ",
    "SJHJ SJHK SJHQ SKHJ SKHK SKHQ SQHJ SQHK SQHQ",
    // Hearts only.
    "HAHK HAHQ",
    "HKHJ HKHQ",
    "HAHJ",
    "HQHJ",
    // Spades only.
    "SASK SASQ",
    "SKSQ SKSJ",
    "SASJ",
    "SQSJ",
};

// Computes the abstraction lookup.
std::vector<int> ConcreteToAbstract() {
  std::vector<int> concrete_to_abstract(kNumPrivates, -1);
  for (int c = 0; c < kNumPrivates; ++c) {
    auto hand = HandString(c);
    for (int ah = 0; ah < kNumAbstractHands; ++ah) {
      if (absl::StrContains(kAbstraction[ah], hand)) {
        concrete_to_abstract[c] = ah;
        break;
      }
    }
    if (concrete_to_abstract[c] == -1) {
      SpielFatalError(
          absl::StrCat("Abstraction not found for concrete hand '", hand, "'"));
    }
  }
  return concrete_to_abstract;
}

// Returns an abstraction.
int ChanceOutcomeToHandAbstraction(int outcome) {
  static std::vector<int> concrete_to_abstract = ConcreteToAbstract();
  return concrete_to_abstract[outcome];
}

// Abstract hand string.
std::string ChanceOutcomeToHandAbstractionString(int outcome) {
  return kAbstraction[ChanceOutcomeToHandAbstraction(outcome)];
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
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {
        {"abstracted",
         GameParameter(GameParameter::Type::kBool, /*is_mandatory=*/false)},
    }};

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
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
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
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/false,
    /*provides_observation_tensor=*/false,
    /*parameter_specification=*/
    {
        {"trumps",
         GameParameter(GameParameter::Type::kString, /*is_mandatory=*/true)},
        {"leader",
         GameParameter(GameParameter::Type::kString, /*is_mandatory=*/true)},
        {"hand_W",
         GameParameter(GameParameter::Type::kString, /*is_mandatory=*/true)},
        {"hand_N",
         GameParameter(GameParameter::Type::kString, /*is_mandatory=*/true)},
        {"hand_E",
         GameParameter(GameParameter::Type::kString, /*is_mandatory=*/true)},
        {"hand_S",
         GameParameter(GameParameter::Type::kString, /*is_mandatory=*/true)},
    }};

std::shared_ptr<const Game> Factory2p(const GameParameters& params) {
  return std::shared_ptr<const Game>(new TinyBridgeGame2p(params));
}

std::shared_ptr<const Game> Factory4p(const GameParameters& params) {
  return std::shared_ptr<const Game>(new TinyBridgeGame4p(params));
}

REGISTER_SPIEL_GAME(kGameType2p, Factory2p);
REGISTER_SPIEL_GAME(kGameType4p, Factory4p);

// Score a played-out hand.
int Score(int contract, int tricks, bool doubled, bool redoubled, int trumps) {
  // -20 per undertrick
  // +10 for 1H/S/NT (+10 extra if overtrick)
  // +30 for 2H/S
  // +35 for 2NT
  const int contract_tricks = 1 + (contract - 1) / 3;
  const int contract_result = tricks - contract_tricks;
  const int double_factor = (1 + doubled) * (1 + redoubled);
  if (contract_result < 0) return 20 * double_factor * contract_result;
  int score = tricks * 10;
  if (contract_tricks == 2) score += 10;
  if (contract_tricks == 2 && trumps == 2) score += 5;
  return score * double_factor;
}

}  // namespace

std::string HandString(Action outcome) {
  auto cards = ChanceOutcomeToCards(outcome);
  return absl::StrCat(CardString(cards.first), CardString(cards.second));
}

std::string SeatString(Seat seat) { return std::string(1, kSeatChar[seat]); }

TinyBridgeGame2p::TinyBridgeGame2p(const GameParameters& params)
    : Game(kGameType2p, params),
      is_abstracted_(ParameterValue<bool>("abstracted", false)) {}

std::unique_ptr<State> TinyBridgeGame2p::NewInitialState() const {
  return std::unique_ptr<State>(
      new TinyBridgeAuctionState(shared_from_this(), is_abstracted_));
}

TinyBridgeGame4p::TinyBridgeGame4p(const GameParameters& params)
    : Game(kGameType4p, params) {}

std::unique_ptr<State> TinyBridgeGame4p::NewInitialState() const {
  return std::unique_ptr<State>(
      new TinyBridgeAuctionState(shared_from_this(), /*is_abstracted=*/false));
}

TinyBridgePlayGame::TinyBridgePlayGame(const GameParameters& params)
    : Game(kGameTypePlay, params) {}

std::unique_ptr<State> TinyBridgePlayGame::NewInitialState() const {
  int trumps = CharToTrumps(ParameterValue<std::string>("trumps")[0]);
  Seat leader = CharToSeat(ParameterValue<std::string>("leader")[0]);
  std::array<Seat, kDeckSize> holder;
  for (Seat i : {kWest, kNorth, kEast, kSouth}) {
    std::string hand = ParameterValue<std::string>(
        absl::StrCat("hand_", std::string(1, kSeatChar[i])));
    for (int j = 0; j < kNumTricks; ++j) {
      int c = StringToCard(hand.substr(j * 2, 2));
      holder[c] = i;
    }
  }
  return std::unique_ptr<State>(
      new TinyBridgePlayState(shared_from_this(), trumps, leader, holder));
}

Seat TinyBridgeAuctionState::PlayerToSeat(Player player) const {
  return num_players_ == 2 ? Seat(player * 2) : Seat(player);
}

Player TinyBridgeAuctionState::SeatToPlayer(Seat seat) const {
  return num_players_ == 2 ? static_cast<int>(seat) / 2
                           : static_cast<int>(seat);
}

std::string TinyBridgeAuctionState::PlayerHandString(Player player,
                                                     bool abstracted) const {
  if (!IsDealt(player)) return "??";
  return abstracted ? ChanceOutcomeToHandAbstractionString(actions_[player])
                    : HandString(actions_[player]);
}

std::string TinyBridgeAuctionState::DealString() const {
  std::string deal;
  for (auto player = Player{0}; player < num_players_; ++player) {
    if (player != 0) deal.push_back(' ');
    absl::StrAppend(&deal, SeatString(PlayerToSeat(player)), ":",
                    PlayerHandString(player, /*abstracted=*/false));
  }
  return deal;
}

TinyBridgeAuctionState::AuctionState TinyBridgeAuctionState::AnalyzeAuction()
    const {
  AuctionState rv;
  rv.last_bid = Call::kPass;
  rv.last_bidder = kInvalidSeat;
  rv.doubler = kInvalidSeat;
  rv.redoubler = kInvalidSeat;
  for (int i = num_players_; i < actions_.size(); ++i) {
    if (actions_[i] == Call::kDouble) {
      rv.doubler = PlayerToSeat(i % num_players_);
    } else if (actions_[i] == Call::kRedouble) {
      rv.redoubler = PlayerToSeat(i % num_players_);
    } else if (actions_[i] != Call::kPass) {
      rv.last_bid = actions_[i];
      rv.last_bidder = PlayerToSeat(i % num_players_);
      rv.doubler = kInvalidSeat;
      rv.redoubler = kInvalidSeat;
    }
  }
  return rv;
}

int Score_p0(std::array<Seat, kDeckSize> holder,
             const TinyBridgeAuctionState::AuctionState& state) {
  if (state.last_bid == Call::kPass) return 0;
  std::shared_ptr<Game> game(new TinyBridgePlayGame({}));
  int trumps = (state.last_bid - 1) % 3;
  Seat leader = Seat((state.last_bidder + 3) % 4);
  Seat decl = Seat(state.last_bidder % 2);
  TinyBridgePlayState play{game, trumps, leader, holder};
  const double tricks =
      algorithms::AlphaBetaSearch(*game, &play, nullptr, -1, decl).first;
  SPIEL_CHECK_GE(tricks, 0);
  SPIEL_CHECK_LE(tricks, kNumTricks);
  const int declarer_score =
      Score(state.last_bid, tricks, state.doubler != kInvalidSeat,
            state.redoubler != kInvalidSeat, trumps);
  return (decl == 0) ? declarer_score : -declarer_score;
}

// Score indexed by [WestHand][EastHand][Contract][LastBidder]
using ScoringTable = std::array<
    std::array<std::array<std::array<double, 2>, kNumActions2p>, kNumPrivates>,
    kNumPrivates>;

// Calculates a single score.
double Score_2p_(Action hand0, Action hand1,
                 const TinyBridgeAuctionState::AuctionState& state) {
  if (state.last_bid == kPass) return 0;
  const double freq = 1. / 6;
  double utility_p0 = 0;
  std::array<Seat, kDeckSize> holders_2p;
  std::fill(holders_2p.begin(), holders_2p.end(), kInvalidSeat);
  const auto cards0 = ChanceOutcomeToCards(hand0);
  holders_2p[cards0.first] = kWest;
  holders_2p[cards0.second] = kWest;
  const auto cards1 = ChanceOutcomeToCards(hand1);
  holders_2p[cards1.first] = kEast;
  holders_2p[cards1.second] = kEast;
  std::array<Seat, kDeckSize> holders_4p;
  for (int n0 = 0; n0 < 3; ++n0) {
    for (int n1 = n0 + 1; n1 < 4; ++n1) {
      int n = 0;
      for (int i = 0; i < kDeckSize; ++i) {
        if (holders_2p[i] == kInvalidSeat) {
          holders_4p[i] = (n == n0 || n == n1) ? kNorth : kSouth;
          ++n;
        } else {
          holders_4p[i] = holders_2p[i];
        }
      }
      utility_p0 += Score_p0(holders_4p, state) * freq;
    }
  }
  return utility_p0;
}

// Returns a cache of scores.
ScoringTable MakeScores() {
  ScoringTable scores;
  for (int hand0 = 0; hand0 < kNumPrivates; ++hand0) {
    for (int hand1 = 0; hand1 < kNumPrivates; ++hand1) {
      if (!IsConsistent(hand0, hand1)) continue;
      for (int contract = k1H; contract < kNumActions2p; ++contract) {
        for (Seat last_bidder : {kWest, kEast}) {
          scores[hand0][hand1][contract][last_bidder / 2] =
              Score_2p_(hand0, hand1,
                        {contract, last_bidder, kInvalidSeat, kInvalidSeat});
        }
      }
    }
  }
  return scores;
}

double Score_2p(Action hand0, Action hand1,
                const TinyBridgeAuctionState::AuctionState& state) {
  if (state.last_bid == kPass) return 0;
  static const ScoringTable scoring_table = MakeScores();
  const double score =
      scoring_table[hand0][hand1][state.last_bid][state.last_bidder / 2];
  return score;
}

std::array<Seat, kDeckSize> TinyBridgeAuctionState::CardHolders() const {
  std::array<Seat, kDeckSize> holder;
  std::fill(holder.begin(), holder.end(), kInvalidSeat);
  for (int i = 0; i < actions_.size() && i < num_players_; ++i) {
    int action_id = actions_[i];
    const auto cards = ChanceOutcomeToCards(action_id);
    holder[cards.first] = Seat(i);
    holder[cards.second] = Seat(i);
  }
  return holder;
}

void TinyBridgeAuctionState::DoApplyAction(Action action) {
  actions_.push_back(action);
  if (num_players_ == 2) {
    if (actions_.size() >= 2 * num_players_ && actions_.back() == Call::kPass) {
      is_terminal_ = true;
      utility_p0 = Score_2p(actions_[0], actions_[1], AnalyzeAuction());
    }
  } else {
    if (actions_.size() >= 2 * num_players_ &&
        actions_[actions_.size() - 1] == Call::kPass &&
        actions_[actions_.size() - 2] == Call::kPass &&
        actions_[actions_.size() - 3] == Call::kPass) {
      is_terminal_ = true;
      utility_p0 = Score_p0(CardHolders(), AnalyzeAuction());
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
    if (num_players_ == 4 && state.last_bidder != kInvalidSeat) {
      if (state.last_bidder % 2 != CurrentPlayer() % 2) {
        if (state.doubler == kInvalidSeat) actions.push_back(Call::kDouble);
      } else {
        if (state.doubler != kInvalidSeat && state.redoubler == kInvalidSeat)
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
  for (int card1 = 0; card1 < kDeckSize; ++card1) {
    if (holder[card1] != kInvalidSeat) continue;
    for (int card2 = card1 + 1; card2 < kDeckSize; ++card2) {
      if (holder[card2] != kInvalidSeat) continue;
      actions.push_back(CardsToChanceOutcome(card2, card1));
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
    return HandString(action_id);
  } else {
    return kActionStr[action_id];
  }
}

int TinyBridgeAuctionState::CurrentPlayer() const {
  if (IsTerminal()) return kTerminalPlayerId;
  return actions_.size() < num_players_ ? kChancePlayerId
                                        : actions_.size() % num_players_;
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

std::string TinyBridgeAuctionState::InformationStateString(
    Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  std::string hand = PlayerHandString(player, is_abstracted_);
  std::string dealer = RelativeSeatString(Seat::kWest, PlayerToSeat(player));
  std::string auction = AuctionString();
  if (!auction.empty())
    return absl::StrCat(hand, " ", dealer, " ", auction);
  else
    return hand;
}

// Observation string is the player's cards plus the most recent bid,
// plus any doubles or redoubles. E.g. "HJSA 2NT:Us Dbl:RH RDbl:Pd"
// This is an observation for a player who holds HJ and SA.
// and redoubled by West.
// The most recent bid is 2NT by this player, which has been doubled by their
// right-hand-opponent and redoubled by their partner. So the most recent few
// calls must be: 2NT-Pass-Pass-Dbl-Pass-Pass-RDbl-Pass.
std::string TinyBridgeAuctionState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  std::string observation = PlayerHandString(player, is_abstracted_);
  if (HasAuctionStarted()) {
    auto state = AnalyzeAuction();
    if (state.last_bid != Call::kPass) {
      absl::StrAppend(
          &observation, " ", ActionToString(state.last_bidder, state.last_bid),
          ":", RelativeSeatString(state.last_bidder, PlayerToSeat(player)));
    }
    if (state.doubler != kInvalidSeat)
      absl::StrAppend(&observation, " ", "Dbl:",
                      RelativeSeatString(state.doubler, PlayerToSeat(player)));
    if (state.redoubler != kInvalidSeat)
      absl::StrAppend(
          &observation, " ",
          "RDbl:", RelativeSeatString(state.redoubler, PlayerToSeat(player)));
  }
  return observation;
}

// Information state vector consists of:
//   kNumCards bits showing which cards the observing player holds
//   For 2p, kNumActions2p*2 bits showing which actions have been taken:
//     For each action, the bits are [1, 0] if we took the action,
//     [0, 1] if our partner took the action, and otherwise [0, 0].
//   For 4p, (kNumBids*3+1)*num_players bits showing which actions have been
//   taken:
//     For each player, 1 bit showing if they passed before the first bid
//     For each bid, 4 bits showing who made it, 4 bits showing who doubled it,
//     and 4 bits showing who redoubled it.
//     Each set of 4 bits is relative the the current player.
void TinyBridgeAuctionState::InformationStateTensor(
    Player player, absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  const int hand_size = is_abstracted_ ? kNumAbstractHands : kDeckSize;
  const int auction_size = (num_players_ == 2)
                               ? kNumActions2p * num_players_
                               : num_players_ + kNumBids * num_players_ * 3;
  std::fill(values.begin(), values.end(), 0);
  SPIEL_CHECK_EQ(values.size(), hand_size + auction_size);
  if (IsDealt(player)) {
    if (is_abstracted_) {
      const int abstraction = ChanceOutcomeToHandAbstraction(actions_[player]);
      values.at(abstraction) = 1;
    } else {
      const auto cards = ChanceOutcomeToCards(actions_[player]);
      values.at(cards.first) = 1;
      values.at(cards.second) = 1;
    }
  }
  if (num_players_ == 2) {
    for (int i = num_players_; i < actions_.size(); ++i) {
      values.at(hand_size + actions_[i] * 2 + (i - player) % num_players_) = 1;
    }
  } else {
    auto last_bid = Call::kPass;
    auto observer = PlayerToSeat(player);
    for (int i = num_players_; i < actions_.size(); ++i) {
      int bidder = RelativeSeatIndex(Seat(i % num_players_), observer);
      if (actions_[i] == Call::kPass) {
        if (last_bid == Call::kPass) {
          values.at(hand_size + bidder) = 1;
        }
      } else if (actions_[i] == Call::kDouble) {
        values.at(hand_size + num_players_ +
                  (last_bid - 1) * (3 * num_players_) + bidder) = 1;
      } else if (actions_[i] == Call::kRedouble) {
        values.at(hand_size + num_players_ +
                  (last_bid - 1) * (3 * num_players_) + num_players_ + bidder) =
            1;
      } else {
        last_bid = Call(actions_[i]);
        values.at(hand_size + num_players_ +
                  (last_bid - 1) * (3 * num_players_) + num_players_ * 2 +
                  bidder) = 1;
      }
    }
  }
}

// Information state vector consists of:
//   kNumCards bits showing which cards the observing player holds
// For 2p:
//   kNumActions2p bits showing the most recent action (one-hot)
// For 4p:
//   kNumBids bits showing the most recent bid (one-hot)
//   4 bits showing who bid it (relative to the observing player)
//   4 bits showing who doubled it (relative to the observing player)
//   4 bits showing who redoubled it (relative to the observing player)
//   4 bits for the dealer
void TinyBridgeAuctionState::ObservationTensor(Player player,
                                               absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  const int hand_size = is_abstracted_ ? kNumAbstractHands : kDeckSize;
  const int auction_size =
      num_players_ == 2 ? kNumActions2p : kNumBids + 4 * num_players_;
  std::fill(values.begin(), values.end(), 0);
  SPIEL_CHECK_EQ(values.size(), hand_size + auction_size);
  if (IsDealt(player)) {
    if (is_abstracted_) {
      const int abstraction = ChanceOutcomeToHandAbstraction(actions_[player]);
      values.at(abstraction) = 1;
    } else {
      const auto cards = ChanceOutcomeToCards(actions_[player]);
      values.at(cards.first) = 1;
      values.at(cards.second) = 1;
    }
  }
  if (num_players_ == 2) {
    if (HasAuctionStarted()) {
      values.at(hand_size + actions_.back()) = 1;
    }
  } else {
    auto state = AnalyzeAuction();
    auto seat = PlayerToSeat(player);
    if (state.last_bidder != kInvalidSeat)
      values.at(hand_size + RelativeSeatIndex(state.last_bidder, seat)) = 1;
    if (state.doubler != kInvalidSeat)
      values.at(hand_size + kNumSeats +
                RelativeSeatIndex(state.doubler, seat)) = 1;
    if (state.redoubler != kInvalidSeat)
      values.at(hand_size + kNumSeats * 2 +
                RelativeSeatIndex(state.redoubler, seat)) = 1;
    values.at(hand_size + kNumSeats * 3 +
              RelativeSeatIndex(Seat::kWest, seat)) = 1;
    if (state.last_bidder != kInvalidSeat)
      values.at(hand_size + kNumSeats * 4 + state.last_bid - 1) = 1;
  }
}

std::unique_ptr<State> TinyBridgeAuctionState::Clone() const {
  return std::unique_ptr<State>{new TinyBridgeAuctionState(*this)};
}

void TinyBridgeAuctionState::UndoAction(Player player, Action action) {
  actions_.pop_back();
  history_.pop_back();
  --move_number_;
  is_terminal_ = false;
}

void TinyBridgePlayState::DoApplyAction(Action action) {
  actions_.emplace_back(CurrentHand(), action);
  if (actions_.size() % 4 == 0) {
    Seat win_hand = actions_[actions_.size() - 4].first;
    int win_card = actions_[actions_.size() - 4].second;
    for (int i = actions_.size() - 3; i < actions_.size(); ++i) {
      Seat hand = actions_[i].first;
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
  for (int i = 0; i < kDeckSize; ++i) {
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

Seat TinyBridgePlayState::CurrentHand() const {
  return Seat(((actions_.size() < 4 ? leader_ : winner_[0]) + actions_.size()) %
              4);
}

std::string TinyBridgePlayState::ActionToString(Player player,
                                                Action action_id) const {
  return CardString(action_id);
}

bool TinyBridgePlayState::IsTerminal() const {
  return actions_.size() == kDeckSize;
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
  --move_number_;
}

std::string TinyBridgePlayState::ToString() const {
  std::array<std::string, kNumSeats> hands;
  for (int i = 0; i < kDeckSize; ++i) {
    hands[holder_[i]].append(CardString(i));
  }
  std::string s;
  for (int i = 0; i < kNumSeats; ++i) {
    if (i > 0) s.push_back(' ');
    s.append(absl::StrCat(std::string(1, kSeatChar[i]), ":", hands[i]));
  }
  s.append(absl::StrCat(" Trumps: ", std::string(1, kSuitChar[trumps_]),
                        " Leader:", std::string(1, kSeatChar[leader_])));
  for (const auto& action : actions_) {
    s.append(absl::StrCat(" ", std::string(1, kSeatChar[action.first]), ":",
                          CardString(action.second)));
  }
  return s;
}

bool IsConsistent(Action hand0, Action hand1) {
  auto cards0 = ChanceOutcomeToCards(hand0);
  auto cards1 = ChanceOutcomeToCards(hand1);
  return cards0.first != cards1.first && cards0.second != cards1.second &&
         cards0.first != cards1.second && cards0.second != cards1.first;
}

}  // namespace tiny_bridge
}  // namespace open_spiel
