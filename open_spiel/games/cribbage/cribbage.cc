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
#include <string>
#include <utility>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace cribbage {

constexpr int kDefaultNumPlayers = 2;
constexpr int kWinScore = 121;

constexpr const std::array<Card, 52> kAllCards = {
		// Clubs
		Card{0, 0, 0}, Card{1, 1, 0}, Card{2, 2, 0}, Card{3, 3, 0},
		Card{4, 4, 0}, Card{5, 5, 0}, Card{6, 6, 0}, Card{7, 7, 0},
		Card{8, 8, 0}, Card{9, 9, 0}, Card{10, 10, 0}, Card{11, 11, 0},
		Card{12, 12, 0},
		// Diamonds
		Card{13, 0, 1}, Card{14, 1, 1}, Card{15, 2, 1}, Card{16, 3, 1},
		Card{17, 4, 1}, Card{18, 5, 1}, Card{19, 6, 1}, Card{20, 7, 1},
		Card{21, 8, 1}, Card{22, 9, 1}, Card{23, 10, 1}, Card{24, 11, 1},
		Card{25, 12, 1},
		// Hearts
		Card{26, 0, 2}, Card{27, 1, 2}, Card{28, 2, 2}, Card{29, 3, 2},
		Card{30, 4, 2}, Card{31, 5, 2}, Card{32, 6, 2}, Card{33, 7, 2},
		Card{34, 8, 2}, Card{35, 9, 2}, Card{36, 10, 2}, Card{37, 11, 2},
		Card{38, 12, 2},
		// Spades
		Card{39, 0, 3}, Card{40, 1, 3}, Card{41, 2, 3}, Card{42, 3, 3},
		Card{43, 4, 3}, Card{44, 5, 3}, Card{45, 6, 3}, Card{46, 7, 3},
		Card{47, 8, 3}, Card{48, 9, 3}, Card{49, 10, 3}, Card{50, 11, 3},
		Card{51, 12, 3},
};

namespace {

// Facts about the game
const GameType kGameType{/*short_name=*/"cribbage",
                         /*long_name=*/"Cribbage",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kImperfectInformation,
                         GameType::Utility::kGeneralSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/2,
                         /*min_num_players=*/4,
                         /*provides_information_state_string=*/false,
                         /*provides_information_state_tensor=*/false,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/
												 {{"players", GameParameter(kDefaultNumPlayers)}}};

static std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new CribbageGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);
}  // namespace

  
bool operator==(const Card& lhs, const Card& rhs) {
  return lhs.id == rhs.id;
}

int CardsPerPlayer(int num_players) {
  switch (num_players) {
	  case 2: return 6;
		case 3: return 5;
		case 4: return 5;
		default: SpielFatalError(absl::StrCat("Unknown number of players: ",
                                          num_players));
	}
}

int CardsToCrib(int num_players) {
  switch (num_players) {
	  case 2: return 0;
		case 3: return 1;
		case 4: return 0;
		default: SpielFatalError(absl::StrCat("Unknown number of players: ",
                                          num_players));
	}
}

Card GetCard(int id) {
  SPIEL_CHECK_GE(id, 0);
	SPIEL_CHECK_LT(id, 52);
	return kAllCards[id];
}

std::string Card::to_string() const {
  std::string str("XX");
	str[0] = kRanks[rank];
	str[1] = kSuitNames[suit];
	return str;
}

std::string CribbageState::ActionToString(Player player,
                                          Action move_id) const {
	if (player == kChancePlayerId) {
	  return absl::StrCat("Deal ", kAllCards[move_id].to_string());
	} else {
	  return "";
	}
}

bool CribbageState::IsTerminal() const { 
  return *std::max_element(scores_.begin(), scores_.end()) >= kWinScore;
}

std::vector<double> CribbageState::Returns() const {
	return {0, 0};
}

std::string CribbageState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, game_->NumPlayers());
  return "";
}

void CribbageState::NextRound() {
	round_++;
	dealer_++;
	if (dealer_ >= num_players_) { dealer_ = 0; }
	start_player_++;
  if (start_player_ >= num_players_) { start_player_ = 0; }
	cur_player_ = kChancePlayerId;	

	deck_.clear();
	deck_.resize(52);
	for (int i = 0; i < 52; ++i) {
		deck_[i] = kAllCards[i];
	}

	for (int p = 0; p < num_players_; ++p) {
		hands_[p].clear();
	}

	phase_ = Phase::kCardPhase;
	crib_.clear();
}

void CribbageState::ObservationTensor(Player player,
                                      absl::Span<float> values) const {
}

CribbageState::CribbageState(std::shared_ptr<const Game> game)
		: State(game),
		  parent_game_(static_cast<const CribbageGame&>(*game)),
		  phase_(kCardPhase),
			scores_(num_players_, 0),
			starter_(std::nullopt),
			hands_(num_players_) {
	NextRound();
}

int CribbageState::CurrentPlayer() const { return cur_player_; }

void CribbageState::DoApplyAction(Action move) {
  SPIEL_CHECK_EQ(IsTerminal(), false);

	if (IsChanceNode()) {
		SPIEL_CHECK_GE(move, 0);
		SPIEL_CHECK_LT(move, 52);
		auto iter = std::find(deck_.begin(), deck_.end(), kAllCards[move]);
		SPIEL_CHECK_TRUE(iter != deck_.end());
		Card card = *iter;
		deck_.erase(iter);
		bool card_dealt = false;

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
			card_dealt = true;
		}

		// Check if we're ready to start choosing cards.
		if (p == (num_players_ - 1) &&
		    hands_[p].size() == parent_game_.cards_per_player() &&
				crib_.size() == parent_game_.cards_to_crib()) {
			cur_player_ = 0;
		} else {
		  cur_player_ = kChancePlayerId;
		}
	} else {
		// Decision node.
	}
}

std::vector<Action> CribbageState::LegalActions() const {
	if (IsChanceNode()) {
		return LegalChanceOutcomes(); 
	} else {
		return {kPassAction};
	}
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
	absl::StrAppend(&str, "Num players: ", num_players_, "\n");
	absl::StrAppend(&str, "Round: ", round_, "\n");
	absl::StrAppend(&str, "Dealer: ", dealer_, "\n");
	absl::StrAppend(&str, "Cur player: ", cur_player_, "\n");
	absl::StrAppend(&str, "Scores:");
	for (int p = 0; p < num_players_; ++p) {
		absl::StrAppend(&str, " ", scores_[p]);
	}
	absl::StrAppend(&str, "\n");
	for (int p = 0; p < num_players_; ++p) {
		absl::StrAppend(&str, "P", p, " Hand:");
		for (int i = 0; i < hands_[p].size(); ++i) {
			absl::StrAppend(&str, " ", hands_[p][i].to_string());
		}
		absl::StrAppend(&str, "\n");
	}
	absl::StrAppend(&str, "Crib:");
	for (int i = 0; i < crib_.size(); ++i) {
		absl::StrAppend(&str, " ", crib_[i].to_string());
	}
	absl::StrAppend(&str, "\n");
	return str;
}

std::unique_ptr<State> CribbageState::Clone() const {
  return std::unique_ptr<State>(new CribbageState(*this));
}

CribbageGame::CribbageGame(const GameParameters& params)
    : Game(kGameType, params),
      num_players_(ParameterValue<int>("players", kDefaultNumPlayers)),
			cards_per_player_(CardsPerPlayer(num_players_)),
			cards_to_crib_(CardsToCrib(num_players_)) {}

}  // namespace blackjack
}  // namespace open_spiel
