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

#include <string>
#include <utility>

#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"

namespace open_spiel {
namespace cribbage {

constexpr int kDefaultNumPlayers = 2;

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

std::string CribbageState::ActionToString(Player player,
                                          Action move_id) const {
	return "";
}

bool CribbageState::IsTerminal() const { return false; }

std::vector<double> CribbageState::Returns() const {
	return {0, 0};
}

std::string CribbageState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, game_->NumPlayers());
  return "";
}

void CribbageState::ObservationTensor(Player player,
                                       absl::Span<float> values) const {
}

CribbageState::CribbageState(std::shared_ptr<const Game> game) : State(game) {
  cur_player_ = kChancePlayerId;
  turn_player_ = 0;
}

int CribbageState::CurrentPlayer() const { return cur_player_; }

void CribbageState::DoApplyAction(Action move) {
  SPIEL_CHECK_EQ(IsTerminal(), false);
}

std::vector<Action> CribbageState::LegalActions() const {
	return {};
}

ActionsAndProbs CribbageState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  ActionsAndProbs outcomes;
  return outcomes;
}

std::string CribbageState::ToString() const {
	return "";
}

std::unique_ptr<State> CribbageState::Clone() const {
  return std::unique_ptr<State>(new CribbageState(*this));
}

CribbageGame::CribbageGame(const GameParameters& params)
    : Game(kGameType, params),
      num_players_(ParameterValue<int>("players")) {}

}  // namespace blackjack
}  // namespace open_spiel
