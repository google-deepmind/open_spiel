// Copyright 2026 DeepMind Technologies Limited
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

#include "open_spiel/games/kayles/kayles.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace kayles {
namespace {

const GameType kGameType{
    /*short_name=*/"kayles",
    /*long_name=*/"Kayles",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/kNumPlayers,
    /*min_num_players=*/kNumPlayers,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    {{"row_length", GameParameter(kDefaultRowLength)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new KaylesGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

}  // namespace

KaylesGame::KaylesGame(const GameParameters& params)
    : Game(kGameType, params),
      row_length_(ParameterValue<int>("row_length")) {
  SPIEL_CHECK_GT(row_length_, 0);
}

KaylesState::KaylesState(std::shared_ptr<const Game> game, int row_length)
    : State(game), row_length_(row_length), pins_(row_length, true) {}

std::vector<Action> KaylesState::LegalActions() const {
  std::vector<Action> actions;
  for (int pin = 0; pin < row_length_; ++pin) {
    if (pins_[pin]) actions.push_back(pin);
  }
  for (int pin = 0; pin + 1 < row_length_; ++pin) {
    if (pins_[pin] && pins_[pin + 1]) {
      actions.push_back(row_length_ + pin);
    }
  }
  return actions;
}

std::string KaylesState::ActionToString(Player player, Action action) const {
  return absl::StrCat("action ", action);
}

std::string KaylesState::ToString() const { return ""; }

bool KaylesState::IsTerminal() const { return false; }

std::vector<double> KaylesState::Returns() const { return {0.0, 0.0}; }

std::string KaylesState::InformationStateString(Player player) const {
  return HistoryString();
}

std::string KaylesState::ObservationString(Player player) const {
  return ToString();
}

void KaylesState::ObservationTensor(Player player,
                                    absl::Span<float> values) const {
  std::fill(values.begin(), values.end(), 0.0);
}

void KaylesState::DoApplyAction(Action action) {
  SpielFatalError("Kayles moves are not implemented yet.");
}

void KaylesState::UndoAction(Player player, Action action) {
  SpielFatalError("Kayles undo is not implemented yet.");
}

std::unique_ptr<State> KaylesState::Clone() const {
  return std::unique_ptr<State>(new KaylesState(*this));
}

}  // namespace kayles
}  // namespace open_spiel
