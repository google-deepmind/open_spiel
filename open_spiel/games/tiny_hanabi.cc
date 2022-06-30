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

#include "open_spiel/games/tiny_hanabi.h"

#include <numeric>

#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace tiny_hanabi {

namespace {

// This is the payoff matrix from the bayesian Action Decoder paper.
constexpr char kDefaultPayoffString[] =
    // Cards: 1, 1
    "10;0;0;4;8;4;10;0;0;"
    // Cards: 1, 2
    "0;0;10;4;8;4;0;0;10;"
    // Cards: 2, 1
    "0;0;10;4;8;4;0;0;0;"
    // Cards: 2, 2
    "10;0;0;4;8;4;10;0;0";

std::vector<int> ParsePayoffString(const std::string& str) {
  std::vector<std::string> pieces = absl::StrSplit(str, ';');
  std::vector<int> payoff;
  for (const auto& piece : pieces) {
    int val;
    if (!absl::SimpleAtoi(piece, &val)) {
      SpielFatalError(absl::StrCat("Could not parse piece '", piece,
                                   "' of payoff string '", str,
                                   "' as an integer"));
    }
    payoff.push_back(val);
  }
  return payoff;
}

// Facts about the game
const GameType kGameType{
    /*short_name=*/"tiny_hanabi",
    /*long_name=*/"Tiny Hanabi",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kIdentical,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/10,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {
        {"num_players", GameParameter(2)},
        {"num_chance", GameParameter(2)},
        {"num_actions", GameParameter(3)},
        {"payoff", GameParameter(std::string(kDefaultPayoffString))},
    }};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new TinyHanabiGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

std::unique_ptr<State> TinyHanabiGame::NewInitialState() const {
  return std::unique_ptr<State>(
      new TinyHanabiState(shared_from_this(), payoff_));
}

TinyHanabiGame::TinyHanabiGame(const GameParameters& params)
    : Game(kGameType, params),
      payoff_(ParameterValue<int>("num_players"),
              ParameterValue<int>("num_chance"),
              ParameterValue<int>("num_actions"),
              ParsePayoffString(ParameterValue<std::string>("payoff"))) {}

Player TinyHanabiState::CurrentPlayer() const {
  const int history_size = history_.size();
  if (history_size < num_players_) return kChancePlayerId;
  if (history_size == 2 * num_players_) return kTerminalPlayerId;
  return history_size - num_players_;
}

std::string TinyHanabiState::ActionToString(Player player,
                                            Action action) const {
  if (player == kChancePlayerId)
    return absl::StrCat("d", action);
  else
    return absl::StrCat("p", player, "a", action);
}

std::vector<std::pair<Action, double>> TinyHanabiState::ChanceOutcomes() const {
  if (!IsChanceNode()) return {};
  std::vector<std::pair<Action, double>> outcomes;
  const int num_outcomes = payoff_.NumChance();
  const double p = 1.0 / num_outcomes;
  outcomes.reserve(num_outcomes);
  for (int i = 0; i < num_outcomes; ++i) outcomes.emplace_back(i, p);
  return outcomes;
}

std::string TinyHanabiState::ToString() const {
  std::string rv;
  for (int i = 0; i < payoff_.NumPlayers() && i < history_.size(); ++i) {
    if (i != 0) absl::StrAppend(&rv, " ");
    absl::StrAppend(&rv, "p", i, ":d", history_[i].action);
  }
  for (int i = payoff_.NumPlayers(); i < history_.size(); ++i) {
    absl::StrAppend(&rv, " p", history_[i].player, ":a", history_[i].action);
  }
  return rv;
}

bool TinyHanabiState::IsTerminal() const {
  return history_.size() == 2 * num_players_;
}

std::vector<double> TinyHanabiState::Returns() const {
  const double value = IsTerminal() ? payoff_(history_) : 0.0;
  return std::vector<double>(payoff_.NumPlayers(), value);
}

std::unique_ptr<State> TinyHanabiState::Clone() const {
  return std::unique_ptr<State>(new TinyHanabiState(*this));
}

std::vector<Action> TinyHanabiState::LegalActions() const {
  if (IsTerminal()) return {};
  std::vector<Action> actions(IsChanceNode() ? payoff_.NumChance()
                                             : payoff_.NumActions());
  std::iota(actions.begin(), actions.end(), 0);
  return actions;
}

std::string TinyHanabiState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  std::string rv = absl::StrCat("p", player);
  if (history_.size() > player)
    absl::StrAppend(&rv, ":d", history_[player].action);
  for (int i = payoff_.NumPlayers(); i < history_.size(); ++i) {
    absl::StrAppend(&rv, " p", i - payoff_.NumPlayers(), ":a",
                    history_[i].action);
  }
  return rv;
}

void TinyHanabiState::InformationStateTensor(Player player,
                                             absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  SPIEL_CHECK_EQ(values.size(), payoff_.NumChance() + payoff_.NumActions() *
                                                          payoff_.NumPlayers());
  std::fill(values.begin(), values.end(), 0);
  if (history_.size() > player) values.at(history_[player].action) = 1;
  for (int i = payoff_.NumPlayers(); i < history_.size(); ++i) {
    values.at(payoff_.NumChance() +
              (i - payoff_.NumPlayers()) * payoff_.NumActions() +
              history_[i].action) = 1;
  }
}

void TinyHanabiState::ObservationTensor(Player player,
                                        absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  InformationStateTensor(player, values);
}

std::string TinyHanabiState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  return InformationStateString(player);
}

void TinyHanabiState::DoApplyAction(Action action) {}

}  // namespace tiny_hanabi
}  // namespace open_spiel
