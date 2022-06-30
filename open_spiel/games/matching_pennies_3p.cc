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

#include "open_spiel/games/matching_pennies_3p.h"

#include <memory>

#include "open_spiel/normal_form_game.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace matching_pennies_3p {

constexpr const Action kHeadsActionId = 0;
constexpr const Action kTailsActionId = 1;

namespace {
const GameType kGameType{/*short_name=*/"matching_pennies_3p",
                         /*long_name=*/"Three-Player Matching Pennies",
                         GameType::Dynamics::kSimultaneous,
                         GameType::ChanceMode::kDeterministic,
                         GameType::Information::kOneShot,
                         GameType::Utility::kGeneralSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/3,
                         /*min_num_players=*/3,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/true,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/{}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new MatchingPennies3pGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace

MatchingPennies3pState::MatchingPennies3pState(std::shared_ptr<const Game> game)
    : NFGState(game), terminal_(false), returns_({0, 0, 0}) {}

std::vector<Action> MatchingPennies3pState::LegalActions(Player player) const {
  if (terminal_)
    return {};
  else
    return {kHeadsActionId, kTailsActionId};
}

std::string MatchingPennies3pState::ActionToString(Player player,
                                                   Action move_id) const {
  switch (move_id) {
    case kHeadsActionId:
      return "Heads";
    case kTailsActionId:
      return "Tails";
    default:
      SpielFatalError("Unrecognized move id");
  }
}

bool MatchingPennies3pState::IsTerminal() const { return terminal_; }

std::vector<double> MatchingPennies3pState::Returns() const { return returns_; }

std::unique_ptr<State> MatchingPennies3pState::Clone() const {
  return std::unique_ptr<State>(new MatchingPennies3pState(*this));
}

void MatchingPennies3pState::DoApplyActions(
    const std::vector<Action>& actions) {
  SPIEL_CHECK_EQ(actions.size(), NumPlayers());

  // Player 1 gets a +1 if they match P2, -1 otherwise
  returns_[0] = (actions[0] == actions[1] ? 1.0 : -1.0);

  // Player 2 gets a +1 if they match P3, -1 otherwise
  returns_[1] = (actions[1] == actions[2] ? 1.0 : -1.0);

  // Player 3 gets a +1 if they don't match P1, -1 otherwise
  returns_[2] = (actions[2] != actions[0] ? 1.0 : -1.0);

  terminal_ = true;
}

MatchingPennies3pGame::MatchingPennies3pGame(const GameParameters& params)
    : NormalFormGame(kGameType, params) {}

}  // namespace matching_pennies_3p
}  // namespace open_spiel
