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

#include <utility>
#include <vector>

#include "open_spiel/path.h"

namespace open_spiel {

Path::Path(std::unique_ptr<State> current_state)
    : State(current_state->GetGame()) {
  if (current_state->IsRoot()) {
    states_.push_back(std::move(current_state));
  } else {
    // Make a rollout from the initial state to the current one.
    states_.reserve(current_state->History().size());
    std::unique_ptr<State> state = game_->NewInitialState();
    for (const Action& action : current_state->History()) {
      states_.push_back(state->Clone());
      state->ApplyAction(action);
    }
    SPIEL_CHECK_EQ(state->History(), current_state->History());
    states_.push_back(std::move(state));
    SPIEL_CHECK_EQ(states_.size(), current_state->History().size());
  }
}

POHistory Path::PublicObservationHistory() const {
  SPIEL_CHECK_TRUE(game_->GetType().provides_factored_observation_string);
  SPIEL_CHECK_TRUE(states_[0]->IsRoot());
  POHistory public_observation_history;
  public_observation_history.reserve(states_.size());
  for (int i = 0; i < states_.size(); i++) {
    public_observation_history.push_back(
        states_[i]->PublicObservationString());
  }
  return public_observation_history;
}


const AOHistory& Path::ActionObservationHistory(Player player) const {
  SPIEL_CHECK_TRUE(game_->GetType().provides_observation_string);
  SPIEL_CHECK_TRUE(states_[0]->IsRoot());

  AOHistory action_observation_history;
  action_observation_history.reserve(2 * states_.size());
  for (int i = 0; i < states_.size(); i++) {
    action_observation_history.push_back(
        states_[i]->PrivateObservationString(player));

    if (i > 0 && states_[i - 1]->CurrentPlayer() == player) {
      action_observation_history.push_back(states_[i]->History().back());
    }
  }
  return action_observation_history;
}

}  // namespace open_spiel
