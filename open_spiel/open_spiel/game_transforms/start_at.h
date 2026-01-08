// Copyright 2021 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_GAME_TRANSFORMS_START_AT_H_
#define OPEN_SPIEL_GAME_TRANSFORMS_START_AT_H_

#include "open_spiel/game_transforms/game_wrapper.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// Transforms a game such that `transformed_game.new_initial_state()` already
// starts at the specified history. This is useful if you want to restrict
// algorithms only to some specific subgames.

namespace open_spiel {

class StartAtTransformationState : public WrappedState {
 public:
  StartAtTransformationState(std::shared_ptr<const Game> game,
                             std::unique_ptr<State> state);
  StartAtTransformationState(const StartAtTransformationState& other) = default;
  std::unique_ptr<State> Clone() const override {
    return std::make_unique<StartAtTransformationState>(*this);
  }
};

class StartAtTransformationGame : public WrappedGame {
 public:
  StartAtTransformationGame(std::shared_ptr<const Game> game,
                            GameType game_type, GameParameters game_parameters);
  std::unique_ptr<State> NewInitialState() const override;
  std::vector<State::PlayerAction> StartAtHistory() const {
    return start_state_->FullHistory();
  }

 private:
  const std::unique_ptr<State> start_state_;
};

}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAME_TRANSFORMS_START_AT_H_
