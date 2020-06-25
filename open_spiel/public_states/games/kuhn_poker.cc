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

#include "open_spiel/games/kuhn_poker.h"

#include "open_spiel/game_parameters.h"
#include "open_spiel/public_states/public_states.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace public_states {
namespace kuhn_poker {
namespace {

const GameWithPublicStatesType kGameType{
    /*short_name=*/"kuhn_poker",
    /*provides_cfr_computation=*/true,
    /*provides_state_compatibility_check=*/true,
};

std::shared_ptr<const GameWithPublicStates> Factory(
    std::shared_ptr<const Game> game) {
  // TODO(author13): This is just a stub.
  SpielFatalError("Not implemented.");
}

REGISTER_SPIEL_GAME_WITH_PUBLIC_STATE_API(kGameType, Factory);
}  // namespace

}  // namespace kuhn_poker
}  // namespace public_states
}  // namespace open_spiel
