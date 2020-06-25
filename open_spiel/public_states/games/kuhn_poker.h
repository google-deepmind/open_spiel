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

#ifndef OPEN_SPIEL_PUBLIC_STATES_GAMES_KUHN_POKER_H_
#define OPEN_SPIEL_PUBLIC_STATES_GAMES_KUHN_POKER_H_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/games/kuhn_poker.h"
#include "open_spiel/public_states/public_states.h"

// This is a public state API version of Kuhn Poker:
// http://en.wikipedia.org/wiki/Kuhn_poker
//
// While public state API describes imperfect recall abstractions, these
// actually coincide with perfect recall on this game.
//
// There is a visualization of world/public/private trees available in [1]
// [1] https://arxiv.org/abs/1906.11110

namespace open_spiel {
namespace public_states {
namespace kuhn_poker {

// TODO(author13): This is just a stub.
class KuhnPrivateInformation : public PrivateInformation {};
class KuhnPublicState : public PublicState {};

}  // namespace kuhn_poker
}  // namespace public_states
}  // namespace open_spiel

#endif  // OPEN_SPIEL_PUBLIC_STATES_GAMES_KUHN_POKER_H_
