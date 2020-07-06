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

#ifndef OPEN_SPIEL_SPIEL_CONSTANTS_H_
#define OPEN_SPIEL_SPIEL_CONSTANTS_H_

#include "open_spiel/spiel_utils.h"

namespace open_spiel {

// Constant representing an invalid action.
inline constexpr Action kInvalidAction = -1;

// Constant representing an invalid observation.
//
// Empty string is an invalid observation on purpose: players always receive
// some observation. If nothing else, players perceive the flow of time:
// in this case, you can return kClockTickObservation (see fog_constants.h).
inline const char* kInvalidObservation = "";

}

#endif  // OPEN_SPIEL_SPIEL_CONSTANTS_H_
