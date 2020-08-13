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

#ifndef OPEN_SPIEL_PUBLIC_STATES_POLICY_H_
#define OPEN_SPIEL_PUBLIC_STATES_POLICY_H_

#include <utility>
#include <vector>

#include "open_spiel/policy.h"
#include "open_spiel/eigen/pyeig.h"
#include "open_spiel/public_states/public_states.h"

namespace open_spiel {
namespace public_states {

// A policy object for public states. In Base API, policy in general
// is a mapping from states to list of (action, prob) pairs for all the
// legal actions at the state.
// A policy for public states additionally defines methods to retrieve
// policy for the whole public state at once.
class PublicStatesPolicy : public Policy {
 public:
  // Returns a vector of probabilities for each private information of the
  // requested player. If the player is not playing at this public state or
  // the policy is not available, returns and empty list.
  virtual std::vector<ArrayXd> GetPublicStatePolicy(
      const PublicState& public_state, Player for_player) const = 0;
};

}  // namespace public_states
}  // namespace open_spiel

#endif  // OPEN_SPIEL_PUBLIC_STATES_POLICY_H_
