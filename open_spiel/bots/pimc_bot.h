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

#ifndef OPEN_SPIEL_BOTS_PIMC_BOT_H_
#define OPEN_SPIEL_BOTS_PIMC_BOT_H_

#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/games/gin_rummy/gin_rummy_utils.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {

class PIMCBot : public Bot {
 public:
  PIMCBot(std::function<double(const State&, Player player)> value_function,
          Player player_id, uint32_t seed, int num_determinizations,
          int depth_limit);

  Action Step(const State& state) override;

  bool ProvidesPolicy() override { return true; }
  std::pair<ActionsAndProbs, Action> StepWithPolicy(
      const State& state) override;
  ActionsAndProbs GetPolicy(const State& state) override;

  bool IsClonable() const override { return false; }

 private:
  ActionsAndProbs PolicyFromBestAction(const State& state,
                                       Action best_action) const;
  std::pair<std::vector<int>, Action> Search(const State& root_state);

  std::mt19937 rng_;
  std::function<double(const State&, Player player)> value_function_;
  const Player player_id_;
  const int num_determinizations_;
  const int depth_limit_;
};

}  // namespace open_spiel

#endif  // OPEN_SPIEL_BOTS_GIN_RUMMY_SIMPLE_GIN_RUMMY_BOT_H_
