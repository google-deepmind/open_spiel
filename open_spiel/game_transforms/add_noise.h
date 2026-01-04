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

#ifndef OPEN_SPIEL_GAME_TRANSFORMS_ADD_NOISE_H_
#define OPEN_SPIEL_GAME_TRANSFORMS_ADD_NOISE_H_

#include <memory>

#include "open_spiel/game_transforms/game_wrapper.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// Transforms game by adding noise to the original utilities.
//
// The noise is sampled from uniform distribution of [-epsilon, epsilon]
// independently for each terminal history.
// The transformation can be seeded for reproducibility.

namespace open_spiel {
namespace add_noise {

class AddNoiseState : public WrappedState {
 public:
  AddNoiseState(std::shared_ptr<const Game> game, std::unique_ptr<State> state);
  AddNoiseState(const AddNoiseState& other) = default;
  std::unique_ptr<State> Clone() const override {
    return std::make_unique<AddNoiseState>(*this);
  }
  std::vector<double> Returns() const override;
  std::vector<double> Rewards() const override;
};

class AddNoiseGame : public WrappedGame {
 public:
  AddNoiseGame(std::shared_ptr<const Game> game, GameType game_type,
               GameParameters game_parameters);
  std::unique_ptr<State> NewInitialState() const override;
  double GetNoise(const AddNoiseState& state);

  double MinUtility() const override;

  double MaxUtility() const override;

 private:
  const double epsilon_;
  std::mt19937 rng_;
  std::unordered_map<std::string, double> noise_table_;
};

}  // namespace add_noise
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAME_TRANSFORMS_ADD_NOISE_H_
