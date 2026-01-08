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

#ifndef OPEN_SPIEL_GAME_TRANSFORMS_ZEROSUM_H_
#define OPEN_SPIEL_GAME_TRANSFORMS_ZEROSUM_H_

#include <numeric>
#include "open_spiel/game_transforms/game_wrapper.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// Transforms a general sum game into a zero sum one by subtracting the mean
// of the rewards and final returns.

namespace open_spiel {

inline std::vector<double> SubtractMean(std::vector<double>&& vec) {
  double mean = std::accumulate(vec.begin(), vec.end(), 0.0) / vec.size();
  std::vector<double> result = std::move(vec);
  for (auto& item : result) item -= mean;
  return result;
}

class ZeroSumState : public WrappedState {
 public:
  ZeroSumState(std::shared_ptr<const Game> game, std::unique_ptr<State> state)
      : WrappedState(game, std::move(state)) {}
  ZeroSumState(const ZeroSumState& other) = default;

  std::vector<double> Rewards() const override {
    return SubtractMean(state_->Rewards());
  }

  std::vector<double> Returns() const override {
    return SubtractMean(state_->Returns());
  }

  std::unique_ptr<State> Clone() const override {
    return std::unique_ptr<State>(new ZeroSumState(*this));
  }
};

class ZeroSumGame : public WrappedGame {
 public:
  ZeroSumGame(std::shared_ptr<const Game> game, GameType game_type,
             GameParameters game_parameters);
  ZeroSumGame(const ZeroSumGame& other) = default;

  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(
        new ZeroSumState(shared_from_this(), game_->NewInitialState()));
  }

  double MaxUtility() const override {
    // The maximum utility is obtained if, in the original game,
    // one player gains game_->MaxUtility() while all other players
    // obtain game_->MinUtility(), because the mean is subtracted.
    double n = static_cast<double>(game_->NumPlayers());
    return (game_->MaxUtility() - game_->MinUtility()) * (n - 1) / n;
  }
  double MinUtility() const override {
    // By symmetry:
    return - MaxUtility();
  }
  absl::optional<double> UtilitySum() const override {
    return 0.0;
  }
};

}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAME_TRANSFORMS_ZEROSUM_H_
