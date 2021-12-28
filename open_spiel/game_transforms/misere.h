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

#ifndef OPEN_SPIEL_GAME_TRANSFORMS_MISERE_H_
#define OPEN_SPIEL_GAME_TRANSFORMS_MISERE_H_

#include "open_spiel/game_transforms/game_wrapper.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// Transforms a game into its Misere version by inverting the sign of the
// rewards / utilities. This is a self-inverse operation.
// https://en.wikipedia.org/wiki/Mis%C3%A8re

namespace open_spiel {

// Flips the sign of a vector.
inline std::vector<double> Negative(std::vector<double>&& vector) {
  std::vector<double> neg = std::move(vector);
  for (auto& item : neg) item = -item;
  return neg;
}

class MisereState : public WrappedState {
 public:
  MisereState(std::shared_ptr<const Game> game, std::unique_ptr<State> state)
      : WrappedState(game, std::move(state)) {}
  MisereState(const MisereState& other) = default;

  std::vector<double> Rewards() const override {
    return Negative(state_->Rewards());
  }

  std::vector<double> Returns() const override {
    return Negative(state_->Returns());
  }

  std::unique_ptr<State> Clone() const override {
    return std::unique_ptr<State>(new MisereState(*this));
  }
};

class MisereGame : public WrappedGame {
 public:
  MisereGame(std::shared_ptr<const Game> game, GameType game_type,
             GameParameters game_parameters);
  MisereGame(const MisereGame& other) = default;

  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(
        new MisereState(shared_from_this(), game_->NewInitialState()));
  }

  double MinUtility() const override { return -game_->MaxUtility(); }
  double MaxUtility() const override { return -game_->MinUtility(); }
  double UtilitySum() const override { return -game_->UtilitySum(); }
};

}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAME_TRANSFORMS_MISERE_H_
