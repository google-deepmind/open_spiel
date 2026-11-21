// Copyright 2026 DeepMind Technologies Limited
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
//


#ifndef OPEN_SPIEL_GAME_TRANSFORMS_EXTRA_INFO_GAME_H_
#define OPEN_SPIEL_GAME_TRANSFORMS_EXTRA_INFO_GAME_H_

#include "open_spiel/game_transforms/game_wrapper.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// Allows adding extra information to the observation tensor,
// observation string, information state tensor, or information stat string.
// Adding tensor information will flatten a tensor
// if it was not already flat.

namespace open_spiel {


class ExtraInfoState : public WrappedState {
 public:
  ExtraInfoState(std::shared_ptr<const Game> game, std::unique_ptr<State> state,
      int extra_observation_tensor_size,
      int extra_information_tensor_size);


  ExtraInfoState(const ExtraInfoState& other) = default;



  std::unique_ptr<State> Clone() const override {
    return std::unique_ptr<State>(new ExtraInfoState(*this));
  }

  void ObservationTensor(Player player,
        absl::Span<float> values) const override;

  void InformationStateTensor(Player player,
        absl::Span<float> values) const override;

  std::string ObservationString(Player player) const override;

  std::string InformationStateString(Player player) const override;

  void SetExtraObservationTensor(
      Player player, const std::vector<float>& values);

  void SetExtraObservationTensor(
      const std::vector<float>& values);

  void SetExtraInformationStateTensor(
      Player player, const std::vector<float>& values);

  void SetExtraInformationStateTensor(
      const std::vector<float>& values);

  void SetExtraInformationStateString(Player player,
                                    const std::string& value);

  void SetExtraObservationString(const std::string& value);

  void SetExtraObservationString(Player player,
                                    const std::string& value);

  void SetExtraInformationStateString(const std::string& value);

  private:
   std::vector<std::vector<float>> extra_information_tensor_;
   std::vector<std::vector<float>> extra_observation_tensor_;
   std::vector<std::string> extra_observation_string_;
   std::vector<std::string> extra_information_state_string_;
};

class ExtraInfoGame : public WrappedGame {
 public:
  ExtraInfoGame(std::shared_ptr<const Game> game, GameType game_type,
             GameParameters game_parameters);
  ExtraInfoGame(const ExtraInfoGame& other) = default;

  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(
        new ExtraInfoState(shared_from_this(), game_->NewInitialState(),
          extra_observation_tensor_size_,extra_information_tensor_size_));
  }

  std::vector<int> ObservationTensorShape() const override;

  std::vector<int> InformationStateTensorShape() const override;

 private:
  int extra_observation_tensor_size_;
  int extra_information_tensor_size_;

};

// Creates an extra info game based on the wrapped game.
std::shared_ptr<const Game> CreateExtraInfoGame(const Game& wrapped_game,
                                               const GameParameters& params);
std::shared_ptr<const Game> CreateExtraInfoGame(
    const std::string& wrapped_game_name, const GameParameters& params);


}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAME_TRANSFORMS_EXTRA_INFO_GAME_H_
