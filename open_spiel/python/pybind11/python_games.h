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

#ifndef OPEN_SPIEL_PYTHON_PYBIND11_PYTHON_GAMES_H_
#define OPEN_SPIEL_PYTHON_PYBIND11_PYTHON_GAMES_H_

// Interface and supporting functions for defining games in Python and using
// them from C++.

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/python/pybind11/pybind11.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {

namespace py = ::pybind11;

// Generic dict-backed SpielStruct wrappers.
// These allow Python games to return plain dicts that get wrapped as proper
// SpielStruct subclasses, enabling JSON serialization without requiring
// Python-side C++ struct subclassing.
class DictStateStruct : public StateStruct {
 public:
  explicit DictStateStruct(nlohmann::json data) : data_(std::move(data)) {}
  nlohmann::json to_json_base() const override { return data_; }

 private:
  nlohmann::json data_;
};

class DictActionStruct : public ActionStruct {
 public:
  explicit DictActionStruct(nlohmann::json data) : data_(std::move(data)) {}
  nlohmann::json to_json_base() const override { return data_; }

 private:
  nlohmann::json data_;
};

class DictObservationStruct : public ObservationStruct {
 public:
  explicit DictObservationStruct(nlohmann::json data)
      : data_(std::move(data)) {}
  nlohmann::json to_json_base() const override { return data_; }

 private:
  nlohmann::json data_;
};

// Trampoline for using Python-defined games from C++.
class PyGame : public Game, public py::trampoline_self_life_support {
 public:
  PyGame(GameType game_type, GameInfo game_info,
         GameParameters game_parameters);

  // Implementation of the Game API.
  std::unique_ptr<State> NewInitialState() const override;
  std::unique_ptr<State> NewInitialState(const std::string& str) const override;
  std::unique_ptr<State> NewInitialState(
      const nlohmann::json& json) const override;
  std::unique_ptr<State> NewInitialStateForPopulation(
      int population) const override;
  int MaxChanceNodesInHistory() const override;
  int NumDistinctActions() const override { return info_.num_distinct_actions; }
  int NumPlayers() const override { return info_.num_players; }
  double MinUtility() const override { return info_.min_utility; }
  double MaxUtility() const override { return info_.max_utility; }
  absl::optional<double> UtilitySum() const override {
    return info_.utility_sum;
  }
  int MaxGameLength() const override { return info_.max_game_length; }
  int MaxChanceOutcomes() const override { return info_.max_chance_outcomes; }
  std::shared_ptr<Observer> MakeObserver(
      absl::optional<IIGObservationType> iig_obs_type,
      const GameParameters& params) const override;
  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override;
  std::unique_ptr<State> DeserializeState(
      const std::string& str) const override;

  // Observers for the old observation API.
  const Observer& default_observer() const;
  const Observer& info_state_observer() const;

 private:
  GameInfo info_;

  // Used to implement the old observation API.
  mutable std::shared_ptr<Observer> default_observer_;
  mutable std::shared_ptr<Observer> info_state_observer_;
};

// Trampoline for using Python-defined states from C++.
class PyState : public State, public py::trampoline_self_life_support {
 public:
  PyState(std::shared_ptr<const Game> game);

  // Implementation of the State API.
  Player CurrentPlayer() const override;
  std::vector<Action> LegalActions() const override;
  std::vector<Action> LegalActions(Player player) const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::vector<double> Rewards() const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<std::string> DistributionSupport() override;
  void UpdateDistribution(const std::vector<double>& distribution) override;
  void DoApplyAction(Action action_id) override;
  void DoApplyActions(const std::vector<Action>& actions) override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  ActionsAndProbs ChanceOutcomes() const override;
  std::string Serialize() const override;
  int MeanFieldPopulation() const override;

  // SpielStruct API overrides.
  // These forward to Python methods (_to_struct, _action_to_struct, etc.)
  // and wrap the returned dicts as generic Dict*Struct objects.
  std::unique_ptr<StateStruct> ToStruct() const override;
  std::unique_ptr<ActionStruct> ActionToStruct(Player player,
                                               Action action_id) const override;
  std::vector<Action> StructToActions(
      const ActionStruct& action_struct) const override;
  std::unique_ptr<ObservationStruct> ToObservationStruct(
      Player player) const override;

  // Python-specific details.
  void Deserialize(const std::string& str);
};

// Register a Python game.
void RegisterPyGame(const GameType& game_type, py::function creator);

// Get the dict for a Python state implementation.
py::dict PyDict(const State& state);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_PYTHON_PYBIND11_PYTHON_GAMES_H_
