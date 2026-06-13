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

#ifndef OPEN_SPIEL_UTILS_TRAJECTORIES_H_
#define OPEN_SPIEL_UTILS_TRAJECTORIES_H_

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace trajectories {

struct Header {
  // Required fields.
  std::string game_string;
  bool terminal;

  // Optional fields.

  // The meta_data is any arbitrary information that the user wants to store
  // with the trajectory. For example, it can be used to store the algorithm
  // used to generate the trajectory, the date it was generated, the identity of
  // the players, etc.
  std::string meta_data;

  // The returns of the trajectory.
  std::vector<double> returns;
};

struct Transition {
  // Required fields.
  Player player;

  // For simultaneous nodes, this will be set to kInvalidAction.
  Action action;

  // Optional fields.
  std::unique_ptr<std::vector<Action>> joint_action;  // For simultaneous games.
  std::unique_ptr<std::vector<Action>> legal_actions;
  std::unique_ptr<ActionsAndProbs> chance_outcomes;
};

// A trajectory is a simple container representing a sequence of states and
// actions and associated with some meta data, used to record games. The string
// representation of the trajectory is the sequence of states and actions.
class Trajectory {
 public:
  Trajectory() = default;
  Trajectory(const std::string& json_str);
  Trajectory(const nlohmann::json& json);

  // Uses the history of the final state to construct the trajectory.
  // Only adds the required fields to the trajectory.
  Trajectory(const State* final_state);

  // Returns a string representation of the trajectory (json-formatted).
  std::string ToString() const;

  std::unique_ptr<State> ReconstructFinalState() const;
  std::vector<std::unique_ptr<State>> ReconstructAllStates() const;

  const Header& header() const { return header_; }
  const std::vector<Transition>& transitions() const { return transitions_; }

 private:
  void ConstructFromJson(const nlohmann::json& json);
  void ConstructFromString(const std::string& json_str);

  // Reconstructs the history of states from the transitions and appends them to
  // the provided history vector (if not null). Returns the final state.
  std::unique_ptr<State> ReconstructHistory(
      std::vector<std::unique_ptr<State>>* states) const;

  Header header_;
  std::vector<Transition> transitions_;
};

}  // namespace trajectories
}  // namespace open_spiel

#endif  // OPEN_SPIEL_UTILS_TRAJECTORIES_H_
