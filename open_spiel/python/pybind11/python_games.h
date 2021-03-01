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

#ifndef OPEN_SPIEL_PYTHON_PYBIND11_PYTHON_GAMES_H_
#define OPEN_SPIEL_PYTHON_PYBIND11_PYTHON_GAMES_H_

// Interface and supporting functions for defining games in Python and using
// them from C++.

#include "open_spiel/spiel.h"
#include "pybind11/include/pybind11/pybind11.h"

namespace open_spiel {

namespace py = ::pybind11;

// Trampoline for using Python-defined games from C++.
class PyGame : public Game {
 public:
  PyGame(py::object py_game, GameType game_type, GameInfo game_info,
         GameParameters game_parameters);

  // Implementation of the Game API.
  std::unique_ptr<State> NewInitialState() const override;
  int NumDistinctActions() const override { return info_.num_distinct_actions; }
  int NumPlayers() const override { return info_.num_players; }
  double MinUtility() const override { return info_.min_utility; }
  double MaxUtility() const override { return info_.max_utility; }
  double UtilitySum() const override { return info_.utility_sum; }
  int MaxGameLength() const override { return info_.max_game_length; }
  int MaxChanceOutcomes() const override { return info_.max_chance_outcomes; }
  std::shared_ptr<Observer> MakeObserver(
      absl::optional<IIGObservationType> iig_obs_type,
      const GameParameters& params) const override;
  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override;

  // Access to the backing Python object.
  py::object py_handle() const { return py_handle_; }

  // Observers for the old observation API.
  const Observer& default_observer() const { return *default_observer_; }
  const Observer& info_state_observer() const { return *info_state_observer_; }

 private:
  py::object py_handle_;
  GameInfo info_;

  // Used to implement the old observation API.
  std::shared_ptr<Observer> default_observer_;
  std::shared_ptr<Observer> info_state_observer_;
};

// Trampoline for using Python-defined states from C++.
class PyState : public State {
 public:
  PyState(py::handle py_state, std::shared_ptr<const Game> game);

  // Implementation of the State API.
  Player CurrentPlayer() const override;
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::unique_ptr<State> Clone() const override;
  void DoApplyAction(Action action_id) override;
  void DoApplyActions(const std::vector<Action>& actions) override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  ActionsAndProbs ChanceOutcomes() const override;

  // Handle the ownership of the Python object.
  void RelinquishOwnership();
  void TakeOwnership();
  ~PyState() override { RelinquishOwnership(); }
  py::handle py_handle() const { return py_handle_; }

 private:
  py::handle py_handle_;
  bool own_;
};

// Returning a std::unique_ptr<State> to Python. Some cleverness is needed in
// the Python game case, but in the regular C++ game case this makes no
// difference.
struct StateDeleter {
  void operator()(State* state);
};
using StateRetPtr = std::unique_ptr<State, StateDeleter>;
StateRetPtr ToPython(std::unique_ptr<State> state);

// Register a Python game.
void RegisterPyGame(const GameType& game_type, py::function creator);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_PYTHON_PYBIND11_PYTHON_GAMES_H_
