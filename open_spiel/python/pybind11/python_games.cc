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

#include "open_spiel/python/pybind11/python_games.h"

#include <memory>

// Interface code for using Python Games and States from C++.

#include "open_spiel/game_parameters.h"
#include "open_spiel/python/pybind11/pybind11.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {

namespace py = ::pybind11;

PyGame::PyGame(GameType game_type, GameInfo game_info,
               GameParameters game_parameters)
    : Game(game_type, game_parameters), info_(game_info) {}

std::unique_ptr<State> PyGame::NewInitialState() const {
  PYBIND11_OVERLOAD_PURE_NAME(std::unique_ptr<State>, Game, "new_initial_state",
                              NewInitialState);
}

const Observer& PyGame::default_observer() const {
  if (!default_observer_) default_observer_ = MakeObserver(kDefaultObsType, {});
  return *default_observer_;
}

const Observer& PyGame::info_state_observer() const {
  if (!info_state_observer_)
    info_state_observer_ = MakeObserver(kInfoStateObsType, {});
  return *info_state_observer_;
}

PyState::PyState(std::shared_ptr<const Game> game) : State(game) {}

Player PyState::CurrentPlayer() const {
  PYBIND11_OVERLOAD_PURE_NAME(Player, State, "current_player", CurrentPlayer);
}

std::vector<Action> PyState::LegalActions() const {
  PYBIND11_OVERLOAD_PURE_NAME(std::vector<Action>, State, "legal_actions",
                              LegalActions);
}

std::string PyState::ActionToString(Player player, Action action_id) const {
  PYBIND11_OVERLOAD_PURE_NAME(std::string, State, "_action_to_string",
                              ActionToString, player, action_id);
}

std::string PyState::ToString() const {
  PYBIND11_OVERLOAD_PURE_NAME(std::string, State, "__str__", ToString);
}

bool PyState::IsTerminal() const {
  PYBIND11_OVERLOAD_PURE_NAME(bool, State, "is_terminal", IsTerminal);
}

std::vector<double> PyState::Returns() const {
  PYBIND11_OVERRIDE_PURE_NAME(std::vector<double>, State, "returns", Returns);
}

std::vector<double> PyState::Rewards() const {
  PYBIND11_OVERRIDE_NAME(std::vector<double>, State, "rewards", Rewards);
}

void PyState::DoApplyAction(Action action_id) {
  PYBIND11_OVERLOAD_PURE_NAME(void, State, "_apply_action", DoApplyAction,
                              action_id);
}

void PyState::DoApplyActions(const std::vector<Action>& actions) {
  PYBIND11_OVERLOAD_PURE_NAME(void, State, "_apply_actions", DoApplyActions,
                              actions);
}

ActionsAndProbs PyState::ChanceOutcomes() const {
  PYBIND11_OVERLOAD_PURE_NAME(ActionsAndProbs, State, "chance_outcomes",
                              ChanceOutcomes);
}

std::unique_ptr<State> PyState::Clone() const {
  auto state = game_->NewInitialState();
  for (auto [p, a] : FullHistory()) {
    state->ApplyAction(a);
  }
  return state;
}

// Register a Python game.
void RegisterPyGame(const GameType& game_type, py::function creator) {
  GameRegisterer::RegisterGame(
      game_type, [game_type, creator](const GameParameters& game_parameters) {
        auto py_game = creator(game_parameters);
        return py::cast<std::shared_ptr<Game>>(py_game);
      });
}

// Observers and observations. We implement the C++ Observer in terms of the
// Python one.

// Wrapper for using a Python observer from C++.
// This is not a 'trampoline' class, just a wrapper.
class PyObserver : public Observer {
 public:
  PyObserver(py::object py_observer);
  void WriteTensor(const State& state, int player,
                   Allocator* allocator) const override;
  std::string StringFrom(const State& state, int player) const override;

 private:
  py::object py_observer_;
  py::function set_from_;
  py::function string_from_;
};

PyObserver::PyObserver(py::object py_observer)
    : Observer(/*has_string=*/true, /*has_tensor=*/true),
      py_observer_(py_observer),
      set_from_(py_observer_.attr("set_from")),
      string_from_(py_observer_.attr("string_from")) {
  has_tensor_ = !py_observer_.attr("tensor").is_none();
}

void PyObserver::WriteTensor(const State& state, int player,
                             Allocator* allocator) const {
  using Array = py::array_t<float, py::array::c_style | py::array::forcecast>;
  const PyState& py_state = open_spiel::down_cast<const PyState&>(state);
  set_from_(py_state, player);
  py::dict dict = py_observer_.attr("dict");
  for (auto [k, v] : dict) {
    auto a = py::cast<Array>(v);
    const int dims = a.ndim();
    absl::InlinedVector<int, 4> shape(dims);
    for (int i = 0; i < dims; ++i) shape[i] = a.shape(i);
    auto out = allocator->Get(k.cast<std::string>(), shape);
    std::copy(a.data(), a.data() + a.size(), out.data.data());
  }
}

std::string PyObserver::StringFrom(const State& state, int player) const {
  const PyState& py_state = open_spiel::down_cast<const PyState&>(state);
  return py::cast<std::string>(string_from_(py_state, player));
}

std::shared_ptr<Observer> PyGame::MakeObserver(
    absl::optional<IIGObservationType> iig_obs_type,
    const GameParameters& params) const {
  py::object h = py::cast(this);
  py::function f = h.attr("make_py_observer");
  if (!f) SpielFatalError("make_py_observer not implemented");
  py::object observer = f(iig_obs_type, params);
  return std::make_shared<PyObserver>(observer);
}

std::string PyState::InformationStateString(Player player) const {
  const PyGame& game = open_spiel::down_cast<const PyGame&>(*game_);
  return game.info_state_observer().StringFrom(*this, player);
}

std::string PyState::ObservationString(Player player) const {
  const PyGame& game = open_spiel::down_cast<const PyGame&>(*game_);
  return game.default_observer().StringFrom(*this, player);
}

void PyState::InformationStateTensor(Player player,
                                     absl::Span<float> values) const {
  ContiguousAllocator allocator(values);
  const PyGame& game = open_spiel::down_cast<const PyGame&>(*game_);
  game.info_state_observer().WriteTensor(*this, player, &allocator);
}

namespace {
std::vector<int> TensorShape(const TrackingVectorAllocator& allocator) {
  switch (allocator.tensors.size()) {
    case 0:
      return {};
    case 1:
      return allocator.tensors.front().shape;
    default: {
      int size = 0;
      for (auto tensor : allocator.tensors) {
        size += std::accumulate(tensor.shape.begin(), tensor.shape.end(), 1,
                                std::multiplies<int>());
      }
      return {size};
    }
  }
}
}  // namespace

std::vector<int> PyGame::InformationStateTensorShape() const {
  TrackingVectorAllocator allocator;
  auto state = NewInitialState();
  info_state_observer().WriteTensor(*state, kDefaultPlayerId, &allocator);
  return TensorShape(allocator);
}

void PyState::ObservationTensor(Player player, absl::Span<float> values) const {
  ContiguousAllocator allocator(values);
  const PyGame& game = open_spiel::down_cast<const PyGame&>(*game_);
  game.default_observer().WriteTensor(*this, player, &allocator);
}

std::vector<int> PyGame::ObservationTensorShape() const {
  TrackingVectorAllocator allocator;
  auto state = NewInitialState();
  default_observer().WriteTensor(*state, kDefaultPlayerId, &allocator);
  return TensorShape(allocator);
}

py::dict PyDict(const State& state) {
  py::object obj = py::cast(&state);
  if (py::hasattr(obj, "__dict__")) {
    return obj.attr("__dict__");
  } else {
    return py::dict();
  }
}

}  // namespace open_spiel
