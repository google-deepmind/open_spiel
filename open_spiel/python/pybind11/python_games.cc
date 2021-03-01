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
// There are two complications which mean we cannot use the standard pybind
// approach:
//
// 1. We wish to use std::unique_ptr to return new objects in the C++ API,
//    which implies that the Python implementations of those APIs must also
//    return std::unique_ptr, which is not supported by pybind11.
//
//
// 2. We wish to create and use Python-backed objects from within C++, which
//    requires that the C++ code has an owning reference to the Python
//    object.
//
// These two factors require separate considerations. (1) means that we must
// create an additional object which the C++ code can own, id addition to the
// trampoline that pybind creates and the underlying Python object. (2) means
// that our trampoline objects must (temporarily) own their backing Python
// objects, unlike the default setup where the C++ trampoline is owned by the
// Python object.
//
// The full workaround is as follows. When a State object is initially created,
// we create a second C++ trampoline, resulting in the following three objects:
//
//    a. Python State object; implicitly governs the lifetime of (b)
//    b. C++ PyState object, created by pybind11; holds a weak reference to (a)
//    c. C++ PyState object, created by StateForCpp; owns (a)
//
// C++ code holds a std::unique_ptr to the object (c).
// If this object is used entirely in C++, then this structure persists
// throughout the object's lifetime. Calls to methods on (c) are routed to (a)
// using the reference it holds to the Python object. The standard pybind11
// macros cannot be used since pybind is unaware of the (a) <-> (c) linkage.
//
// If the object is returned to Python, then we no longer need (c), and so in
// `ToPython`, we copy (c) to (b), return (b) and delete (c). In doing so, we
// need to make sure that the Python object (a) sticks around until the
// Python-side code takes ownership of it. This is managed by temporarily giving
// (b) ownership of the Python object, and then relinquishing it using a custom
// deleter when ownership of (b) is taken by the pybind11 code, by which time it
// will have ownership of (a) also. This then results in the canonical pybind
// setup:
//
//    a. Python State object; implicitly governs the lifetime of (b); owned by
//       the Python caller
//    b. C++ PyState object, created by pybind11
//
// Additionally, (b) holds a weak reference to (a) - we use this in place of
// the normal pybind mechanism (same codepath as above).
//
// TODO(author11) Adopt a similar scheme for Game objects (currently leaked).
// TODO(author11) Simplify as and when pybind11 enhancements are submitted.

#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "pybind11/include/pybind11/functional.h"
#include "pybind11/include/pybind11/numpy.h"
#include "pybind11/include/pybind11/operators.h"
#include "pybind11/include/pybind11/pybind11.h"
#include "pybind11/include/pybind11/stl.h"

namespace open_spiel {

namespace py = ::pybind11;

// Alternate macros to PYBIND11_OVERRIDE_NAME and PYBIND11_OVERRIDE_PURE_NAME.
// We use our own because the trampoline class on which the method is invoked
// may be the for-use-in-C++ copy rather than the original that pybind11 knows
// about, which would mean that pybind11 is unable to find the Python object.
#define PYSPIEL_OVERRIDE_IMPL(ret_type, cname, name, ...)                  \
  do {                                                                     \
    pybind11::gil_scoped_acquire gil;                                      \
    pybind11::function override = py_handle_.attr(name);                   \
    if (override) {                                                        \
      auto o = override(__VA_ARGS__);                                      \
      if (pybind11::detail::cast_is_temporary_value_reference<             \
              ret_type>::value) {                                          \
        static pybind11::detail::override_caster_t<ret_type> caster;       \
        return pybind11::detail::cast_ref<ret_type>(std::move(o), caster); \
      } else {                                                             \
        return pybind11::detail::cast_safe<ret_type>(std::move(o));        \
      }                                                                    \
    }                                                                      \
  } while (false)

#define PYSPIEL_OVERRIDE_PURE(ret_type, cname, name, fn, ...)                  \
  do {                                                                         \
    PYSPIEL_OVERRIDE_IMPL(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), name, \
                          __VA_ARGS__);                                        \
    pybind11::pybind11_fail(                                                   \
        "Tried to call pure virtual function \"" PYBIND11_STRINGIFY(           \
            cname) "::" name "\"");                                            \
  } while (false)

#define PYSPIEL_OVERRIDE(ret_type, cname, name, fn, ...)                       \
  do {                                                                         \
    PYSPIEL_OVERRIDE_IMPL(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), name, \
                          __VA_ARGS__);                                        \
    return cname::fn(__VA_ARGS__);                                             \
  } while (false)

PyGame::PyGame(py::object py_game, GameType game_type, GameInfo game_info,
               GameParameters game_parameters)
    : Game(game_type, game_parameters),
      py_handle_(std::move(py_game)),
      info_(game_info) {
  default_observer_ = MakeObserver(kDefaultObsType, {});
  info_state_observer_ = MakeObserver(kInfoStateObsType, {});
}

// Creates an additional C++ wrapper class. See discussion at top of file.
std::unique_ptr<State> StateForCpp(py::object py_state,
                                   std::shared_ptr<const Game> game) {
  auto rv = std::make_unique<PyState>(py_state, game);
  rv->TakeOwnership();
  return rv;
}

std::unique_ptr<State> PyGame::NewInitialState() const {
  // Can't use the standard macro because we need to create a second State
  // object (for use in C++).
  py::gil_scoped_acquire gil;
  py::function f = py_handle_.attr("new_initial_state");
  return StateForCpp(f(), shared_from_this());
}

PyState::PyState(py::handle py_state, std::shared_ptr<const Game> game)
    : State(game), py_handle_(py_state), own_(false) {}

Player PyState::CurrentPlayer() const {
  PYSPIEL_OVERRIDE_PURE(Player, State, "current_player", CurrentPlayer);
}

std::vector<Action> PyState::LegalActions() const {
  PYSPIEL_OVERRIDE_PURE(std::vector<Action>, State, "legal_actions",
                        LegalActions);
}

std::string PyState::ActionToString(Player player, Action action_id) const {
  PYSPIEL_OVERRIDE_PURE(std::string, State, "action_to_string", ActionToString,
                        player, action_id);
}

std::string PyState::ToString() const {
  PYSPIEL_OVERRIDE_PURE(std::string, State, "__str__", ToString);
}

bool PyState::IsTerminal() const {
  PYSPIEL_OVERRIDE_PURE(bool, State, "is_terminal", IsTerminal);
}

std::vector<double> PyState::Returns() const {
  PYSPIEL_OVERRIDE_PURE(std::vector<double>, State, "returns", Returns);
}

void PyState::DoApplyAction(Action action_id) {
  PYSPIEL_OVERRIDE(void, State, "do_apply_action", DoApplyAction, action_id);
}

void PyState::DoApplyActions(const std::vector<Action>& actions) {
  PYSPIEL_OVERRIDE(void, State, "do_apply_actions", DoApplyActions, actions);
}

ActionsAndProbs PyState::ChanceOutcomes() const {
  PYSPIEL_OVERRIDE_PURE(ActionsAndProbs, State, "chance_outcomes",
                        ChanceOutcomes);
}

std::unique_ptr<State> PyState::Clone() const {
  auto state = game_->NewInitialState();
  for (auto [p, a] : FullHistory()) {
    state->ApplyAction(a);
  }
  return state;
}

// Handle the ownership of the Python object.
void PyState::RelinquishOwnership() {
  if (own_) py_handle_.dec_ref();
  own_ = false;
}

void PyState::TakeOwnership() {
  if (!own_) py_handle_.inc_ref();
  own_ = true;
}

// Returning states to Python, with deferred relinquishing of ownership.
StateRetPtr ToPython(std::unique_ptr<State> state) {
  if (auto pystate = dynamic_cast<PyState*>(state.get())) {
    // Python created this in the first place, and we created a copy to pass
    // around the C++ library. Return the original and let the copy be deleted.
    // Retain the reference to the Python object held by the original until the
    // pybind11 code takes ownership.
    auto original = py::cast<PyState*>(pystate->py_handle());
    original->TakeOwnership();
    *original = *pystate;
    return StateRetPtr(original);
  } else {
    // This is a C++ object; Python takes ownership.
    return StateRetPtr(state.release());
  }
}

// Custom deleter which frees the Python-side object instead of the C++ one in
// the case of Python-defined states.
void StateDeleter::operator()(State* state) {
  if (auto pystate = dynamic_cast<PyState*>(state)) {
    // Once the Python object is deleted, pybind11 will tidy up the C++ side.
    pystate->RelinquishOwnership();
  } else {
    // Should never happen.
    delete state;
  }
}

// Register a Python game.
void RegisterPyGame(const GameType& game_type, py::function creator) {
  GameRegisterer::RegisterGame(
      game_type, [game_type, creator](const GameParameters& game_parameters) {
        auto py_game = creator(game_parameters);
        // TODO(author11) Fix leak
        // (circular reference between C++ Game and Python Game)
        return py::cast<std::shared_ptr<Game>>(py_game);
      });
}

// Observers and observations. We implement the C++ Observer in terms of the
// Python one.

// Wrapper for using a Python observer from C++.
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
  set_from_(py_state.py_handle(), player);
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
  return py::cast<std::string>(string_from_(py_state.py_handle(), player));
}

std::shared_ptr<Observer> PyGame::MakeObserver(
    absl::optional<IIGObservationType> iig_obs_type,
    const GameParameters& params) const {
  py::function f = py_handle_.attr("make_py_observer");
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

}  // namespace open_spiel
