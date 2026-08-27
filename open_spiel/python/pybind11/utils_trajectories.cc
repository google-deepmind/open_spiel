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

#include "open_spiel/python/pybind11/utils_trajectories.h"

#include <string>

// Python bindings for policies and algorithms handling them.

#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/trajectories.h"
#include "pybind11/include/pybind11/detail/common.h"
#include "open_spiel/pybind11_json/include/pybind11_json/pybind11_json.hpp"

namespace open_spiel {
namespace {

using ::open_spiel::State;
using ::open_spiel::trajectories::Header;
using ::open_spiel::trajectories::Trajectory;
using ::open_spiel::trajectories::Transition;

namespace py = ::pybind11;
}  // namespace

void init_pyspiel_utils_trajectories(py::module& m) {
  py::class_<Header> header(m, "Header");
  header.def(py::init<>())
      .def_readwrite("game_string", &Header::game_string)
      .def_readwrite("terminal", &Header::terminal)
      .def_readwrite("meta_data", &Header::meta_data)
      .def_readwrite("returns", &Header::returns);

  py::class_<Transition> transition(m, "Transition");
  transition.def(py::init<>())
      .def_readwrite("player", &Transition::player)
      .def_readwrite("action", &Transition::action)
      // Optional fields are std::unique_ptr, so they can return nullptr.
      .def("get_joint_action",
           [](const Transition& transition) {
             return transition.joint_action.get();
           })
      .def("get_legal_actions",
           [](const Transition& transition) {
             return transition.legal_actions.get();
           })
      .def("get_chance_outcomes", [](const Transition& transition) {
        return transition.chance_outcomes.get();
      });

  py::classh<Trajectory> trajectory(m, "Trajectory");
  trajectory.def(py::init<const std::string&>())
      .def(py::init<const nlohmann::json&>())
      .def(py::init<const State*>())
      .def("header", &Trajectory::header)
      .def("transitions", &Trajectory::transitions)
      .def("to_string", &Trajectory::ToString)
      .def("__str__", &Trajectory::ToString)
      // Returns the final state of the trajectory.
      .def("reconstruct_final_state", &Trajectory::ReconstructFinalState)
      // Returns all states of the trajectory.
      .def("reconstruct_all_states", &Trajectory::ReconstructAllStates);
}
}  // namespace open_spiel
