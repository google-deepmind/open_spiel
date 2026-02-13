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

#include "open_spiel/python/pybind11/games_gomoku.h"

#include <memory>
#include <string>
#include <utility> 

#include "open_spiel/games/gomoku/gomoku.h"
#include "open_spiel/spiel.h"
#include "pybind11/include/pybind11/cast.h"
#include "pybind11/include/pybind11/pybind11.h"

namespace py = ::pybind11;
using open_spiel::Game;
using open_spiel::Action;
using open_spiel::State;
using open_spiel::gomoku::SymmetryPolicy;
using open_spiel::gomoku::GomokuGame;
using open_spiel::gomoku::GomokuState;

void open_spiel::init_pyspiel_games_gomoku(::pybind11::module &m) {
  // py::module_ gomoku = m.def_submodule("gomoku");
	py::classh<GomokuGame, Game>(m, "GomokuGame")
    .def("move_to_action",
      [](const GomokuGame& g, const std::vector<int>& coord) {
        return g.MoveToAction(coord);
      },
      py::arg("coord"))
    .def("action_to_move",
      [](const GomokuGame& g, Action action) {
         return g.ActionToMove(action);
      },
      py::arg("action"))
	  .def("size", &GomokuGame::Size)
    .def("dims", &GomokuGame::Dims)
    .def("connect", &GomokuGame::Connect)
    .def("wrap", &GomokuGame::Wrap)
	  .def("anti", &GomokuGame::Anti)
		.def(py::pickle(
       [](std::shared_ptr<const GomokuGame> game) {
         return game->ToString();
       },
       [](const std::string& data) {
         return std::dynamic_pointer_cast<GomokuGame>(
           std::const_pointer_cast<Game>(LoadGame(data)));
       }));

  py::classh<GomokuState, State>(m, "GomokuState")
    .def("hash_value", &GomokuState::HashValue)
    .def("symmetric_hash", &GomokuState::SymmetricHash)
    .def("pretty", &GomokuState::Pretty)
		.def("winning_line", &GomokuState::WinningLine,
       py::return_value_policy::reference_internal)

		.def(py::pickle(
       [](const GomokuState& state) {
       return SerializeGameAndState(*state.GetGame(), state);
      },
      [](const std::string& data) {
      auto game_and_state = DeserializeGameAndState(data);
      return dynamic_cast<GomokuState*>(
          game_and_state.second.release());
      }))
	  .def("set_symmetry_policy",
         &GomokuState::SetSymmetryPolicy)
    .def("get_symmetry_policy",
         &GomokuState::GetSymmetryPolicy,
         py::return_value_policy::reference_internal);

  py::class_<SymmetryPolicy>(m, "SymmetryPolicy")
    .def(py::init<>())
    .def_readwrite("allow_reflections",
                   &SymmetryPolicy::allow_reflections)
    .def_readwrite("allow_reflection_rotations",
                   &SymmetryPolicy::allow_reflection_rotations);

}
