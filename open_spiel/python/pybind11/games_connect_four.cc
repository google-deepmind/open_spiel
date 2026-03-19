// Copyright 2019 DeepMind Technologies Limited
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

#include "open_spiel/python/pybind11/games_connect_four.h"

#include <memory>
#include <string>
#include <utility>

#include "open_spiel/games/connect_four/connect_four.h"
#include "open_spiel/json/include/nlohmann/json.hpp"  // IWYU pragma: keep
#include "open_spiel/python/pybind11/pybind11.h"
#include "open_spiel/spiel.h"

namespace py = ::pybind11;
using open_spiel::Game;
using open_spiel::State;
using open_spiel::connect_four::ConnectFourActionStruct;
using open_spiel::connect_four::ConnectFourGameParams;
using open_spiel::connect_four::ConnectFourObservationStruct;
using open_spiel::connect_four::ConnectFourState;
using open_spiel::connect_four::ConnectFourStateStruct;

namespace {
template <typename T, typename... Args>
void DefReadWriteConnectFourFields(py::class_<T, Args...>& c) {
  c.def_readwrite("current_player", &T::current_player)
      .def_readwrite("board", &T::board)
      .def_readwrite("is_terminal", &T::is_terminal)
      .def_readwrite("winner", &T::winner);
}
}  // namespace

void open_spiel::init_pyspiel_games_connect_four(py::module& m) {
  py::module_ connect_four = m.def_submodule("connect_four");

  auto state_struct_class =
      bind_spiel_struct<ConnectFourStateStruct, open_spiel::StateStruct>(
          connect_four, "ConnectFourStateStruct");
  DefReadWriteConnectFourFields(state_struct_class);

  auto obs_struct_class = bind_spiel_struct<ConnectFourObservationStruct,
                                            open_spiel::ObservationStruct>(
      connect_four, "ConnectFourObservationStruct");
  DefReadWriteConnectFourFields(obs_struct_class);

  bind_spiel_struct<ConnectFourActionStruct, open_spiel::ActionStruct>(
      connect_four, "ConnectFourActionStruct")
      .def_readwrite("column", &ConnectFourActionStruct::column);

  py::class_<ConnectFourGameParams, open_spiel::GameParametersStruct>(
      connect_four, "ConnectFourGameParams")
      .def(py::init<>())
      .def_readwrite("game_name", &ConnectFourGameParams::game_name)
      .def_readwrite("rows", &ConnectFourGameParams::rows)
      .def_readwrite("columns", &ConnectFourGameParams::columns)
      .def_readwrite("x_in_row", &ConnectFourGameParams::x_in_row)
      .def_readwrite("egocentric_obs_tensor",
                     &ConnectFourGameParams::egocentric_obs_tensor)
      .def("to_json", &ConnectFourGameParams::ToJson);

  py::classh<ConnectFourState, State>(connect_four, "ConnectFourState")
      // Pickle support
      .def(py::pickle(
          [](const ConnectFourState& state) {  // __getstate__
            return SerializeGameAndState(*state.GetGame(), state);
          },
          [](const std::string& data) {  // __setstate__
            std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
                game_and_state = DeserializeGameAndState(data);
            return dynamic_cast<ConnectFourState*>(
                game_and_state.second.release());
          }));
}
