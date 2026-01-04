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

#include "open_spiel/python/pybind11/games_spades.h"

#include <memory>
#include <string>
#include <utility>

#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/games/spades/spades.h"
#include "open_spiel/python/pybind11/pybind11.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {

namespace py = ::pybind11;
using spades::SpadesGame;
using spades::SpadesState;

void init_pyspiel_games_spades(py::module& m) {
  py::classh<SpadesState, State>(m, "SpadesState")
      .def("get_current_scores", &SpadesState::GetCurrentScores)
      .def("set_current_scores", &SpadesState::SetCurrentScores)
      .def("is_game_over", &SpadesState::IsGameOver)
      .def("set_current_player", &SpadesState::SetCurrentPlayer)
      .def("contract_indexes", &SpadesState::ContractIndexes)
      .def("possible_contracts", &SpadesState::PossibleContracts)
      .def("current_phase", &SpadesState::CurrentPhase)
      .def("write_observation_tensor",
           [](const SpadesState& state,
              py::array_t<float, py::array::c_style> array) {
             py::buffer_info buf = array.request();
             SPIEL_CHECK_EQ(buf.ndim, 1);
             SPIEL_CHECK_EQ(buf.strides.front(), buf.itemsize);
             state.WriteObservationTensor(
                 state.CurrentPlayer(),
                 absl::MakeSpan(static_cast<float*>(buf.ptr),
                                buf.shape.front()));
           })
      .def("private_observation_tensor", &SpadesState::PrivateObservationTensor)
      .def("public_observation_tensor", &SpadesState::PublicObservationTensor)
      // Pickle support
      .def(py::pickle(
          [](const SpadesState& state) {  // __getstate__
            return SerializeGameAndState(*state.GetGame(), state);
          },
          [](const std::string& data) {  // __setstate__
            std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
                game_and_state = DeserializeGameAndState(data);
            return dynamic_cast<SpadesState*>(game_and_state.second.release());
          }));

  py::classh<SpadesGame, Game>(m, "SpadesGame")
      .def("num_possible_contracts", &SpadesGame::NumPossibleContracts)
      .def("contract_string", &SpadesGame::ContractString)
      .def("private_observation_tensor_size",
           &SpadesGame::PrivateObservationTensorSize)
      .def("public_observation_tensor_size",
           &SpadesGame::PublicObservationTensorSize)
      // Pickle support
      .def(py::pickle(
          [](std::shared_ptr<const SpadesGame> game) {  // __getstate__
            return game->ToString();
          },
          [](const std::string& data) {  // __setstate__
            return std::dynamic_pointer_cast<SpadesGame>(
                std::const_pointer_cast<Game>(LoadGame(data)));
          }));
}
}  // namespace open_spiel
