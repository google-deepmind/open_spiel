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

#include "open_spiel/python/pybind11/games_bridge.h"

#include <memory>

#include "open_spiel/games/bridge.h"
#include "open_spiel/python/pybind11/pybind11.h"
#include "open_spiel/spiel.h"

PYBIND11_SMART_HOLDER_TYPE_CASTERS(open_spiel::bridge::BridgeGame);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(open_spiel::bridge::BridgeState);

namespace open_spiel {

namespace py = ::pybind11;
using bridge::BridgeGame;
using bridge::BridgeState;

void init_pyspiel_games_bridge(py::module& m) {
  py::classh<BridgeState, State>(m, "BridgeState")
      .def("contract_index", &BridgeState::ContractIndex)
      .def("possible_contracts", &BridgeState::PossibleContracts)
      .def("score_by_contract", &BridgeState::ScoreByContract)
      .def("score_for_contracts", &BridgeState::ScoreForContracts)
      .def("current_phase", &BridgeState::CurrentPhase)
      .def("write_observation_tensor",
           [](const BridgeState& state,
              py::array_t<float, py::array::c_style> array) {
             py::buffer_info buf = array.request();
             SPIEL_CHECK_EQ(buf.ndim, 1);
             SPIEL_CHECK_EQ(buf.strides.front(), buf.itemsize);
             state.WriteObservationTensor(
                 state.CurrentPlayer(),
                 absl::MakeSpan(static_cast<float*>(buf.ptr),
                                buf.shape.front()));
           })
      .def("private_observation_tensor", &BridgeState::PrivateObservationTensor)
      .def("public_observation_tensor", &BridgeState::PublicObservationTensor)
      // Pickle support
      .def(py::pickle(
          [](const BridgeState& state) {  // __getstate__
            return SerializeGameAndState(*state.GetGame(), state);
          },
          [](const std::string& data) {  // __setstate__
            std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
                game_and_state = DeserializeGameAndState(data);
            return dynamic_cast<BridgeState*>(game_and_state.second.release());
          }));

  py::classh<BridgeGame, Game>(m, "BridgeGame")
      .def("num_possible_contracts", &BridgeGame::NumPossibleContracts)
      .def("contract_string", &BridgeGame::ContractString)
      .def("private_observation_tensor_size",
           &BridgeGame::PrivateObservationTensorSize)
      .def("public_observation_tensor_size",
           &BridgeGame::PublicObservationTensorSize)
      // Pickle support
      .def(py::pickle(
          [](std::shared_ptr<const BridgeGame> game) {  // __getstate__
            return game->ToString();
          },
          [](const std::string& data) {  // __setstate__
            return std::dynamic_pointer_cast<BridgeGame>(
                std::const_pointer_cast<Game>(LoadGame(data)));
          }));
}
}  // namespace open_spiel
