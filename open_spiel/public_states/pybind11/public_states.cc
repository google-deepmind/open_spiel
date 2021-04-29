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

#include "open_spiel/public_states/public_states.h"

#include <memory>

#include "open_spiel/python/pybind11/pybind11.h"
#include "pybind11/include/pybind11/eigen.h"

namespace open_spiel {

namespace py = ::pybind11;
namespace ps = open_spiel::public_states;

void init_pyspiel_public_states(::pybind11::module& m) {
  py::class_<ps::GameWithPublicStatesType> game_with_public_states_type(
      m, "GameWithPublicStatesType");
  game_with_public_states_type
      .def(py::init<std::string, bool, bool>(), py::arg("short_name"),
           py::arg("provides_cfr_computation"),
           py::arg("provides_state_compatibility_check"))
      .def_readonly("short_name", &ps::GameWithPublicStatesType::short_name)
      .def_readonly("provides_cfr_computation",
                    &ps::GameWithPublicStatesType::provides_cfr_computation)
      .def_readonly(
          "provides_state_compatibility_check",
          &ps::GameWithPublicStatesType::provides_state_compatibility_check)
      .def("__repr__", [](const ps::GameWithPublicStatesType& gt) {
        return "<GameWithPublicStatesType '" + gt.short_name + "'>";
      });

  py::class_<ps::PrivateInformation> private_information(m,
                                                         "PrivateInformation");
  private_information.def("get_player", &ps::PrivateInformation::GetPlayer)
      .def("reach_probs_index", &ps::PrivateInformation::ReachProbsIndex)
      .def("network_index", &ps::PrivateInformation::NetworkIndex)
      .def("is_state_compatible", &ps::PrivateInformation::IsStateCompatible)
      .def("__str__", &ps::PrivateInformation::ToString)
      .def("clone", &ps::PrivateInformation::Clone)
      .def("serialize", &ps::PrivateInformation::Serialize)
      .def("get_game", &ps::PrivateInformation::GetGame);

  py::class_<ps::CfPrivValues> cf_priv_values(m, "CfPrivValues");
  cf_priv_values.def_readonly("player", &ps::CfPrivValues::player)
      .def_readwrite("cfvs", &ps::CfPrivValues::cfvs);

  py::class_<ps::CfActionValues> cf_action_values(m, "CfActionValues");
  cf_action_values.def_readonly("player", &ps::CfActionValues::player)
      .def_readwrite("cfavs", &ps::CfActionValues::cfavs);

  py::class_<ps::ReachProbs> reach_probs(m, "ReachProbs");
  reach_probs.def_readonly("player", &ps::ReachProbs::player)
      .def_readwrite("probs", &ps::ReachProbs::probs);

  py::class_<ps::GameWithPublicStates,
             std::shared_ptr<ps::GameWithPublicStates>>
      game(m, "GameWithPublicStates");
  game.def("new_initial_public_state",
           &ps::GameWithPublicStates::NewInitialPublicState)
      .def("new_initial_reach_probs",
           &ps::GameWithPublicStates::NewInitialReachProbs)
      .def("max_distinct_private_informations_count",
           &ps::GameWithPublicStates::MaxDistinctPrivateInformationsCount)
      .def("sum_max_distinct_private_informations",
           &ps::GameWithPublicStates::SumMaxDistinctPrivateInformations)
      .def("num_public_features", &ps::GameWithPublicStates::NumPublicFeatures)
      .def("network_input_size", &ps::GameWithPublicStates::NetworkInputSize)
      .def("deserialize_private_information",
           &ps::GameWithPublicStates::DeserializePrivateInformation)
      .def("deserialize_public_state",
           &ps::GameWithPublicStates::DeserializePublicState)
      .def("get_base_game", &ps::GameWithPublicStates::GetBaseGame);

  py::class_<ps::PublicState> public_state(m, "PublicState");
  public_state
      .def("get_public_observation_history",
           &ps::PublicState::GetPublicObservationHistory)
      .def("num_distinct_private_informations",
           &ps::PublicState::NumDistinctPrivateInformations)
      .def("get_private_informations", &ps::PublicState::GetPrivateInformations)
      .def("get_public_set", &ps::PublicState::GetPublicSet)
      .def("get_information_state", &ps::PublicState::GetInformationState)
      .def("get_information_set", &ps::PublicState::GetInformationSet)
      .def("get_world_state", &ps::PublicState::GetWorldState)
      .def("resample_from_public_set", &ps::PublicState::ResampleFromPublicSet)
      .def("resample_from_information_set",
           &ps::PublicState::ResampleFromInformationSet)
      .def("apply_public_transition", &ps::PublicState::ApplyPublicTransition)
      .def("apply_state_action", &ps::PublicState::ApplyStateAction)
      .def("child", &ps::PublicState::Child)
      .def("legal_transitions", &ps::PublicState::LegalTransitions)
      .def("get_private_actions", &ps::PublicState::GetPrivateActions)
      .def("undo_transition", &ps::PublicState::UndoTransition)
      .def("is_public_transition_legal", &ps::PublicState::IsTransitionLegal)
      .def("is_chance", &ps::PublicState::IsChance)
      .def("is_terminal", &ps::PublicState::IsTerminal)
      .def("is_player", &ps::PublicState::IsPlayer)
      .def("acting_players", &ps::PublicState::ActingPlayers)
      .def("terminal_returns", &ps::PublicState::TerminalReturns)
      .def("compute_reach_probs", &ps::PublicState::ComputeReachProbs)
      .def("terminal_cf_values", &ps::PublicState::TerminalCfValues)
      .def("compute_cf_priv_values", &ps::PublicState::ComputeCfPrivValues)
      .def("compute_cf_action_values", &ps::PublicState::ComputeCfActionValues)
      .def("public_features_tensor", &ps::PublicState::PublicFeaturesTensor)
      .def("reach_probs_tensor", &ps::PublicState::ReachProbsTensor)
      .def("to_tensor", &ps::PublicState::ToTensor)
      .def("to_string", &ps::PublicState::ToString)
      .def("move_number", &ps::PublicState::MoveNumber)
      .def("is_root", &ps::PublicState::IsRoot)
      .def("clone", &ps::PublicState::Clone)
      .def("serialize", &ps::PublicState::Serialize)
      .def("get_base_game", &ps::PublicState::GetBaseGame)
      .def("get_public_game", &ps::PublicState::GetPublicGame);

  m.def("load_game_with_public_states",
        py::overload_cast<const std::string&>(&ps::LoadGameWithPublicStates),
        "Returns a new game object for the specified short name using default "
        "parameters");

  m.def("load_game_with_public_states",
        py::overload_cast<const std::string&, const GameParameters&>(
            &ps::LoadGameWithPublicStates),
        "Returns a new game object for the specified short name using given "
        "parameters");

  m.def("registered_names_with_public_states",
        ps::RegisteredGamesWithPublicStates,
        "Returns the names of all available games.");

  m.def("registered_games_with_public_states",
        ps::RegisteredGameTypesWithPublicStates,
        "Returns the details of all available games.");

  m.def("serialize_game_with_public_state", ps::SerializeGameWithPublicState,
        "A general implementation of game with public state "
        "and public state serialization.");

  m.def("deserialize_game_with_public_state",
        ps::DeserializeGameWithPublicState,
        "A general implementation of deserialization of "
        "string serialized by serialize_game_with_public_state.");
}

}  // namespace open_spiel
