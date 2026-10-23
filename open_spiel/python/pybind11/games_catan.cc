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

#include "open_spiel/python/pybind11/games_catan.h"

#include "open_spiel/games/catan/catan.h"
#include "open_spiel/json/include/nlohmann/json.hpp"  // IWYU pragma: keep
#include "open_spiel/pybind11_json/include/pybind11_json/pybind11_json.hpp"
#include "open_spiel/python/pybind11/pybind11.h"
#include "open_spiel/spiel.h"

namespace py = ::pybind11;

void open_spiel::init_pyspiel_games_catan(py::module& m) {
  py::module_ catan = m.def_submodule("catan");
  bind_spiel_struct<catan::CatanStateStruct, StateStruct>(catan,
                                                          "CatanStateStruct");
  bind_spiel_struct<catan::CatanObservationStruct, ObservationStruct>(
      catan, "CatanObservationStruct");
  bind_spiel_struct<catan::CatanActionStruct, ActionStruct>(
      catan, "CatanActionStruct");
}
