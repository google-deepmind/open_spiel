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

#include "open_spiel/python/pybind11/games_kuhn_poker.h"

#include "open_spiel/games/kuhn_poker.h"
#include "open_spiel/python/pybind11/pybind11.h"

namespace py = ::pybind11;

void open_spiel::init_pyspiel_games_kuhn_poker(py::module& m) {
  py::module sub = m.def_submodule("kuhn_poker");
  sub.def("get_optimal_policy", &kuhn_poker::GetOptimalPolicy);
}
