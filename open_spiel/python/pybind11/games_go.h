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

#ifndef OPEN_SPIEL_PYTHON_PYBIND11_GAMES_GO_H_
#define OPEN_SPIEL_PYTHON_PYBIND11_GAMES_GO_H_

#include "open_spiel/python/pybind11/pybind11.h"

// Initialize the Python interface for games/connect_four.
namespace open_spiel {

void init_pyspiel_games_go(::pybind11::module &m);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_PYTHON_PYBIND11_GAMES_GO_H_
