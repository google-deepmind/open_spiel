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

#include "open_spiel/games/gamut/gamut.h"
#include "open_spiel/python/pybind11/pybind11.h"

namespace open_spiel {

namespace py = ::pybind11;

void init_pyspiel_gamut(::pybind11::module& m) {
  py::class_<gamut::GamutGenerator> gamut_generator(m, "GamutGenerator");
  gamut_generator.def(py::init<std::string>())
      .def(py::init<std::string, std::string>())
      .def("generate_game", py::overload_cast<const std::string&>(
                                &gamut::GamutGenerator::GenerateGame))
      .def("generate_game", py::overload_cast<const std::vector<std::string>&>(
                                &gamut::GamutGenerator::GenerateGame))
      .def("generate_matrix_game", &gamut::GamutGenerator::GenerateMatrixGame)
      .def("generate_tensor_game", &gamut::GamutGenerator::GenerateTensorGame);
}

}  // namespace open_spiel
