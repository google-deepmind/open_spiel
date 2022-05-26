// Copyright 2022 DeepMind Technologies Limited
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
#include "open_spiel/python/pybind11/utils.h"

#include "open_spiel/python/pybind11/pybind11.h"
#include "open_spiel/utils/file.h"

namespace open_spiel {

namespace py = ::pybind11;

void init_pyspiel_utils(py::module& m) {
  // read_contents_from_file(string filename, string mode)
  m.def("read_contents_from_file", file::ReadContentsFromFile,
        "Read the entire contents of a file.");

  // write_contents_to_file(string filename, string mode, string contents)
  m.def("write_contents_to_file", open_spiel::file::WriteContentsToFile,
        "Write the contents of the string to the specified filename.");
}

}  // namespace open_spiel
