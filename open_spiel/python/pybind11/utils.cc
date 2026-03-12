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

#include <string>
#include <vector>

#include "open_spiel/utils/sgf_reader.h"
#include "open_spiel/utils/status.h"
#include "open_spiel/utils/file.h"
#include "open_spiel/python/pybind11/pybind11.h"  // NOLINT

namespace open_spiel {

namespace py = ::pybind11;
using open_spiel::SgfProperty;
using open_spiel::SgfNode;

void init_pyspiel_utils(py::module& m) {
  py::class_<open_spiel::StatusWithValue<std::vector<SgfNode>>>(m,
      "StatusWithSgfNodes")
      .def(py::init<open_spiel::StatusValue, std::string,
              std::vector<SgfNode>>(),
           py::arg("status_value"),
           py::arg("message"),
           py::arg("nodes"))
      .def("ok", &open_spiel::Status::ok)
      .def("message", &open_spiel::Status::message)
      .def("to_string", &open_spiel::Status::ToString)
      .def("value", &open_spiel::StatusWithValue<std::vector<SgfNode>>::value);

  py::class_<SgfProperty>(m, "SgfProperty")
      .def(py::init<>())
      .def_readonly("name", &SgfProperty::name)
      .def_readonly("values", &SgfProperty::values);

  py::class_<SgfNode>(m, "SgfNode")
      .def(py::init<>())
      .def_readonly("properties", &SgfNode::properties)
      .def_readonly("children", &SgfNode::children);

  // read_contents_from_file(string filename, string mode)
  m.def("read_contents_from_file", file::ReadContentsFromFile,
        "Read the entire contents of a file.");

  // write_contents_to_file(string filename, string mode, string contents)
  m.def("write_contents_to_file", open_spiel::file::WriteContentsToFile,
        "Write the contents of the string to the specified filename.");

  m.def("read_sgf_string", open_spiel::ReadSgfString,
        "Read an SGF string and return a StatusWithValue of SgfNodes.");
}
}  // namespace open_spiel
