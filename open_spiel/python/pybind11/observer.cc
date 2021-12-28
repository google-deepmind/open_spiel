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

#include "open_spiel/python/pybind11/observer.h"

// Python bindings for observers.

#include "open_spiel/game_transforms/normal_form_extensive_game.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"
#include "open_spiel/observer.h"
#include "open_spiel/python/pybind11/pybind11.h"

namespace open_spiel {
namespace py = ::pybind11;

void init_pyspiel_observer(py::module& m) {
  // C++ Observer, intended only for the Python Observation class, not
  // for general Python code.
  py::class_<Observer, std::shared_ptr<Observer>>(m, "Observer")
      .def("__str__", [](const Observer& self) { return "Observer()"; });

  py::class_<SpanTensorInfo>(m, "SpanTensorInfo")
      .def_property_readonly(
          "name", [](const SpanTensorInfo& info) { return info.name(); })
      .def_property_readonly(
          "shape",
          [](const SpanTensorInfo& info) { return info.vector_shape(); })
      .def("__str__", &SpanTensorInfo::DebugString);

  py::class_<SpanTensor>(m, "SpanTensor")
      .def_property_readonly(
          "name", [](const SpanTensor& tensor) { return tensor.info().name(); })
      .def_property_readonly(
          "shape",
          [](const SpanTensor& tensor) { return tensor.info().vector_shape(); })
      .def_property_readonly("data",
                             [](const SpanTensor& tensor) {
                               // absl::Span requires pybind11_abseil which
                               // open spiel forbids. Thus copy the data
                               // and expose a vector through pybind.
                               std::vector<float> data(tensor.data().begin(),
                                                       tensor.data().end());
                               return data;
                             })
      .def("__str__", &SpanTensor::DebugString);

  // C++ Observation, intended only for the Python Observation class, not
  // for general Python code.
  py::class_<Observation>(m, "_Observation", py::buffer_protocol())
      .def(py::init<const Game&, std::shared_ptr<Observer>>(), py::arg("game"),
           py::arg("observer"))
      .def("tensors", &Observation::tensors)
      .def("tensors_info", &Observation::tensors_info)
      .def("string_from", &Observation::StringFrom)
      .def("set_from", &Observation::SetFrom)
      .def("has_string", &Observation::HasString)
      .def("has_tensor", &Observation::HasTensor)
      .def("compress",
           [](const Observation& self) { return py::bytes(self.Compress()); })
      .def("decompress", &Observation::Decompress)
      .def_buffer([](Observation& buffer_observer) -> py::buffer_info {
        return py::buffer_info(
            buffer_observer.Tensor().data(),         // Pointer to buffer
            sizeof(float),                           // Size of one scalar
            py::format_descriptor<float>::format(),  // Format descriptor
            1,                                       // Num dimensions
            {buffer_observer.Tensor().size()},       // Dimensions
            {sizeof(float)}                          // Stride
        );
      });
}

}  // namespace open_spiel
