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

#include "open_spiel/python/pybind11/algorithms_trajectories.h"

// Python bindings for trajectories.h

#include "open_spiel/algorithms/trajectories.h"
#include "open_spiel/python/pybind11/pybind11.h"

namespace open_spiel {
namespace py = ::pybind11;

void init_pyspiel_algorithms_trajectories(py::module& m) {
  py::class_<open_spiel::algorithms::BatchedTrajectory>(m, "BatchedTrajectory")
      .def(py::init<int>())
      .def_readwrite("observations",
                     &open_spiel::algorithms::BatchedTrajectory::observations)
      .def_readwrite("state_indices",
                     &open_spiel::algorithms::BatchedTrajectory::state_indices)
      .def_readwrite("legal_actions",
                     &open_spiel::algorithms::BatchedTrajectory::legal_actions)
      .def_readwrite("actions",
                     &open_spiel::algorithms::BatchedTrajectory::actions)
      .def_readwrite(
          "player_policies",
          &open_spiel::algorithms::BatchedTrajectory::player_policies)
      .def_readwrite("player_ids",
                     &open_spiel::algorithms::BatchedTrajectory::player_ids)
      .def_readwrite("rewards",
                     &open_spiel::algorithms::BatchedTrajectory::rewards)
      .def_readwrite("valid", &open_spiel::algorithms::BatchedTrajectory::valid)
      .def_readwrite(
          "next_is_terminal",
          &open_spiel::algorithms::BatchedTrajectory::next_is_terminal)
      .def_readwrite("batch_size",
                     &open_spiel::algorithms::BatchedTrajectory::batch_size)
      .def_readwrite(
          "max_trajectory_length",
          &open_spiel::algorithms::BatchedTrajectory::max_trajectory_length)
      .def("resize_fields",
           &open_spiel::algorithms::BatchedTrajectory::ResizeFields);

  m.def("record_batched_trajectories",
        py::overload_cast<
            const Game&, const std::vector<open_spiel::TabularPolicy>&,
            const std::unordered_map<std::string, int>&, int, bool, int, int>(
            &open_spiel::algorithms::RecordBatchedTrajectory),
        "Records a batch of trajectories.");

  py::class_<open_spiel::algorithms::TrajectoryRecorder>(m,
                                                         "TrajectoryRecorder")
      .def(py::init<const Game&, const std::unordered_map<std::string, int>&,
                    int>())
      .def("record_batch",
           &open_spiel::algorithms::TrajectoryRecorder::RecordBatch);
}

}  // namespace open_spiel
