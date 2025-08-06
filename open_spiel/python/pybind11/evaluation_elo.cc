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

#include "open_spiel/python/pybind11/evaluation_elo.h"

#include <string>

#include "open_spiel/evaluation/elo.h"

namespace py = ::pybind11;

using open_spiel::evaluation::ComputeRatingsFromMatchRecords;
using open_spiel::evaluation::ComputeRatingsFromMatrices;
using open_spiel::evaluation::DefaultEloOptions;
using open_spiel::evaluation::DoubleArray2D;
using open_spiel::evaluation::EloOptions;
using open_spiel::evaluation::IntArray2D;
using open_spiel::evaluation::MatchOutcome;
using open_spiel::evaluation::MatchRecord;

using open_spiel::evaluation::kDefaultConvergenceDelta;
using open_spiel::evaluation::kDefaultMaxIterations;
using open_spiel::evaluation::kDefaultSmoothingFactor;
using open_spiel::evaluation::kStandardScaleFactor;
using open_spiel::evaluation::kDefaultMinimumRating;

namespace open_spiel {

void init_pyspiel_evaluation_elo(py::module& m) {
  py::module_ elo = m.def_submodule("elo");

  elo.attr("DEFAULT_SMOOTHING_FACTOR") = py::float_(kDefaultSmoothingFactor);
  elo.attr("DEFAULT_MAX_ITERATIONS") = py::int_(kDefaultMaxIterations);
  elo.attr("DEFAULT_CONVERGENCE_DELTA") = py::float_(kDefaultConvergenceDelta);
  elo.attr("STANDARD_SCALE_FACTOR") = py::float_(kStandardScaleFactor);
  elo.attr("DEFAULT_MINIMUM_RATING") = py::float_(kDefaultMinimumRating);

  py::class_<EloOptions>(elo, "EloOptions")
      .def_readwrite("smoothing_factor", &EloOptions::smoothing_factor)
      .def_readwrite("max_iterations", &EloOptions::max_iterations)
      .def_readwrite("convergence_delta", &EloOptions::convergence_delta)
      .def_readwrite("scale_factor", &EloOptions::scale_factor)
      .def_readwrite("minimum_rating", &EloOptions::minimum_rating);

  py::enum_<MatchOutcome>(elo, "MatchOutcome")
      .value("FIRST_PLAYER_WIN", MatchOutcome::kFirstPlayerWin)
      .value("FIRST_PLAYER_LOSS", MatchOutcome::kFirstPlayerLoss)
      .value("DRAW", MatchOutcome::kDraw)
      .export_values();

  py::class_<MatchRecord>(elo, "MatchRecord")
      .def(py::init<std::string, std::string, MatchOutcome>(),
              py::arg("first_player_name"),
              py::arg("second_player_name"),
              py::arg("outcome") = MatchOutcome::kFirstPlayerWin)
      .def_readwrite("first_player_name", &MatchRecord::first_player_name)
      .def_readwrite("second_player_name", &MatchRecord::second_player_name)
      .def_readwrite("outcome", &MatchRecord::outcome);

  elo.def("default_elo_options", DefaultEloOptions,
          "Return default EloOptions (see elo.h for values).");

  elo.def("compute_ratings_from_matrices", ComputeRatingsFromMatrices,
          "Compute Elo ratings from a win matrix and a draw matrix.",
          py::arg("win_matrix"), py::arg("draw_matrix") = py::list({}),
          py::arg("options") = DefaultEloOptions());

  elo.def("compute_ratings_from_match_records", ComputeRatingsFromMatchRecords,
          "Compute Elo ratings from a list of match records.",
          py::arg("match_records"), py::arg("options") = DefaultEloOptions());
}

}  // namespace open_spiel
