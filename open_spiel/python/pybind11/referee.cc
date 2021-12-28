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

#include "open_spiel/python/pybind11/referee.h"

// Python bindings for referee and tournament between bots.

#include "open_spiel/higc/referee.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace {

namespace py = ::pybind11;
}  // namespace

void init_pyspiel_referee(py::module& m) {
  py::class_<higc::TournamentSettings> settings(m, "TournamentSettings");
  settings.def(py::init<int, int, int, int, int, int, int, double>(),
               py::arg("timeout_ready") = 200, py::arg("timeout_start") = 100,
               py::arg("timeout_act") = 100, py::arg("timeout_ponder") = 50,
               py::arg("timeout_match_over") = 100,
               py::arg("time_tournament_over") = 100,
               py::arg("max_invalid_behaviors") = 1,
               py::arg("disqualification_rate") = .1);
  settings
      .def_readonly("timeout_ready", &higc::TournamentSettings::timeout_ready)
      .def_readonly("timeout_start", &higc::TournamentSettings::timeout_start)
      .def_readonly("timeout_act", &higc::TournamentSettings::timeout_act)
      .def_readonly("timeout_ponder", &higc::TournamentSettings::timeout_ponder)
      .def_readonly("timeout_match_over",
                    &higc::TournamentSettings::timeout_match_over)
      .def_readonly("time_tournament_over",
                    &higc::TournamentSettings::time_tournament_over)
      .def_readonly("max_invalid_behaviors",
                    &higc::TournamentSettings::max_invalid_behaviors)
      .def_readonly("disqualification_rate",
                    &higc::TournamentSettings::disqualification_rate);

  py::class_<higc::TournamentResults> results(m, "TournamentResults");
  results.def_readonly("num_bots", &higc::TournamentResults::num_bots)
      .def_readonly("matches", &higc::TournamentResults::matches)
      .def_readonly("returns_mean", &higc::TournamentResults::returns_mean)
      .def("returns_var", &higc::TournamentResults::returns_var)
      .def_readonly("history_len_mean",
                    &higc::TournamentResults::history_len_mean)
      .def_readonly("corrupted_matches",
                    &higc::TournamentResults::corrupted_matches)
      .def_readonly("disqualified", &higc::TournamentResults::disqualified)
      .def_readonly("restarts", &higc::TournamentResults::restarts)
      .def("__repr__", &higc::TournamentResults::ToString);

  py::class_<higc::MatchResult> match(m, "MatchResult");
  match.def_readonly("terminal", &higc::MatchResult::terminal)
      .def_readonly("errors", &higc::MatchResult::errors)
      .def("__repr__", &higc::MatchResult::ToString);

  py::class_<higc::BotErrors> errors(m, "BotErrors");
  errors.def_readonly("protocol_error", &higc::BotErrors::protocol_error)
      .def_readonly("illegal_actions", &higc::BotErrors::illegal_actions)
      .def_readonly("ponder_error", &higc::BotErrors::ponder_error)
      .def_readonly("time_over", &higc::BotErrors::time_over)
      .def("total_errors", &higc::BotErrors::total_errors);

  // TODO(author13): expose ostream in Python for logging.
  // Now all logging is printed to stdout.
  // Maybe something like this:
  // https://gist.github.com/asford/544323a5da7dddad2c9174490eb5ed06
  py::class_<higc::Referee> referee(m, "Referee");
  referee
      .def(py::init<const std::string&, const std::vector<std::string>&, int,
                    higc::TournamentSettings>(),
           py::arg("game_name"), py::arg("executables"), py::arg("seed") = 42,
           py::arg("settings") = higc::TournamentSettings())
      .def("play_tournament", &higc::Referee::PlayTournament,
           py::arg("num_matches"));
}

}  // namespace open_spiel
