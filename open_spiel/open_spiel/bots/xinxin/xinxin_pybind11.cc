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

#include "open_spiel/bots/xinxin/xinxin_bot.h"
#include "open_spiel/spiel_bots.h"
#include "open_spiel/python/pybind11/pybind11.h"

namespace open_spiel {

namespace py = ::pybind11;

void init_pyspiel_xinxin(::pybind11::module& m) {
  m.def("make_xinxin_bot", open_spiel::hearts::MakeXinxinBot, py::arg("params"),
        py::arg("uct_num_runs") = 50, py::arg("uct_c_val") = 0.4,
        py::arg("iimc_num_worlds") = 20, py::arg("use_threads") = true,
        "Make the XinXin bot.");
}

}  // namespace open_spiel
