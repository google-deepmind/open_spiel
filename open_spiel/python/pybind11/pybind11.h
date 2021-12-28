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

#ifndef OPEN_SPIEL_PYTHON_PYBIND11_PYBIND11_H_
#define OPEN_SPIEL_PYTHON_PYBIND11_PYBIND11_H_

// Common definitions and includes for pybind code.

#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "pybind11/include/pybind11/detail/common.h"
#include "pybind11/include/pybind11/detail/descr.h"
#include "pybind11/include/pybind11/functional.h"
#include "pybind11/include/pybind11/numpy.h"
#include "pybind11/include/pybind11/operators.h"
#include "pybind11/include/pybind11/pybind11.h"
#include "pybind11/include/pybind11/pytypes.h"
#include "pybind11/include/pybind11/smart_holder.h"
#include "pybind11/include/pybind11/stl.h"

// Runtime errors happen if we're inconsistent about whether or not a type has
// PYBIND11_SMART_HOLDER_TYPE_CASTERS applied to it or not. So we do it mostly
// in one place to help with consistency.

namespace open_spiel {
class NormalFormGame;

namespace matrix_game {
class MatrixGame;
}

namespace tensor_game {
class TensorGame;
}
}  // namespace open_spiel

PYBIND11_SMART_HOLDER_TYPE_CASTERS(open_spiel::State);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(open_spiel::Game);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(open_spiel::NormalFormGame);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(open_spiel::matrix_game::MatrixGame);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(open_spiel::tensor_game::TensorGame);

// Custom caster for GameParameter (essentially a variant).
namespace pybind11 {
namespace detail {

template <>
struct type_caster<open_spiel::GameParameter> {
 public:
  PYBIND11_TYPE_CASTER(open_spiel::GameParameter, _("GameParameter"));

  bool load(handle src, bool convert) {
    if (src.is_none()) {
      // value is default-constructed to an unset value
      return true;
    } else if (PyBool_Check(src.ptr())) {
      value = open_spiel::GameParameter(src.cast<bool>());
      return true;
    } else if (auto str_val = maybe_load<std::string>(src, convert)) {
      value = open_spiel::GameParameter(*str_val);
      return true;
    } else if (PyFloat_Check(src.ptr())) {
      value = open_spiel::GameParameter(src.cast<double>());
      return true;
    } else if (PyLong_Check(src.ptr())) {
      value = open_spiel::GameParameter(src.cast<int>());
      return true;
    } else {
      auto dict = src.cast<pybind11::dict>();
      std::map<std::string, open_spiel::GameParameter> d;
      for (const auto& [k, v] : dict) {
        d[k.cast<std::string>()] = v.cast<open_spiel::GameParameter>();
      }
      value = open_spiel::GameParameter(d);
      return true;
    }
  }

  static handle cast(const open_spiel::GameParameter& gp,
                     return_value_policy policy, handle parent) {
    if (gp.has_bool_value()) {
      return pybind11::bool_(gp.bool_value()).release();
    } else if (gp.has_double_value()) {
      return pybind11::float_(gp.double_value()).release();
    } else if (gp.has_string_value()) {
      return pybind11::str(gp.string_value()).release();
    } else if (gp.has_int_value()) {
      return pybind11::int_(gp.int_value()).release();
    } else if (gp.has_game_value()) {
      pybind11::dict d;
      for (const auto& [k, v] : gp.game_value()) {
        d[pybind11::str(k)] = pybind11::cast(v);
      }
      return d.release();
    } else {
      return pybind11::none();
    }
  }

 private:
  template <typename T>
  absl::optional<T> maybe_load(handle src, bool convert) {
    auto caster = pybind11::detail::make_caster<T>();
    if (caster.load(src, convert)) {
      return cast_op<T>(caster);
    } else {
      return absl::nullopt;
    }
  }
};

}  // namespace detail
}  // namespace pybind11

#endif  // OPEN_SPIEL_PYTHON_PYBIND11_PYBIND11_H_
