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

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"
#include "open_spiel/spiel_utils.h"
#include "pybind11/include/pybind11/cast.h"
#include "pybind11/include/pybind11/detail/common.h"
#include "pybind11/include/pybind11/detail/descr.h"
#include "pybind11/include/pybind11/functional.h"  // IWYU pragma: keep
#include "pybind11/include/pybind11/numpy.h"  // IWYU pragma: keep
#include "pybind11/include/pybind11/pybind11.h"
#include "pybind11/include/pybind11/stl.h"  // IWYU pragma: keep

namespace open_spiel {

class Policy;
class TabularPolicy;
class PartialTabularPolicy;
class UniformPolicy;
class PreferredActionPolicy;

class NormalFormGame;
class Bot;

namespace matrix_game {
class MatrixGame;
}

namespace tensor_game {
class TensorGame;
}

namespace algorithms {
class MCTSBot;
class ISMCTSBot;
}  // namespace algorithms
}  // namespace open_spiel

namespace open_spiel {
// Trampoline helper class to allow implementing Bots in Python. See
// https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python
template <class BotBase = Bot>
class PyBot : public BotBase, public ::pybind11::trampoline_self_life_support {
 public:
  // We need the bot constructor
  using BotBase::BotBase;
  ~PyBot() override = default;

  // Choose and execute an action in a game. The bot should return its
  // distribution over actions and also its selected action.
  open_spiel::Action Step(const State& state) override {
    PYBIND11_OVERLOAD_PURE_NAME(
        open_spiel::Action,  // Return type (must be simple token)
        BotBase,             // Parent class
        "step",              // Name of function in Python
        Step,                // Name of function in C++
        state                // Arguments
    );
  }

  // Restart at the specified state.
  void Restart() override {
    PYBIND11_OVERLOAD_NAME(
        void,       // Return type (must be a simple token for macro parser)
        BotBase,    // Parent class
        "restart",  // Name of function in Python
        Restart,    // Name of function in C++
        // The trailing coma after Restart is necessary to say "No argument"
    );
  }
  bool ProvidesForceAction() override {
    PYBIND11_OVERLOAD_NAME(
        bool,     // Return type (must be a simple token for macro parser)
        BotBase,  // Parent class
        "provides_force_action",  // Name of function in Python
        ProvidesForceAction,      // Name of function in C++
                                  // Arguments
    );
  }
  void ForceAction(const State& state, Action action) override {
    PYBIND11_OVERLOAD_NAME(
        void,     // Return type (must be a simple token for macro parser)
        BotBase,  // Parent class
        "force_action",  // Name of function in Python
        ForceAction,     // Name of function in C++
        state,           // Arguments
        action);
  }
  void InformAction(const State& state, Player player_id,
                    Action action) override {
    PYBIND11_OVERLOAD_NAME(
        void,     // Return type (must be a simple token for macro parser)
        BotBase,  // Parent class
        "inform_action",  // Name of function in Python
        InformAction,     // Name of function in C++
        state,            // Arguments
        player_id, action);
  }
  void InformActions(const State& state,
                     const std::vector<Action>& actions) override {
    PYBIND11_OVERLOAD_NAME(
        void,     // Return type (must be a simple token for macro parser)
        BotBase,  // Parent class
        "inform_actions",  // Name of function in Python
        InformActions,     // Name of function in C++
        state,             // Arguments
        actions);
  }

  void RestartAt(const State& state) override {
    PYBIND11_OVERLOAD_NAME(
        void,          // Return type (must be a simple token for macro parser)
        BotBase,       // Parent class
        "restart_at",  // Name of function in Python
        RestartAt,     // Name of function in C++
        state          // Arguments
    );
  }
  bool ProvidesPolicy() override {
    PYBIND11_OVERLOAD_NAME(
        bool,     // Return type (must be a simple token for macro parser)
        BotBase,  // Parent class
        "provides_policy",  // Name of function in Python
        ProvidesPolicy,     // Name of function in C++
                            // Arguments
    );
  }
  ActionsAndProbs GetPolicy(const State& state) override {
    PYBIND11_OVERLOAD_NAME(ActionsAndProbs,  // Return type (must be a simple
                                             // token for macro parser)
                           BotBase,          // Parent class
                           "get_policy",     // Name of function in Python
                           GetPolicy,        // Name of function in C++
                           state);
  }
  std::pair<ActionsAndProbs, Action> StepWithPolicy(
      const State& state) override {
    using step_retval_t = std::pair<ActionsAndProbs, open_spiel::Action>;
    PYBIND11_OVERLOAD_NAME(
        step_retval_t,  // Return type (must be a simple token for macro parser)
        BotBase,        // Parent class
        "step_with_policy",  // Name of function in Python
        StepWithPolicy,      // Name of function in C++
        state                // Arguments
    );
  }

  bool IsClonable() const override {
    PYBIND11_OVERLOAD_NAME(
        bool,           // Return type (must be a simple token for macro parser)
        BotBase,        // Parent class
        "is_clonable",  // Name of function in Python
        IsClonable,     // Name of function in C++
    );
  }

  std::unique_ptr<Bot> Clone() override {
    using BotUniquePtr = std::unique_ptr<Bot>;
    PYBIND11_OVERLOAD_NAME(
        BotUniquePtr,  // Return type (must be a simple token for macro parser)
        BotBase,       // Parent class
        "clone",       // Name of function in Python
        Clone,         // Name of function in C++
    );
  }
};
}  // namespace open_spiel

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
