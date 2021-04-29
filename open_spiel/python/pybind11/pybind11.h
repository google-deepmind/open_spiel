// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPEN_SPIEL_PYTHON_PYBIND11_PYBIND11_H_
#define OPEN_SPIEL_PYTHON_PYBIND11_PYBIND11_H_

// Common definitions and includes for pybind code.

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

#endif  // OPEN_SPIEL_PYTHON_PYBIND11_PYBIND11_H_
