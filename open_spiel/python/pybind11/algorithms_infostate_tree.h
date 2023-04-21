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

#ifndef OPEN_SPIEL_PYTHON_PYBIND11_INFOSTATE_TREE_H
#define OPEN_SPIEL_PYTHON_PYBIND11_INFOSTATE_TREE_H

#include "open_spiel/python/pybind11/pybind11.h"
#include "pybind11_abseil/absl_casters.h"

namespace open_spiel {

void init_pyspiel_infostate_tree(::pybind11::module& m);

void init_pyspiel_infostate_node(::pybind11::module& m);

template < typename T >
void init_pyspiel_treevector_bundle(::pybind11::module& m, std::string& typestr);

template < typename Self >
void init_pyspiel_node_id(::pybind11::module& m, const std::string& class_name);

// Bind the Range class
template < class Id >
void init_pyspiel_range(::pybind11::module& m, const std::string& name);


}  // namespace open_spiel

// include the template definition file
#include "open_spiel/python/pybind11/algorithms_infostate_tree.tcc"

#endif  // OPEN_SPIEL_PYTHON_PYBIND11_INFOSTATE_TREE_H
