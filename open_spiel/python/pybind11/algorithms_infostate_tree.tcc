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

#ifndef OPEN_SPIEL_PYTHON_PYBIND11_INFOSTATE_TREE_TCC
#define OPEN_SPIEL_PYTHON_PYBIND11_INFOSTATE_TREE_TCC

#include "open_spiel/algorithms/infostate_tree.h"
#include "open_spiel/python/pybind11/algorithms_infostate_tree.h"

namespace open_spiel {

using namespace algorithms;

namespace detail {
template < typename T, template < typename > class TreeVectorDerived, typename IdType >
void _init_pyspiel_treevector_bundle_impl(
   ::pybind11::module &m,
   const std::string &template_name,
   const std::string &type_name
)
{
   ::pybind11::class_< TreeVectorDerived< T > >(m, (template_name + type_name).c_str())
      .def(::pybind11::init< const InfostateTree * >(), ::pybind11::arg("tree"))
      .def(
         ::pybind11::init< const InfostateTree *, std::vector< T > >(),
         ::pybind11::arg("tree"),
         ::pybind11::arg("vec")
      )
      .def(
         "__getitem__",
         [](const TreeVectorDerived< T > &self, const IdType &id) { return self[id]; },
         ::pybind11::arg("id")
      )
      .def("__len__", &TreeVectorDerived< T >::size)
      .def("__repr__", &TreeVectorDerived< T >::operator<<);
}
}  // namespace detail

template < typename T >
void init_pyspiel_treevector_bundle(::pybind11::module &m, const std::string &typestr)
{
   detail::_init_pyspiel_treevector_bundle_impl< T, TreeplexVector, SequenceId >(
      m, "TreeplexVector", typestr
   );
   detail::_init_pyspiel_treevector_bundle_impl< T, LeafVector, LeafId >(m, "LeafVector", typestr);
   detail::_init_pyspiel_treevector_bundle_impl< T, DecisionVector, DecisionId >(
      m, "DecisionVector", typestr
   );
}

template < typename Self >
void init_pyspiel_node_id(::pybind11::module &m, const std::string &class_name)
{
   ::pybind11::class_< Self >(m, class_name.c_str())
      .def(
         ::pybind11::init< size_t, const InfostateTree * >(),
         ::pybind11::arg("id_value"),
         ::pybind11::arg("tree")
      )
      .def("id", &Self::id)
      .def("is_undefined", &Self::is_undefined)
      .def("next", &Self::next)
      .def("__eq__", &Self::operator==)
      .def("__ne__", &Self::operator!=);
}

template < class Id >
void init_pyspiel_range(::pybind11::module &m, const std::string &name)
{
   ::pybind11::class_< Range< Id > >(m, name.c_str())
      .def(
         ::pybind11::init< size_t, size_t, const InfostateTree * >(),
         ::pybind11::arg("start"),
         ::pybind11::arg("end"),
         ::pybind11::arg("tree")
      )
      .def(
         "__iter__",
         [](const Range< Id > &r) { return ::pybind11::make_iterator(r.begin(), r.end()); },
         ::pybind11::keep_alive< 0, 1 >()
      );
}

}  // namespace open_spiel

#endif  // OPEN_SPIEL_PYTHON_PYBIND11_INFOSTATE_TREE_TCC