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

template < typename T >
void init_pyspiel_treevector_bundle(::pybind11::module &m, std::string typestr)
{
   ::pybind11::class_< TreeplexVector< T > >(m, (std::string("TreeplexVector") + typestr).c_str())
      .def(::pybind11::init< const InfostateTree * >())
      .def(::pybind11::init< const InfostateTree *, std::vector< T > >())
      .def("view", [](const TreeplexVector< T > &self, const SequenceId &id) { return self[id]; })
      .def("__getitem__", [](TreeplexVector< T > &self, const SequenceId &id) { return self[id]; })
      .def("__len__", &TreeplexVector< T >::size)
      .def("__repr__", &TreeplexVector< T >::operator<<);

   ::pybind11::class_< LeafVector< T > >(m, (std::string("LeafVector") + typestr).c_str())
      .def(::pybind11::init< const InfostateTree * >())
      .def(::pybind11::init< const InfostateTree *, std::vector< T > >())
      .def("__getitem__", [](const LeafVector< T > &self, const LeafId &id) { return self[id]; })
      .def("__len__", &LeafVector< T >::size)
      .def("__repr__", &LeafVector< T >::operator<<);

   ::pybind11::class_< DecisionVector< T > >(m, (std::string("DecisionVector") + typestr).c_str())
      .def(::pybind11::init< const InfostateTree * >())
      .def(::pybind11::init< const InfostateTree *, std::vector< T > >())
      .def(
         "__getitem__",
         [](const DecisionVector< T > &self, const DecisionId &id) { return self[id]; }
      )
      .def("__len__", &DecisionVector< T >::size)
      .def("__repr__", &DecisionVector< T >::operator<<);
}

template < typename Self >
void init_pyspiel_node_id(::pybind11::module &m, const std::string &class_name)
{
   ::pybind11::class_< Self >(m, class_name.c_str())
      .def(::pybind11::init< size_t, const InfostateTree * >())
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
      .def(::pybind11::init< size_t, size_t, const InfostateTree * >())
      .def(
         "__iter__",
         [](const Range< Id > &r) -> RangeIterator< Id > {
            return ::pybind11::make_iterator(r.begin(), r.end());
         },
         ::pybind11::keep_alive< 0, 1 >()
      )
      .def("__repr__", [](const Range< Id > &r) {
         return "<" + std::string(typeid(Range< Id >).name()) + " object at "
                + std::to_string(reinterpret_cast< std::uintptr_t >(&r)) + ">";
      });
}

template < class Id >
void init_pyspiel_range_iterator(::pybind11::module &m, const std::string &name)
{
   ::pybind11::class_< RangeIterator< Id > >(m, name.c_str())
      .def(::pybind11::init< size_t, const InfostateTree * >())
      .def(
         "__iter__", [](const RangeIterator< Id > &it) -> const RangeIterator< Id > & { return it; }
      )
      .def(
         "__next__",
         [](RangeIterator< Id > &it) -> Id {
            if(it.id_ == it.tree_->num_nodes()) {
               throw ::pybind11::stop_iteration();
            }
            return *it++;
         }
      )
      .def("__eq__", &RangeIterator< Id >::operator==)
      .def("__ne__", &RangeIterator< Id >::operator!=)
      .def("__repr__", [](const RangeIterator< Id > &it) {
         return "<" + std::string(typeid(RangeIterator< Id >).name()) + " object at "
                + std::to_string(reinterpret_cast< std::uintptr_t >(&it)) + ">";
      });
}

}  // namespace open_spiel

#endif  // OPEN_SPIEL_PYTHON_PYBIND11_INFOSTATE_TREE_TCC