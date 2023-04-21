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

#include "open_spiel/python/pybind11/algorithms_infostate_tree.h"

#include "open_spiel/algorithms/infostate_tree.h"

namespace open_spiel {

using namespace algorithms;
namespace py = ::pybind11;

void init_pyspiel_sequence_id(::pybind11::module &m)
{
   init_pyspiel_node_id< SequenceId >(m, "SequenceId");
   m.attr("kUndefinedSequenceId") = ::pybind11::cast(kUndefinedSequenceId);
}

void init_pyspiel_decision_id(::pybind11::module &m)
{
   init_pyspiel_node_id< DecisionId >(m, "DecisionId");
   m.attr("kUndefinedDecisionId") = ::pybind11::cast(kUndefinedDecisionId);
}

void init_pyspiel_leaf_id(::pybind11::module &m)
{
   init_pyspiel_node_id< LeafId >(m, "LeafId");
   m.attr("kUndefinedLeafId") = ::pybind11::cast(kUndefinedLeafId);
}

void init_pyspiel_infostate_tree(::pybind11::module &m)
{
   py::enum_< InfostateNodeType >(m, "InfostateNodeType")
      .value("kDecisionInfostateNode", InfostateNodeType::kDecisionInfostateNode)
      .value("kObservationInfostateNode", InfostateNodeType::kObservationInfostateNode)
      .value("kTerminalInfostateNode", InfostateNodeType::kTerminalInfostateNode)
      .export_values();

   m.attr("k_dummy_root_node_infostate") = algorithms::kDummyRootNodeInfostate;
   m.attr("k_filler_infostate") = algorithms::kFillerInfostate;
   m.attr("kUndefinedLeafId") = kUndefinedLeafId;

   m.def("IsValidSfStrategy", &IsValidSfStrategy);

   // the suffix is float despite using double, since it python the default floating point type is a
   // double.
   init_pyspiel_treevector_bundle< double >(m, "Float");
   // a generic tree vector bundle holding any type of python object
   init_pyspiel_treevector_bundle< py::object >(m, "");

   py::class_< InfostateTree >(m, "InfostateTree")
      .def("root", &InfostateTree::root, py::return_value_policy::reference)
      .def("mutable_root", &InfostateTree::mutable_root, py::return_value_policy::reference)
      .def("root_branching_factor", &InfostateTree::root_branching_factor)
      .def("acting_player", &InfostateTree::acting_player)
      .def("tree_height", &InfostateTree::tree_height)
      .def("num_decisions", &InfostateTree::num_decisions)
      .def("num_sequences", &InfostateTree::num_sequences)
      .def("num_leaves", &InfostateTree::num_leaves)
      .def("empty_sequence", &InfostateTree::empty_sequence)
      .def(
         "observation_infostate",
         py::overload_cast< const SequenceId & >(&InfostateTree::observation_infostate),
         py::return_value_policy::reference
      )
      .def(
         "observation_infostate",
         py::overload_cast< const SequenceId & >(&InfostateTree::observation_infostate, py::const_),
         py::return_value_policy::reference
      )
      .def("AllSequenceIds", &InfostateTree::AllSequenceIds)
      .def("DecisionIdsWithParentSeq", &InfostateTree::DecisionIdsWithParentSeq)
      .def("DecisionIdForSequence", &InfostateTree::DecisionIdForSequence)
      .def(
         "DecisionForSequence",
         &InfostateTree::DecisionForSequence,
         py::return_value_policy::reference
      )
      .def("IsLeafSequence", &InfostateTree::IsLeafSequence)
      .def(
         "decision_infostate",
         py::overload_cast< const DecisionId & >(&InfostateTree::decision_infostate),
         py::return_value_policy::reference
      )
      .def(
         "decision_infostate",
         py::overload_cast< const DecisionId & >(&InfostateTree::decision_infostate, py::const_),
         py::return_value_policy::reference
      )
      .def(
         "AllDecisionInfostates",
         &InfostateTree::AllDecisionInfostates,
         py::return_value_policy::reference
      )
      .def("AllDecisionIds", &InfostateTree::AllDecisionIds)
      .def("DecisionIdFromInfostateString", &InfostateTree::DecisionIdFromInfostateString)
      .def("leaf_nodes", &InfostateTree::leaf_nodes, py::return_value_policy::reference)
      .def("leaf_node", &InfostateTree::leaf_node, py::return_value_policy::reference)
      .def("nodes_at_depths", &InfostateTree::nodes_at_depths, py::return_value_policy::reference)
      .def("nodes_at_depth", &InfostateTree::nodes_at_depth, py::return_value_policy::reference)
      .def("BestResponse", &InfostateTree::BestResponse)
      .def("BestResponseValue", &InfostateTree::BestResponseValue);
}


}  // namespace open_spiel
