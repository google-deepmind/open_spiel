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

#include <utility>

#include "open_spiel/algorithms/infostate_tree.h"

namespace open_spiel {

using namespace algorithms;
namespace py = ::pybind11;

void init_pyspiel_sequence_id(::pybind11::module &m)
{
   init_pyspiel_node_id< SequenceId >(m, "SequenceId");
   m.attr("UNDEFINED_SEQUENCE_ID") = ::pybind11::cast(kUndefinedSequenceId);
}

void init_pyspiel_decision_id(::pybind11::module &m)
{
   init_pyspiel_node_id< DecisionId >(m, "DecisionId");
   m.attr("UNDEFINED_DECISION_ID") = ::pybind11::cast(kUndefinedDecisionId);
}

void init_pyspiel_leaf_id(::pybind11::module &m)
{
   init_pyspiel_node_id< LeafId >(m, "LeafId");
   m.attr("UNDEFINED_LEAF_ID") = ::pybind11::cast(kUndefinedLeafId);
}

void init_pyspiel_infostate_node(::pybind11::module &m)
{
   py::class_< InfostateNode, std::unique_ptr< InfostateNode, py::nodelete > >(
      m, "InfostateNode", py::is_final()
   )
      .def_property_readonly("tree", &InfostateNode::tree, py::return_value_policy::reference)
      .def_property_readonly(
         "parent",
         [](const InfostateNode &node) {
            return std::unique_ptr< InfostateNode, py::nodelete >{node.parent()};
         }
      )
      .def_property_readonly("incoming_index", &InfostateNode::incoming_index)
      .def_property_readonly("type", &InfostateNode::type)
      .def_property_readonly("depth", &InfostateNode::depth)
      .def_property_readonly("is_root_node", &InfostateNode::is_root_node)
      .def_property_readonly("has_infostate_string", &InfostateNode::has_infostate_string)
      .def_property_readonly("infostate_string", &InfostateNode::infostate_string)
      .def(
         "child_at", &InfostateNode::child_at, py::arg("index"), py::return_value_policy::reference
      )
      .def_property_readonly("num_children", &InfostateNode::num_children)
      .def_property_readonly("child_iterator", &InfostateNode::child_iterator);

   py::enum_< InfostateNodeType >(m, "InfostateNodeType")
      .value("decision", InfostateNodeType::kDecisionInfostateNode)
      .value("observation_infostate_node", InfostateNodeType::kObservationInfostateNode)
      .value("terminal", InfostateNodeType::kTerminalInfostateNode)
      .export_values();
}

void init_pyspiel_vec_with_uniqptrs_iterator(::pybind11::module &m)
{
   py::class_< VecWithUniquePtrsIterator< InfostateNode > >(m, "InfostateNodeVecIterator")
      .def(
         "__iter__",
         [](VecWithUniquePtrsIterator< InfostateNode > &it
         ) -> VecWithUniquePtrsIterator< InfostateNode > & { return it; }
      )
      .def(
         "__next__",
         &VecWithUniquePtrsIterator< InfostateNode >::operator++,
         py::return_value_policy::reference
      )
      .def("__eq__", &VecWithUniquePtrsIterator< InfostateNode >::operator==)
      .def("__ne__", &VecWithUniquePtrsIterator< InfostateNode >::operator!=);
}

void init_pyspiel_infostate_tree(::pybind11::module &m)
{
   m.attr("DUMMY_ROOT_NODE_INFOSTATE") = algorithms::kDummyRootNodeInfostate;
   m.attr("FILLER_INFOSTATE") = algorithms::kFillerInfostate;
   m.attr("UNDEFINED_LEAF_ID") = kUndefinedLeafId;

   m.def("is_valid_sf_strategy", &IsValidSfStrategy);

   // the suffix is float despite using double, since it python the default floating point type is a
   // double.
   init_pyspiel_treevector_bundle< double >(m, "Float");
   // a generic tree vector bundle holding any type of python object
   init_pyspiel_treevector_bundle< py::object >(m, "");

   py::class_< InfostateTree, std::shared_ptr< InfostateTree > >(m, "InfostateTree")
      .def(
         py::init([](const Game &game, Player acting_player, int max_move_limit) {
            return MakeInfostateTree(game, acting_player, max_move_limit);
         }),
         py::arg("game"),
         py::arg("acting_player"),
         py::arg("max_move_limit") = 1000
      )
      .def(
         py::init([](const std::vector< const State * > &start_states,
                     const std::vector< double > &chance_reach_probs,
                     std::shared_ptr< Observer > infostate_observer,
                     Player acting_player,
                     int max_move_ahead_limit) {
            return MakeInfostateTree(
               start_states,
               chance_reach_probs,
               std::move(infostate_observer),
               acting_player,
               max_move_ahead_limit
            );
         }),
         py::arg("start_states"),
         py::arg("chance_reach_probs"),
         py::arg("infostate_observer"),
         py::arg("acting_player"),
         py::arg("max_move_ahead_limit") = 1000
      )
      .def(
         py::init([](const std::vector< const InfostateNode * > &start_nodes,
                     int max_move_ahead_limit) {
            return MakeInfostateTree(start_nodes, max_move_ahead_limit);
         }),
         py::arg("start_nodes"),
         py::arg("max_move_ahead_limit") = 1000
      )
      .def(
         py::init([](const std::vector< InfostateNode * > &start_nodes, int max_move_ahead_limit) {
            return MakeInfostateTree(start_nodes, max_move_ahead_limit);
         }),
         py::arg("start_nodes"),
         py::arg("max_move_ahead_limit") = 1000
      )
      .def(
         "root",
         [](InfostateTree &tree) {
            return std::unique_ptr< InfostateNode, py::nodelete >{tree.mutable_root()};
         }
      )
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
      .def("all_sequence_ids", &InfostateTree::AllSequenceIds)
      .def("decision_ids_with_parent_seq", &InfostateTree::DecisionIdsWithParentSeq)
      .def("decision_id_for_sequence", &InfostateTree::DecisionIdForSequence)
      .def(
         "decision_for_sequence",
         &InfostateTree::DecisionForSequence,
         py::return_value_policy::reference
      )
      .def("is_leaf_sequence", &InfostateTree::IsLeafSequence)
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
         "all_decision_infostates",
         &InfostateTree::AllDecisionInfostates,
         py::return_value_policy::reference
      )
      .def("all_decision_ids", &InfostateTree::AllDecisionIds)
      .def("decision_id_from_infostate_string", &InfostateTree::DecisionIdFromInfostateString)
      .def("leaf_nodes", &InfostateTree::leaf_nodes, py::return_value_policy::reference)
      .def("leaf_node", &InfostateTree::leaf_node, py::return_value_policy::reference)
      .def("nodes_at_depths", &InfostateTree::nodes_at_depths, py::return_value_policy::reference)
      .def("nodes_at_depth", &InfostateTree::nodes_at_depth, py::return_value_policy::reference)
      .def("best_response", &InfostateTree::BestResponse, py::arg("gradient"))
      .def("best_response_value", &InfostateTree::BestResponseValue, py::arg("gradient"))
      .def("__repr__", [](const InfostateTree &tree) {
         std::ostringstream oss;
         oss << tree;
         return oss.str();
      });
}

}  // namespace open_spiel
