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
#include "pybind11/stl_bind.h"

namespace py = ::pybind11;


namespace open_spiel {

using namespace algorithms;

using infostatenode_holder_ptr = MockUniquePtr< InfostateNode >;
using const_node_uniq_ptr = MockUniquePtr< const InfostateNode >;

void init_pyspiel_infostate_node(::pybind11::module &m)
{
   py::class_< InfostateNode, infostatenode_holder_ptr >(m, "InfostateNode", py::is_final())
      .def("tree", [](const InfostateNode &node) { return node.tree().shared_ptr(); })
      .def(
         "parent", [](const InfostateNode &node) { return infostatenode_holder_ptr{node.parent()}; }
      )
      .def("incoming_index", &InfostateNode::incoming_index)
      .def("type", &InfostateNode::type)
      .def("depth", &InfostateNode::depth)
      .def("is_root_node", &InfostateNode::is_root_node)
      .def("has_infostate_string", &InfostateNode::has_infostate_string)
      .def("infostate_string", &InfostateNode::infostate_string)
      .def("num_children", &InfostateNode::num_children)
      .def("child_iterator", &InfostateNode::child_iterator)
      .def(
         "terminal_history",
         &InfostateNode::TerminalHistory,
         py::return_value_policy::reference_internal
      )
      .def("sequence_id", &InfostateNode::sequence_id)
      .def("start_sequence_id", &InfostateNode::start_sequence_id)
      .def("end_sequence_id", &InfostateNode::end_sequence_id)
      .def("all_sequence_ids", &InfostateNode::AllSequenceIds)
      .def("decision_id", &InfostateNode::decision_id)
      .def(
         "legal_actions", &InfostateNode::legal_actions, py::return_value_policy::reference_internal
      )
      .def("is_leaf_node", &InfostateNode::is_leaf_node)
      .def("terminal_utility", &InfostateNode::terminal_utility)
      .def("terminal_chance_reach_prob", &InfostateNode::terminal_chance_reach_prob)
      .def("corresponding_states_size", &InfostateNode::corresponding_states_size)
      .def(
         "corresponding_states",
         &InfostateNode::corresponding_states,
         py::return_value_policy::reference_internal
      )
      .def(
         "corresponding_chance_reach_probs",
         &InfostateNode::corresponding_chance_reach_probs,
         py::return_value_policy::reference_internal
      )
      .def(
         "child_at",
         [](const InfostateNode &node, int index) {
            return infostatenode_holder_ptr{node.child_at(index)};
         },
         py::arg("index")
      )
      .def("make_certificate", &InfostateNode::MakeCertificate)
      .def(
         "__copy__",
         [](const InfostateNode &node) {
            throw ForbiddenException(
               "InfostateNode cannot be copied, because its "
               "lifetime is managed by the owning "
               "InfostateTree. Store a variable naming the "
               "associated tree to ensure the node's "
               "lifetime."
            );
         }
      )
      .def(
         "__deepcopy__",
         [](const InfostateNode &node) {
            throw ForbiddenException(
               "InfostateNode cannot be copied, because its "
               "lifetime is managed by the owning "
               "InfostateTree. Store a variable naming the "
               "associated tree to ensure the node's "
               "lifetime."
            );
         }
      )
      .def("address_str", [](const InfostateNode &node) {
         std::stringstream ss;
         ss << &node;
         return ss.str();
      });

   py::enum_< InfostateNodeType >(m, "InfostateNodeType")
      .value("decision", InfostateNodeType::kDecisionInfostateNode)
      .value("observation", InfostateNodeType::kObservationInfostateNode)
      .value("terminal", InfostateNodeType::kTerminalInfostateNode)
      .export_values();
}

struct ToUniquePtrFunctor {
   auto operator()(InfostateNode *ptr) const noexcept { return infostatenode_holder_ptr{ptr}; }
};

template <
   typename ContainerOut,
   typename TransformFunctor = ToUniquePtrFunctor,
   template < class... > class Container = std::vector,
   typename Value = InfostateNode *,
   typename... RestTs >
auto to_unique_ptr_vec(
   const Container< Value, RestTs... > &node_container,
   TransformFunctor transformer
)
{
   ContainerOut internal_vec{};
   internal_vec.reserve(node_container.size());
   std::transform(
      node_container.begin(),
      node_container.end(),
      std::back_insert_iterator(internal_vec),
      transformer
   );
   return internal_vec;
}

void init_pyspiel_infostate_tree(::pybind11::module &m)
{
   // Infostate-Tree nodes and NodeType enum
   init_pyspiel_infostate_node(m);
   // suffix is float despite using double, since python's floating point type
   // is double precision.
   init_pyspiel_treevector_bundle< double >(m, "Float");
   // a generic tree vector bundle holding any type of python object
   init_pyspiel_treevector_bundle< py::object >(m, "");
   // bind a range for every possible id type
   init_pyspiel_range< SequenceId >(m, "SequenceIdRange");
   init_pyspiel_range< DecisionId >(m, "DecisionIdRange");
   init_pyspiel_range< LeafId >(m, "LeafIdRange");

   init_pyspiel_node_id< SequenceId >(m, "SequenceId");
   init_pyspiel_node_id< DecisionId >(m, "DecisionId");
   init_pyspiel_node_id< LeafId >(m, "LeafId");

   m.attr("UNDEFINED_DECISION_ID") = ::pybind11::cast(kUndefinedDecisionId);
   m.attr("UNDEFINED_LEAF_ID") = ::pybind11::cast(kUndefinedLeafId);
   m.attr("UNDEFINED_SEQUENCE_ID") = ::pybind11::cast(kUndefinedSequenceId);
   m.attr("DUMMY_ROOT_NODE_INFOSTATE") = ::pybind11::cast(algorithms::kDummyRootNodeInfostate);
   m.attr("FILLER_INFOSTATE") = ::pybind11::cast(algorithms::kFillerInfostate);

   m.def("is_valid_sf_strategy", &IsValidSfStrategy);

   py::bind_vector< std::vector< std::vector< infostatenode_holder_ptr > > >(
      m, "InfostateNodeVector2D"
   );

   py::class_< InfostateTree, std::shared_ptr< InfostateTree > >(m, "InfostateTree", py::is_final())
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
         py::arg("max_move_limit") = 1000
      )
      .def(
         py::init([](const std::vector< const InfostateNode * > &start_nodes,
                     int max_move_ahead_limit) {
            return MakeInfostateTree(start_nodes, max_move_ahead_limit);
         }),
         py::arg("start_nodes"),
         py::arg("max_move_limit") = 1000
      )
      .def(
         "root", [](InfostateTree &tree) { return infostatenode_holder_ptr{tree.mutable_root()}; }
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
         [](InfostateTree &tree, const SequenceId &id) {
            return infostatenode_holder_ptr{tree.observation_infostate(id)};
         }
      )
      .def(
         "observation_infostate_view",
         [](const InfostateTree &tree, const SequenceId &id) {
            return std::unique_ptr< const InfostateNode, py::nodelete >{
               tree.observation_infostate(id)};
         }
      )
      .def("all_sequence_ids", &InfostateTree::AllSequenceIds)
      .def("decision_ids_with_parent_seq", &InfostateTree::DecisionIdsWithParentSeq)
      .def("decision_id_for_sequence", &InfostateTree::DecisionIdForSequence)
      .def(
         "decision_for_sequence",
         [](InfostateTree &tree, const SequenceId &id) {
            auto node_opt = tree.DecisionForSequence(id);
            if(not node_opt.has_value()) {
               return absl::optional< infostatenode_holder_ptr >{};
            } else {
               return absl::optional< infostatenode_holder_ptr >{
                  infostatenode_holder_ptr{*node_opt}};
            }
         }
      )
      .def("is_leaf_sequence", &InfostateTree::IsLeafSequence)
      .def(
         "decision_infostate",
         [](const InfostateTree &tree, const DecisionId &id) {
            return const_node_uniq_ptr{tree.decision_infostate(id)};
         }
      )
      .def(
         "all_decision_infostates",
         [](const InfostateTree &tree) {
            return to_unique_ptr_vec< std::vector< infostatenode_holder_ptr > >(
               tree.AllDecisionInfostates(), ToUniquePtrFunctor{}
            );
         }
      )
      .def("all_decision_ids", &InfostateTree::AllDecisionIds)
      .def("decision_id_from_infostate_string", &InfostateTree::DecisionIdFromInfostateString)
      .def(
         "leaf_nodes",
         [](const InfostateTree &tree) {
            return to_unique_ptr_vec< std::vector< infostatenode_holder_ptr > >(
               tree.leaf_nodes(), ToUniquePtrFunctor{}
            );
         }
      )
      .def(
         "leaf_node",
         [](const InfostateTree &tree, const LeafId &id) {
            return infostatenode_holder_ptr{tree.leaf_node(id)};
         }
      )
      .def(
         "nodes_at_depths",
         [](const InfostateTree &tree) {
            return to_unique_ptr_vec< std::vector< std::vector< infostatenode_holder_ptr > > >(
               tree.nodes_at_depths(),
               [](const auto &internal_vec) {
                  return to_unique_ptr_vec< std::vector< infostatenode_holder_ptr > >(
                     internal_vec, ToUniquePtrFunctor{}
                  );
               }
            );
         }
      )
      .def(
         "nodes_at_depth",
         [](const InfostateTree &tree, const py::int_ &depth) {
            // we accept a py::int_ here instead of directly asking for a
            // size_t, since whatever pybind11 would cast to size_t in order to
            // fulifll the type requirement would simply be byte-cast into
            // size_t. This would turn negative values into high integers,
            // instead of throwing an error.
            if(depth < py::int_(0)) {
               throw std::invalid_argument("'depth' must be non-negative.");
            }
            // convert the raw node vector again into a vector of non-deleting
            // node unique pointer.
            return to_unique_ptr_vec< std::vector< infostatenode_holder_ptr > >(
               tree.nodes_at_depth(py::cast< size_t >(depth)), ToUniquePtrFunctor{}
            );
         },
         py::arg("depth")
      )
      .def("best_response", &InfostateTree::BestResponse, py::arg("gradient"))
      .def("best_response_value", &InfostateTree::BestResponseValue, py::arg("gradient"))
      .def(
         "__repr__",
         [](const InfostateTree &tree) {
            std::ostringstream oss;
            oss << tree;
            return oss.str();
         }
      )
      .def(
         "__copy__",
         [](const InfostateTree &) {
            throw ForbiddenException(
               "InfostateTree cannot be copied, because its "
               "internal structure is entangled during construction. "
               "Create a new tree instead."
            );
         }
      )
      .def("__deepcopy__", [](const InfostateTree &) {
         throw ForbiddenException(
            "InfostateTree cannot be copied, because its "
            "internal structure is entangled during construction. "
            "Create a new tree instead."
         );
      });
}

}  // namespace open_spiel
