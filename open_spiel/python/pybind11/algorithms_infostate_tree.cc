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

#include <algorithm>
#include <iterator>
#include <memory>
#include <sstream>
#include <utility>

#include "open_spiel/algorithms/infostate_tree.h"
#include "pybind11/stl_bind.h"

namespace py = ::pybind11;

namespace open_spiel {
using namespace algorithms;

// Hold nodes with an *aliasing shared_ptr* that shares ownership with the tree.
// In short, the member (a node) shares the ownership of its parent object
// (the Tree), guaranteeing that any Python-held node keeps the owning tree (and
// therefore the node storage) alive.
std::shared_ptr<InfostateNode> make_aliasing_node_ptr(InfostateNode *node) {
  return {/*owner=*/node->tree().shared_ptr(), /*pointed_to=*/node};
}

std::shared_ptr<const InfostateNode>
make_aliasing_node_ptr(const InfostateNode *node) {
  return {/*owner=*/node->tree().shared_ptr(), /*pointed_to=*/node};
}

class InfostateNodeChildIterator {
  using iter_type = VecWithUniquePtrsIterator<InfostateNode>;

  iter_type iter_;

public:
  explicit InfostateNodeChildIterator(iter_type it) : iter_(it) {}

  InfostateNodeChildIterator &operator++() {
    ++iter_;
    return *this;
  }

  bool operator==(const InfostateNodeChildIterator &other) const {
    return iter_ == other.iter_;
  }
  bool operator!=(const InfostateNodeChildIterator &other) const {
    return !(*this == other);
  }

  // each yielded element must keep the tree alive.
  auto operator*() { return make_aliasing_node_ptr(*iter_); }

  auto begin() const { return InfostateNodeChildIterator{iter_.begin()}; }
  auto end() const { return InfostateNodeChildIterator{iter_.end()}; }
};

void init_pyspiel_infostate_node(::pybind11::module &m) {
  py::class_<InfostateNode, std::shared_ptr<InfostateNode>>(m, "InfostateNode",
                                                            py::is_final())
      .def("tree",
           [](const InfostateNode &node) { return node.tree().shared_ptr(); })
      .def("parent",
           [](const InfostateNode &node) {
             return make_aliasing_node_ptr(node.parent());
           })
      .def("incoming_index", &InfostateNode::incoming_index)
      .def("type", &InfostateNode::type)
      .def("depth", &InfostateNode::depth)
      .def("is_root_node", &InfostateNode::is_root_node)
      .def("is_filler_node", &InfostateNode::is_filler_node)
      .def("has_infostate_string", &InfostateNode::has_infostate_string)
      .def("infostate_string", &InfostateNode::infostate_string)
      .def("num_children", &InfostateNode::num_children)
      .def("terminal_history", &InfostateNode::TerminalHistory,
           py::return_value_policy::reference_internal)
      .def("sequence_id", &InfostateNode::sequence_id)
      .def("start_sequence_id", &InfostateNode::start_sequence_id)
      .def("end_sequence_id", &InfostateNode::end_sequence_id)
      .def("all_sequence_ids", &InfostateNode::AllSequenceIds)
      .def("decision_id", &InfostateNode::decision_id)
      .def("legal_actions", &InfostateNode::legal_actions,
           py::return_value_policy::reference_internal)
      .def("is_leaf_node", &InfostateNode::is_leaf_node)
      .def("terminal_utility", &InfostateNode::terminal_utility)
      .def("terminal_chance_reach_prob",
           &InfostateNode::terminal_chance_reach_prob)
      .def("corresponding_states_size",
           &InfostateNode::corresponding_states_size)
      .def("corresponding_states", &InfostateNode::corresponding_states,
           py::return_value_policy::reference_internal)
      .def("corresponding_chance_reach_probs",
           &InfostateNode::corresponding_chance_reach_probs,
           py::return_value_policy::reference_internal)
      .def(
          "child_at",
          [](const InfostateNode &node, int index) {
            return make_aliasing_node_ptr(node.child_at(index));
          },
          py::arg("index"))
      .def("make_certificate", &InfostateNode::MakeCertificate)
      .def("address_str",
           [](const InfostateNode &node) {
             std::stringstream ss;
             ss << &node;
             return ss.str();
           })
      .def("__iter__",
           [](const InfostateNode &node) {
             return py::make_iterator(
                 InfostateNodeChildIterator{node.child_iterator().begin()},
                 InfostateNodeChildIterator{node.child_iterator().end()});
           })
      .def("__copy__",
           [](const InfostateNode &) {
             throw ForbiddenException(
                 "InfostateNode cannot be copied or deep-copied. It is an "
                 "immutable view onto node storage owned by an InfostateTree.");
           })
      .def("__deepcopy__", [](const InfostateNode &) {
        throw ForbiddenException(
            "InfostateNode cannot be copied or deep-copied. It is an immutable "
            "view onto node storage owned by an InfostateTree.");
      });

  py::enum_<InfostateNodeType>(m, "InfostateNodeType")
      .value("decision", InfostateNodeType::kDecisionInfostateNode)
      .value("observation", InfostateNodeType::kObservationInfostateNode)
      .value("terminal", InfostateNodeType::kTerminalInfostateNode)
      .export_values();
}

struct ToHolderPtrFunctor {
  auto operator()(InfostateNode *ptr) const noexcept {
    return make_aliasing_node_ptr(ptr);
  }
  auto operator()(const InfostateNode *ptr) const noexcept {
    return make_aliasing_node_ptr(ptr);
  }
};

template <typename ContainerOut, typename TransformFunctor = ToHolderPtrFunctor,
          template <class...> class Container = std::vector, typename... RestTs>
auto to(const Container<RestTs...> &node_container,
        TransformFunctor transformer) {
  ContainerOut internal_vec{};
  internal_vec.reserve(node_container.size());
  std::transform(node_container.begin(), node_container.end(),
                 std::back_inserter(internal_vec), transformer);
  return internal_vec;
}

void init_pyspiel_infostate_tree(::pybind11::module &m) {
  // Infostate-Tree nodes and NodeType enum
  init_pyspiel_infostate_node(m);
  // suffix is float despite using double, since python's floating point type
  // is double precision.
  init_pyspiel_treevector_bundle<double>(m, "Float");
  // a generic tree vector bundle holding any type of python object
  init_pyspiel_treevector_bundle<py::object>(m, "");
  // bind a range for every possible id type
  init_pyspiel_range<SequenceId>(m, "SequenceIdRange");
  init_pyspiel_range<DecisionId>(m, "DecisionIdRange");
  init_pyspiel_range<LeafId>(m, "LeafIdRange");

  init_pyspiel_node_id<SequenceId>(m, "SequenceId");
  init_pyspiel_node_id<DecisionId>(m, "DecisionId");
  init_pyspiel_node_id<LeafId>(m, "LeafId");

  m.attr("UNDEFINED_DECISION_ID") = ::pybind11::cast(kUndefinedDecisionId);
  m.attr("UNDEFINED_LEAF_ID") = ::pybind11::cast(kUndefinedLeafId);
  m.attr("UNDEFINED_SEQUENCE_ID") = ::pybind11::cast(kUndefinedSequenceId);
  m.attr("DUMMY_ROOT_NODE_INFOSTATE") =
      ::pybind11::cast(algorithms::kDummyRootNodeInfostate);
  m.attr("FILLER_INFOSTATE") = ::pybind11::cast(algorithms::kFillerInfostate);

  m.def("is_valid_sf_strategy", &IsValidSfStrategy);

  py::bind_vector<std::vector<std::vector<std::shared_ptr<InfostateNode>>>>(
      m, "InfostateNodeVector2D");

  py::class_<InfostateTree, std::shared_ptr<InfostateTree>>(m, "InfostateTree",
                                                            py::is_final())
      .def(py::init(
               [](const Game &game, Player acting_player, int max_move_limit) {
                 return MakeInfostateTree(game, acting_player, max_move_limit);
               }),
           py::arg("game"), py::arg("acting_player"),
           py::arg("max_move_limit") = 1000)
      .def(py::init([](const std::vector<const State *> &start_states,
                       const std::vector<double> &chance_reach_probs,
                       std::shared_ptr<Observer> infostate_observer,
                       Player acting_player, int max_move_ahead_limit) {
             return MakeInfostateTree(start_states, chance_reach_probs,
                                      std::move(infostate_observer),
                                      acting_player, max_move_ahead_limit);
           }),
           py::arg("start_states"), py::arg("chance_reach_probs"),
           py::arg("infostate_observer"), py::arg("acting_player"),
           py::arg("max_move_limit") = 1000)
      .def(py::init([](const std::vector<const InfostateNode *> &start_nodes,
                       int max_move_ahead_limit) {
             return MakeInfostateTree(start_nodes, max_move_ahead_limit);
           }),
           py::arg("start_nodes"), py::arg("max_move_limit") = 1000)
      .def("root",
           [](InfostateTree &tree) {
             return make_aliasing_node_ptr(tree.mutable_root());
           })
      .def("root_branching_factor", &InfostateTree::root_branching_factor)
      .def("acting_player", &InfostateTree::acting_player)
      .def("tree_height", &InfostateTree::tree_height)
      .def("num_decisions", &InfostateTree::num_decisions)
      .def("num_sequences", &InfostateTree::num_sequences)
      .def("num_leaves", &InfostateTree::num_leaves)
      .def("empty_sequence", &InfostateTree::empty_sequence)
      .def(
          "observation_infostate",
          [](const InfostateTree &tree, const SequenceId &id) {
            return make_aliasing_node_ptr(tree.observation_infostate(id));
          },
          py::arg("sequence_id"))
      .def("all_sequence_ids", &InfostateTree::AllSequenceIds)
      .def("decision_ids_with_parent_seq",
           &InfostateTree::DecisionIdsWithParentSeq, py::arg("sequence_id"))
      .def("decision_id_for_sequence", &InfostateTree::DecisionIdForSequence,
           py::arg("sequence_id"))
      .def(
          "decision_for_sequence",
          [](InfostateTree &tree,
             const SequenceId &id) -> std::shared_ptr<const InfostateNode> {
            auto node_opt = tree.DecisionForSequence(id);
            if (!node_opt.has_value()) {
              return {};
            }
            return {make_aliasing_node_ptr(*node_opt)};
          },
          py::arg("sequence_id"))
      .def("is_leaf_sequence", &InfostateTree::IsLeafSequence)
      .def(
          "decision_infostate",
          [](const InfostateTree &tree, const DecisionId &id) {
            return make_aliasing_node_ptr(tree.decision_infostate(id));
          },
          py::arg("decision_id"))
      .def("all_decision_infostates",
           [](const InfostateTree &tree) {
             return to<std::vector<std::shared_ptr<const InfostateNode>>>(
                 tree.AllDecisionInfostates(), ToHolderPtrFunctor{});
           })
      .def("all_decision_ids", &InfostateTree::AllDecisionIds)
      .def("decision_id_from_infostate_string",
           &InfostateTree::DecisionIdFromInfostateString,
           py::arg("infostate_string"))
      .def("leaf_nodes",
           [](const InfostateTree &tree) {
             return to<std::vector<std::shared_ptr<const InfostateNode>>>(
                 tree.leaf_nodes(), ToHolderPtrFunctor{});
           })
      .def(
          "leaf_node",
          [](const InfostateTree &tree, const LeafId &id) {
            return make_aliasing_node_ptr(tree.leaf_node(id));
          },
          py::arg("leaf_id"))
      .def(
          "nodes_at_depths",
          [](const InfostateTree &tree) {
            return to<
                std::vector<std::vector<std::shared_ptr<const InfostateNode>>>>(
                tree.nodes_at_depths(), [](const auto &internal_vec) {
                  return to<std::vector<std::shared_ptr<const InfostateNode>>>(
                      internal_vec, ToHolderPtrFunctor{});
                });
          })
      .def(
          "nodes_at_depth",
          [](const InfostateTree &tree, long depth) {
            if (depth < 0) {
              throw std::invalid_argument("'depth' must be non-negative.");
            }
            return to<std::vector<std::shared_ptr<const InfostateNode>>>(
                tree.nodes_at_depth(static_cast<size_t>(depth)),
                ToHolderPtrFunctor{});
          },
          py::arg("depth"))
      .def("best_response", &InfostateTree::BestResponse, py::arg("gradient"))
      .def("best_response_value", &InfostateTree::BestResponseValue,
           py::arg("gradient"))
      .def("__repr__",
           [](const InfostateTree &tree) {
             std::ostringstream oss;
             oss << tree;
             return oss.str();
           })
      .def("__copy__",
           [](const InfostateTree &) {
             throw ForbiddenException(
                 "InfostateTree cannot be copied or deep-copied. A correct "
                 "copy would require rebuilding all internal node storage and "
                 "index structures. Construct a new tree instead.");
           })
      .def("__deepcopy__", [](const InfostateTree &) {
        throw ForbiddenException(
            "InfostateTree cannot be copied or deep-copied. A correct "
            "copy would require rebuilding all internal node storage and "
            "index structures. Construct a new tree instead.");
      });
}
} // namespace open_spiel
