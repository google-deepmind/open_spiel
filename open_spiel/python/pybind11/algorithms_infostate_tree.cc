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

#include <cstddef>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "open_spiel/algorithms/infostate_tree.h"
#include "open_spiel/observer.h"
#include "open_spiel/python/pybind11/algorithms_infostate_tree_templates.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "pybind11/include/pybind11/stl_bind.h"

namespace py = ::pybind11;

namespace open_spiel {
using algorithms::InfostateNode;
using algorithms::InfostateNodeType;
using algorithms::InfostateTree;
using algorithms::IsValidSfStrategy;
using algorithms::kUndefinedDecisionId;
using algorithms::kUndefinedLeafId;
using algorithms::kUndefinedSequenceId;
using algorithms::MakeInfostateTree;
using algorithms::SequenceId;
using algorithms::VecWithUniquePtrsIterator;

// Python-facing view type for InfostateNode, keeping tree owner alive.
struct InfostateNodeView {
  std::shared_ptr<InfostateTree> owner;
  const InfostateNode* node;
};

InfostateNodeView make_node_view(std::shared_ptr<InfostateTree> owner,
                                 const InfostateNode* node) {
  SPIEL_CHECK_TRUE(owner != nullptr);
  SPIEL_CHECK_TRUE(node != nullptr);
  return {std::move(owner), node};
}

InfostateNodeView make_node_view(const InfostateNode* node) {
  SPIEL_CHECK_TRUE(node != nullptr);
  return {std::const_pointer_cast<InfostateTree>(node->tree().shared_ptr()),
          node};
}

py::object maybe_node_view(std::shared_ptr<InfostateTree> owner,
                           const InfostateNode* node) {
  if (node == nullptr) {
    return py::none();
  }
  return py::cast(make_node_view(std::move(owner), node));
}

py::object maybe_node_view(const InfostateNode* node) {
  if (node == nullptr) {
    return py::none();
  }
  return py::cast(make_node_view(node));
}

template <typename NodePtr>
py::list node_view_list(std::shared_ptr<InfostateTree> owner,
                        const std::vector<NodePtr>& nodes) {
  py::list out;
  for (NodePtr node : nodes) {
    out.append(py::cast(make_node_view(owner, node)));
  }
  return out;
}

template <typename NodePtr>
py::list node_view_list_2d(
    std::shared_ptr<InfostateTree> owner,
    const std::vector<std::vector<NodePtr>>& nested_nodes) {
  py::list out;
  for (const auto& nodes : nested_nodes) {
    out.append(node_view_list(owner, nodes));
  }
  return out;
}

std::ostream& operator<<(std::ostream& os, const InfostateNodeView& view) {
  os << view.node;
  return os;
}

class InfostateNodeChildIterator {
  using iter_type = VecWithUniquePtrsIterator<InfostateNode>;

  std::shared_ptr<InfostateTree> owner_;
  iter_type iter_;

 public:
  InfostateNodeChildIterator(std::shared_ptr<InfostateTree> owner, iter_type it)
      : owner_(std::move(owner)), iter_(it) {}

  InfostateNodeChildIterator& operator++() {
    ++iter_;
    return *this;
  }

  bool operator==(const InfostateNodeChildIterator& other) const {
    return iter_ == other.iter_;
  }

  bool operator!=(const InfostateNodeChildIterator& other) const {
    return !(*this == other);
  }

  auto operator*() { return make_node_view(owner_, *iter_); }

  auto begin() const {
    return InfostateNodeChildIterator{owner_, iter_.begin()};
  }
  auto end() const { return InfostateNodeChildIterator{owner_, iter_.end()}; }
};

void init_pyspiel_infostate_node(::pybind11::module& m) {
  py::class_<InfostateNodeView>(m, "InfostateNode", py::is_final())
      .def("tree", [](const InfostateNodeView& view) { return view.owner; })
      .def(
          "__eq__",
          [](const InfostateNodeView& lhs, const InfostateNodeView& rhs) {
            return lhs.node == rhs.node && lhs.owner.get() == rhs.owner.get();
          },
          py::is_operator())
      .def(
          "__ne__",
          [](const InfostateNodeView& lhs, const InfostateNodeView& rhs) {
            return !(lhs.node == rhs.node &&
                     lhs.owner.get() == rhs.owner.get());
          },
          py::is_operator())
      .def("__hash__",
           [](const InfostateNodeView& view) {
             return std::hash<const InfostateNode*>{}(view.node);
           })
      .def("parent",
           [](const InfostateNodeView& view) {
             return maybe_node_view(view.owner, view.node->parent());
           })
      .def("incoming_index",
           [](const InfostateNodeView& view) {
             return view.node->incoming_index();
           })
      .def("type",
           [](const InfostateNodeView& view) { return view.node->type(); })
      .def("depth",
           [](const InfostateNodeView& view) { return view.node->depth(); })
      .def("is_root_node",
           [](const InfostateNodeView& view) {
             return view.node->is_root_node();
           })
      .def("is_filler_node",
           [](const InfostateNodeView& view) {
             return view.node->is_filler_node();
           })
      .def("has_infostate_string",
           [](const InfostateNodeView& view) {
             return view.node->has_infostate_string();
           })
      .def("infostate_string",
           [](const InfostateNodeView& view) {
             return view.node->infostate_string();
           })
      .def("num_children",
           [](const InfostateNodeView& view) {
             return view.node->num_children();
           })
      .def(
          "terminal_history",
          [](const InfostateNodeView& view) -> const std::vector<Action>& {
            return view.node->TerminalHistory();
          },
          py::return_value_policy::reference_internal)
      .def("sequence_id",
           [](const InfostateNodeView& view) {
             return view.node->sequence_id();
           })
      .def("start_sequence_id",
           [](const InfostateNodeView& view) {
             return view.node->start_sequence_id();
           })
      .def("end_sequence_id",
           [](const InfostateNodeView& view) {
             return view.node->end_sequence_id();
           })
      .def("all_sequence_ids",
           [](const InfostateNodeView& view) {
             return view.node->AllSequenceIds();
           })
      .def("decision_id",
           [](const InfostateNodeView& view) {
             return view.node->decision_id();
           })
      .def(
          "legal_actions",
          [](const InfostateNodeView& view) -> const std::vector<Action>& {
            return view.node->legal_actions();
          },
          py::return_value_policy::reference_internal)
      .def("is_leaf_node",
           [](const InfostateNodeView& view) {
             return view.node->is_leaf_node();
           })
      .def("terminal_utility",
           [](const InfostateNodeView& view) {
             return view.node->terminal_utility();
           })
      .def("terminal_chance_reach_prob",
           [](const InfostateNodeView& view) {
             return view.node->terminal_chance_reach_prob();
           })
      .def("corresponding_states_size",
           [](const InfostateNodeView& view) {
             return view.node->corresponding_states_size();
           })
      .def(
          "corresponding_states",
          [](const InfostateNodeView& view)
              -> const std::vector<std::shared_ptr<const State>>& {
            return view.node->corresponding_states();
          },
          py::return_value_policy::reference_internal)
      .def(
          "corresponding_chance_reach_probs",
          [](const InfostateNodeView& view) -> const std::vector<double>& {
            return view.node->corresponding_chance_reach_probs();
          },
          py::return_value_policy::reference_internal)
      .def(
          "child_at",
          [](const InfostateNodeView& view, int index) {
            return make_node_view(view.owner, view.node->child_at(index));
          },
          py::arg("index"))
      .def("make_certificate",
           [](const InfostateNodeView& view) {
             return view.node->MakeCertificate();
           })
      .def("address_str",
           [](const InfostateNodeView& view) {
             std::stringstream ss;
             ss << view.node;
             return ss.str();
           })
      .def(
          "__iter__",
          [](const InfostateNodeView& view) {
            return py::make_iterator(
                InfostateNodeChildIterator{view.owner,
                                           view.node->child_iterator().begin()},
                InfostateNodeChildIterator{view.owner,
                                           view.node->child_iterator().end()});
          },
          py::keep_alive<0, 1>())
      .def("__copy__",
           [](const InfostateNodeView&) {
             throw ForbiddenException(
                 "InfostateNode cannot be copied or deep-copied. It is an "
                 "immutable view onto node storage owned by an InfostateTree.");
           })
      .def("__deepcopy__", [](const InfostateNodeView&, py::dict) {
        throw ForbiddenException(
            "InfostateNode cannot be copied or deep-copied. It is an "
            "immutable view onto node storage owned by an InfostateTree.");
      });

  py::enum_<InfostateNodeType>(m, "InfostateNodeType")
      .value("decision", InfostateNodeType::kDecisionInfostateNode)
      .value("observation", InfostateNodeType::kObservationInfostateNode)
      .value("terminal", InfostateNodeType::kTerminalInfostateNode)
      .export_values();
}

void init_pyspiel_infostate_tree(::pybind11::module& m) {
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

  py::bind_vector<std::vector<InfostateNodeView>>(m, "InfostateNodeVector");
  py::bind_vector<std::vector<std::vector<InfostateNodeView>>>(
      m, "InfostateNodeVector2D");

  py::class_<InfostateTree, std::shared_ptr<InfostateTree>>(m, "InfostateTree",
                                                            py::is_final())
      .def(py::init([](const Game& game, Player acting_player,
                       bool store_world_states, int max_move_limit) {
             return MakeInfostateTree(game, acting_player, store_world_states,
                                      max_move_limit);
           }),
           py::arg("game"), py::arg("acting_player"), py::kw_only(),
           py::arg("store_world_states") = false,
           py::arg("max_move_limit") = 1000)
      .def(py::init([](const std::vector<const State*>& start_states,
                       const std::vector<double>& chance_reach_probs,
                       std::shared_ptr<Observer> infostate_observer,
                       Player acting_player, bool store_world_states,
                       int max_move_ahead_limit) {
             return MakeInfostateTree(start_states, chance_reach_probs,
                                      std::move(infostate_observer),
                                      acting_player, store_world_states,
                                      max_move_ahead_limit);
           }),
           py::arg("start_states"), py::arg("chance_reach_probs"),
           py::arg("infostate_observer"), py::arg("acting_player"),
           py::kw_only(), py::arg("store_world_states") = false,
           py::arg("max_move_limit") = 1000)
      .def(py::init([](const std::vector<InfostateNodeView>& start_nodes,
                       bool store_world_states, int max_move_ahead_limit) {
             std::vector<const InfostateNode*> raw_start_nodes;
             raw_start_nodes.reserve(start_nodes.size());
             for (const auto& start_node : start_nodes) {
               raw_start_nodes.push_back(start_node.node);
             }
             return MakeInfostateTree(raw_start_nodes, store_world_states,
                                      max_move_ahead_limit);
           }),
           py::arg("start_nodes"), py::kw_only(),
           py::arg("store_world_states") = false,
           py::arg("max_move_limit") = 1000)
      .def("root",
           [](const std::shared_ptr<InfostateTree>& tree) {
             return make_node_view(tree, tree->mutable_root());
           })
      .def("root_branching_factor", &InfostateTree::root_branching_factor)
      .def("acting_player", &InfostateTree::acting_player)
      .def("tree_height", &InfostateTree::tree_height)
      .def("num_decisions", &InfostateTree::num_decisions)
      .def("num_sequences", &InfostateTree::num_sequences)
      .def("num_leaves", &InfostateTree::num_leaves)
      .def("empty_sequence", &InfostateTree::empty_sequence)
      .def("stores_all_world_states", &InfostateTree::stores_all_world_states)
      .def(
          "observation_infostate",
          [](const std::shared_ptr<InfostateTree>& tree, const SequenceId& id) {
            return make_node_view(tree, tree->observation_infostate(id));
          },
          py::arg("sequence_id"))
      .def("all_sequence_ids", &InfostateTree::AllSequenceIds)
      .def("decision_ids_with_parent_seq",
           &InfostateTree::DecisionIdsWithParentSeq, py::arg("sequence_id"))
      .def("decision_id_for_sequence", &InfostateTree::DecisionIdForSequence,
           py::arg("sequence_id"))
      .def(
          "decision_for_sequence",
          [](const std::shared_ptr<InfostateTree>& tree,
             const SequenceId& id) -> py::object {
            auto node_opt = tree->DecisionForSequence(id);
            if (!node_opt.has_value()) {
              return py::none();
            }
            return py::cast(make_node_view(tree, *node_opt));
          },
          py::arg("sequence_id"))
      .def("is_leaf_sequence", &InfostateTree::IsLeafSequence)
      .def(
          "decision_infostate",
          [](const std::shared_ptr<InfostateTree>& tree, const DecisionId& id) {
            return make_node_view(tree, tree->decision_infostate(id));
          },
          py::arg("decision_id"))
      .def("all_decision_infostates",
           [](const std::shared_ptr<InfostateTree>& tree) {
             return node_view_list(tree, tree->AllDecisionInfostates());
           })
      .def("all_decision_ids", &InfostateTree::AllDecisionIds)
      .def("decision_id_from_infostate_string",
           &InfostateTree::DecisionIdFromInfostateString,
           py::arg("infostate_string"))
      .def("leaf_nodes",
           [](const std::shared_ptr<InfostateTree>& tree) {
             return node_view_list(tree, tree->leaf_nodes());
           })
      .def(
          "leaf_node",
          [](const std::shared_ptr<InfostateTree>& tree, const LeafId& id) {
            return make_node_view(tree, tree->leaf_node(id));
          },
          py::arg("leaf_id"))
      .def("nodes_at_depths",
           [](const std::shared_ptr<InfostateTree>& tree) {
             return node_view_list_2d(tree, tree->nodes_at_depths());
           })
      .def(
          "nodes_at_depth",
          [](const std::shared_ptr<InfostateTree>& tree, int depth) {
            if (depth < 0) {
              throw std::invalid_argument("'depth' must be non-negative.");
            }
            return node_view_list(
                tree, tree->nodes_at_depth(static_cast<size_t>(depth)));
          },
          py::arg("depth"))
      .def("best_response", &InfostateTree::BestResponse, py::arg("gradient"))
      .def("best_response_value", &InfostateTree::BestResponseValue,
           py::arg("gradient"))
      .def("__repr__",
           [](const InfostateTree& tree) {
             std::ostringstream oss;
             oss << tree;
             return oss.str();
           })
      .def("__copy__",
           [](const InfostateTree&) {
             throw ForbiddenException(
                 "InfostateTree cannot be copied or deep-copied. A correct "
                 "copy would require rebuilding all internal node storage and "
                 "index structures. Construct a new tree instead.");
           })
      .def("__deepcopy__", [](const InfostateTree&) {
        throw ForbiddenException(
            "InfostateTree cannot be copied or deep-copied. A correct "
            "copy would require rebuilding all internal node storage and "
            "index structures. Construct a new tree instead.");
      });
}
}  // namespace open_spiel
