# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test Python bindings for infostate tree and related classes."""

import copy
import gc
from typing import Iterable
from typing import Iterator

from absl.testing import absltest
from absl.testing import parameterized

import pyspiel


class InfostateTreeTest(parameterized.TestCase):

  def test_tree_binding(self):
    game = pyspiel.load_game("kuhn_poker")
    tree = pyspiel.InfostateTree(game, 0)
    self.assertEqual(tree.num_sequences(), 13)

    # disallowing copying is enforced
    with self.assertRaises(pyspiel.ForbiddenError) as _:
      copy.copy(tree)
      copy.deepcopy(tree)

  def test_node_keeps_tree_alive(self):
    game = pyspiel.load_game("kuhn_poker")
    tree = pyspiel.InfostateTree(game, 0)
    height = tree.tree_height()
    num_sequences = tree.num_sequences()
    node = tree.root()
    del tree
    gc.collect()  # force garbage collection
    # the Python wrapper referenced by `tree` may be gone, but the underlying
    # c++ tree must still be alive if nodes alias-own it.
    tree_from_node = node.tree()
    self.assertIsNotNone(tree_from_node)
    # assert its attributes are identical
    # (unnecessary, but shows it is the same tree)
    self.assertEqual(tree_from_node.num_sequences(), num_sequences)
    self.assertEqual(tree_from_node.tree_height(), height)

  @parameterized.parameters([
      # test for matrix mp
      dict(
          game=pyspiel.load_game("matrix_mp"),
          players=[0, 1],
          expected_certificate="([({}{})({}{})])",
      ),
      # test for imperfect info goofspiel
      dict(
          game=pyspiel.load_game(
              "goofspiel",
              {"num_cards": 2, "imp_info": True, "points_order": "ascending"},
          ),
          players=[0, 1],
          expected_certificate="([({}{})({}{})])",
      ),
      # test for kuhn poker (0 player only)
      dict(
          game=pyspiel.load_game("kuhn_poker"),
          players=[0],
          expected_certificate=(
              "(("  # Root node, p1 is getting a card
              "("  # p2 is getting card
              "["  # p1 acts
              "(("  # p1 bet, and p2 acts
              "(({}))"
              "(({}))"
              "(({}))"
              "(({}))"
              "))"
              "(("  # p1 checks, and p2 acts
              # p2 checked
              "(({}))"
              "(({}))"
              # p2 bet
              "[({}"
              "{})"
              "({}"
              "{})]"
              "))"
              "]"
              ")"
              # Just 2 more copies.
              "([(((({}))(({}))(({}))(({}))))(((({}))(({}))[({}{})({}{})]))])"
              "([(((({}))(({}))(({}))(({}))))(((({}))(({}))[({}{})({}{})]))])"
              "))"
          ),
      ),
  ])
  def test_root_certificates(self, game, players, expected_certificate):
    for i in players:
      tree = pyspiel.InfostateTree(game, i)
      self.assertEqual(tree.root().make_certificate(), expected_certificate)

  def check_tree_leaves(self, tree, move_limit):
    for leaf_node in tree.leaf_nodes():
      self.assertTrue(leaf_node.is_leaf_node())
      self.assertTrue(leaf_node.has_infostate_string())
      self.assertNotEmpty(leaf_node.corresponding_states())

      num_states = len(leaf_node.corresponding_states())
      terminal_cnt = 0
      max_move_number = float("-inf")
      min_move_number = float("inf")
      for state in leaf_node.corresponding_states():
        if state.is_terminal():
          terminal_cnt += 1
        max_move_number = max(max_move_number, state.move_number())
        min_move_number = min(min_move_number, state.move_number())
      self.assertTrue(terminal_cnt == 0 or terminal_cnt == num_states)
      self.assertEqual(max_move_number, min_move_number)
      if terminal_cnt == 0:
        self.assertEqual(max_move_number, move_limit)
      else:
        self.assertLessEqual(max_move_number, move_limit)

  def check_continuation(self, tree):
    leaves = tree.nodes_at_depth(tree.tree_height())
    continuation = pyspiel.InfostateTree(leaves)
    self.assertLen(leaves, continuation.root_branching_factor())
    for i in range(len(leaves)):
      leaf_node = leaves[i]
      root_node = continuation.root().child_at(i)
      self.assertTrue(leaf_node.is_leaf_node())
      if leaf_node.type() != pyspiel.InfostateNodeType.terminal:
        self.assertEqual(leaf_node.type(), root_node.type())
        self.assertEqual(
            leaf_node.has_infostate_string(), root_node.has_infostate_string()
        )
        if leaf_node.has_infostate_string():
          self.assertEqual(
              leaf_node.infostate_string(), root_node.infostate_string()
          )
      else:
        terminal_continuation = continuation.root().child_at(i)
        while (
            terminal_continuation.type()
            == pyspiel.InfostateNodeType.observation
        ):
          self.assertFalse(terminal_continuation.is_leaf_node())
          self.assertEqual(terminal_continuation.num_children(), 1)
          terminal_continuation = terminal_continuation.child_at(0)
        self.assertEqual(
            terminal_continuation.type(), pyspiel.InfostateNodeType.terminal
        )
        self.assertEqual(
            leaf_node.has_infostate_string(),
            terminal_continuation.has_infostate_string(),
        )
        if leaf_node.has_infostate_string():
          self.assertEqual(
              leaf_node.infostate_string(),
              terminal_continuation.infostate_string(),
          )
        self.assertEqual(
            leaf_node.terminal_utility(),
            terminal_continuation.terminal_utility(),
        )
        self.assertEqual(
            leaf_node.terminal_chance_reach_prob(),
            terminal_continuation.terminal_chance_reach_prob(),
        )
        self.assertEqual(
            leaf_node.terminal_history(),
            terminal_continuation.terminal_history(),
        )

  def test_depth_limited_tree_kuhn_poker(self):
    # Test MakeTree for Kuhn Poker with depth limit 2
    expected_certificate = (
        "("  # <dummy>
        "("  # 1st is getting a card
        "("  # 2nd is getting card
        "["  # 1st acts - Node J
        # Depth cutoff.
        "]"
        ")"
        # Repeat the same for the two other cards.
        "([])"  # Node Q
        "([])"  # Node K
        ")"
        ")"  # </dummy>
    )
    tree = pyspiel.InfostateTree(
        pyspiel.load_game("kuhn_poker"), 0, max_move_limit=2
    )
    self.assertEqual(tree.root().make_certificate(), expected_certificate)

    # Test leaf nodes in Kuhn Poker tree
    for acting in tree.leaf_nodes():
      self.assertTrue(acting.is_leaf_node())
      self.assertEqual(acting.type(), pyspiel.InfostateNodeType.decision)
      self.assertLen(acting.corresponding_states(), 2)
      self.assertTrue(acting.has_infostate_string())

  @parameterized.parameters([
      "kuhn_poker",
      "kuhn_poker(players=3)",
      "leduc_poker",
      "goofspiel(players=2,num_cards=3,imp_info=True)",
      "goofspiel(players=3,num_cards=3,imp_info=True)",
  ])
  def test_depth_limited_trees_all_depths(self, game_name):
    game = pyspiel.load_game(game_name)
    max_moves = game.max_move_number()
    for move_limit in range(max_moves):
      for pl in range(game.num_players()):
        tree = pyspiel.InfostateTree(game, pl, max_move_limit=move_limit)
        self.check_tree_leaves(tree, move_limit)
        self.check_continuation(tree)

  def test_node_binding(self):
    with self.assertRaises(TypeError) as _:
      # No constructor defined for InfostateNode.
      pyspiel.InfostateNode()
    # disallowing copying is enforced
    tree = pyspiel.InfostateTree(pyspiel.load_game("kuhn_poker"), 0)
    root = tree.root()
    with self.assertRaises(pyspiel.ForbiddenError) as _:
      copy.copy(root)
      copy.deepcopy(root)

    self.assertIsInstance(root, Iterable)
    self.assertIsInstance(iter(root), Iterator)
    self.assertEqual(root, next(iter(root)).parent())

    for _ in root:
      pass

  def test_treevector_binding(self):
    game = pyspiel.load_game("kuhn_poker")
    tree = pyspiel.InfostateTree(game, 0)
    # ensure constructors are bound with the respective args
    treeplex_vec = pyspiel.TreeplexVector(tree)
    leaf_vec = pyspiel.LeafVector(tree)
    decision_vec = pyspiel.DecisionVector(tree)

    self.assertLen(treeplex_vec, 13)
    self.assertLen(leaf_vec, 30)
    self.assertLen(decision_vec, 6)

    tree.all_decision_ids()
    seq_id_range = tree.all_sequence_ids()
    n_ids = 0
    for _ in seq_id_range:
      n_ids += 1
    self.assertEqual(n_ids, 13)
    seq_id = next(iter(tree.all_sequence_ids()))
    seq_id_copy = copy.copy(seq_id)
    self.assertEqual(seq_id.id(), 0)
    self.assertFalse(seq_id.is_undefined())
    self.assertIsNone(seq_id.next())
    self.assertNotEqual(seq_id, seq_id_copy)

  def test_sequence_id_labeling(self):
    for pl in range(2):
      tree = pyspiel.InfostateTree(pyspiel.load_game("kuhn_poker"), pl)

      for depth in range(tree.tree_height() + 1):
        for node in tree.nodes_at_depth(depth):
          self.assertLessEqual(
              node.start_sequence_id().id(), node.sequence_id().id()
          )
          self.assertLessEqual(
              node.end_sequence_id().id(), node.sequence_id().id()
          )

      # Check labeling was done from the deepest nodes.
      depth = float("inf")  # Some large number.
      for seq_id in tree.all_sequence_ids():
        node = tree.observation_infostate(seq_id)
        self.assertLessEqual(node.depth(), depth)
        depth = node.depth()
        # Longer sequences (extensions) must have the corresponding
        # infostate nodes placed deeper.
        for extension in node.all_sequence_ids():
          child = tree.observation_infostate(extension)
          self.assertLess(node.depth(), child.depth())

  def test_best_response(self):
    tree0 = pyspiel.InfostateTree(pyspiel.load_game("matrix_mp"), 0)
    tree1 = pyspiel.InfostateTree(pyspiel.load_game("matrix_mp"), 1)
    for alpha in range(0, 10):
      alpha /= 10.0
      br_value = max(2 * alpha - 1, -2 * alpha + 1)
      grad0 = pyspiel.LeafVectorFloat(
          tree0,
          [
              1.0 * alpha,
              -1.0 * (1.0 - alpha),
              -1.0 * alpha,
              1.0 * (1.0 - alpha),
          ],
      )
      self.assertAlmostEqual(tree0.best_response_value(grad0), br_value)

      grad1 = pyspiel.LeafVectorFloat(
          tree1,
          [
              -1.0 * alpha,
              1.0 * (1.0 - alpha),
              1.0 * alpha,
              -1.0 * (1.0 - alpha),
          ],
      )
      self.assertAlmostEqual(tree1.best_response_value(grad1), br_value)

      grad0_tp = pyspiel.TreeplexVectorFloat(
          tree0, [-1.0 + 2.0 * alpha, 1.0 - 2.0 * alpha, 0.0]
      )
      actual_response = tree0.best_response(grad0_tp)
      self.assertAlmostEqual(actual_response[0], br_value)

      grad1_tp = pyspiel.TreeplexVectorFloat(
          tree1, [1.0 - 2.0 * alpha, -1.0 + 2.0 * alpha, 0.0]
      )
      actual_response = tree1.best_response(grad1_tp)
      self.assertAlmostEqual(actual_response[0], br_value)


if __name__ == "__main__":
  absltest.main()
