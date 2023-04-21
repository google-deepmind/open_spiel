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

from absl.testing import absltest, parameterized

import pyspiel
import gc
from copy import copy, deepcopy
import weakref


class InfostateTreeTest(parameterized.TestCase):
    def test_tree_binding(self):
        game = pyspiel.load_game("kuhn_poker")
        tree = pyspiel.InfostateTree(game, 0)
        self.assertEqual(tree.num_sequences(), 13)

        # disallowing copying is enforced
        with self.assertRaises(pyspiel.ForbiddenError) as context:
            copy(tree)
            deepcopy(tree)

    def test_node_tree_lifetime_management(self):
        root0 = tree.root()
        # let's maintain a weak ref to the tree to see when the tree object is deallocated
        wptr = weakref.ref(tree)
        # ensure we can get a shared_ptr from root that keeps tree alive if we lose the 'tree' name
        wptr_node = weakref.ref(root0)
        tree_sptr = root0.tree()
        # grab the tree id
        id_tree0 = id(tree)
        # now delete the initial tree ptr
        del tree
        # ensure that we still hold the object
        gc.collect()  # force garbage collection
        self.assertIsNotNone(wptr())
        self.assertEqual(id(tree_sptr), id_tree0)
        # now delete the last pointer as well
        del tree_sptr
        gc.collect()  # force garbage collection
        self.assertIsNone(wptr())

    @parameterized.parameters(
        [
            # test for matrix mp
            dict(
                game=pyspiel.load_game("matrix_mp"),
                players=[0, 1],
                expected_certificate="([" "({}{})" "({}{})" "])",
            ),
            # test for imperfect info goofspiel
            dict(
                game=pyspiel.load_game(
                    "goofspiel",
                    {"num_cards": 2, "imp_info": True, "points_order": "ascending"},
                ),
                players=[0, 1],
                expected_certificate="([" "({}{})" "({}{})" "])",
            ),
            # test for kuhn poker (0 player only)
            dict(
                game=pyspiel.load_game("kuhn_poker"),
                players=[0],
                expected_certificate=(
                    "(("  # Root node, 1st is getting a card
                    "("  # 2nd is getting card
                    "["  # 1st acts
                    "(("  # 1st bet, and 2nd acts
                    "(({}))"
                    "(({}))"
                    "(({}))"
                    "(({}))"
                    "))"
                    "(("  # 1st checks, and 2nd acts
                    # 2nd checked
                    "(({}))"
                    "(({}))"
                    # 2nd betted
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
        ]
    )
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
            self.assertTrue(max_move_number == min_move_number)
            if terminal_cnt == 0:
                self.assertEqual(max_move_number, move_limit)
            else:
                self.assertLessEqual(max_move_number, move_limit)

    def check_continuation(self, tree):
        leaves = tree.nodes_at_depth(tree.tree_height())
        continuation = pyspiel.InfostateTree(leaves)
        self.assertEqual(continuation.root_branching_factor(), len(leaves))
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
        tree = pyspiel.InfostateTree(pyspiel.load_game("kuhn_poker"), 0, 2)
        self.assertEqual(tree.root().make_certificate(), expected_certificate)

        # Test leaf nodes in Kuhn Poker tree
        for acting in tree.leaf_nodes():
            self.assertTrue(acting.is_leaf_node())
            self.assertEqual(acting.type(), pyspiel.InfostateNodeType.decision)
            self.assertEqual(len(acting.corresponding_states()), 2)
            self.assertTrue(acting.has_infostate_string())

    @parameterized.parameters(
        [
            "kuhn_poker",
            "kuhn_poker(players=3)",
            "leduc_poker",
            "goofspiel(players=2,num_cards=3,imp_info=True)",
            "goofspiel(players=3,num_cards=3,imp_info=True)",
        ]
    )
    def test_depth_limited_trees_all_depths(self, game_name):
        game = pyspiel.load_game(game_name)
        max_moves = game.max_move_number()
        for move_limit in range(max_moves):
            for pl in range(game.num_players()):
                tree = pyspiel.InfostateTree(game, pl, move_limit)
                self.check_tree_leaves(tree, move_limit)
                self.check_continuation(tree)

    def test_node_binding(self):
        with self.assertRaises(TypeError) as context:
            pyspiel.InfostateNode()
            self.assertTrue("No constructor defined" in context.exception)


if __name__ == "__main__":
    absltest.main()
