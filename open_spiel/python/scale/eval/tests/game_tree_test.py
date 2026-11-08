# Copyright 2026 DeepMind Technologies Limited
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

"""Tests of GameTree's arrays for exact belief.

GameTree's arrays for exact belief: seat-local infosets, their keys, chance
reach and per-player path edges reproduce the tree's own expected returns and
the analytic kuhn posterior.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import game_tree
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.envs import leduc


@pytest.fixture(name="tree", scope="module", params=["kuhn", "leduc"])
def _tree_fixture(request):
  """The kuhn and leduc GameTrees.

  Args:
    request: pytest's request (param: "kuhn" or "leduc").

  Returns:
    the tree.
  """
  env = kuhn.KuhnEnv() if request.param == "kuhn" else leduc.LeducEnv()
  return game_tree.GameTree(env)


@pytest.fixture(name="ktree", scope="module")
def _ktree_fixture():
  """The kuhn GameTree.

  Returns:
    the tree.
  """
  return game_tree.GameTree(kuhn.KuhnEnv())


def _random_table(tree, seed):
  """A random table over each infoset's legal actions.

  Args:
    tree: the GameTree.
    seed: the numpy seed.

  Returns:
    the table f64[I, A].
  """
  rng = np.random.default_rng(seed)
  t = rng.random((tree.num_infosets, tree.num_actions)) * tree.legal
  return t / t.sum(1, keepdims=True)


def test_seat_infosets_partition_the_infosets(tree):
  """The seats' infosets partition the infosets."""
  rows = np.concatenate(
      [tree.seat_infosets(p) for p in range(tree.num_players)]
  )
  assert sorted(rows.tolist()) == list(range(tree.num_infosets))
  for p in range(tree.num_players):
    assert (tree.player[tree.seat_infosets(p)] == p).all()


def test_infoset_nodes_and_members_agree(tree):
  """infoset_nodes and infoset_members agree with the tree."""
  for p in range(tree.num_players):
    dense = tree.infoset_members(p)
    for i, row in enumerate(tree.seat_infosets(p)):
      nodes = tree.infoset_nodes(p, i)
      np.testing.assert_array_equal(nodes, tree.members[row])
      np.testing.assert_array_equal(dense[i, : nodes.size], nodes)
      assert (dense[i, nodes.size :] == -1).all()
      assert (tree.infoset[nodes] == row).all()
  with pytest.raises(IndexError):
    tree.infoset_nodes(0, tree.seat_infosets(0).size)


def test_infoset_keys_are_unique_and_match_the_node_info(tree):
  """Infoset keys are unique and match each node's info."""
  states = tree.node_states()
  for p in range(tree.num_players):
    keys = tree.infoset_keys(p)
    assert (
        keys.dtype == np.float64 and keys.shape[0] == tree.seat_infosets(p).size
    )
    assert len({k.tobytes() for k in keys}) == keys.shape[0]
    for i in (0, keys.shape[0] - 1):
      n = int(tree.infoset_nodes(p, i)[0])
      s = jax.tree_util.tree_map(lambda x, n=n: jnp.asarray(x[n]), states)
      info = tree.env.info_state(s, jnp.int32(p))
      np.testing.assert_array_equal(
          np.asarray(game_tree.GameTree.flatten_info(info)), keys[i]
      )


def test_kuhn_decision_nodes_have_chance_reach_one_sixth(ktree):
  """Every kuhn decision node has chance reach 1/6."""
  reach = ktree.chance_reach()
  assert reach[0] == 1.0
  dec = ktree.kind == game_tree.DECISION
  np.testing.assert_allclose(reach[dec], 1.0 / 6.0, rtol=0, atol=1e-15)


def test_path_edges_reproduce_expected_returns(tree):
  """Path edges reproduce GameTree.expected_returns.

  Sum over terminals of chance reach x each player's path probability x
  returns equals GameTree.expected_returns.

  Args:
    tree: the kuhn or leduc GameTree (fixture).
  """
  tables = [_random_table(tree, 1), _random_table(tree, 2)]
  reach = tree.chance_reach()
  prob = reach.copy()
  for p in range(tree.num_players):
    edges = tree.player_edges_on_path(p)
    flat = tables[p].reshape(-1)
    prob *= np.where(edges >= 0, flat[np.maximum(edges, 0)], 1.0).prod(1)
  term = tree.kind == game_tree.TERMINAL
  got = (prob[term, None] * tree.node_returns[term]).sum(0)
  np.testing.assert_allclose(
      got, tree.expected_returns(tables), rtol=0, atol=1e-12
  )


def test_kuhn_posterior_from_path_edges(ktree):
  """The kuhn posterior from path edges matches the analytic one.

  After player 0 bets, player 1 holding the queen believes player 0 holds
  the king with probability P(bet | K) / (P(bet | J) + P(bet | K)).

  Args:
    ktree: the kuhn GameTree (fixture).
  """
  bet = {
      0: 0.2,
      1: 0.5,
      2: 0.9,
  }  # P(bet) at player 0's opening decision, by card
  table = ktree.uniform_table().copy()
  info = jax.tree_util.tree_map(np.asarray, ktree.info)
  for row in ktree.seat_infosets(0):
    if info.num_bets[row] == 0:
      p = bet[int(info.card[row])]
      table[row] = [1.0 - p, p]
  edges = ktree.player_edges_on_path(0)
  flat = table.reshape(-1)
  weight = ktree.chance_reach() * np.where(
      edges >= 0, flat[np.maximum(edges, 0)], 1.0
  ).prod(1)
  p0_card = np.asarray(ktree.node_states().p0_card)
  rows = ktree.seat_infosets(1)
  (i,) = [
      i
      for i, r in enumerate(rows)
      if info.card[r] == 1 and info.num_bets[r] == 1 and info.bets[r][0] == 1
  ]
  nodes = ktree.infoset_nodes(1, i)
  post = dict(zip(p0_card[nodes].tolist(), weight[nodes] / weight[nodes].sum()))
  assert sorted(post) == [0, 2]
  np.testing.assert_allclose(
      post[2], bet[2] / (bet[0] + bet[2]), rtol=0, atol=1e-12
  )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
