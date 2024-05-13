# Copyright 2023 DeepMind Technologies Limited
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
"""Ranked Pairs A.K.A. the Tideman method.

Based on https://en.wikipedia.org/wiki/Ranked_pairs.
"""

from typing import List, Tuple
import numpy as np
from open_spiel.python.voting import base

# TODO(author5): either one of the following: (i) change graph representation to
# adjacency lists for more efficient cycle checking, (ii) use a graph library
# such as networkx to represent the graph and support graph functions.


class RankedPairsRankOutcome(base.RankOutcome):
  """A custom RankOutcome class for Ranked Pairs.

  Provides an extra method to get the graph.
  """

  def __init__(
      self,
      rankings: List[base.AlternativeId],
      scores: List[float],
      graph: np.ndarray,
  ):
    super().__init__(rankings, scores)
    self._graph = graph

  @property
  def graph(self) -> np.ndarray:
    return self._graph


class RankedPairsVoting(base.AbstractVotingMethod):
  """Implements Ranked Pairs / Tideman's method."""

  def __init__(self):
    pass

  def name(self) -> str:
    return "ranked_pairs"

  def _would_create_cycle(
      self,
      alternatives: List[base.AlternativeId],
      graph: np.ndarray,
      from_idx: int,
      to_idx: int,
  ) -> bool:
    """Checks if adding a specific directed edge would result in a cycle.

    Args:
      alternatives: list of alternatives.
      graph: 2D adjacency matrix representing a directed acyclic graph. Row is
        the from node index, column the to node index.
      from_idx: the edge to add (from index).
      to_idx: the edge to add (to index).

    Returns:
      True if adding the specified edge would result in a cycle in the graph.
    """
    # Perform a breadth-first flood fill using a status table.
    # Values in the status table represent:
    #    0  means it does not exist in the flood yet
    #    1  means it needs to be expanded
    #    -1 means it has been expanded (now closed, do not revisit)
    m = len(alternatives)
    status_table = np.zeros(m)
    status_table[to_idx] = 1
    num_expanded = 1
    while num_expanded > 0:
      num_expanded = 0
      for i in np.where(status_table == 1)[0]:
        num_expanded += 1
        for j in np.where(graph[i][:] == 1)[0]:
          if status_table[j] == 0:
            if j == from_idx:
              return True
            status_table[j] = 1
        status_table[i] = -1
    return False

  def _is_source(self, graph: np.ndarray, idx: int):
    """Returns true if this node is a source, false otherwise."""
    num_incoming = np.sum(graph[:, idx])
    num_outgoing = np.sum(graph[idx])
    return num_outgoing > 0 and num_incoming == 0

  def _remove_node(self, graph: np.ndarray, idx: int):
    """Removes a node from the graph."""
    graph[idx, :] = 0
    graph[:, idx] = 0

  def _get_score(
      self, graph: np.ndarray, margin_matrix: np.ndarray, node_idx: int
  ) -> int:
    """Computes the score of an alternative.

    The score is defined as the sum of the margins between the subgraph
    containing all reachable nodes from this node.

    Args:
      graph: 2D adjacency matrix representing a directed acyclic graph. Row is
        the from node index, column the to node index.
      margin_matrix: the margin matrix from the profile
      node_idx: the node index in question.

    Returns:
      the score of the alternative represented by this node index.
    """
    # Flood fill to compute score from a source
    score = 0
    open_list = {node_idx: True}
    closed_list = {}
    while open_list:
      i = list(open_list.keys())[0]
      open_list.pop(i)
      outgoing_edges = np.where(graph[i][:] == 1)[0]
      for j in outgoing_edges:
        score += margin_matrix[i, j]
        if j not in open_list and j not in closed_list:
          open_list[j] = True
      closed_list[i] = True
    return score

  def _get_ranked_pairs(
      self, alternatives: List[base.AlternativeId], margin_matrix: np.ndarray
  ) -> List[Tuple[Tuple[base.AlternativeId, base.AlternativeId], int]]:
    """Returns the positively-valued ranked pairs coupled with their values.
 
    Arguments:
      alternatives: the list of alternatives ids.
      margin_matrix: the margin matrix we use to get the values for each ranked
        pair.

    Returns:
      A list of tuples of the form ((x, y), value) indicating x beating y by
      the specified value.
    """
    ranked_pairs = {}
    rows, cols = np.where(margin_matrix > 0)
    for i, j in zip(rows, cols):
      key_tup = (alternatives[i], alternatives[j])
      ranked_pairs[key_tup] = margin_matrix[i, j]
    return sorted(ranked_pairs.items(), key=lambda item: item[1], reverse=True)

  def run_election(
      self, profile: base.PreferenceProfile
  ) -> RankedPairsRankOutcome:
    assert self.is_valid_profile(profile)
    alternatives = profile.alternatives
    m = len(alternatives)
    alt_idx = profile.alternatives_dict
    margin_matrix = profile.margin_matrix()

    # First, get the ranked pairs annotated with their values (delta(a,b)).
    sorted_pairs = self._get_ranked_pairs(alternatives, margin_matrix)

    # Now, create the graph: add edges that do not create cycles.
    graph = np.zeros(shape=(m, m), dtype=np.int32)
    if sorted_pairs:
      # Create the top-ranked pair. This needs to be in a conditional block,
      # because some profiles can legitimately lead to a graph with no edges (no
      # positively-valued ranked pairs)
      first_pair = sorted_pairs[0][0]
      p0_idx = alt_idx[first_pair[0]]
      p1_idx = alt_idx[first_pair[1]]
      graph[p0_idx, p1_idx] = 1
    for j in range(1, len(sorted_pairs)):
      pair = sorted_pairs[j][0]
      p0_idx = alt_idx[pair[0]]
      p1_idx = alt_idx[pair[1]]
      if not self._would_create_cycle(alternatives, graph, p0_idx, p1_idx):
        graph[p0_idx, p1_idx] = 1
    full_graph = graph.copy()  # Make a copy to return later.

    # Now, remove sources nodes in sequence to get the ranking.
    ranking = []
    scores = []
    alt_idx_remaining = []
    for i in range(m):
      alt_idx_remaining.append(i)
    while len(ranking) < m:
      has_source = False
      for j in range(m):
        if self._is_source(graph, j):
          ranking.append(alternatives[j])
          scores.append(self._get_score(graph, margin_matrix, j))
          self._remove_node(graph, j)
          alt_idx_remaining.remove(j)
          has_source = True
          break
      if not has_source:
        # At the end, it can happen that there are a number of disconnected
        # nodes (no incoming nor outgoing edges). Take the first one from the
        # graph.
        j = alt_idx_remaining[0]
        ranking.append(alternatives[j])
        scores.append(0)
        self._remove_node(graph, j)
        alt_idx_remaining.remove(j)

    # Finally, return the ranking and scores.
    outcome = RankedPairsRankOutcome(
        rankings=ranking, scores=scores, graph=full_graph
    )
    return outcome
