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

"""Population mixtures as behaviour policies, and exact symmetric payoffs.

A meta-strategy over a population is a mixed strategy (play member k for the
whole game with probability w_k). Kuhn's theorem gives the behaviour policy
with the same outcome distribution in a perfect-recall game, so the mixture's
exact exploitability can be computed with the same tools as a single policy.
Everything here works on `eval.exploitability.GameTree` tables of games
small enough to enumerate (kuhn, leduc).

    tables = [tree.policy_table(policy, params, rule)
              for policy, params in members]
    mix = mixture.mixture_table(tree, tables, weights)

  own_reach(tree, table, p)        f64[num_infosets]: seat p's own reach
                                   probability of each of its infosets (the
                                   product of p's action probabilities on the
                                   path; 0 for other seats' rows).
  mixture_table(tree, tables, w)   the behaviour table realization-equivalent
                                   to playing table k with probability w[k]
                                   for the whole game (Kuhn's theorem):
                                   pi(a|I) = sum_k w_k r_k(I) pi_k(a|I) /
                                   sum_k w_k r_k(I), and the plain w-average
                                   where no member reaches I.
  symmetric_payoff(tree, a, b)     a's exact expected return against b,
                                   averaged over the two seat assignments (the
                                   Evaluator's duplicate-deal estimand).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from open_spiel.python.scale.eval import exploitability as expl

F64 = np.float64


def own_reach(tree: expl.GameTree, table, player: int) -> np.ndarray:
  """Seat `player`'s own reach probability of each of its infosets.

  The product of player's action probabilities on the path; 0 for other
  seats' rows.

  Args:
    tree: the game tree.
    table: a behaviour table over the tree's infosets.
    player: the seat.

  Returns:
    f64[num_infosets].

  Raises:
    ValueError: an infoset's histories have different own reach (the game is
      not perfect recall).
  """
  t = tree.check_table(table)
  out = np.zeros(tree.num_infosets, F64)
  for row in range(tree.num_infosets):
    if int(tree.player[row]) != int(player):
      continue
    vals = set()
    for n in tree.members[row]:
      r, m = 1.0, int(n)
      while tree.parent[m] >= 0:
        par = int(tree.parent[m])
        if tree.kind[par] == expl.DECISION and int(tree.actor[par]) == int(
            player
        ):
          r *= t[int(tree.infoset[par]), int(tree.in_action[m])]
        m = par
      vals.add(r)
    if max(vals) - min(vals) > 1e-12:
      raise ValueError(
          f"{tree.game}: infoset {row} has histories with different own reach "
          "(not perfect recall)"
      )
    out[row] = vals.pop()
  return out


def mixture_table(
    tree: expl.GameTree, tables: Sequence[np.ndarray], weights
) -> np.ndarray:
  """The behaviour table realization-equivalent to a mixture of tables.

  Playing table k with probability w[k] for the whole game (Kuhn's theorem):
  pi(a|I) = sum_k w_k r_k(I) pi_k(a|I) / sum_k w_k r_k(I), and the plain
  w-average where no member reaches I.

  Args:
    tree: the game tree.
    tables: the members' behaviour tables.
    weights: the mixture weights, one per table, summing to 1.

  Returns:
    The mixture's behaviour table.

  Raises:
    ValueError: the weights do not fit the tables or do not sum to 1.
  """
  w = np.asarray(weights, F64)
  if (
      w.ndim != 1
      or w.shape[0] != len(tables)
      or not np.all(np.isfinite(w))
      or np.any(w < 0)
  ):
    raise ValueError(f"weights {w} do not fit {len(tables)} tables")
  if abs(w.sum() - 1.0) > 1e-9:
    raise ValueError(f"weights sum to {w.sum()}, not 1")
  ts = [tree.check_table(t) for t in tables]
  reach = np.zeros((len(ts), tree.num_infosets), F64)
  for k, t in enumerate(ts):
    for p in range(tree.num_players):
      reach[k] += own_reach(tree, t, p)
  num = sum(w[k] * reach[k][:, None] * ts[k] for k in range(len(ts)))
  den = (w[:, None] * reach).sum(0)
  plain = sum(w[k] * ts[k] for k in range(len(ts)))
  out = np.where(
      den[:, None] > 0.0, num / np.where(den > 0.0, den, 1.0)[:, None], plain
  )
  out = out / out.sum(1, keepdims=True)
  return tree.check_table(out, "mixture table")


def symmetric_payoff(tree: expl.GameTree, a, b) -> float:
  """A's exact expected return against b, averaged over the seat assignments.

  This is the Evaluator's duplicate-deal estimand.

  Args:
    tree: a two-player game tree.
    a: a behaviour table.
    b: another behaviour table.

  Returns:
    The expected return.
  """
  assert tree.num_players == 2, tree.num_players
  ta, tb = tree.check_table(a), tree.check_table(b)
  return float(
      0.5
      * (
          tree.expected_returns([ta, tb])[0]
          + tree.expected_returns([tb, ta])[1]
      )
  )
