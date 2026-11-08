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

"""Kuhn poker tabular populations with known equilibria, for population tests.

Mixture evaluation, population exploitability and PSRO tests need members
whose exact values are known. Kuhn poker's Nash family (Kuhn 1950) gives
them: player 0 bets a jack with probability alpha in [0, 1/3] and the rest of
the strategy follows from alpha; every member has NashConv 0.

  pol = populations.kuhn_tabular()
  params = pol.params_from_logits(
      populations.bet_logits(populations.kuhn_nash(1 / 6)))
  a, b = populations.kuhn_populations()       # two lists of match.Entrant

Strategies are written as P(bet) per kuhn info index card * 4 + history,
with cards J, Q, K = 0, 1, 2 and histories "", "p", "b", "pb" = 0..3.
"""

import numpy as np

from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.eval import match
from open_spiel.python.scale.policies import tabular

#: "module:attr" of kuhn's info index: info_index in the module of the
#: kuhn_poker env, resolved through the env registry's game table.
KUHN_INDEX = env_registry.game("kuhn_poker").ctor.split(":")[0] + ":info_index"
NUM_INFO_STATES = 12


def kuhn_nash(alpha: float) -> np.ndarray:
  """Returns P(bet) at every kuhn infoset under the Nash strategy alpha.

  Args:
    alpha: Player 0's jack betting probability, in [0, 1/3].

  Returns:
    f64[12].
  """
  p = np.zeros(NUM_INFO_STATES)
  for card, row in (
      (0, (alpha, 1 / 3, 0.0, 0.0)),
      (1, (0.0, 0.0, 1 / 3, alpha + 1 / 3)),
      (2, (3 * alpha, 1.0, 1.0, 1.0)),
  ):
    p[card * 4 : card * 4 + 4] = row
  return p


def bet_logits(bet) -> np.ndarray:
  """Returns logits (pass, bet) whose softmax is P(bet) = bet.

  Args:
    bet: P(bet) per infoset.

  Returns:
    f32[12, 2].
  """
  b = np.clip(np.asarray(bet, np.float64), 1e-30, 1.0)
  nb = np.clip(1.0 - np.asarray(bet, np.float64), 1e-30, 1.0)
  return np.stack([np.log(nb), np.log(b)], 1).astype(np.float32)


def kuhn_tabular() -> tabular.TabularPolicy:
  """Returns a kuhn TabularPolicy over kuhn.info_index.

  Returns:
    The policy.
  """
  return tabular.TabularPolicy("kuhn_poker", NUM_INFO_STATES, 2, 2, KUHN_INDEX)


def entrant(
    name: str, bet, rule: types.DecisionRule = types.SAMPLE, policy=None
):
  """Returns a match.Entrant playing P(bet) = bet with a kuhn TabularPolicy.

  Args:
    name: The entrant's name.
    bet: P(bet) per infoset.
    rule: The decision rule.
    policy: The TabularPolicy; a new one when None.

  Returns:
    The Entrant.
  """
  pol = kuhn_tabular() if policy is None else policy
  return match.Entrant(name, pol, pol.params_from_logits(bet_logits(bet)), rule)


def kuhn_populations():
  """Returns a three-member and a two-member kuhn population of Entrants.

  They share one policy object, so each population stacks into one slot.

  Returns:
    (a, b).
  """
  pol = kuhn_tabular()
  a = [
      entrant("a0", kuhn_nash(0.0), policy=pol),
      entrant("a1", np.full(12, 0.9), policy=pol),
      entrant("a2", np.zeros(12), policy=pol),
  ]
  b = [
      entrant("b0", kuhn_nash(1 / 3), policy=pol),
      entrant("b1", np.full(12, 0.5), policy=pol),
  ]
  return a, b
