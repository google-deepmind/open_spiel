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

"""Stochastically stable distributions (SSD) of perturbed Markov processes.

  "An Algorithm for Computing Stochastically Stable Distributions with
  Applications to Multiagent Learning in Repeated Games"
  John R. Wicks and Amy Greenwald, UAI 2005. https://arxiv.org/abs/1207.1424
"""

import numpy as np
import scipy.sparse as sps
from scipy.sparse import csgraph
import scipy.sparse.linalg as spla

from open_spiel.python.egt import alpharank
from open_spiel.python.egt import utils

# Resistances closer than this are considered to be equal.
_RESISTANCE_TOL = 1e-9


def _add_terms(term, other):
  """Returns the leading term of the sum of two (cost, resistance) terms."""
  if term is None:
    return other
  if abs(term[1] - other[1]) <= _RESISTANCE_TOL:
    return (term[0] + other[0], min(term[1], other[1]))
  return term if term[1] < other[1] else other


class PerturbedMarkovChain(object):
  """Leading-order description of a perturbed Markov process.

  The probability of moving from state `source` to a different state `target`
  is cost * eps^resistance + o(eps^resistance), with cost > 0 and
  resistance >= 0. Transitions that are not added have probability 0 (infinite resistance).
  Self-transitions implicitly take the missing probability.
  """

  def __init__(self, num_states):
    self.num_states = num_states
    # _out[source][target] = (cost, resistance), _in[target] = {sources}.
    self._out = [{} for _ in range(num_states)]
    self._in = [set() for _ in range(num_states)]

  def add_transition(self, source, target, cost, resistance):
    """Adds cost * eps^resistance to the transition source -> target."""
    if cost <= 0 or resistance < 0:
      raise ValueError("Transitions need cost > 0 and resistance >= 0, got "
                       "cost={} and resistance={}.".format(cost, resistance))
    if source == target:
      return
    term = (float(cost), float(resistance))
    self._out[source][target] = _add_terms(self._out[source].get(target), term)
    self._in[target].add(source)

  def transitions(self, source):
    return dict(self._out[source])

  def _outflow(self, state):
    """Returns the leading term of 1 - M_eps[state, state], or None if it is 0."""
    total = None
    for term in self._out[state].values():
      total = _add_terms(total, term)
    return total

  def _classes(self, states):
    """Finds the communicating classes of the unperturbed process M_0.

    Args:
      states: The states that have not been eliminated yet.

    Returns:
      A list with the closed communicating classes (each a list of states).
    """
    sources, targets = [], []
    for source in states:
      for target, (_, resistance) in self._out[source].items():
        if resistance <= _RESISTANCE_TOL:
          sources.append(source)
          targets.append(target)
    sources, targets = np.asarray(sources, int), np.asarray(targets, int)
    graph = sps.csr_matrix((np.ones(len(sources)), (sources, targets)),
                           shape=(self.num_states, self.num_states))
    _, labels = csgraph.connected_components(graph, connection="strong")

    # A class is closed iff no zero-resistance transition leaves it
    open_labels = set(labels[sources][labels[sources] != labels[targets]])
    closed = {}
    for state in states:
      if labels[state] not in open_labels:
        closed.setdefault(labels[state], []).append(state)
    return list(closed.values())

  def _collapse(self, closed):
    """Collapses a closed class of M_0 into its first state.

    Args:
      closed: The states of a non-trivial closed communicating class of M_0.

    Returns:
      The expected number of visits to each of closed[1:] between two visits to
      closed[0], in M_0. This is the inclusion operator i_0: multiplying by the
      stochastically stable mass of closed[0] gives the mass of closed[1:].
    """
    keep, removed = closed[0], closed[1:]
    index = {state: i for i, state in enumerate(removed)}

    # Solve (I - M_0) visits = M_0[keep -> removed], with M_0 restricted to the
    # removed states.
    rows, cols, data = [], [], []
    entries = np.zeros(len(removed))
    for state, i in index.items():
      for target, (cost, resistance) in self._out[state].items():
        # Transitions of resistance 0 stay within the class, as it is closed.
        if resistance <= _RESISTANCE_TOL:
          rows.append(i)
          cols.append(i)
          data.append(cost)
          if target != keep:
            rows.append(index[target])
            cols.append(i)
            data.append(-cost)
    for target, (cost, resistance) in self._out[keep].items():
      if resistance <= _RESISTANCE_TOL:
        entries[index[target]] = cost
    matrix = sps.csc_matrix((data, (rows, cols)), shape=2 * (len(removed),))
    visits = np.atleast_1d(spla.spsolve(matrix, entries))

    for state, i in index.items():
      # Transitions that leave the class now leave `keep`. Transitions within the class are dropped.
      for target, (cost, resistance) in self._out[state].items():
        self._in[target].discard(state)
        if target != keep and target not in index:
          self.add_transition(keep, target, cost * visits[i], resistance)
      # Transitions that enter the class now enter `keep`.
      for source in self._in[state]:
        cost, resistance = self._out[source].pop(state)
        if source not in index:
          self.add_transition(source, keep, cost, resistance)
      self._out[state] = {}
      self._in[state] = set()
    return visits

  def _scale(self, absorbing_states):
    """ 'Speeds up time' in the absorbing states of M_0.

    Args:
      absorbing_states: The states that cannot be left in M_0.
    """
    outflows = [self._outflow(state) for state in absorbing_states]
    if any(outflow is None for outflow in outflows):
      raise ValueError("The perturbed Markov process is not regular: it has "
                       "more than one closed class for eps > 0.")
    min_resistance = min(resistance for _, resistance in outflows)
    scale = 2 * max(cost for cost, resistance in outflows
                    if resistance - min_resistance <= _RESISTANCE_TOL)
    for state in absorbing_states:
      for target, (cost, resistance) in self._out[state].items():
        resistance -= min_resistance
        if resistance <= _RESISTANCE_TOL:
          resistance = 0.
        self._out[state][target] = (cost / scale, resistance)

  def stochastically_stable_distribution(self):
    """Computes the SSD. This modifies the process.

    Returns:
      A numpy array with the stochastically stable distribution. States in its
      support are the stochastically stable states.

    Raises:
      ValueError: If the process is not regular, i.e. if it has more than one
        closed communicating class for eps > 0.
    """
    states = set(range(self.num_states))
    # The inclusion operators, as ("collapse", closed class, i_0) and
    # ("zero", transient states) steps. Applied in reverse to the final SSD.
    steps = []
    while True:
      closed_classes = self._classes(states)
      if any(len(closed) > 1 for closed in closed_classes):
        for closed in closed_classes:
          if len(closed) > 1:
            steps.append(("collapse", closed, self._collapse(closed)))
            states.difference_update(closed[1:])
      elif len(closed_classes) > 1:
        # All closed classes are absorbing states. Any other state is transient
        # in M_0, thus not stochastically stable.
        absorbing_states = [closed[0] for closed in closed_classes]
        steps.append(("zero", states.difference(absorbing_states)))
        self._scale(absorbing_states)
      else:
        break

    # M_0 is left with a unique absorbing state.
    distribution = np.zeros(self.num_states)
    distribution[closed_classes[0][0]] = 1.
    for step in reversed(steps):
      if step[0] == "zero":
        distribution[list(step[1])] = 0.
      else:
        # The transient states of M_0 also enter the class. They do not show up
        # here because their mass is 0 by now.
        _, closed, visits = step
        distribution[closed[1:]] = visits * distribution[closed[0]]
    return distribution / np.sum(distribution)


def _get_alpharank_perturbed_markov_chain(payoff_tables,
                                          payoffs_are_hpt_format,
                                          use_sparse=False):
  """Returns the infinite-alpha Alpha-Rank chain as a perturbed Markov process.

  Args:
    payoff_tables: List of game payoff tables, one for each agent identity.
    payoffs_are_hpt_format: Boolean indicating whether each payoff table is a
      _PayoffTableInterface object (AKA Heuristic Payoff Table or HPT), or a
      numpy array. True indicates HPT format, False indicates numpy array.
    use_sparse: If true, build the Alpha-Rank transition matrices sparsely.
  """
  def transition_matrix(inf_alpha_eps):
    if len(payoff_tables) == 1:
      c, _ = alpharank._get_singlepop_transition_matrix(  # pylint: disable=protected-access
          payoff_tables[0], payoffs_are_hpt_format, m=None, alpha=None,
          game_is_constant_sum=None, use_local_selection_model=True,
          payoff_sum=None, use_inf_alpha=True, inf_alpha_eps=inf_alpha_eps,
          use_sparse=use_sparse)
    else:
      c, _ = alpharank._get_multipop_transition_matrix(  # pylint: disable=protected-access
          payoff_tables, payoffs_are_hpt_format, m=None, alpha=None,
          use_inf_alpha=True, inf_alpha_eps=inf_alpha_eps,
          use_sparse=use_sparse)
    return sps.coo_matrix(c)

  # The transition probabilities are affine in eps. With eps = 0 only the
  # resistance 0 transitions are left, with eps = 1 (also) those of resistance 1
  c_0 = transition_matrix(inf_alpha_eps=0.)
  c_1 = transition_matrix(inf_alpha_eps=1.)
  chain = PerturbedMarkovChain(c_0.shape[0])
  free_transitions = set()
  for source, target, cost in zip(c_0.row, c_0.col, c_0.data):
    if source != target and cost > 0:
      chain.add_transition(source, target, cost, resistance=0)
      free_transitions.add((source, target))
  for source, target, cost in zip(c_1.row, c_1.col, c_1.data):
    if (source != target and cost > 0 and
        (source, target) not in free_transitions):
      chain.add_transition(source, target, cost, resistance=1)
  return chain


def compute_ssd(payoff_tables, verbose=False, use_sparse=False,
                **unused_kwargs):
  """Computes the SSD of the infinite-alpha Alpha-Rank Markov chain of a game.

  Args:
    payoff_tables: List of game payoff tables, one for each agent identity. Each
      payoff_table may be either a numpy array, or a _PayoffTableInterface
      object.
    verbose: Set to True to print intermediate results.
    use_sparse: If true, build the Alpha-Rank transition matrices sparsely.
    **unused_kwargs: Ignored, so that PSRO solver kwargs can be passed along.

  Returns:
    The stochastically stable distribution over strategy profiles (strategies
    in the single-population case), indexed like Alpha-Rank's pi.
  """
  payoffs_are_hpt_format = utils.check_payoffs_are_hpt(payoff_tables)
  num_strats_per_population = utils.get_num_strats_per_population(
      payoff_tables, payoffs_are_hpt_format)

  # trivial case: Markov chain with one state
  if np.array_equal(num_strats_per_population,
                    np.ones(len(num_strats_per_population))):
    return np.asarray([1.])

  if verbose:
    print("Constructing perturbed Markov chain")
    print("num_strats_per_population:", num_strats_per_population)

  chain = _get_alpharank_perturbed_markov_chain(
      payoff_tables, payoffs_are_hpt_format, use_sparse=use_sparse)
  ssd = chain.stochastically_stable_distribution()

  if verbose:
    print("\nStochastically stable distribution:\n", ssd)
  return ssd
