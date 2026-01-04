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

"""Approximate the limiting logit equilbrium (Nash) of a large normal-form game.

This is a python implementation of the Nash solver for normal-form games,
Average Deviation Incentive Descent with Adaptive Sampling (ADIDAS), from
"Sample-based Approximation of Nash in Large Many-player Games via Gradient
Descent" [Gemp et al, AAMAS 2022].

Link to paper: https://arxiv.org/abs/2106.01285.

The limiting logit equilibrium (LLE) was originally defined in "Quantal Response
Equilibria for Normal Form Games" [McKelvey & Palfrey, Games and Economic
Behavior 1995]. The LLE is a Nash equilibrium that is uniquely defined for
*almost* all games.
"""

import itertools

import time

import numpy as np

from open_spiel.python.algorithms.adidas_utils.helpers import misc
from open_spiel.python.algorithms.adidas_utils.helpers import simplex
from open_spiel.python.algorithms.adidas_utils.helpers.nonsymmetric import exploitability as nonsym_exp
from open_spiel.python.algorithms.adidas_utils.helpers.nonsymmetric import game_runner as nonsym_game_runner
from open_spiel.python.algorithms.adidas_utils.helpers.symmetric import exploitability as sym_exp
from open_spiel.python.algorithms.adidas_utils.helpers.symmetric import game_runner as sym_game_runner


class ADIDAS(object):
  """Average Deviation Incentive Descent with Adaptive Sampling.

  Approximate the limiting logit equilibrium of a normal-form game.

  Attributes:
    experiment_seed: int, seed for random number generator
    random: numpy.random.RandomState object
    results: dictionary of results populated upon completion of solver
  """

  def __init__(self, seed=0):
    self.experiment_seed = seed
    self.random = np.random.RandomState(self.experiment_seed)

    self.results = None

  def estimate_exploitability_sym(self, dist, num_eval_samples, num_ckpts,
                                  num_players, game, policies):
    """Estimate exploitability via monte carlo.

    Args:
      dist: 1-d np.array, estimate of nash distribution
      num_eval_samples: int, number of samples to estimate exploitability
      num_ckpts: int, number of checkpoints (actions, policies, ...)
      num_players: int, number of players
      game: game with minimal functionality (see games/small.py)
      policies: list mapping checkpoints to policies
    Returns:
      list of exploitabilities computed using [index] monte carlo samples
    """
    pg_mean = np.zeros_like(dist)
    exps_estimated = []
    for s in range(num_eval_samples):
      base_profile = tuple([
          self.random.choice(num_ckpts, p=dist) for _ in range(num_players)
      ])
      game_queries = sym_game_runner.construct_game_queries_for_exp(
          base_profile, num_ckpts)
      game_results = sym_game_runner.run_games_and_record_payoffs(
          game_queries, game.get_payoffs_for_strategies, policies)
      pg_s = np.zeros_like(dist)
      for query, payoffs in game_results.items():
        pg_s[query[0]] = payoffs[0]
      pg_mean = (pg_mean * float(s) + pg_s) / float(s + 1)
      exps_estimated.append(pg_mean.max() - pg_mean.dot(dist))

    return exps_estimated

  def estimate_exploitability_nonsym(self, dist, num_eval_samples, num_ckpts,
                                     num_players, game, policies):
    """Estimate exploitability via monte carlo.

    Args:
      dist: list of 1-d np.arrays, estimate of nash distribution
      num_eval_samples: int, number of samples to estimate exploitability
      num_ckpts: int, number of checkpoints (actions, policies, ...)
      num_players: int, number of players
      game: game with minimal functionality (see games/small.py)
      policies: list mapping checkpoints to policies
    Returns:
      list of exploitabilities computed using [index] monte carlo samples
    """
    pg_mean = [np.zeros_like(dist_i) for dist_i in dist]
    exps_estimated = []
    for s in range(num_eval_samples):
      base_profile = tuple([
          self.random.choice(num_ckpts[i], p=dist[i])
          for i in range(num_players)
      ])
      game_queries = nonsym_game_runner.construct_game_queries_for_exp(
          base_profile, num_ckpts)
      game_results = nonsym_game_runner.run_games_and_record_payoffs(
          game_queries, game.get_payoffs_for_strategies, policies)
      for pi_query, payoffs in game_results.items():
        pi, query = pi_query
        ai = query[pi]
        pg_mean[pi][ai] += (payoffs[pi] - pg_mean[pi][ai]) / float(s + 1)
      exp_is = []
      for i in range(num_players):
        exp_is.append(pg_mean[i].max() - pg_mean[i].dot(dist[i]))
      exps_estimated.append(np.mean(exp_is))

    return exps_estimated

  def update_payoff_matrices(self, payoff_matrices, payoff_matrices_new, s):
    """Update mean of payoff matrices.

    Args:
      payoff_matrices: dictionary with keys as tuples of agents (i, j) and
          values of (2 x A x A) np.arrays, payoffs for each joint action. keys
          are sorted and arrays should be indexed in the same order
          **current mean
      payoff_matrices_new: **new sample
      s: int, sample number
    Returns:
      payoff_matrices with updated means
    """
    if payoff_matrices:
      for key in payoff_matrices_new:
        new = payoff_matrices_new[key]
        old = payoff_matrices[key]
        payoff_matrices[key] += (new - old) / float(s + 1)
    else:
      payoff_matrices = payoff_matrices_new

    return payoff_matrices

  def construct_payoff_matrices_from_samples_sym(
      self, game, dist, num_samples, policies, num_players, num_ckpts):
    """Construct payoff matrices (approx. sym. polymatrix game) from samples.

    Args:
      game: game with minimal functionality (see games/small.py)
      dist: 1-d np.array, estimate of nash distribution
      num_samples: int, `minibatch' size for stochastic gradient
      policies: list mapping checkpoints to policies
      num_players: int, number of players
      num_ckpts: int, number of checkpoints (actions, policies, ...)
    Returns:
      payoff_matrices (2 x num_ckpts x num_ckpts array) to compute adidas grad
    """
    payoff_matrices = np.zeros((2, num_ckpts, num_ckpts))
    for _ in range(num_samples):
      base_profile = tuple([
          self.random.choice(num_ckpts, p=dist) for _ in range(num_players)
      ])
      game_queries = sym_game_runner.construct_game_queries(
          base_profile, num_ckpts)
      game_results = sym_game_runner.run_games_and_record_payoffs(
          game_queries, game.get_payoffs_for_strategies, policies)
      payoff_matrices += sym_game_runner.form_payoff_matrices(
          game_results, num_ckpts) / float(num_samples)
    return payoff_matrices

  def construct_payoff_matrices_exactly_sym(
      self, game, dist, num_players):
    """Construct payoff matrices exactly (expected sym. polymatrix game).

    Args:
      game: game with minimal functionality (see games/small.py)
      dist: 1-d np.array, estimate of nash distribution
      num_players: int, number of players
    Returns:
      payoff_matrices (2 x A x A array) to compute adidas gradient
    """
    sym_nash = [dist for _ in range(num_players)]
    pt = game.payoff_tensor()
    payoff_matrix_exp_0 = misc.pt_reduce(pt[0], sym_nash, [0, 1])
    payoff_matrix_exp_1 = misc.pt_reduce(pt[1], sym_nash, [0, 1])
    payoff_matrices = np.stack((payoff_matrix_exp_0, payoff_matrix_exp_1))
    return payoff_matrices

  def construct_payoff_matrices_from_samples_nonsym(
      self, game, dist, num_samples, policies, num_players, num_ckpts):
    """Construct payoff matrices (approx. nonsym. polymatrix) from samples.

    Args:
      game: game with minimal functionality (see games/small.py)
      dist: list of 1-d np.arrays, estimate of nash distribution
      num_samples: int, `minibatch' size for stochastic gradient
      policies: list mapping checkpoints to policies
      num_players: int, number of players
      num_ckpts: int, number of checkpoints (actions, policies, ...)
    Returns:
      payoff_matrices: dictionary with keys as tuples of agents (i, j) and
          values of (2 x A x A) np.arrays, payoffs for each joint action. keys
          are sorted and arrays should be indexed in the same order
    """
    payoff_matrices = None
    for s in range(num_samples):
      base_profile = tuple([
          self.random.choice(num_ckpts[i], p=dist[i])
          for i in range(num_players)
      ])
      game_queries = nonsym_game_runner.construct_game_queries(
          base_profile, num_ckpts)
      game_results = nonsym_game_runner.run_games_and_record_payoffs(
          game_queries, game.get_payoffs_for_strategies, policies)
      payoff_matrices_new = nonsym_game_runner.form_payoff_matrices(
          game_results, num_ckpts)
      payoff_matrices = self.update_payoff_matrices(payoff_matrices,
                                                    payoff_matrices_new,
                                                    s)
    return payoff_matrices

  def construct_payoff_matrices_exactly_nonsym(
      self, game, dist, num_players):
    """Construct payoff matrices exactly (expected nonsym. polymatrix game).

    Args:
      game: game with minimal functionality (see games/small.py)
      dist: list of 1-d np.arrays, estimate of nash distribution
      num_players: int, number of players
    Returns:
      payoff_matrices: dictionary with keys as tuples of agents (i, j) and
          values of (2 x A x A) np.arrays, payoffs for each joint action. keys
          are sorted and arrays should be indexed in the same order
    """
    pt = game.payoff_tensor()
    payoff_matrices = {}
    for pi, pj in itertools.combinations(range(num_players), 2):
      key = (pi, pj)
      pt_i = misc.pt_reduce(pt[pi], dist, [pi, pj])
      pt_j = misc.pt_reduce(pt[pj], dist, [pi, pj])
      payoff_matrices[key] = np.stack((pt_i, pt_j), axis=0)
    return payoff_matrices

  def approximate_nash(self, game, solver, sym,
                       num_iterations=10000, num_samples=1,
                       num_eval_samples=int(10e4), approx_eval=False,
                       exact_eval=False, avg_trajectory=False,
                       return_trajectory=False):
    """Runs solver on game.

    Args:
      game: game with minimal functionality (see games/small.py)
      solver: gradient solver (see utils/updates.py)
      sym: bool, true if the game is symmetric across players
      num_iterations: int, number of incremental updates
      num_samples: int, `minibatch' size for stochastic gradient
      num_eval_samples: int, number of samples to estimate exploitability
        default = # of samples for P[|sample_payoff-true| > C/100] < ~5e-7%
        where C = pt.max() - pt.min();
        P[|pt_grad|_inf <= C/100] > (1-5e-7)^num_actions
      approx_eval: bool, whether to evaluate exploitability during
        descent with stochastic samples
      exact_eval: bool, whether to evaluate exploitability during
        descent with exact expectation (req. full payoff tensor)
      avg_trajectory: bool, whether to evaluate w.r.t. the average distribution
        up to time t instead of the distribution at time t
      return_trajectory: bool, whether to record all parameters (e.g., dist)
        during learning and return them -- see solver code for details
    Returns:
      None -- dict of results stored in `results` attribute upon completion
        (key=name of metric, value=[m_0, ..., m_{last_iter}])
    """
    num_players = game.num_players()
    num_strats = game.num_strategies()

    if sym:
      if len(set(num_strats)) != 1:
        raise ValueError('Each player should have the same number of actions.')
      num_strats = num_strats[0]

    params = solver.init_vars(num_strats, num_players)  # dist = params[0]
    if sym:
      dist_avg = np.zeros_like(params[0])
      policies = list(range(num_strats))
      num_ckpts = len(policies)
      form_payoffs_appx = self.construct_payoff_matrices_from_samples_sym
      form_payoffs_exact = self.construct_payoff_matrices_exactly_sym
      exp = sym_exp
      estimate_exploitability = self.estimate_exploitability_sym
    else:
      dist_avg = [np.zeros_like(dist_i) for dist_i in params[0]]
      policies = [list(range(num_strats_i)) for num_strats_i in num_strats]
      num_ckpts = [len(policy_i) for policy_i in policies]
      form_payoffs_appx = self.construct_payoff_matrices_from_samples_nonsym
      form_payoffs_exact = self.construct_payoff_matrices_exactly_nonsym
      exp = nonsym_exp
      estimate_exploitability = self.estimate_exploitability_nonsym

    exps_exact = []
    exps_solver_exact = []
    exps_approx = []
    exps_solver_approx = []
    grad_norms = []

    if return_trajectory:
      params_traj = []

    has_temp = False
    if hasattr(solver, 'temperature') or hasattr(solver, 'p'):
      has_temp = True
      temperatures = []
      if hasattr(solver, 'temperature'):
        temp_attr = 'temperature'
      else:
        temp_attr = 'p'

    early_exit = False

    start = time.time()

    # search for nash (sgd)
    for t in range(num_iterations + 1):
      dist = params[0]
      if return_trajectory:
        params_traj.append(params)

      if return_trajectory:
        params_traj.append(params)

      if has_temp:
        temperatures.append(getattr(solver, temp_attr))

      if num_samples < np.inf:
        payoff_matrices = form_payoffs_appx(game, dist, num_samples,
                                            policies, num_players, num_ckpts)
      else:
        payoff_matrices = form_payoffs_exact(game, dist, num_players)

      grads, exp_sto, exp_solver_sto = solver.compute_gradients(params,
                                                                payoff_matrices)

      if sym:
        grads_dist = grads[0]
        grad_norms.append(simplex.grad_norm(dist, grads_dist))
      else:
        grad_norm = 0.
        grads_dist = grads[0]
        for dist_i, grads_i in zip(dist, grads_dist[0]):
          grad_norm += simplex.grad_norm(dist_i, grads_i)**2.
        grad_norm = np.sqrt(grad_norm)
        grad_norms.append(grad_norm)

      if solver.has_aux:
        solver.record_aux_errors(grads)

      if sym:
        dist_avg += (dist - dist_avg) / float(t + 1)
      else:
        for i, dist_i in enumerate(dist):
          dist_avg[i] += (dist_i - dist_avg[i]) / float(t + 1)

      if avg_trajectory:
        dist_eval = dist_avg
      else:
        dist_eval = dist

      if approx_eval:
        exps_approx.append(exp_sto)
        exps_solver_approx.append(exp_solver_sto)
      if exact_eval:
        pt = game.payoff_tensor()
        exps_exact.append(exp.unreg_exploitability(dist_eval, pt))
        exps_solver_exact.append(solver.exploitability(dist_eval, pt))

      # skip the last update so to avoid computing the matching exploitability
      # and gradient norm information outside the loop
      if t < num_iterations:
        params = solver.update(params, grads, t)
        if misc.isnan(params):
          print('Warning: NaN detected in params post-update. Exiting loop.')
          early_exit = True
          break

    end = time.time()
    solve_runtime = end - start
    start = end

    # evaluating exploitability (monte-carlo)
    exp_estimated = estimate_exploitability(dist_eval, num_eval_samples,
                                            num_ckpts, num_players,
                                            game, policies)

    eval_runtime = time.time() - start

    results = {'exps_approx': exps_approx,
               'exps_solver_approx': exps_solver_approx,
               'exps_exact': exps_exact,
               'exps_solver_exact': exps_solver_exact,
               'exp_estimated': exp_estimated,
               'grad_norms': grad_norms,
               'dist': dist,
               'dist_avg': dist_avg,
               'solve_runtime': solve_runtime,
               'eval_runtime': eval_runtime,
               'early_exit': early_exit}

    if solver.has_aux:
      results.update({'aux_errors': solver.aux_errors})

    if return_trajectory:
      results.update({'params_trajectory': params_traj})

    if has_temp:
      results.update({'temperatures': temperatures})

    self.results = results
