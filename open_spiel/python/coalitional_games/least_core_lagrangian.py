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

"""Methods to compute the core based on stochastic saddle point programming.

This file contains methods to compute the core using a Lagrangian formulation
referred to in Gemp et al AAMAS '24:
"Approximating the Core via Iterative Coalition Sampling"

TODO:
- add a link to arXiv when it's live 
- add the least core algorithm to the algorithms.md also when that link is live
"""

import dataclasses
import functools
import itertools
import time
from typing import Any, Dict, Tuple

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
import optax

from open_spiel.python.coalitional_games import coalitional_game


@dataclasses.dataclass(frozen=True)
class LeastCoreValue:
  payoff: np.ndarray
  lcv: float
  duration: float
  meta: Dict[Any, Any]


def compute_least_core_value(
    cvc: coalitional_game.CoalitionalGame, alg_config
) -> LeastCoreValue:
  """Computes the least core value of a game."""
  opt_primal = optax.adam(learning_rate=alg_config.init.lr_primal)
  opt_dual = optax.adam(learning_rate=alg_config.init.lr_dual)
  evaluation_iterations = alg_config.eval.evaluation_iterations
  evaluate_every = 2 * alg_config.solve.n_iter  # do not evaluate
  cl = CoreLagrangian(cvc, opt_primal, opt_dual)
  payoffs, epsilons, _, duration = cl.solve(
      evaluation_iterations=evaluation_iterations,
      evaluate_every=evaluate_every,
      **alg_config.solve,
  )
  lcvs = np.full(payoffs.shape[0], np.inf)
  payoff = payoffs[-1]
  lcv = np.inf
  for i in range(payoffs.shape[0]):
    payoff = payoffs[i]
    epsilon = epsilons[i]
    max_violation = payoff_evaluation(
        cvc, payoff, epsilon, evaluation_iterations)
    lcv = epsilon + max_violation
    lcvs[i] = lcv
  meta = dict(payoffs=payoffs, epsilons=epsilons, lcvs=lcvs)
  return LeastCoreValue(payoff, lcv, duration, meta)


def payoff_evaluation(
    cv_calc: coalitional_game.CoalitionalGame,
    payoffs: np.ndarray,
    epsilon: float,
    batch_size: int,
    max_exponent: int = 13,
) -> float:
  """Evaluate deficit over a set of random coalitions.

  Args:
    cv_calc: the game to work on
    payoffs: the payoff vector to test
    epsilon: desired approximation of the epsilon-core
    batch_size: number of random coalitions to sample
    max_exponent: examine at maxixum 2**max_exponent constraints in one batch
      default 13, assume 2**13 ~ 10k coalitions is mem limit for single batch

  Returns:
    Expected loss, relu(deficit), over random batch of coalitions
  """
  max_batch = 2**max_exponent
  num_players = cv_calc.num_players()
  violation = 0.
  if batch_size >= 2**num_players:
    num_suffix_repeats = min(max_exponent, num_players)
    num_prefix_repeats = max(0, num_players - num_suffix_repeats)
    zo = [0, 1]
    suffix = np.array(list(itertools.product(zo, repeat=num_suffix_repeats)))
    prefixes = itertools.product(zo, repeat=num_prefix_repeats)
    for prefix in prefixes:
      if prefix:
        prefix_rep = np.repeat([prefix], suffix.shape[0], axis=0)
        coalitions = np.concatenate([prefix_rep, suffix], axis=1)
      else:
        coalitions = suffix
      batch_contributions = cv_calc.coalition_values(coalitions)
      batch_payouts = np.dot(coalitions, payoffs)
      batch_deficits = batch_contributions - batch_payouts - epsilon
      batch_deficits = np.clip(batch_deficits, 0., np.inf)
      violation = max(violation, np.max(batch_deficits))
  else:
    q, r = divmod(batch_size, max_batch)
    num_loops = q + (r > 0)
    for _ in range(num_loops):
      coalitions = np.random.randint(2, size=(max_batch, num_players))
      batch_contributions = cv_calc.coalition_values(coalitions)
      batch_payouts = np.dot(coalitions, payoffs)
      batch_deficits = batch_contributions - batch_payouts - epsilon
      batch_deficits = np.clip(batch_deficits, 0., np.inf)
      violation = max(violation, np.max(batch_deficits))
  return float(violation)


class CoreSolver():
  """Find an epsilon-core."""

  def __init__(self,
               cvc: coalitional_game.CoalitionalGame):
    self.cvc = cvc
    self.num_players = cvc.num_players()
    # we assume grand_coalition is optimal coalition
    grand_coalition = np.full(cvc.num_players(), 1, dtype=np.int32)
    self.grand_coalition_value = cvc.coalition_value(grand_coalition)

    self.payoffs = None
    self.losses = None
    self.max_deficits = None
    self.evaluation_losses = None

  def logits_to_payoff(self, logits):
    logits_ext = jnp.append(logits, 0.)
    payoff = jax.nn.softmax(logits_ext)
    payoff *= self.grand_coalition_value
    return payoff

  def loss_deficit(self, current_payoff, coalitions, coalition_values, epsilon):
    """Compute Mean Loss and Max Deficit."""
    coalition_payment = jnp.dot(coalitions, current_payoff)
    deficit = coalition_values - epsilon - coalition_payment
    coalition_size = jnp.sum(coalitions, axis=1)
    weight = 1.0 / jnp.clip(coalition_size, 1, self.num_players)
    losses = 0.5 * weight * jax.nn.relu(deficit) ** 2.0
    return jnp.mean(losses, axis=0), jnp.max(jax.nn.relu(deficit))


class CoreOptimization(CoreSolver):
  """Find an epsilon-core via optimization."""

  def __init__(self,
               cvc: coalitional_game.CoalitionalGame,
               opt,
               epsilon):
    super().__init__(cvc)
    self.opt = opt
    self.epsilon = epsilon

  @functools.partial(jax.jit, static_argnums=[0])
  def loss(self, params, data):
    """Compute Loss."""
    current_payoff = params
    coalitions, coalition_values = data

    return self.loss_deficit(current_payoff, coalitions, coalition_values,
                             self.epsilon)

  @functools.partial(jax.jit, static_argnums=[0])
  def update_step(self, params, data, opt_state):
    """GD update step."""

    # data = (coalitions, coalition_values)

    # Convert losses into pure functions.
    loss_fn = lambda p: self.loss(p, data)[0]

    # Compute saddle-point gradients.
    grads_fn = jax.grad(loss_fn, argnums=0)
    grads = grads_fn(params)

    updates, opt_state = self.opt.update(grads, opt_state, params)

    params = optax.apply_updates(params, updates)

    params = jnp.clip(params, 0, np.inf)
    scale = self.grand_coalition_value / jnp.sum(params)
    params = params * scale

    return params, opt_state

  def solve(self, n_iter: int, batch_size: int = 100,
            save_every: int = 2,
            evaluate_every: int = 2, evaluation_iterations: int = 100,
            seed: int = 0
            ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Find a least-core via Lagrange multipliers.

    Additional optimization metrics are stored as class variables:
      self.payoffs
      self.losses
      self.max_deficits
      self.evaluation_losses
      self.duration

    Args:
      n_iter: number of iterations
      batch_size: number of constraints to evaluate at each step
      save_every: int, how often to record optimization variables
      evaluate_every: int, how often to evaluate the max constraint violation
      evaluation_iterations: int, how many constraints to measure violations
        for, if number if less than number of coalitions a batch of constraints
        is sampled randomly. otherwise, all constraints are evaluated
      seed: int, for sampling minibatches of constraints

    Returns:
      payoffs over training
      max deficit over training
      runtime duration (sec)
    """

    qe, re = divmod(n_iter, evaluate_every)
    num_eval = qe + (re > 0)
    qs, rs = divmod(n_iter, save_every)
    num_save = qs + (rs > 0)

    max_violations = np.empty(num_eval, dtype=np.float32)
    losses = np.empty(num_save, dtype=np.float32)
    max_deficits = np.empty(num_save, dtype=np.float32)
    payoffs = np.empty((num_save, self.num_players), dtype=np.float32)

    scale = self.grand_coalition_value / self.num_players
    grand_coalition = np.full(self.num_players, 1, dtype=np.int32)
    current_payoff = jnp.array(grand_coalition * scale)
    params = current_payoff

    opt_state = self.opt.init(params)

    logging.debug('Uniform payoff %s', current_payoff)

    rng = jax.random.PRNGKey(seed)

    start = time.time()
    for iter_id in range(n_iter):
      if batch_size < 2**self.num_players:
        rng, key = jax.random.split(rng, 2)
        coalitions = jax.random.randint(key,
                                        shape=(batch_size, self.num_players),
                                        minval=0,
                                        maxval=2,
                                        dtype=jnp.int32)
      else:
        prod_space = itertools.product([0, 1], repeat=self.num_players)
        coalitions = np.stack(list(prod_space))
      coalition_values = self.cvc.coalition_values(np.array(coalitions))

      data = (coalitions, coalition_values)
      loss, max_deficit = self.loss(params, data)
      params, opt_state = self.update_step(params, data, opt_state)

      # Done updating, save if needed
      if iter_id % save_every == 0:
        logging.debug('Saving...')
        idx = iter_id // save_every
        losses[idx] = loss
        max_deficits[idx] = max_deficit
        current_payoff = params
        payoffs[idx] = current_payoff
        logging.debug('Loss was %f, Max deficit was %f, New payoff %s',
                      loss, max_deficit, current_payoff)

      # Done updating, evaluate if needed
      if (evaluate_every < n_iter) and (iter_id % evaluate_every == 0):
        logging.debug('Evaluating...')
        estimated_loss = payoff_evaluation(
            self.cvc,
            current_payoff,
            self.epsilon,
            evaluation_iterations,
        )
        max_violations[iter_id // evaluate_every] = estimated_loss
        logging.debug('Estimated loss %f', estimated_loss)
    end = time.time()
    duration = end - start

    self.payoffs = np.array(payoffs)
    self.losses = np.array(losses)
    self.max_deficits = np.array(max_deficits)
    self.max_violations = np.array(max_violations)
    self.duration = duration

    return (np.array(payoffs),
            np.array(max_deficits),
            duration)


class CoreOptimizationLogits(CoreSolver):
  """Find an epsilon-core via optimization over logits."""

  def __init__(self,
               cvc: coalitional_game.CoalitionalGame,
               opt,
               epsilon):
    super().__init__(cvc)
    self.opt = opt
    self.epsilon = epsilon

  @functools.partial(jax.jit, static_argnums=[0])
  def loss(self, params, data):
    """Compute Loss."""
    current_payoff = self.logits_to_payoff(params)
    coalitions, coalition_values = data

    return self.loss_deficit(current_payoff, coalitions, coalition_values,
                             self.epsilon)

  @functools.partial(jax.jit, static_argnums=[0])
  def update_step(self, params, data, opt_state):
    """GD update step."""

    # data = (coalitions, coalition_values)

    # Convert losses into pure functions.
    loss_fn = lambda p: self.loss(p, data)[0]

    # Compute saddle-point gradients.
    grads_fn = jax.grad(loss_fn, argnums=0)
    grads = grads_fn(params)

    updates, opt_state = self.opt.update(grads, opt_state, params)

    params = optax.apply_updates(params, updates)

    return params, opt_state

  def solve(self, n_iter: int, batch_size: int = 100,
            save_every: int = 2,
            evaluate_every: int = 2, evaluation_iterations: int = 100,
            seed: int = 0
            ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Find a least-core via Lagrange multipliers.

    Additional optimization metrics are stored as class variables:
      self.payoffs
      self.losses
      self.max_deficits
      self.evaluation_losses
      self.duration

    Args:
      n_iter: number of iterations
      batch_size: number of constraints to evaluate at each step
      save_every: int, how often to record optimization variables
      evaluate_every: int, how often to evaluate the max constraint violation
      evaluation_iterations: int, how many constraints to measure violations
        for, if number if less than number of coalitions a batch of constraints
        is sampled randomly. otherwise, all constraints are evaluated
      seed: int, for sampling minibatches of constraints

    Returns:
      payoffs over training
      max deficit over training
      runtime duration (sec)
    """

    qe, re = divmod(n_iter, evaluate_every)
    num_eval = qe + (re > 0)
    qs, rs = divmod(n_iter, save_every)
    num_save = qs + (rs > 0)

    max_violations = np.empty(num_eval, dtype=np.float32)
    losses = np.empty(num_save, dtype=np.float32)
    max_deficits = np.empty(num_save, dtype=np.float32)
    payoffs = np.empty((num_save, self.num_players), dtype=np.float32)

    current_logits = jnp.zeros(self.num_players - 1, dtype=jnp.float32)
    current_payoff = np.asarray(self.logits_to_payoff(current_logits))
    params = current_logits

    opt_state = self.opt.init(params)

    logging.debug('Uniform payoff %s', current_payoff)

    rng = jax.random.PRNGKey(seed)

    start = time.time()
    for iter_id in range(n_iter):
      if batch_size < 2**self.num_players:
        rng, key = jax.random.split(rng, 2)
        coalitions = jax.random.randint(key,
                                        shape=(batch_size, self.num_players),
                                        minval=0,
                                        maxval=2,
                                        dtype=jnp.int32)
      else:
        prod_space = itertools.product([0, 1], repeat=self.num_players)
        coalitions = np.stack(list(prod_space))
      coalition_values = self.cvc.coalition_values(np.array(coalitions))

      data = (coalitions, coalition_values)
      loss, max_deficit = self.loss(params, data)
      params, opt_state = self.update_step(params, data, opt_state)

      # Done updating, save if needed
      if iter_id % save_every == 0:
        logging.debug('Saving...')
        idx = iter_id // save_every
        losses[idx] = loss
        max_deficits[idx] = max_deficit
        current_logits = params
        current_payoff = np.asarray(self.logits_to_payoff(current_logits))
        payoffs[idx] = current_payoff
        logging.debug('Loss was %f, Max deficit was %f, New payoff %s',
                      loss, max_deficit, current_payoff)

      # Done updating, evaluate if needed
      if (evaluate_every < n_iter) and (iter_id % evaluate_every == 0):
        logging.debug('Evaluating...')
        estimated_loss = payoff_evaluation(
            self.cvc,
            current_payoff,
            self.epsilon,
            evaluation_iterations,
        )
        max_violations[iter_id // evaluate_every] = estimated_loss
        logging.debug('Estimated loss %f', estimated_loss)
    end = time.time()
    duration = end - start
    self.payoffs = np.array(payoffs)
    self.losses = np.array(losses)
    self.max_deficits = np.array(max_deficits)
    self.max_violations = np.array(max_violations)
    self.duration = duration

    return (np.array(payoffs),
            np.array(max_deficits),
            duration)


class CoreLagrangian(CoreSolver):
  """Find a least-core via Lagrange multipliers."""

  def __init__(self,
               cvc: coalitional_game.CoalitionalGame,
               opt_primal,
               opt_dual):
    super().__init__(cvc)
    self.opt_primal = opt_primal
    self.opt_dual = opt_dual

    current_logits_keys = ['current_logits' for _ in range(self.num_players)]
    keys_primal = {'current_logits': current_logits_keys,
                   'epsilon': 'epsilon'}
    keys_dual = {'mu': 'mu'}
    self.keys = (keys_primal, keys_dual)
    self.nonnegative_keys = ('epsilon', 'mu')

    self.epsilons = None
    self.mus = None
    self.lagrangians = None

  @functools.partial(jax.jit, static_argnums=[0])
  def lagrangian(self, primal, dual, data):
    """Compute Lagrangian."""
    current_logits, epsilon = primal['current_logits'], primal['epsilon']
    mu = dual['mu']
    coalitions, coalition_values, gamma_adj = data

    current_payoff = self.logits_to_payoff(current_logits)
    mean_loss, max_deficit = self.loss_deficit(current_payoff,
                                               coalitions,
                                               coalition_values,
                                               epsilon)
    lagrangian = epsilon + mu * (mean_loss - gamma_adj)
    lagrangian = jnp.sum(lagrangian)  # just for converting (1,) array to scalar
    return lagrangian, (mean_loss, max_deficit)

  @functools.partial(jax.jit, static_argnums=[0])
  def update_step(self, params, data, opt_state):
    """SimGD update step."""

    # data = (coalitions, coalition_values, gamma_adj)
    params_primal, params_dual = params
    opt_state_primal, opt_state_dual = opt_state

    # Convert losses into pure functions.
    loss_primal_fn = lambda p, d: self.lagrangian(p, d, data)[0]
    loss_dual_fn = lambda p, d: -self.lagrangian(p, d, data)[0]

    # Compute saddle-point gradients.
    grads_primal_fn = jax.grad(loss_primal_fn, argnums=0)
    grads_primal = grads_primal_fn(params_primal, params_dual)
    grads_dual_fn = jax.grad(loss_dual_fn, argnums=1)
    grads_dual = grads_dual_fn(params_primal, params_dual)

    updates_primal, opt_state_primal = self.opt_primal.update(grads_primal,
                                                              opt_state_primal,
                                                              params_primal)
    updates_dual, opt_state_dual = self.opt_dual.update(grads_dual,
                                                        opt_state_dual,
                                                        params_dual)

    params_primal = optax.apply_updates(params_primal, updates_primal)
    params_dual = optax.apply_updates(params_dual, updates_dual)

    params = (params_primal, params_dual)
    opt_state = (opt_state_primal, opt_state_dual)

    clip = (
        lambda x, k: jnp.clip(x, 0, np.inf) if k in self.nonnegative_keys else x
    )
    params = jax.tree_util.tree_map(clip, params, self.keys)

    return params, opt_state

  def solve(self, n_iter: int, batch_size: int = 100, gamma: float = 1e-2,
            mu_init: float = 1000.,
            save_every: int = 2,
            evaluate_every: int = 2, evaluation_iterations: int = 100,
            seed: int = 0,
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Find a least-core via Lagrange multipliers.

    Additional optimization metrics are stored as class variables:
      self.payoffs
      self.epsilons
      self.mus
      self.lagrangians
      self.losses
      self.max_deficits
      self.evaluation_losses
      self.duration

    Args:
      n_iter: number of iterations
      batch_size: number of constraints to evaluate at each step
      gamma: float, slack allowed in core constraints
      mu_init: float, initialize the lagrange multiplier to this value
      save_every: int, how often to record optimization variables
      evaluate_every: int, how often to evaluate the max constraint violation
      evaluation_iterations: int, how many constraints to measure violations
        for, if number if less than number of coalitions a batch of constraints
        is sampled randomly. otherwise, all constraints are evaluated
      seed: int, for sampling minibatches of constraints

    Returns:
      payoffs over training
      epsilon over training
      max deficit over training
      runtime duration (sec)
    """

    qe, re = divmod(n_iter, evaluate_every)
    num_eval = qe + (re > 0)
    qs, rs = divmod(n_iter, save_every)
    num_save = qs + (rs > 0)

    max_violations = np.empty(num_eval, dtype=np.float32)
    lagrangians = np.empty(num_save, dtype=np.float32)
    losses = np.empty(num_save, dtype=np.float32)
    max_deficits = np.empty(num_save, dtype=np.float32)
    epsilons = np.empty(num_save, dtype=np.float32)
    payoffs = np.empty((num_save, self.num_players), dtype=np.float32)
    mus = np.empty(num_save, dtype=np.float32)

    current_logits = jnp.zeros(self.num_players - 1, dtype=jnp.float32)
    epsilon = self.grand_coalition_value * jnp.ones(1, dtype=jnp.float32)
    mu = jnp.ones(1, dtype=jnp.float32) * mu_init

    params_primal = {'current_logits': current_logits,
                     'epsilon': epsilon}
    params_dual = {'mu': mu}
    params = (params_primal, params_dual)

    opt_state_primal = self.opt_primal.init(params_primal)
    opt_state_dual = self.opt_dual.init(params_dual)
    opt_state = (opt_state_primal, opt_state_dual)

    current_payoff = np.asarray(self.logits_to_payoff(current_logits))
    logging.debug('Uniform payoff %s', current_payoff)

    if self.num_players < 30:
      gamma_adj = gamma**2.0 / (2**self.num_players - 1)
    else:
      # Set arbitrary value if the above would result in a too tiny number.
      gamma_adj = 1e-6

    rng = jax.random.PRNGKey(seed)

    start = time.time()
    for iter_id in range(n_iter):
      if batch_size < 2**self.num_players:
        rng, key = jax.random.split(rng, 2)
        coalitions = jax.random.randint(key,
                                        shape=(batch_size, self.num_players),
                                        minval=0,
                                        maxval=2,
                                        dtype=jnp.int32)
      else:
        prod_space = itertools.product([0, 1], repeat=self.num_players)
        coalitions = np.stack(list(prod_space))
      coalition_values = self.cvc.coalition_values(np.array(coalitions))

      data = (coalitions, coalition_values, gamma_adj)
      lagrangian, (loss, max_deficit) = self.lagrangian(*params, data)
      params, opt_state = self.update_step(params, data, opt_state)

      params_primal, params_dual = params

      # Done updating, save if needed
      if iter_id % save_every == 0:
        logging.debug('Saving...')
        idx = iter_id // save_every
        lagrangians[idx] = lagrangian
        losses[idx] = loss
        max_deficits[idx] = max_deficit
        epsilons[idx] = params_primal['epsilon'].item()
        mus[idx] = params_dual['mu'].item()
        current_payoff = np.asarray(self.logits_to_payoff(
            params_primal['current_logits']))
        payoffs[idx] = current_payoff
        logging.debug('Loss was %f, Max deficit was %f, New payoff %s',
                      loss, max_deficit, current_payoff)

      # Done updating, evaluate if needed
      if (evaluate_every < n_iter) and (iter_id % evaluate_every == 0):
        logging.debug('Evaluating...')
        estimated_loss = payoff_evaluation(
            self.cvc,
            current_payoff,
            params_primal['epsilon'].item(),
            evaluation_iterations,
        )
        max_violations[iter_id // evaluate_every] = estimated_loss
        logging.debug('Estimated loss %f', estimated_loss)
    end = time.time()
    duration = end - start

    self.payoffs = np.array(payoffs)
    self.epsilons = np.array(epsilons)
    self.mus = np.array(mus)
    self.lagrangians = np.array(lagrangians)
    self.losses = np.array(losses)
    self.max_deficits = np.array(max_deficits)
    self.max_violations = np.array(max_violations)
    self.duration = duration

    return (np.array(payoffs),
            np.array(epsilons),
            np.array(max_deficits),
            duration)
