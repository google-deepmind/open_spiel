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

"""Mean-Field Bandit Regret Minimizers from Muller et al."""

from typing import Optional

import numpy as np
import scipy.optimize
import scipy.sparse.linalg

from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms import utils


# pylint: disable=invalid-name
def get_proba_constraints_positivity(nus):
  A = np.zeros((nus.shape[0], 1 + nus.shape[0]))
  A[:, 1:] = -np.eye(nus.shape[0])
  return A, np.zeros(A.shape[0])


def get_proba_constraint_sum_eq(nus):
  A = np.ones((1, 1 + nus.shape[0]))
  A[0, 0] = 0.0
  return A, np.array([1.0])


def compress_internal_weights(nus, regrets):
  """Compress internal weights.

  Via optimization, identify which regret timesteps are useful and which aren't
  for internal regret.

  Args:
    nus: Distribution per timestep.
    regrets: Regret value per timestep and action.

  Returns:
    Weights over nus which can be used to average the no-regret distribution.
  """

  def get_c(nus):
    return np.concatenate((np.array([1.0]), np.zeros(nus.shape[0])))

  def get_max_constraint(regrets):
    regrets = np.transpose(np.array(regrets), axes=[0, 2, 1])
    regrets = regrets.reshape(-1, regrets.shape[-1])
    A = np.zeros((regrets.shape[0], 1 + regrets.shape[1]))
    A[:, 1:] = regrets
    A[:, 0] = -1.0

    b = np.zeros(A.shape[0])
    return A, b

  def get_a_ub(nus, regrets):
    Amax, bmax = get_max_constraint(regrets)
    Apos, bpos = get_proba_constraints_positivity(nus)
    return np.concatenate((Amax, Apos), axis=0), np.concatenate(
        (bmax, bpos), axis=0
    )

  c = get_c(nus)

  A_ub, b_ub = get_a_ub(nus, regrets)
  A_eq, b_eq = get_proba_constraint_sum_eq(nus)

  res = scipy.optimize.linprog(
      c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, options={"tol": 1e-10}
  )
  new_weights = res.x
  return new_weights[1:]


def compress_external_weights(nus, regrets, lbd=0.0):
  """Compress internal weights.

  Via optimization, identify which regret timesteps are useful and which aren't
  for external regret.

  Args:
    nus: Distribution per timestep.
    regrets: Regret value per timestep and action.
    lbd: Sparsity penalty.

  Returns:
    Weights over nus which can be used to average the no-regret distribution.
  """

  def get_c(nus):
    return np.concatenate((np.array([1.0]), np.zeros(nus.shape[0])))

  def get_max_constraints(nus, regrets, lbd):
    A = np.zeros((regrets.shape[1], 1 + nus.shape[0]))
    A[:, 0] = -1.0
    A[:, 1:] = np.transpose(
        regrets
        - np.sum(regrets * nus, axis=1).reshape(-1, 1)
        - lbd * np.abs(regrets)
    )
    return A, np.zeros(A.shape[0])

  def get_a_ub(nus, regrets, lbd):
    Amax, bmax = get_max_constraints(nus, regrets, lbd)
    Apos, bpos = get_proba_constraints_positivity(nus)
    return np.concatenate((Amax, Apos), axis=0), np.concatenate(
        (bmax, bpos), axis=0
    )

  c = get_c(nus)

  A_ub, b_ub = get_a_ub(nus, regrets, lbd)
  A_eq, b_eq = get_proba_constraint_sum_eq(nus)

  res = scipy.optimize.linprog(
      c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, options={"tol": 1e-10}
  )
  new_weights = res.x
  return new_weights[1:]


# Faster than using scipy.linalg.eig.
def power_method(w_nus):
  """Quick implementation of the power method.

  Args:
    w_nus:

  Returns:
    Highest eigenvalue of the system.

  Raises:
    ValueError: when the power method did not converge after 10.000 trials.
  """
  p = np.ones(len(w_nus))
  pprime = np.dot(p, w_nus)
  n_trials = 10000
  i = 0
  while np.sum(np.abs(pprime - p)) > 1e-8 and i < n_trials:
    p = pprime
    pprime = np.dot(p, w_nus)
    pprime[pprime < 0] = 0.0
    pprime /= np.sum(pprime)
    i += 1

  if np.sum(np.abs(pprime - p)) > 1e-8 and i >= n_trials:
    raise ValueError(
        "Power method did not converge after {} trials.".format(n_trials)
    )

  p[p < 0] = 0.0
  return p / np.sum(p)


class RegretMinimizer(object):
  """Base class for Regret Minimizers.

  Implements base functions for regret minimizers to implement.

  Attributes:
    _game: Pyspiel game.
    _regret_steps_per_step: Number of regret steps per `step` call (Maximum
      number in case `stop_early` is true)
    _rho_tol: If `_compress_nus` is true, minimum probability threshold (
      Probabilities below `rho_tol` will be filtered out).
    _compress_nus: Whether to compress nus (Remove nus with low selection
      probability) or not.
    _compress_lbd: Penalty term in L1 minimization when compressing nus.
    _stop_early: Whether to stop regret computation when average regret is lower
      than `_stop_regret_threshold` or to keep going until
      `_regret_steps_per_step` steps have been accomplished.
    _stop_regret_threshold: If `stop_early` is true, average regret threshold
      under which the algorithm will stop.
    _policies: List of Policies
    _value_estimator: Value estimation function.
    _value_estimation_n: Number of runs to average _value_estimator's result on.
  """

  def __init__(
      self,
      game,
      policies,
      regret_steps_per_step: int = 1,
      rho_tol: float = 1e-4,
      compress_nus: bool = True,
      compress_every: int = 1,
      compress_lbd: float = 0.0,
      stop_early: bool = True,
      stop_regret_threshold: float = 1e-3,
      value_estimator=utils.sample_value,
      value_estimation_n: int = 1,
      compute_internal_regret: bool = False,
  ):
    self._game = game
    self._regret_steps_per_step = regret_steps_per_step

    self._compress_nus = compress_nus
    self._compress_every = compress_every
    self._compress_lbd = compress_lbd

    self._stop_early = stop_early
    self._stop_regret_threshold = stop_regret_threshold

    self._rho_tol = rho_tol
    self._policies = policies

    self._value_estimator = value_estimator
    self._value_estimation_n = value_estimation_n

    self._compute_internal_regret = compute_internal_regret

  def update_policy_mus(self):
    """Update the stored distributions of our policies."""
    self._policy_mus = [
        distribution.DistributionPolicy(self._game, policy)
        for policy in self._policies
    ]

  def get_nu(self):
    """Returns current Population Distribution."""
    raise NotImplementedError

  def step(self):
    """Make a regret minimization step."""
    raise NotImplementedError

  def step_for(self, T):
    """Do `T` steps."""
    raise NotImplementedError

  def compute_average_regret(self):
    raise NotImplementedError

  def compute_regrets(self):
    raise NotImplementedError

  def reset(self, policies):
    """Restart the bandit with new policies."""
    raise NotImplementedError


def polynomial_weight_update(weights, rewards, eta):
  return weights * (1 + eta * rewards)


class PolynomialWeightAlgorithm(RegretMinimizer):
  """Implements the Polynomial Weight Algorithm Regret minimizer.

  This is an external-regret minimizer, adapted here to the Mean-Field,
  Partially-Observable case.
  """

  def __init__(
      self,
      game,
      policies,
      eta: Optional[float] = None,
      regret_steps_per_step: int = 1,
      rho_tol: float = 1e-4,
      compress_nus: bool = True,
      compress_every: int = 1,
      compress_lbd: float = 0.0,
      stop_early: bool = True,
      stop_regret_threshold: float = 1e-3,
      value_estimator=utils.sample_value,
      value_estimation_n: int = 1,
      compute_internal_regret: bool = False,
  ):
    super().__init__(
        game,
        policies,
        regret_steps_per_step=regret_steps_per_step,
        rho_tol=rho_tol,
        compress_nus=compress_nus,
        compress_every=compress_every,
        compress_lbd=compress_lbd,
        stop_early=stop_early,
        stop_regret_threshold=stop_regret_threshold,
        value_estimator=value_estimator,
        value_estimation_n=value_estimation_n,
        compute_internal_regret=compute_internal_regret,
    )
    if self._compute_internal_regret:
      self._ws = [np.ones(len(policies)) for _ in range(len(policies))]
      self._p = np.ones(len(policies)) / (1.0 * len(policies))
    else:
      self._w = np.ones(len(policies))

    if eta is None:
      assert regret_steps_per_step is not None, (
          "Both `eta` and "
          "`regret_steps_per_step` were "
          "None, whereas our algorithm "
          "requires either value to be "
          "set."
      )
      self.compute_optimal_eta()
    else:
      self._eta = eta

    self._nus = []
    self._rewards = []
    self._policy_mus = []
    self._nu_weights = []

  def get_all_w_nus(self):
    assert self._compute_internal_regret
    return [w / np.sum(w) for w in list(self._ws)]

  def get_nu(self):
    if self._compute_internal_regret:
      return np.sum(
          self._p.reshape(-1, 1) * np.array(self.get_all_w_nus()), axis=0
      )
    else:
      return self._w / np.sum(self._w)

  def compute_p(self):
    assert (
        self._compute_internal_regret
    ), "`p` does not exist when computing external regret."
    w_nus = np.array(self.get_all_w_nus())

    p = power_method(w_nus)
    self._p = p

  def _update_weights(self, rewards):
    if self._compute_internal_regret:
      self._ws = [
          w * (1 + self._eta * rewards * p) for w, p in zip(self._ws, self._p)
      ]
      self.compute_p()
    else:
      self._w = self._w * (1 + self._eta * rewards)

  def step(self):
    rewards = np.zeros(len(self._policies))
    nu = self.get_nu()
    self._nus.append(nu)
    self._nu_weights = list(self._nu_weights)
    self._nu_weights.append(1.0)

    mu = utils.MixedDistribution(self._policy_mus, nu)
    for _ in range(self._value_estimation_n):
      for index, policy in enumerate(self._policies):
        rewards[index] += self._value_estimator(policy, mu, self._game)
    rewards /= self._value_estimation_n

    self._update_weights(rewards)
    self._rewards.append(rewards)

  def step_for(self, T):
    if self._compute_internal_regret:
      print("Minimizing Internal Regret")
    else:
      print("Minimizing External Regret")
    for t in range(T):
      self.step()
      if self._stop_early and (t % self._compress_every == 0):
        try:
          regret, weights = self.get_post_compression_regret_and_weights()
          # print("{}".format(regret))
          assert np.abs(np.sum(weights) - 1.0) < 1e-8
        except:  # pylint: disable=bare-except
          print("Simplex method encountered an error.")
          continue
        if regret < self._stop_regret_threshold:
          break
    self.compress_nus_and_weights(weights)

  def get_post_compression_regret_and_weights(self):
    """Compress the regret and weights."""
    if self._compute_internal_regret:
      nu_weights = compress_internal_weights(
          self.get_nus(), self.compute_regrets()
      )
      regret = np.max([
          np.max(np.sum(nu_weights.reshape(-1, 1) * a, axis=0))
          for a in self.compute_regrets()
      ])
    else:
      nu_weights = compress_external_weights(
          self.get_nus(), self.compute_regrets(), lbd=self._compress_lbd
      )
      regret = np.max(
          np.sum(nu_weights.reshape(-1, 1) * self.compute_regrets(), axis=0)
      )
    return regret, nu_weights

  def compress_nus_and_weights(self, nu_weights):
    """Run L1 optimization to only keep important members of `nus`."""
    if self._compress_nus:
      try:
        assert np.abs(np.sum(nu_weights) - 1.0) < 1e-8
      except:  # pylint: disable=bare-except
        # If the optimization was unsuccessful, do *not* compress.
        return

      new_nus = [
          nu
          for weight, nu in zip(nu_weights, self._nus)
          if weight > self._rho_tol
      ]
      new_rewards = [
          reward
          for weight, reward in zip(nu_weights, self._rewards)
          if weight > self._rho_tol
      ]
      new_nu_weights = [
          weight for weight in nu_weights if weight > self._rho_tol
      ]
      new_nu_weights = np.array(new_nu_weights) / np.sum(new_nu_weights)

      self._nus = new_nus
      self._rewards = new_rewards
      self._nu_weights = new_nu_weights

  def normalize_nu_weights(self):
    self._nu_weights = np.array(self._nu_weights) / np.sum(self._nu_weights)

  def get_normalized_nu_weights(self):
    return np.array(self._nu_weights) / np.sum(self._nu_weights)

  def compute_regrets(self):
    if self._compute_internal_regret:
      regrets = []
      nus = np.array(self._nus)
      rewards = np.array(self._rewards)
      for action in range(rewards.shape[1]):
        on_policy_values = (rewards[:, action] * nus[:, action]).reshape(-1, 1)
        action_values = rewards * nus[:, action].reshape(-1, 1)
        regrets.append(action_values - on_policy_values)
    else:
      on_policy_value = np.sum(
          self._rewards * np.array(self._nus), axis=1, keepdims=True
      )
      policy_value = self._rewards
      regrets = policy_value - on_policy_value
    return regrets

  def compute_average_regret(self):
    nu_weights = self.get_normalized_nu_weights()
    if self._compute_internal_regret:
      regrets = 0.0
      nus = np.array(self._nus)
      rewards = np.array(self._rewards)
      for action in range(rewards.shape[1]):
        on_policy_values = (rewards[:, action] * nus[:, action]).reshape(-1, 1)
        action_values = rewards * nus[:, action].reshape(-1, 1)
        regrets += np.max(
            np.sum(
                nu_weights.reshape(-1, 1) * (action_values - on_policy_values),
                axis=0,
            )
        )
    else:
      regrets = np.sum(
          nu_weights.reshape(-1, 1) * self.compute_regrets(), axis=0
      )
    return np.max(regrets) / len(self._nus)

  def get_nus(self):
    return np.array(self._nus)

  def get_mus(self):
    mus = []
    for nu in self._nus:
      mu = utils.MixedDistribution(self._policy_mus, nu)
      mus.append(mu)
    return mus

  def get_rewards(self):
    return self._rewards

  def get_mus_and_weights(self):
    mus = self.get_mus()
    self.normalize_nu_weights()
    return mus, self._nu_weights

  def compute_optimal_eta(self):
    if self._regret_steps_per_step is not None:
      self._eta = min(
          np.sqrt(np.log(len(self._policies)) / self._regret_steps_per_step),
          0.5,
      )

  def reset(self, policies):
    if self._compute_internal_regret:
      self._ws = [np.ones(len(policies)) for _ in range(len(policies))]
      self._p = np.ones(len(policies)) / (1.0 * len(policies))
    else:
      self._w = np.ones(len(policies))
    self._policies = policies
    self._nus = []
    self._rewards = []
    self._policy_mus = []
    self._nu_weights = []
    self.update_policy_mus()
    self.compute_optimal_eta()


class Hedge(PolynomialWeightAlgorithm):
  """Hedge algorithm implementation."""

  def __init__(
      self,
      game,
      policies,
      eta: Optional[float] = None,
      regret_steps_per_step: int = 1,
      rho_tol: float = 1e-4,
      compress_nus: bool = True,
      compress_lbd: float = 0.0,
      compress_every: int = 1,
      stop_early: bool = True,
      stop_regret_threshold: float = 1e-3,
      value_estimator=utils.sample_value,
      value_estimation_n: int = 1,
      compute_internal_regret: bool = False,
  ):
    super().__init__(
        game,
        policies,
        eta=eta,
        regret_steps_per_step=regret_steps_per_step,
        rho_tol=rho_tol,
        compress_nus=compress_nus,
        compress_lbd=compress_lbd,
        stop_early=stop_early,
        stop_regret_threshold=stop_regret_threshold,
        value_estimator=value_estimator,
        value_estimation_n=value_estimation_n,
        compute_internal_regret=compute_internal_regret,
    )

    if self._compute_internal_regret:
      self._ws = [np.ones(len(policies)) for _ in range(len(policies))]
      self._p = np.ones(len(policies)) / (1.0 * len(policies))
    else:
      self._w = np.ones(len(policies))

    if eta is None:
      assert regret_steps_per_step is not None, (
          "Both `eta` and "
          "`regret_steps_per_step` were "
          "None, whereas our algorithm "
          "requires either value to be "
          "set."
      )
      self.compute_optimal_eta()
    else:
      self._eta = eta

    self._compress_every = compress_every

    self._nus = []
    self._rewards = []
    self._policy_mus = []
    self._nu_weights = []

  def _update_weights(self, rewards):
    if self._compute_internal_regret:
      self._ws = [
          w * np.exp(self._eta * rewards * p) for w, p in zip(self._ws, self._p)
      ]
      self.compute_p()
    else:
      self._w = self._w * np.exp(self._eta * rewards)
