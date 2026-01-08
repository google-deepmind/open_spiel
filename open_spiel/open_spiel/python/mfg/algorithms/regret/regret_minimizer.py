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

"""Base class for regret minimizers."""

import numpy as np

from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms import utils
from open_spiel.python.mfg.algorithms.regret import c_ce_optimization


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

    self._nus = []
    self._rewards = []
    self._true_rewards = []
    self._policy_mus = []
    self._nu_weights = []

  def update_policy_mus(self):
    """Update the stored distributions of our policies."""
    self._policy_mus = [
        distribution.DistributionPolicy(self._game, policy)
        for policy in self._policies
    ]

  def get_nu(self):
    """Returns current Population Distribution."""
    raise NotImplementedError

  def step(self, welfare_bonus=0.0):
    raise NotImplementedError

  def step_for(
      self,
      T,  # pylint: disable=invalid-name
      initial_welfare_bonus=None,
      welfare_decay=None,
      use_true_rewards_when_compressing=True,
  ):
    """Call `step` method `T` times maximum, potentially stop early.

    Args:
      T: Maximum number of `step` calls to run.
      initial_welfare_bonus: How much to initially reward high-welfare-inducing
        actions.
      welfare_decay: Welfare decay term.
      use_true_rewards_when_compressing: Compress and compute optimal (C)CE
        according to true rewards (= True) or according to modified rewards (=
        False)
    """
    welfare_bonus = 0.0
    if initial_welfare_bonus is not None:
      assert welfare_decay is not None
      welfare_bonus = initial_welfare_bonus

    weights = None
    for t in range(T):
      if welfare_decay is not None:
        welfare_bonus = max(0.0, welfare_bonus - welfare_decay * t / T)
      self.step(welfare_bonus=welfare_bonus)
      if self._stop_early and (t % self._compress_every == 0):
        try:
          regret, weights = self.get_post_compression_regret_and_weights(
              use_true_rewards_when_compressing=use_true_rewards_when_compressing
          )
          # print("\t\t{}".format(regret))
          assert np.abs(np.sum(weights) - 1.0) < 1e-8, np.sum(weights)
        except:  # pylint: disable=bare-except
          print("Simplex method encountered an error.")
          continue
        if regret < self._stop_regret_threshold:
          break
    if weights is None and self._compress_nus:
      regret, weights = self.get_post_compression_regret_and_weights(
          use_true_rewards_when_compressing=use_true_rewards_when_compressing
      )
    if self._compress_nus:
      self.compress_nus_and_weights(weights)

  def get_post_compression_regret_and_weights(
      self, use_true_rewards_when_compressing=True
  ):
    """Computes optimized (C)CE by varying the temporal weight on each `nu`.

    Args:
      use_true_rewards_when_compressing: compute optimal (C)CE according to true
        rewards (= True) or according to modified rewards (= False)

    Returns:
      Regret for new temporal weights, new temporal weights
    """
    if self._compute_internal_regret:
      nu_weights = c_ce_optimization.compress_internal_weights(
          self.get_nus(),
          self.compute_regrets(
              use_true_rewards=use_true_rewards_when_compressing
          ),
          rewards=self._rewards,
          lbd=self._compress_lbd,
      )
      regret = np.max([
          np.max(np.sum(nu_weights.reshape(-1, 1) * a, axis=0))
          for a in self.compute_regrets(
              use_true_rewards=use_true_rewards_when_compressing
          )
      ])
    else:
      nu_weights = c_ce_optimization.compress_external_weights(
          self.get_nus(),
          self.compute_regrets(
              use_true_rewards=use_true_rewards_when_compressing
          ),
          rewards=self._rewards,
          lbd=self._compress_lbd,
      )
      regret = np.max(
          np.sum(
              nu_weights.reshape(-1, 1)
              * self.compute_regrets(
                  use_true_rewards=use_true_rewards_when_compressing
              ),
              axis=0,
          )
      )
    return regret, nu_weights

  def compress_nus_and_weights(self, nu_weights):
    """Run L1 optimization to only keep important members of `nus`."""
    if self._compress_nus:
      if np.abs(np.sum(nu_weights) - 1.0) > 1e-8:
        # If the optimization was unsuccessful, do *not* compress.
        print(
            "Unsuccessful optimization, weights sum to {}".format(
                np.sum(nu_weights)
            )
        )
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
      new_true_rewards = [
          reward
          for weight, reward in zip(nu_weights, self._true_rewards)
          if weight > self._rho_tol
      ]

      new_nu_weights = [
          weight for weight in nu_weights if weight > self._rho_tol
      ]
      new_nu_weights = np.array(new_nu_weights) / np.sum(new_nu_weights)

      self._nus = new_nus
      self._rewards = new_rewards
      self._true_rewards = new_true_rewards
      self._nu_weights = new_nu_weights

  def reset(self, policies):
    """Restart the bandit with new policies."""
    raise NotImplementedError

  def increase_precision_x_fold(self, x):
    self._stop_regret_threshold /= x
    self._rho_tol /= x
    self._regret_steps_per_step *= x

  def compute_p(self):
    """Computes `p` as presented in Blum's External to Internal Regret."""
    assert (
        self._compute_internal_regret
    ), "`p` does not exist when computing external regret."
    w_nus = np.array(self.get_all_w_nus())

    p = np.ones(len(self._policies))
    pprime = np.dot(p, w_nus)
    n_trials = 100000
    i = 0
    while np.sum(np.abs(pprime - p)) > 1e-8 and i < n_trials:
      p = pprime
      pprime = np.dot(p, w_nus)
      i += 1

    if np.sum(np.abs(pprime - p)) > 1e-8 and i >= n_trials:
      raise ValueError(
          "Power method did not converge after {} trials.".format(n_trials)
      )
    self._p = p / np.sum(p)

  def get_all_w_nus(self):
    """returns all nus for all times and all policies."""
    raise NotImplementedError

  def compute_regrets(self, use_true_rewards=False):
    """Computes the algorithm's current external/internal regrets.

    Args:
      use_true_rewards: Whether to use altered game rewards, or true game
        rewards.

    Returns:
      Internal regret of shape [T, P, P] if `self._compute_internal_regret` is
        true, otherwise external regret of shape [T, P], where T is the current
        number of iterations and P the number of policies.
    """
    if use_true_rewards:
      rewards = self._true_rewards
    else:
      rewards = self._rewards

    if self._compute_internal_regret:
      regrets = []
      nus = np.array(self._nus)
      rewards = np.array(rewards)
      for action in range(rewards.shape[1]):
        on_policy_values = (rewards[:, action] * nus[:, action]).reshape(-1, 1)
        action_values = rewards * nus[:, action].reshape(-1, 1)
        regrets.append(action_values - on_policy_values)
    else:
      on_policy_value = np.sum(
          rewards * np.array(self._nus), axis=1, keepdims=True
      )
      policy_value = rewards
      regrets = policy_value - on_policy_value
    return regrets

  def compute_average_regret(self, use_true_rewards=True):
    """Computes the algorithm's average external/internal regrets.

    Args:
      use_true_rewards: Whether to use altered game rewards, or true game
        rewards.

    Returns:
      Internal regret if `self._compute_internal_regret` is true, otherwise
        external regret.
    """

    if use_true_rewards:
      rewards = self._true_rewards
    else:
      rewards = self._rewards

    nu_weights = self.get_normalized_nu_weights()
    if self._compute_internal_regret:
      regrets = 0.0
      nus = np.array(self._nus)
      rewards = np.array(rewards)
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
          nu_weights.reshape(-1, 1)
          * self.compute_regrets(use_true_rewards=use_true_rewards),
          axis=0,
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

  def normalize_nu_weights(self):
    self._nu_weights = np.array(self._nu_weights) / np.sum(self._nu_weights)

  def get_normalized_nu_weights(self):
    return np.array(self._nu_weights) / np.sum(self._nu_weights)

  def restart(self):
    self._nu_weights = list(self._nu_weights)
