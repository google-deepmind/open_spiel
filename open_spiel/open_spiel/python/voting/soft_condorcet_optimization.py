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

"""Find ratings by minimizing a differentiable Kendall-tau distance.

This is an idea for an optimization algorithm motivated by work on
voting-as-evaluation (VasE); see go/gdm-vase.  Please contact lanctot@ if you
have any comments or questions.

This method was first inspired by Condorcet's vision that there is an underlying
true ranking and that voting rules are simply noisy estimates of the ground
truth rank. As shown in Section 8.3 of the Handbook of Computational Social
Choice (https://www.cse.unsw.edu.au/~haziz/comsoc.pdf) the maximum likelihood
estimate leads to a ranking that minimized the Kendall-tau distance to the
votes, which is precisely what the Kemeny-Young voting method does. However,
its complexity is O(m!) where m is the number of alternatives. Also, it is not
clear how the social choice methods handle sparsity in the ranking data.

This method, Soft Condorcet Optimization (SCO), assigns a numerical rating to
each alternative: r_i, and defines the loss to be:

    sum_{v in Votes} sum_{alternatives a, b in v s.t. a > b} D(r_a, r_b)

If D is defined to be:
        0  if r_a - r_b > 0    (r_a > r_b -> correct ordering)
  or    1  otherwise           (r_b >= r_a -> incorrect ordering)

then the loss above is the sum of Kendall-tau distances and the minimum
corresponds to the solution Kemeny-Young would find. But, it's not
differentiable. In SCO, we replace D by a sigmoid (smooth step function).

   D(r_a, r_b) = sigmoid((r_b - r_a) / tau)
               = sigmoid(Delta_{ab})

where sigmoid(x) = 1.0 / (1.0 + exp(-x)). The partial derivatives of D(r_a, r_b)
  - w.r.t r_a: is sigmoid(Delta_{ab}) (1 - sigmoid(Delta_{ab})) (-1/tau)
  - w.r.t r_b: is sigmoid(Delta_{ab}) (1 - sigmoid(Delta_{ab})) (1/tau).

which makes the losses easy to compute for any batch of votes.

We call this loss the "sigmoid loss", and it is implemented in the
SoftCondorcetOptimizer class. There is also the Fenchel-Young loss, as described
in Section 3.3 of the paper, which uses a similar gradient descent form but
optimizes a different loss based on perturbed optimizers in machine learning.
The optimizer using the Fenchel-Young loss is implemented in the
FenchelYoungOptimizer class.

Note: this python implementation was the one used for the results in the
original paper. For a faster version, see the C++ implementation in
evaluation/soft_condorcet_optimization.h which is exposed via python bindings
in pyspiel (for an example use from Python, see voting/examples/atari.py).
"""

import abc
import collections
from absl import logging
import numpy as np
from open_spiel.python.voting import base


class Optimizer(abc.ABC):
  """Optimizer without a gradient."""

  def __init__(
      self,
      profile: base.PreferenceProfile,
      batch_size: int,
      rating_lower_bound: float,
      rating_upper_bound: float,
      compute_norm_freq: int,
      initial_param_noise: float,
      verbose: bool = False,
  ):
    self._verbose = verbose
    self._profile = profile
    # Ungroup the profile (make all the votes have weight 1) to make it easier
    # to sample from when assembling batches.
    self._profile.ungroup()
    self._num_alternatives = self._profile.num_alternatives()
    if rating_upper_bound <= rating_lower_bound:
      raise ValueError(
          f"Upper bound ({rating_upper_bound}) must be higher than lower"
          f" bound ({rating_lower_bound})."
      )

    self._rating_ub = rating_upper_bound
    self._rating_lb = rating_lower_bound
    self._batch_size = batch_size
    self._compute_norm_freq = compute_norm_freq
    midpoint_rating = (
        self._rating_ub - self._rating_lb
    ) / 2.0 + self._rating_lb
    self._ratings = np.ones(self._profile.num_alternatives(), dtype=np.float32)
    self._ratings.fill(midpoint_rating)
    self._initial_noise = np.zeros(self._num_alternatives, dtype=np.float32)
    if initial_param_noise > 0.0:
      self._initial_noise = (
          np.random.rand(self._num_alternatives) * initial_param_noise
      )
      self._ratings = self._ratings + self._initial_noise
    self._avg_l2_grad_norm = 0
    self._avg_l1_sum_grad_norm = 0
    self._total_iterations = 0

  @property
  def ratings(self) -> np.ndarray:
    return self._ratings

  @property
  def initial_noise(self) -> np.ndarray:
    return self._initial_noise

  @property
  def total_iterations(self) -> int:
    return self._total_iterations

  @property
  def avg_l2_grad_norm(self) -> float:
    return self._avg_l2_grad_norm

  @property
  def avg_l1_sum_grad_norm(self) -> float:
    return self._avg_l1_sum_grad_norm

  def _gradient(self, ratings: np.ndarray, batch: np.ndarray) -> np.ndarray:
    raise NotImplementedError()

  def step(self, learning_rate: float, batch: np.ndarray) -> np.ndarray:
    """Applies one step of gradient descent on the batch.

    Args:
      learning_rate: a step size for the update.
      batch: the batch of votes (integer indices)

    Returns:
      gradient: the gradient over all parameters.
    """
    gradient = self._gradient(self._ratings, batch)
    self._ratings = self._ratings - learning_rate * gradient
    self._ratings = np.clip(self._ratings, self._rating_lb, self._rating_ub)
    return gradient

  def ranking(self) -> base.PreferenceList:
    """Return a sorted list by decreasing rating."""
    sorted_indices = np.argsort(-self._ratings)
    return [self._profile.alternatives[i] for i in sorted_indices]

  def run_solver(self,
                 iterations: int = 1000,
                 learning_rate: float = 0.01) -> tuple[np.ndarray,
                                                       base.PreferenceList]:
    """Soft Condorcet optimizer."""

    l1_sum_norms = []
    l2_norms = []
    batch = np.arange(self._profile.num_votes(), dtype=int)
    for i in range(iterations):
      self._total_iterations += 1
      if self._batch_size > 0:
        # SGD case: Sample a batch of votes.
        batch = np.random.randint(self._profile.num_votes(),
                                  size=self._batch_size)
      gradient = self.step(learning_rate, batch)
      l2_norms.append(np.linalg.norm(gradient))
      l1_sum_norms.append(np.absolute(gradient).sum())
      if (i - 1) % self._compute_norm_freq == 0:
        self._avg_l1_sum_grad_norm = (
            np.asarray(l1_sum_norms).sum() / self._compute_norm_freq)
        self._avg_l2_grad_norm = (
            np.asarray(l2_norms).sum() / self._compute_norm_freq)
        l2_norms = []
        l1_sum_norms = []
        if self._verbose:
          logging.info("L1 gradient norm = %d", self._avg_l1_sum_grad_norm)
          logging.info("L2 gradient norm = %d", self._avg_l2_grad_norm)
    return self._ratings, self.ranking()

  def approximate_posterior(self,
                            num_posterior_samples: int,
                            num_cov_samples: int) -> np.ndarray:
    """Stochastic Gradient Descent as Approximate Bayesian Inference."""

    gradients = []
    for _ in range(num_cov_samples):
      batch = np.random.randint(self._profile.num_votes(),
                                size=self._batch_size)
      gradient = np.asarray(self._gradient(self._ratings, batch))
      gradients.append(gradient)
    gradients = np.stack(gradients, axis=0)

    gradients_centered = gradients - gradients.mean(axis=0, keepdims=True)
    cov = np.dot(gradients_centered.T, gradients_centered) / num_cov_samples
    # cov_factor = np.linalg.cholesky(cov)  # cov = cov_factor.dot(cov_factor.T)

    coeff = 2 * self._batch_size / self._profile.num_votes()
    precon = coeff * np.linalg.pinv(cov)

    samples = []

    sample = np.array(self._ratings)
    for _ in range(num_posterior_samples):
      batch = np.random.randint(self._profile.num_votes(),
                                size=self._batch_size)
      gradient = self._gradient(sample, batch)
      sample -= precon.dot(gradient)
      sample = np.clip(sample, self._rating_lb, self._rating_ub)
      samples.append(sample)

    samples = np.stack(samples, axis=0)

    return samples


class SoftCondorcetOptimizer(Optimizer):
  """Soft Condorcet optimizer."""

  def __init__(
      self,
      profile: base.PreferenceProfile,
      batch_size: int = 0,  # full GD by default
      rating_lower_bound: float = 0.0,
      rating_upper_bound: float = 1000.0,
      compute_norm_freq: int = 1000,
      initial_param_noise: float = 0.0,
      temperature: float = 1.0,
  ):
    super().__init__(
        profile,
        batch_size,
        rating_lower_bound,
        rating_upper_bound,
        compute_norm_freq,
        initial_param_noise,
    )
    self._temperature = temperature

  def _gradient(self, ratings: np.ndarray, batch: np.ndarray) -> np.ndarray:
    """Compute the gradient of a batch of data. Explained above."""

    alt_idx = self._profile.alternatives_dict
    wins_dict = collections.defaultdict(lambda: 0)
    grad = np.zeros(self._profile.num_alternatives(), dtype=np.float32)
    for idx in batch:
      vote = self._profile.votes[idx]
      vote_len = len(vote.vote)
      for i in range(vote_len):
        for j in range(i+1, vote_len):
          wins_dict[(vote.vote[i], vote.vote[j])] += vote.weight
    # print(dict(wins_dict))
    for alt_tuple, weight in wins_dict.items():
      alt_a, alt_b = alt_tuple
      a_idx = alt_idx[alt_a]
      b_idx = alt_idx[alt_b]
      # print(f"{alt_a} ({a_idx}) {alt_b} ({b_idx}) {weight}")
      delta_ab = ((ratings[b_idx] - ratings[a_idx])
                  / self._temperature)
      sigma_ab = 1.0 / (1.0 + np.exp(-delta_ab))
      grad[a_idx] -= (weight * sigma_ab * (1.0 - sigma_ab)
                      / self._temperature)
      grad[b_idx] += (weight * sigma_ab * (1.0 - sigma_ab)
                      / self._temperature)
    grad /= len(batch)
    return grad


class FenchelYoungOptimizer(Optimizer):
  """Replace gradient by Fenchel Young loss gradient."""

  def __init__(
      self,
      profile: base.PreferenceProfile,
      batch_size: int = 0,  # full GD by default
      rating_lower_bound: float = 0.0,
      rating_upper_bound: float = 1000.0,
      compute_norm_freq: int = 1000,
      initial_param_noise: float = 0.0,
      sigma: float = 100.0,
  ):
    super().__init__(
        profile,
        batch_size,
        rating_lower_bound,
        rating_upper_bound,
        compute_norm_freq,
        initial_param_noise,
    )
    self._sigma = sigma

  def _gradient(self, ratings: np.ndarray, batch: np.ndarray) -> np.ndarray:
    """Compute FY gradient y_eps - y."""
    alt_idx = self._profile.alternatives_dict
    grad = np.zeros(self._profile.num_alternatives(), dtype=np.float32)
    for idx in batch:
      vote = self._profile.votes[idx]
      if vote.weight != 1:
        raise ValueError("Fenchel Young Optimizer only works with weight 1.")
      vote_len = len(vote.vote)
      target_ranking = np.arange(vote_len).astype(np.float32)
      player_ids = [alt_idx[a] for a in vote.vote]
      gumbel_noise = np.random.gumbel(loc=0.0, scale=1.0, size=vote_len).astype(
          np.float32
      )
      # Sample one perturbed ranking. Could do averages of multiple.
      predicted_ratings = ratings[player_ids] + gumbel_noise * self._sigma
      # Randomize tie-breaking by shuffling and unshuffling.
      shuffled = np.random.permutation(len(player_ids))
      unshuffle = np.argsort(shuffled)
      predicted_ranking = np.argsort(np.argsort(-predicted_ratings[shuffled]))[
          unshuffle
      ].astype(np.float32)

      local_grad = predicted_ranking - target_ranking
      # Flipping the sign due to argmin in ranking: Since the loss was derived
      # for f(x)=argmax x, and g(x)=argmin x=f(-x), the gradient g'=-f'(-x).
      grad[player_ids] += -local_grad
    grad /= len(batch)
    return grad
