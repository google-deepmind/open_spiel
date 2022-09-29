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
"""Python implementation of R-NaD."""

import functools
from typing import Any, Dict, Optional, Sequence, Tuple

import chex
import haiku as hk
import jax
from jax import lax
from jax import numpy as jnp
from jax import tree_util as tree
import numpy as np
import optax

from open_spiel.python import policy as policy_lib
import pyspiel


class EntropySchedule:
  """An increasing list of steps where the regularisation network is updated.

  Example
    EntropySchedule([3, 5, 10], [2, 4, 1])
    =>   [0, 3, 6, 11, 16, 21, 26, 10]
          | 3 x2 |      5 x4     | 10 x1
  """

  def __init__(self, *, sizes: Sequence[int], repeats: Sequence[int]):
    """Constructs a schedule of entropy iterations.

    Args:
      sizes: the list of iteration sizes.
      repeats: the list, parallel to sizes, with the number of times for each
        size from `sizes` to repeat.
    """
    try:
      if len(repeats) != len(sizes):
        raise ValueError("`repeats` must be parallel to `sizes`.")
      if not sizes:
        raise ValueError("`sizes` and `repeats` must not be empty.")
      if any([(repeat <= 0) for repeat in repeats]):
        raise ValueError("All repeat values must be strictly positive")
      if repeats[-1] != 1:
        raise ValueError("The last value in `repeats` must be equal to 1, "
                         "ince the last iteration size is repeated forever.")
    except ValueError as e:
      raise ValueError(
          f"Entropy iteration schedule: repeats ({repeats}) and sizes ({sizes})."
      ) from e

    schedule = [0]
    for size, repeat in zip(sizes, repeats):
      schedule.extend([schedule[-1] + (i + 1) * size for i in range(repeat)])

    self.schedule = np.array(schedule, dtype=np.int32)

  def __call__(self, t: int) -> Tuple[float, bool]:
    """Entropy scheduling parameters for a given step `t`.

    Args:
      t: The current learning step.
    Returns:
      alpha_t: The mixing weight (from [0, 1]) of the previous policy with
        the one before for computing the intrinsic reward.
      update_target_net: A boolean indicator for updating the target network
        with the current network.
    """

    # The complexity below is because at some point we might go past
    # the explicit schedule, and then we'd need to just use the last step
    # in the schedule and apply ((t - last_step) % last_iteration) == 0) logic.

    # The schedule might look like this:
    # X----X-----X--X--X--X--------X
    # `t` might  |  be here  ^         |
    # or there   ^                     |
    # or even past the schedule        ^

    # We need to deal with two cases below.
    # Instead of going for the complicated conditional, let's just
    # compute both and then do the A * s + B * (1 - s) with s being a bool
    # selector between A and B.

    # 1. assume t is past the schedule, ie schedule[-1] <= t.
    last_size = self.schedule[-1] - self.schedule[-2]
    last_start = self.schedule[-1] + (
        t - self.schedule[-1]) // last_size * last_size
    # 2. assume t is within the schedule.
    start = jnp.amax(self.schedule * (self.schedule <= t))
    finish = jnp.amin(
        self.schedule * (t < self.schedule),
        initial=self.schedule[-1],
        where=(t < self.schedule))
    size = finish - start

    # Now select between the two.
    beyond = (self.schedule[-1] <= t)  # Are we past the schedule?
    iteration_start = (last_start * beyond + start * (1 - beyond))
    iteration_size = (last_size * beyond + size * (1 - beyond))

    update_target_net = jnp.logical_and(t > 0, jnp.sum(t == iteration_start))
    alpha_t = jnp.minimum((2.0 * (t - iteration_start)) / iteration_size, 1.0)

    return alpha_t, update_target_net


@chex.dataclass(frozen=True)
class PolicyPostProcessing:
  """Policy post-processing options.

  Even when fully trained, the resulting softmax-based policy may put
  a small probability mass on bad actions. This results in an agent
  waiting for the opponent (itself in self-play) to commit an error.

  To address that the policy is post-processed using:
  - thresholding: any action with probability smaller than self.threshold
    is simply removed from the policy.
  - discretization: the probability values are rounded to the closest
    multiple of 1/self.discretization.

  The post-processing is used on the learner, and thus must be jit-friendly.
  """
  # All policy probabilities below `threshold` are zeroed out. Thresholding
  # is disabled if this value is non-positive.
  threshold: float = 0.03
  # Rounds the policy probabilities to the "closest"
  # multiple of 1/`self.discretization`.
  # Discretization is disabled for non-positive values.
  discretization: int = 32

  def __call__(self, policy: chex.Array, mask: chex.Array) -> chex.Array:
    """A jax friendly post-processing of a policy."""
    policy = self._threshold(policy, mask)
    policy = self._discretize(policy)
    return policy

  def _threshold(self, policy: chex.Array, mask: chex.Array) -> chex.Array:
    """Remove from the support the actions 'a' where policy(a) < threshold."""
    if self.threshold <= 0:
      return policy

    mask = mask * (
        # Values over the threshold.
        (policy >= self.threshold) +
        # Degenerate case is when policy is less than threshold *everywhere*.
        # In that case we just keep the policy as-is.
        (jnp.max(policy, axis=-1, keepdims=True) < self.threshold))
    return mask * policy / jnp.sum(mask * policy, axis=-1, keepdims=True)

  def _discretize(self, policy: chex.Array) -> chex.Array:
    """Round all action probabilities to a multiple of 1/self.discretize."""
    if self.discretization <= 0:
      return policy

    # The unbatched/single policy case:
    if len(policy.shape) == 1:
      return self._discretize_single(policy)

    # policy may be [B, A] or [T, B, A], etc. Thus add hk.BatchApply.
    dims = len(policy.shape) - 1

    # TODO(perolat): avoid mixing vmap and BatchApply since the two could
    # be folded into either a single BatchApply or a sequence of vmaps, but
    # not the mix.
    vmapped = jax.vmap(self._discretize_single)
    policy = hk.BatchApply(vmapped, num_dims=dims)(policy)

    return policy

  def _discretize_single(self, mu: chex.Array) -> chex.Array:
    """A version of self._discretize but for the unbatched data."""
    # TODO(perolat): try to merge _discretize and _discretize_single
    # into one function that handles both batched and unbatched cases.
    if len(mu.shape) == 2:
      mu_ = jnp.squeeze(mu, axis=0)
    else:
      mu_ = mu
    n_actions = mu_.shape[-1]
    roundup = jnp.ceil(mu_ * self.discretization).astype(jnp.int32)
    result = jnp.zeros_like(mu_)
    order = jnp.argsort(-mu_)  # Indices of descending order.
    weight_left = self.discretization

    def f_disc(i, order, roundup, weight_left, result):
      x = jnp.minimum(roundup[order[i]], weight_left)
      result = jax.numpy.where(weight_left >= 0, result.at[order[i]].add(x),
                               result)
      weight_left -= x
      return i + 1, order, roundup, weight_left, result

    def f_scan_scan(carry, x):
      i, order, roundup, weight_left, result = carry
      i_next, order_next, roundup_next, weight_left_next, result_next = f_disc(
          i, order, roundup, weight_left, result)
      carry_next = (i_next, order_next, roundup_next, weight_left_next,
                    result_next)
      return carry_next, x

    (_, _, _, weight_left_next, result_next), _ = jax.lax.scan(
        f_scan_scan,
        init=(jnp.asarray(0), order, roundup, weight_left, result),
        xs=None,
        length=n_actions)

    result_next = jnp.where(
        weight_left_next > 0, result_next.at[order[0]].add(weight_left_next),
        result_next)
    if len(mu.shape) == 2:
      result_next = jnp.expand_dims(result_next, axis=0)
    return result_next / self.discretization


def play_chance(state: pyspiel.State, rng: np.random.RandomState):
  """Plays the chance nodes until we end up at another type of node."""
  while state.is_chance_node():
    chance_outcome, chance_proba = zip(*state.chance_outcomes())
    action = rng.choice(chance_outcome, p=chance_proba)
    state.apply_action(action)
  return state


def legal_policy(logits: chex.Array,
                 legal_actions: chex.Array,
                 temperature: float = 1.0) -> chex.Array:
  """A soft-max policy that respects legal_actions and temperature."""
  # Fiddle a bit to make sure we don't generate NaNs or Inf in the middle.
  l_min = logits.min(axis=-1, keepdims=True)
  logits = jnp.where(legal_actions, logits, l_min)
  logits -= logits.max(axis=-1, keepdims=True)
  logits *= legal_actions
  exp_logits = jnp.where(legal_actions,
                         jnp.exp(temperature * logits),
                         0)  # Illegal actions become 0.
  exp_logits_sum = jnp.sum(exp_logits, axis=-1, keepdims=True)
  return exp_logits / exp_logits_sum


def player_others(player_ids, valid, player):
  """A vector of 1 for the current player and -1 for others.

  Args:
    player_ids: Tensor [...] containing player ids (0 <= player_id < N).
    valid: Tensor [...] containing whether these states are valid.
    player: The player id.

  Returns:
    player_other: is 1 for the current player and -1 for others [..., 1].
  """
  current_player_tensor = (player_ids == player).astype(jnp.int32)

  res = 2 * current_player_tensor - 1
  res = res * valid
  return jnp.expand_dims(res, axis=-1)


def _select_action(actions, pi, valid):
  return jnp.sum(actions * pi, axis=-1, keepdims=False) * valid + (1 - valid)


def _policy_ratio(pi, mu, actions, valid):
  """Returns a ratio of policy pi/mu when selecting action a.

  By convention, this ratio is 1 on non valid states
  Args:
    pi: the policy of shape [..., A].
    mu: the sampling policy of shape [..., A].
    actions: an array of the current actions of shape [..., A].
    valid: 0 if the state is not valid and else 1 of shape [...].

  Returns:
    policy_ratio: pi/mu and 1 on non valid states (the shape is [..., 1]).
  """
  pi_actions = _select_action(actions, pi, valid)
  mu_actions = _select_action(actions, mu, valid)
  return pi_actions / mu_actions


def _where(pred, true_data, false_data):
  """Similar to jax.where that treats `pred` as a broadcastable prefix."""

  def _where_one(t, f):
    chex.assert_equal_rank((t, f))
    # Expand the dimensions of pred if true_data and false_data are higher rank.
    p = jnp.reshape(pred, pred.shape + (1,) * (len(t.shape) - len(pred.shape)))
    return jnp.where(p, t, f)

  return tree.tree_map(_where_one, true_data, false_data)


def has_played_with_state(state: chex.Array, valid: chex.Array,
                          player_id: chex.Array,
                          player: int) -> Tuple[chex.Array, chex.Array]:
  """Compute a mask of states which have a next state in the sequence."""
  if state is None:
    state = jnp.zeros_like(player_id[-1])

  def _loop_has_played(carry, x):
    valid, player_id = x
    chex.assert_equal_shape((valid, player_id))

    our_res = jnp.ones_like(player_id)
    opp_res = carry
    reset_res = jnp.zeros_like(carry)

    our_carry = carry
    opp_carry = carry
    reset_carry = jnp.zeros_like(player_id)

    # pyformat: disable
    return _where(valid, _where((player_id == player),
                                (our_carry, our_res),
                                (opp_carry, opp_res)),
                  (reset_carry, reset_res))
    # pyformat: enable

  return lax.scan(
      f=_loop_has_played,
      init=state,
      xs=(valid, player_id),
      reverse=True)


# V-Trace
#
# Custom implementation of VTrace to handle trajectories having a mix of
# different player steps. The standard rlax.vtrace can't be applied here
# out of the box because a trajectory could look like '121211221122'.


@chex.dataclass(frozen=True)
class LoopVTraceCarry:
  """An internal carry-over between chunks related to v-trace computations."""
  reward: chex.Array
  # The cumulated reward until the end of the episode. Uncorrected (v-trace).
  # Gamma discounted and includes eta_reg_entropy.
  reward_uncorrected: chex.Array
  next_value: chex.Array
  next_v_target: chex.Array
  importance_sampling: chex.Array


@chex.dataclass(frozen=True)
class VTraceState:
  """An internal carry-over between chunks related to v-trace computations."""
  has_played: Any = None
  v_trace: Optional[LoopVTraceCarry] = None


def v_trace(
    state: Optional[VTraceState],
    v,
    valid,
    player_id,
    acting_policy,
    merged_policy,
    merged_log_policy,
    player_others_,
    actions,
    reward,
    player,
    # Scalars below.
    eta,
    lambda_,
    c,
    rho,
    gamma=1.0,
    estimate_all=False,
) -> Tuple[VTraceState, Tuple[Any, Any, Any]]:
  """Custom VTrace for trajectories with a mix of different player steps."""
  if not state:
    state = VTraceState()

  # pylint: disable=g-long-lambda
  if estimate_all:
    player_id_step = player * jnp.ones_like(player_id)
  else:
    player_id_step = player_id

  new_state_has_played, has_played_ = has_played_with_state(
      state.has_played, valid, player_id_step, player)

  policy_ratio = _policy_ratio(merged_policy, acting_policy, actions, valid)
  inv_mu = _policy_ratio(
      jnp.ones_like(merged_policy), acting_policy, actions, valid)

  eta_reg_entropy = (-eta *
                     jnp.sum(merged_policy * merged_log_policy, axis=-1) *
                     jnp.squeeze(player_others_, axis=-1))
  eta_log_policy = -eta * merged_log_policy * player_others_

  init_state_v_trace = LoopVTraceCarry(
      reward=jnp.zeros_like(reward[-1]),
      reward_uncorrected=jnp.zeros_like(reward[-1]),
      next_value=jnp.zeros_like(v[-1]),
      next_v_target=jnp.zeros_like(v[-1]),
      importance_sampling=jnp.ones_like(policy_ratio[-1]))

  def _loop_v_trace(carry: LoopVTraceCarry, x) -> Tuple[LoopVTraceCarry, Any]:
    (cs, player_id, v, reward, eta_reg_entropy, valid, inv_mu, actions,
     eta_log_policy) = x

    reward_uncorrected = (
        reward + gamma * carry.reward_uncorrected + eta_reg_entropy)
    discounted_reward = reward + gamma * carry.reward

    # V-target:
    our_v_target = (
        v + jnp.expand_dims(
            jnp.minimum(rho, cs * carry.importance_sampling), axis=-1) *
        (jnp.expand_dims(reward_uncorrected, axis=-1) +
         gamma * carry.next_value - v) + lambda_ * jnp.expand_dims(
             jnp.minimum(c, cs * carry.importance_sampling), axis=-1) * gamma *
        (carry.next_v_target - carry.next_value))

    opp_v_target = jnp.zeros_like(our_v_target)
    reset_v_target = jnp.zeros_like(our_v_target)

    # Learning output:
    our_learning_output = (
        v +  # value
        eta_log_policy +  # regularisation
        actions * jnp.expand_dims(inv_mu, axis=-1) *
        (jnp.expand_dims(discounted_reward, axis=-1) + gamma * jnp.expand_dims(
            carry.importance_sampling, axis=-1) * carry.next_v_target - v))

    opp_learning_output = jnp.zeros_like(our_learning_output)
    reset_learning_output = jnp.zeros_like(our_learning_output)

    # State carry:
    our_carry = LoopVTraceCarry(
        reward=jnp.zeros_like(carry.reward),
        next_value=v,
        next_v_target=our_v_target,
        reward_uncorrected=jnp.zeros_like(carry.reward_uncorrected),
        importance_sampling=jnp.ones_like(carry.importance_sampling))
    opp_carry = LoopVTraceCarry(
        reward=eta_reg_entropy + cs * discounted_reward,
        reward_uncorrected=reward_uncorrected,
        next_value=gamma * carry.next_value,
        next_v_target=gamma * carry.next_v_target,
        importance_sampling=cs * carry.importance_sampling)
    reset_carry = init_state_v_trace

    # Invalid turn: init_state_v_trace and (zero target, learning_output)
    # pyformat: disable
    return _where(valid,
                  _where((player_id == player),
                         (our_carry, (our_v_target, our_learning_output)),
                         (opp_carry, (opp_v_target, opp_learning_output))),
                  (reset_carry, (reset_v_target, reset_learning_output)))
    # pyformat: enable

  state_v_trace = state.v_trace or init_state_v_trace
  new_state_v_trace, (v_target_, learning_output) = lax.scan(
      f=_loop_v_trace,
      init=state_v_trace,
      xs=(policy_ratio, player_id_step, v, reward, eta_reg_entropy, valid,
          inv_mu, actions, eta_log_policy),
      reverse=True)

  new_state = VTraceState(
      has_played=new_state_has_played,
      v_trace=new_state_v_trace)
  return new_state, (v_target_, has_played_, learning_output)


def legal_log_policy(logits, legal_actions):
  """Return the log of the policy on legal action, 0 on illegal action."""
  # logits_masked has illegal actions set to -inf.
  logits_masked = logits + jnp.log(legal_actions)
  max_legal_logit = logits_masked.max(axis=-1, keepdims=True)
  logits_masked = logits_masked - max_legal_logit
  # exp_logits_masked is 0 for illegal actions.
  exp_logits_masked = jnp.exp(logits_masked)

  baseline = jnp.log(jnp.sum(exp_logits_masked, axis=-1, keepdims=True))
  # Subtract baseline from logits. We do not simply return
  #     logits_masked - baseline
  # because that has -inf for illegal actions, or
  #     legal_actions * (logits_masked - baseline)
  # because that leads to 0 * -inf == nan for illegal actions.
  log_policy = jnp.multiply(
      legal_actions,
      (logits - max_legal_logit - baseline))
  return log_policy


def get_loss_v(v_list,
               v_target_list,
               mask_list,
               normalization_list=None):
  """Define the loss function for the critic."""
  if normalization_list is None:
    normalization_list = [jnp.sum(mask) for mask in mask_list]
  loss_v_list = []
  for (v_n, v_target, mask, normalization) in zip(
      v_list, v_target_list, mask_list, normalization_list):
    assert v_n.shape[0] == v_target.shape[0]

    loss_v = jnp.expand_dims(mask, axis=-1) * (
        v_n - lax.stop_gradient(v_target))**2
    loss_v = jnp.sum(loss_v) / (normalization + (normalization == 0.0))

    loss_v_list.append(loss_v)
  return sum(loss_v_list)


def apply_force_with_threshold(decision_outputs,
                               force,
                               threshold,
                               threshold_center):
  """Apply the force with below a given threshold."""
  can_decrease = decision_outputs - threshold_center > -threshold
  can_increase = decision_outputs - threshold_center < threshold
  force_negative = jnp.minimum(force, 0.0)
  force_positive = jnp.maximum(force, 0.0)
  clipped_force = can_decrease * force_negative + can_increase * force_positive
  return decision_outputs * lax.stop_gradient(clipped_force)


def renormalize(loss, mask, normalization=None):
  """The `normalization` is the number of steps over which loss is computed."""
  loss_ = jnp.sum(loss * mask)
  if normalization is None:
    normalization = jnp.sum(mask)
  loss_ = loss_ / (normalization + (normalization == 0.0))
  return loss_


def get_loss_nerd(logit_list,
                  policy_list,
                  q_vr_list,
                  valid,
                  player_ids,
                  legal_actions,
                  importance_sampling_correction,
                  clip=100,
                  threshold=2,
                  threshold_center=None,
                  normalization_list=None):
  """Define the nerd loss."""
  assert isinstance(importance_sampling_correction, list)
  if normalization_list is None:
    normalization_list = [None] * len(logit_list)
  loss_pi_list = []
  for k, (logit_pi, pi, q_vr, is_c, normalization) in enumerate(
      zip(logit_list, policy_list, q_vr_list, importance_sampling_correction,
          normalization_list)):
    assert logit_pi.shape[0] == q_vr.shape[0]
    # loss policy
    adv_pi = q_vr - jnp.sum(pi * q_vr, axis=-1, keepdims=True)
    adv_pi = is_c * adv_pi  # importance sampling correction
    adv_pi = jnp.clip(adv_pi, a_min=-clip, a_max=clip)
    adv_pi = lax.stop_gradient(adv_pi)

    logits = logit_pi - jnp.mean(
        logit_pi * legal_actions, axis=-1, keepdims=True)

    if threshold_center is None:
      threshold_center = jnp.zeros_like(logits)
    else:
      threshold_center = threshold_center - jnp.mean(
          threshold_center * legal_actions, axis=-1, keepdims=True)

    nerd_loss = jnp.sum(legal_actions *
                        apply_force_with_threshold(
                            logits, adv_pi, threshold, threshold_center),
                        axis=-1)
    nerd_loss = -renormalize(nerd_loss,
                             valid * (player_ids == k), normalization)
    loss_pi_list.append(nerd_loss)
  return sum(loss_pi_list)


@chex.dataclass(frozen=True)
class AdamConfig:
  """Adam optimizer related params."""
  b1: float = 0.0
  b2: float = 0.999
  eps: float = 10e-8


@chex.dataclass(frozen=True)
class NerdConfig:
  beta: float = 2.0
  clip: float = 10_000


@chex.dataclass(frozen=True)
class RNaDConfig:
  """Configuration parameters for the RNaDSolver."""
  game_name: str
  batch_size: int = 256
  adam: AdamConfig = AdamConfig()
  nerd: NerdConfig = NerdConfig()
  c_vtrace: float = 1.0
  clip_gradient: float = 10_000
  entropy_schedule_repeats: Sequence[int] = (1,)
  entropy_schedule_size: Sequence[int] = (20_000,)
  eta_reward_transform: float = 0.2
  finetune_from: int = -1
  learning_rate: float = 0.00005
  policy_network_layers: Sequence[int] = (256, 256)
  policy_post_processing: PolicyPostProcessing = PolicyPostProcessing()
  seed: int = 42
  state_representation: str = "info_set"  # or "observation"
  target_network_avg: float = 0.001
  trajectory_max: int = 10


class RNaDSolver(policy_lib.Policy):
  """Implements a solver for the R-NaD Algorithm.

  See https://arxiv.org/abs/2206.15378.

  Define all networks. Derive losses & learning steps. Initialize the game
  state and algorithmic variables.
  """

  def __init__(self, config: RNaDConfig):
    self.config = config

    # Learner and actor step counters.
    self._t = 0
    self._step_counter = 0

    self.init()

  def init(self):
    """Initialize the network and losses."""
    self._game = pyspiel.load_game(self.config.game_name)
    self._entropy_schedule = EntropySchedule(
        sizes=self.config.entropy_schedule_size,
        repeats=self.config.entropy_schedule_repeats)

    # Initialize the random facilities for jax and numpy.
    self._rngkey = jax.random.PRNGKey(self.config.seed)
    self._np_rng = np.random.RandomState(self.config.seed)
    # TODO(etar): serialize both above to get fully deterministic behaviour.

    self._num_actions = self._game.num_distinct_actions()

    def network(x, legal):
      mlp_torso = hk.nets.MLP(self.config.policy_network_layers)
      mlp_policy_head = hk.nets.MLP([self._num_actions])
      mlp_policy_value = hk.nets.MLP([1])
      torso = mlp_torso(x)
      logit, v = mlp_policy_head(torso), mlp_policy_value(torso)
      pi = legal_policy(logit, legal)
      log_pi = legal_log_policy(logit, legal)
      return pi, v, log_pi, logit

    self.network = hk.without_apply_rng(hk.transform(network))
    self.network_jit = tree.tree_map(jax.jit, self.network)

    s = play_chance(self._game.new_initial_state(), self._np_rng)
    x = self._get_state_representation(s)
    self._state_representation_shape = x.shape
    x = np.expand_dims(x, axis=0)
    legal = np.expand_dims(s.legal_actions_mask(), axis=0)
    key = self._next_rng_key()
    self._params = self.network.init(key, x, legal)
    self._params_target = self.network.init(key, x, legal)
    self._params_prev = self.network.init(key, x, legal)
    self._params_prev_ = self.network.init(key, x, legal)

    def loss(params, params_target, params_prev, params_prev_, observation,
             legal, action, policy_actor, player_id, valid, rewards, alpha,
             finetune):
      pi, v, log_pi, logit = jax.vmap(self.network.apply, (None, 0, 0),
                                      0)(params, observation, legal)

      pi_pprocessed = self.config.policy_post_processing(pi, legal)
      merged_policy_pprocessed = jnp.where(finetune, pi_pprocessed, pi)

      _, v_target, _, _ = jax.vmap(self.network.apply, (None, 0, 0),
                                   0)(params_target, observation, legal)
      _, _, log_pi_prev, _ = jax.vmap(self.network.apply, (None, 0, 0),
                                      0)(params_prev, observation, legal)
      _, _, log_pi_prev_, _ = jax.vmap(self.network.apply, (None, 0, 0),
                                       0)(params_prev_, observation, legal)
      player_others_list = [
          player_others(player_id, valid, player)
          for player in range(self._game.num_players())
      ]
      # This line creates the reward transform log(pi(a|x)/pi_reg(a|x)).
      # For the stability reasons, reward changes smoothly between iterations.
      # The mixing between old and new reward transform is a convex combination
      # parametrised by alpha.
      log_policy_reg = log_pi - (
          alpha * log_pi_prev + (1 - alpha) * log_pi_prev_)

      new_v_trace_states = []
      v_target_list, has_played_list, v_trace_policy_target_list = [], [], []
      for i, (player_others_, reward) in enumerate(
          zip(player_others_list, rewards)):
        new_state, (v_target_, has_played_, policy_target_) = v_trace(
            None,
            v_target,
            valid,
            player_id,
            policy_actor,
            merged_policy_pprocessed,
            log_policy_reg,
            player_others_,
            action,
            reward,
            i,
            lambda_=1.0,
            c=self.config.c_vtrace,
            rho=np.inf,
            estimate_all=False,
            eta=self.config.eta_reward_transform,
            gamma=1.0)
        new_v_trace_states.append(new_state)
        v_target_list.append(v_target_)
        has_played_list.append(has_played_)
        v_trace_policy_target_list.append(policy_target_)
      loss_v = get_loss_v(
          [v] * self._game.num_players(),
          v_target_list,
          has_played_list,
          normalization_list=None)

      is_vector = jnp.expand_dims(jnp.ones_like(valid), axis=-1)
      importance_sampling_correction = [is_vector] * self._game.num_players()
      # Uses v-trace to define q-values for Nerd
      loss_nerd = get_loss_nerd(
          [logit] * self._game.num_players(), [pi] * self._game.num_players(),
          v_trace_policy_target_list,
          valid,
          player_id,
          legal,
          importance_sampling_correction,
          clip=self.config.nerd.clip,
          threshold=self.config.nerd.beta,
          threshold_center=None,
          normalization_list=None)
      return loss_v + loss_nerd

    self._loss_and_grad = jax.value_and_grad(loss, has_aux=False)

    ## Optimizer state
    opt_init, opt_update = optax.chain(
        optax.scale_by_adam(
            eps_root=0.0,
            **self.config.adam,
        ), optax.scale(-self.config.learning_rate),
        optax.clip(self.config.clip_gradient))
    self._opt_update_fn = self._get_update_func(opt_update)
    self._opt_state = opt_init(self._params)

    ## Target network update SGD
    opt_init_target, opt_update_target = optax.sgd(
        self.config.target_network_avg)
    self._opt_update_target_fn = self._get_update_func(opt_update_target)
    self._opt_state_target = opt_init_target(self._params_target)

    def update(params, params_target, params_prev, params_prev_, opt_state,
               opt_state_target, observation, legal, action, policy_actor,
               player_id, valid, rewards, alpha, finetune, update_target_net):
      loss_val, grad = self._loss_and_grad(params, params_target, params_prev,
                                           params_prev_, observation, legal,
                                           action, policy_actor, player_id,
                                           valid, rewards, alpha, finetune)
      # Update params using the computed gradient.
      (next_params, next_opt_state) = self._opt_update_fn(
          params, opt_state, grad)

      # Also update the `params_target` a tiny bit towards `params`.
      diff = tree.tree_map(lambda a, b: a - b, params_target, next_params)
      (next_params_target, next_opt_state_target) = self._opt_update_target_fn(
          params_target, opt_state_target, diff)

      # Rolls forward the prev and prev_ params if update_target_net is 1.
      # I.e. if update_target_net then
      # params_prev <= params_target
      # params_prev_ <= params_prev
      next_params_prev = tree.tree_map(
          lambda x, y: jnp.where(update_target_net, x, y),
          next_params_target, params_prev)
      next_params_prev_ = tree.tree_map(
          lambda x, y: jnp.where(update_target_net, x, y),
          params_prev, params_prev_)

      return (loss_val, next_params, next_params_target, next_params_prev,
              next_params_prev_, next_opt_state, next_opt_state_target)

    self._update = jax.jit(update)

  def __getstate__(self) -> Dict[str, Any]:
    """To serialize the agent."""
    return dict(
        # RNaD config.
        config=self.config,

        # Learner and actor step counters.
        t=self._t,
        step_counter=self._step_counter,

        # Network params.
        params=self._params,
        params_target=self._params_target,
        params_prev=self._params_prev,
        params_prev_=self._params_prev_,

        # Optimizer state.
        opt_state=self._opt_state,
        opt_state_target=self._opt_state_target,
    )

  def __setstate__(self, state: Dict[str, Any]):
    """To deserialize the agent."""
    # RNaD config.
    self.config = state["config"]

    # Learner and actor step counters.
    self._t = state["t"]
    self._step_counter = state["step_counter"]

    self.init()

    # Network params.
    self._params = state["params"]
    self._params_target = state["params_target"]
    self._params_prev = state["params_prev"]
    self._params_prev_ = state["params_prev_"]
    # Optimizer state.
    self._opt_state = state["opt_state"]
    self._opt_state_target = state["opt_state_target"]

  def step(self):
    (observation, legal, action, policy, player_id, valid,
     rewards) = self.collect_batch_trajectory()
    alpha, update_target_net = self._entropy_schedule(self._t)
    finetune = (self.config.finetune_from >= 0) and (self._t >
                                                     self.config.finetune_from)
    (_, self._params, self._params_target, self._params_prev,
     self._params_prev_, self._opt_state, self._opt_state_target
     ) = self._update(self._params, self._params_target, self._params_prev,
                      self._params_prev_, self._opt_state,
                      self._opt_state_target, observation, legal, action,
                      policy, player_id, valid, rewards, alpha, finetune,
                      update_target_net)
    self._t += 1

  def _get_update_func(self, opt_update):

    def update_param_state(params, opt_state, gradient):
      """Learning rule (stochastic gradient descent)."""
      updates, opt_state = opt_update(gradient, opt_state)
      new_params = optax.apply_updates(params, updates)
      return new_params, opt_state

    return update_param_state

  def _next_rng_key(self):
    """Get the next rng subkey from class rngkey."""
    self._rngkey, subkey = jax.random.split(self._rngkey)
    return subkey

  def _get_state_representation(self, state):
    if self.config.state_representation == "observation":
      return np.asarray(state.observation_tensor())
    elif self.config.state_representation == "info_set":
      return np.asarray(state.information_state_tensor())
    else:
      raise ValueError(
          f"Invalid state_representation: {self.config.state_representation}. "
          "Must be either 'info_set' or 'observation'.")

  def action_probabilities(self, state: pyspiel.State) -> Dict[int, float]:
    """Returns action probabilities dict for a single batch."""
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)
    x = self._get_state_representation(state)
    legal_actions_mask = np.array(
        state.legal_actions_mask(cur_player), dtype=jnp.float32)
    probs = self._network_jit_apply(self._params_target, x, legal_actions_mask,
                                    self.config.policy_post_processing)
    return {action: probs[action] for action in legal_actions}

  def sample_batch_action(self, x, legal):
    pi = self._network_jit_apply(self._params, x, legal)
    pi = np.asarray(pi).astype("float64")
    pi = pi / np.sum(pi, axis=-1, keepdims=True)
    a = np.apply_along_axis(
        lambda x: self._np_rng.choice(range(pi.shape[1]), p=x), axis=-1, arr=pi)
    action_vec = np.zeros(pi.shape, dtype="float64")
    action_vec[range(pi.shape[0]), a] = 1.0
    return pi, action_vec, a

  @functools.partial(jax.jit, static_argnums=(0,))
  def _network_jit_apply(
      self,
      params,
      x: chex.Array,
      legal: chex.Array,
      policy_post_processing: Optional[PolicyPostProcessing] = None):
    pi, _, _, _ = self.network.apply(params, x, legal)
    if policy_post_processing is not None:
      pi = policy_post_processing(pi, legal)
    return pi

  def collect_batch_trajectory(self):
    observation = np.zeros(
        (self.config.trajectory_max, self.config.batch_size) +
        self._state_representation_shape,
        dtype="float64")
    legal = np.ones((self.config.trajectory_max, self.config.batch_size,
                     self._num_actions),
                    dtype="float64")
    action = np.zeros_like(legal)
    policy = np.ones_like(action) / (1.0 * self._num_actions)
    player_id = np.zeros((self.config.trajectory_max, self.config.batch_size),
                         dtype="float64")
    valid = np.zeros((self.config.trajectory_max, self.config.batch_size),
                     dtype="float64")
    rewards = [
        np.zeros((self.config.trajectory_max, self.config.batch_size),
                 dtype="float64") for p in range(self._game.num_players())
    ]

    states = [
        play_chance(self._game.new_initial_state(), self._np_rng)
        for _ in range(self.config.batch_size)
    ]

    for t in range(self.config.trajectory_max):
      for i, state in enumerate(states):
        if not state.is_terminal():
          observation[t, i, :] = self._get_state_representation(state)
          legal[t, i, :] = state.legal_actions_mask()
          player_id[t, i] = state.current_player()
          valid[t, i] = 1.0
      (policy[t, :, :], action[t, :, :], a
       ) = self.sample_batch_action(observation[t, :, :], legal[t, :, :])
      for i, state in enumerate(states):
        if not state.is_terminal():
          state.apply_action(a[i])
          self._step_counter += 1
          state = play_chance(state, self._np_rng)
          returns = state.returns()
          for p in range(self._game.num_players()):
            rewards[p][t, i] = returns[p]
    return observation, legal, action, policy, player_id, valid, rewards

  def get_actor_step_counter(self) -> int:
    return self._step_counter

  def get_learner_step_counter(self) -> int:
    return self._t
