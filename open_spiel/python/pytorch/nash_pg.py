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

"""Nash Policy Gradient (NashPG) agent implemented in PyTorch.

See the paper https://arxiv.org/abs/2510.18183 (Yu et al., 2026) for details.

NashPG is a policy-gradient Nash solver: each player runs an independent PPO
update with an extra regularization term pulling its policy toward a magnet
(reference) policy that is periodically re-copied from the current policy.

Notes:
  - Each player's magnet refreshes on its own schedule; for a synchronized
    refresh, set `auto_update_magnet=False` and call `update_magnet()` yourself.
  - Supports both sequential and simultaneous-move games. However, dense
    rewards in a sequential game need the driver to step every agent at every
    time step; see the example.

Example: see `open_spiel/python/examples/nash_pg_example.py`
"""

import collections
import copy

import numpy as np
import torch
from torch import distributions
from torch import nn
from torch import optim

from open_spiel.python import rl_agent

# Acting player's decisions, plus the reward and terminal flag for
# the interval up to its next decision (or the end of the episode).
Transition = collections.namedtuple(
    "Transition",
    [
        "info_state",
        "legal_actions_mask",
        "action",
        "logprob",
        "value",
        "reward",
        "done",
    ],
)

LossInfo = collections.namedtuple("LossInfo", "policy value entropy magnet_kl")

_ILLEGAL_ACTION_LOGIT = torch.finfo(torch.float32).min


def _legal_actions_mask(legal_actions, num_actions):
  """Boolean [num_actions] tensor, True on legal actions."""
  mask = torch.zeros(num_actions, dtype=torch.bool)
  mask[legal_actions] = True
  return mask


class _ActorCritic(nn.Module):
  """Separate tanh-MLP policy and value heads."""

  def __init__(self, info_state_size, num_actions, hidden_sizes):
    super().__init__()
    self.actor = self._mlp(info_state_size, hidden_sizes, num_actions, 0.01)
    self.critic = self._mlp(info_state_size, hidden_sizes, 1, 1.0)

  @staticmethod
  def _mlp(in_size, hidden_sizes, out_size, out_std):
    """A tanh MLP with orthogonally-initialized layers (PPO convention)."""

    def linear(fan_in, fan_out, std):
      layer = nn.Linear(fan_in, fan_out)
      nn.init.orthogonal_(layer.weight, std)
      nn.init.zeros_(layer.bias)
      return layer

    layers, last = [], in_size
    for size in hidden_sizes:
      layers += [linear(last, size, np.sqrt(2)), nn.Tanh()]
      last = size
    layers.append(linear(last, out_size, out_std))
    return nn.Sequential(*layers)

  def policy(self, obs, legal_actions_mask):
    """Masked categorical policy at `obs`."""
    logits = torch.where(
        legal_actions_mask, self.actor(obs), _ILLEGAL_ACTION_LOGIT
    )
    return distributions.Categorical(logits=logits)

  def value(self, obs):
    """State value with shape [batch]."""
    return self.critic(obs).squeeze(-1)


class NashPG(rl_agent.AbstractAgent):
  """Nash Policy Gradient agent.

  Note: Unlike the paper, which shares one network between players,
  each player here has its own policy/value network.
  """

  def __init__(
      self,
      player_id,
      info_state_size,
      num_actions,
      hidden_layers_sizes=(128, 128),
      batch_size=1024,
      update_epochs=4,
      num_minibatches=4,
      learning_rate=1e-3,
      magnet_coef=0.2,
      magnet_update_period=40,
      auto_update_magnet=True,
      gamma=1.0,
      gae_lambda=0.95,
      clip_coef=0.2,
      entropy_coef=0.1,
      value_coef=0.5,
      max_grad_norm=0.5,
      device="cpu",
      seed=None,
      name="nash_pg",
  ):
    """Initializes the agent.

    Args:
      player_id: int, index of the player this learner controls.
      info_state_size: int, length of the flat information-state vector.
      num_actions: int, size of the action space.
      hidden_layers_sizes: iterable of int, hidden sizes of the MLPs.
      batch_size: int, number of this player's own decisions per update.
      update_epochs: int, optimization epochs per update.
      num_minibatches: int, minibatches per epoch.
      learning_rate: float, Adam learning rate.
      magnet_coef: float, weight of the KL-to-magnet penalty.
      magnet_update_period: int, updates between magnet refreshes.
      auto_update_magnet: bool, whether to refresh the magnet automatically.
      gamma: float, discount factor.
      gae_lambda: float, GAE(lambda) parameter.
      clip_coef: float, PPO clip range for the policy and value losses.
      entropy_coef: float, entropy bonus weight.
      value_coef: float, value loss weight.
      max_grad_norm: float, global gradient-norm clip.
      device: str, torch device.
      seed: int or None, seeds the action-sampling RNG.
      name: str, agent name.
    """
    self.player_id = player_id
    self._name = name
    self._num_actions = num_actions
    self._device = torch.device(device)
    self._rng = np.random.RandomState(seed)

    self._batch_size = batch_size
    self._update_epochs = update_epochs
    self._minibatch_size = max(1, batch_size // num_minibatches)
    self._magnet_coef = magnet_coef
    self._magnet_update_period = magnet_update_period
    self._auto_update_magnet = auto_update_magnet
    self._gamma = gamma
    self._gae_lambda = gae_lambda
    self._clip_coef = clip_coef
    self._entropy_coef = entropy_coef
    self._value_coef = value_coef
    self._max_grad_norm = max_grad_norm

    hidden_sizes = tuple(int(h) for h in hidden_layers_sizes)
    self._network = _ActorCritic(info_state_size, num_actions, hidden_sizes)
    self._network.to(self._device)
    self._magnet = copy.deepcopy(self._network).requires_grad_(False)
    self._optimizer = optim.Adam(
        self._network.parameters(), lr=learning_rate, eps=1e-5
    )

    self._buffer = []
    self._pending = None  # This player's last decision, awaiting its reward.
    self._pending_reward = 0.0
    self._total_updates = 0
    self._magnet_refreshes = 0
    self._last_loss = None

  def step(self, time_step, is_evaluation=False):
    """Returns an action for the state and, unless evaluating, also learns."""
    my_turn = (not time_step.last()) and (
        time_step.is_simultaneous_move()
        or time_step.current_player() == self.player_id
    )

    action, probs = None, []
    if my_turn:
      info_state = time_step.observations["info_state"][self.player_id]
      legal_actions = time_step.observations["legal_actions"][self.player_id]
      action, probs, logprob, value = self._act(info_state, legal_actions)

    if is_evaluation:
      return rl_agent.StepOutput(action=action, probs=probs)

    # Attribute the reward observed since our last decision to that decision.
    if self._pending is not None and time_step.rewards is not None:
      self._pending_reward += time_step.rewards[self.player_id]
      if my_turn or time_step.last():
        self._buffer.append(
            self._pending._replace(
                reward=self._pending_reward, done=time_step.last()
            )
        )
        self._pending = None

    if time_step.last():
      self._pending, self._pending_reward = None, 0.0
      self._maybe_learn(bootstrap_value=0.0)
    elif my_turn:
      mask = _legal_actions_mask(legal_actions, self._num_actions)
      self._pending = Transition(
          info_state=np.asarray(info_state, dtype=np.float32),
          legal_actions_mask=mask,
          action=action,
          logprob=logprob,
          value=value,
          reward=0.0,
          done=False,
      )
      self._pending_reward = 0.0
      self._maybe_learn(bootstrap_value=value)

    return rl_agent.StepOutput(action=action, probs=probs)

  def update_magnet(self):
    """Copies the current policy into the magnet (reference) policy."""
    self._magnet.load_state_dict(self._network.state_dict())
    self._magnet_refreshes += 1

  @property
  def loss(self):
    """The most recent `LossInfo`, or None before the first update."""
    return self._last_loss

  @property
  def magnet_refreshes(self):
    """How many times the magnet has been refreshed so far."""
    return self._magnet_refreshes

  def _act(self, info_state, legal_actions):
    """Samples an action; returns (action, full_probs, logprob, value)."""
    obs = torch.as_tensor(
        np.asarray(info_state, dtype=np.float32).reshape(1, -1),
        device=self._device,
    )
    mask = _legal_actions_mask(legal_actions, self._num_actions)
    mask = mask.to(self._device).unsqueeze(0)
    with torch.no_grad():
      dist = self._network.policy(obs, mask)
      value = self._network.value(obs).item()
    probs = dist.probs[0].cpu().numpy().astype(np.float64)
    probs /= probs.sum()
    action = int(self._rng.choice(self._num_actions, p=probs))
    logprob = dist.log_prob(torch.as_tensor(action, device=self._device)).item()
    return action, probs, logprob, value

  def _maybe_learn(self, bootstrap_value):
    """Runs an update once `batch_size` transitions have accumulated."""
    if len(self._buffer) < self._batch_size:
      return
    self._update(bootstrap_value)
    self._buffer = []
    self._total_updates += 1
    if (
        self._auto_update_magnet
        and self._total_updates % self._magnet_update_period == 0
    ):
      self.update_magnet()

  def _update(self, bootstrap_value):
    """One PPO update with the added KL-to-magnet policy penalty."""
    device = self._device
    batch = self._buffer
    cols = Transition(*zip(*batch))

    def to_tensor(seq):
      return torch.as_tensor(np.asarray(seq, np.float32), device=device)

    obs = to_tensor(cols.info_state)
    masks = torch.stack(cols.legal_actions_mask).to(device)
    actions = torch.as_tensor(cols.action, device=device)
    old_logprob = to_tensor(cols.logprob)
    values = to_tensor(cols.value)
    rewards = to_tensor(cols.reward)
    not_done = 1.0 - to_tensor(cols.done)

    # Generalized advantage estimation.
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(batch))):
      next_value = bootstrap_value if t == len(batch) - 1 else values[t + 1]
      delta = rewards[t] + self._gamma * next_value * not_done[t] - values[t]
      gae = delta + self._gamma * self._gae_lambda * not_done[t] * gae
      advantages[t] = gae
    returns = advantages + values
    if advantages.numel() > 1:
      advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # PPO update.
    indices = np.arange(len(batch))
    for _ in range(self._update_epochs):
      self._rng.shuffle(indices)
      for start in range(0, len(batch), self._minibatch_size):
        mb = indices[start : start + self._minibatch_size]
        dist = self._network.policy(obs[mb], masks[mb])
        with torch.no_grad():
          magnet_dist = self._magnet.policy(obs[mb], masks[mb])

        log_ratio = dist.log_prob(actions[mb]) - old_logprob[mb]
        ratio = log_ratio.exp()
        clipped_ratio = ratio.clamp(1 - self._clip_coef, 1 + self._clip_coef)
        pg_loss = torch.max(
            -advantages[mb] * ratio, -advantages[mb] * clipped_ratio
        ).mean()

        new_value = self._network.value(obs[mb])
        value_clipped = values[mb] + (new_value - values[mb]).clamp(
            -self._clip_coef, self._clip_coef
        )
        v_loss_unclipped = (new_value - returns[mb]) ** 2
        v_loss_clipped = (value_clipped - returns[mb]) ** 2
        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

        entropy = dist.entropy().mean()
        magnet_kl = distributions.kl_divergence(dist, magnet_dist).mean()

        loss = (
            pg_loss
            + self._value_coef * v_loss
            - self._entropy_coef * entropy
            + self._magnet_coef * magnet_kl
        )

        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self._network.parameters(), self._max_grad_norm
        )
        self._optimizer.step()

    self._last_loss = LossInfo(
        policy=pg_loss.item(),
        value=v_loss.item(),
        entropy=entropy.item(),
        magnet_kl=magnet_kl.item(),
    )
