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

"""Nash Policy Gradient implemented as an OpenSpiel PyTorch agent.

NashPG combines PPO with a KL penalty toward a periodically refreshed copy of
the current policy. Each player owns one agent, so the class can be used in the
same self-play loops as the other agents derived from `AbstractAgent`.

The implementation assumes terminal rewards for turn-based games. In that
setting, each player's decisions form an on-policy trajectory even when the
opponent acts between two decisions by that player. Intermediate rewards are
not supported because the `AbstractAgent` interface does not expose the exact
transition that produced a reward to an agent that was not acting.

Reference:
  https://arxiv.org/abs/2510.18183
"""

import copy
import pathlib
from typing import Iterable, NamedTuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from open_spiel.python import rl_agent
from open_spiel.python import rl_environment
from open_spiel.python.pytorch import dqn


class _PendingTransition(NamedTuple):
  """A transition waiting for the following timestep's reward."""

  info_state: np.ndarray
  action: int
  legal_actions_mask: np.ndarray
  log_prob: float
  value: float


class _EpisodeTransition(NamedTuple):
  """A transition with its observed reward."""

  pending: _PendingTransition
  reward: float
  terminal: bool


class _TrainingSample(NamedTuple):
  """A transition with PPO targets computed at episode end."""

  info_state: np.ndarray
  action: int
  legal_actions_mask: np.ndarray
  log_prob: float
  value: float
  advantage: float
  return_: float


class NashPG(rl_agent.AbstractAgent):
  """Nash Policy Gradient agent for OpenSpiel.

  One instance should be created per player. The training loop is intentionally
  compatible with the standard OpenSpiel agent pattern:

  ```python
  output = agents[player_id].step(time_step)
  time_step = env.step([output.action])
  ```

  Args:
    player_id: Player controlled by this agent.
    state_representation_size: Flattened information-state size.
    num_actions: Number of distinct actions in the game.
    hidden_layers_sizes: Hidden MLP layer sizes.
    batch_size: Maximum number of on-policy samples per minibatch.
    learning_rate: Adam learning rate.
    discount_factor: Discount factor used for returns and GAE.
    gae_lambda: GAE lambda parameter.
    clip_epsilon: PPO clipping range.
    entropy_coefficient: Entropy bonus coefficient.
    magnet_coefficient: KL penalty coefficient toward the magnet policy.
    ppo_epochs: Number of passes over each on-policy batch.
    learn_every: Number of completed episodes between updates.
    magnet_update_period: Number of policy updates between magnet refreshes.
    auto_update_magnet: Whether to refresh the magnet automatically.
    gradient_clipping: Optional maximum global gradient norm.
    seed: Random seed for network initialization and sampling.
    device: Torch device.
  """

  def __init__(
      self,
      player_id: int,
      state_representation_size: int | tuple[int, ...],
      num_actions: int,
      hidden_layers_sizes: Iterable[int] = (128,),
      batch_size: int = 128,
      learning_rate: float = 3e-4,
      discount_factor: float = 1.0,
      gae_lambda: float = 0.95,
      clip_epsilon: float = 0.2,
      entropy_coefficient: float = 0.01,
      magnet_coefficient: float = 0.01,
      ppo_epochs: int = 4,
      learn_every: int = 1,
      magnet_update_period: int = 1,
      auto_update_magnet: bool = True,
      gradient_clipping: float | None = 0.5,
      seed: int = 42,
      device: str = "cpu",
  ) -> None:
    if not isinstance(player_id, int):
      raise TypeError("player_id must be an int")
    if not isinstance(num_actions, int) or num_actions <= 0:
      raise ValueError("num_actions must be a positive int")
    if batch_size <= 0 or ppo_epochs <= 0 or learn_every <= 0:
      raise ValueError("batch_size, ppo_epochs, and learn_every must be > 0")
    if magnet_update_period <= 0:
      raise ValueError("magnet_update_period must be > 0")
    if not 0.0 <= discount_factor <= 1.0:
      raise ValueError("discount_factor must be in [0, 1]")
    if not 0.0 <= gae_lambda <= 1.0:
      raise ValueError("gae_lambda must be in [0, 1]")
    if clip_epsilon < 0.0:
      raise ValueError("clip_epsilon must be non-negative")
    if entropy_coefficient < 0.0 or magnet_coefficient < 0.0:
      raise ValueError("regularization coefficients must be non-negative")

    dqn.set_seed(seed)
    self.player_id = player_id
    self._num_actions = num_actions
    self._batch_size = batch_size
    self._discount_factor = discount_factor
    self._gae_lambda = gae_lambda
    self._clip_epsilon = clip_epsilon
    self._entropy_coefficient = entropy_coefficient
    self._magnet_coefficient = magnet_coefficient
    self._ppo_epochs = ppo_epochs
    self._learn_every = learn_every
    self._magnet_update_period = magnet_update_period
    self._auto_update_magnet = auto_update_magnet
    self._gradient_clipping = gradient_clipping
    self._device = torch.device(device)

    if isinstance(state_representation_size, int):
      input_size = state_representation_size
    else:
      input_size = int(np.prod(state_representation_size))
    if input_size <= 0:
      raise ValueError("state_representation_size must be non-empty")

    if isinstance(hidden_layers_sizes, int):
      hidden_layers_sizes = [hidden_layers_sizes]
    hidden_layers_sizes = tuple(hidden_layers_sizes)
    if any(size <= 0 for size in hidden_layers_sizes):
      raise ValueError("hidden layer sizes must be positive")

    self._policy_network = dqn.MLP(
        input_size, hidden_layers_sizes, num_actions, seed=seed
    ).to(self._device)
    self._value_network = dqn.MLP(
        input_size, hidden_layers_sizes, 1, seed=seed + 1
    ).to(self._device)
    self._magnet_network = copy.deepcopy(self._policy_network).to(self._device)
    self._freeze_magnet()

    self._optimizer = torch.optim.Adam(
        list(self._policy_network.parameters())
        + list(self._value_network.parameters()),
        lr=learning_rate,
        eps=1e-5,
    )

    self._pending_transition = None
    self._episode = []
    self._training_samples = []
    self._step_counter = 0
    self._episode_counter = 0
    self._num_updates = 0
    self._last_loss_value = None
    self._last_metrics = {}

  def _freeze_magnet(self) -> None:
    self._magnet_network.eval()
    for parameter in self._magnet_network.parameters():
      parameter.requires_grad_(False)

  @staticmethod
  def _masked_logits(
      logits: torch.Tensor, legal_actions_mask: torch.Tensor
  ) -> torch.Tensor:
    mask_value = torch.finfo(logits.dtype).min
    return logits.masked_fill(~legal_actions_mask, mask_value)

  def _distribution(
      self,
      network: nn.Module,
      info_states: torch.Tensor,
      legal_actions_mask: torch.Tensor,
  ) -> Categorical:
    logits = network(info_states)
    return Categorical(logits=self._masked_logits(logits, legal_actions_mask))

  def _legal_actions_mask(self, legal_actions) -> np.ndarray:
    mask = np.zeros(self._num_actions, dtype=bool)
    mask[np.asarray(legal_actions, dtype=np.int64)] = True
    if not mask.any():
      raise ValueError("OpenSpiel returned no legal actions")
    return mask

  def _act(
      self, time_step: rl_environment.TimeStep, record_transition: bool
  ) -> rl_agent.StepOutput:
    info_state = np.asarray(
        time_step.observations["info_state"][self.player_id],
        dtype=np.float32,
    ).reshape(-1)
    legal_actions_mask = self._legal_actions_mask(
        time_step.observations["legal_actions"][self.player_id]
    )
    info_state_tensor = torch.as_tensor(
        info_state, dtype=torch.float32, device=self._device
    ).unsqueeze(0)
    legal_mask_tensor = torch.as_tensor(
        legal_actions_mask, dtype=torch.bool, device=self._device
    ).unsqueeze(0)

    with torch.no_grad():
      distribution = self._distribution(
          self._policy_network, info_state_tensor, legal_mask_tensor
      )
      action = distribution.sample()
      probs = distribution.probs.squeeze(0).cpu().numpy()
      log_prob = distribution.log_prob(action).item()
      value = self._value_network(info_state_tensor).squeeze().item()

    action_id = int(action.item())
    if record_transition:
      self._pending_transition = _PendingTransition(
          info_state=info_state,
          action=action_id,
          legal_actions_mask=legal_actions_mask,
          log_prob=log_prob,
          value=value,
      )
    return rl_agent.StepOutput(action=action_id, probs=probs)

  def _can_act(self, time_step: rl_environment.TimeStep) -> bool:
    return not time_step.last() and (
        time_step.is_simultaneous_move()
        or self.player_id == time_step.current_player()
    )

  def _record_pending_transition(
      self, time_step: rl_environment.TimeStep
  ) -> None:
    if self._pending_transition is None:
      return
    reward = 0.0
    if time_step.rewards is not None:
      reward = float(time_step.rewards[self.player_id])
    self._episode.append(
        _EpisodeTransition(
            pending=self._pending_transition,
            reward=reward,
            terminal=time_step.last(),
        )
    )
    self._pending_transition = None

  def _finish_episode(self) -> None:
    if not self._episode:
      return

    advantages = np.zeros(len(self._episode), dtype=np.float32)
    returns = np.zeros(len(self._episode), dtype=np.float32)
    gae = 0.0
    next_value = 0.0
    for index in reversed(range(len(self._episode))):
      transition = self._episode[index]
      non_terminal = 0.0 if transition.terminal else 1.0
      delta = (
          transition.reward
          + self._discount_factor * next_value * non_terminal
          - transition.pending.value
      )
      gae = (
          delta + self._discount_factor * self._gae_lambda * non_terminal * gae
      )
      advantages[index] = gae
      returns[index] = gae + transition.pending.value
      next_value = transition.pending.value

    for transition, advantage, return_ in zip(
        self._episode, advantages, returns
    ):
      self._training_samples.append(
          _TrainingSample(
              info_state=transition.pending.info_state,
              action=transition.pending.action,
              legal_actions_mask=transition.pending.legal_actions_mask,
              log_prob=transition.pending.log_prob,
              value=transition.pending.value,
              advantage=float(advantage),
              return_=float(return_),
          )
      )

    self._episode.clear()
    self._episode_counter += 1
    if self._episode_counter % self._learn_every == 0:
      self._learn()

  def _learn(self) -> None:
    if not self._training_samples:
      return

    samples = self._training_samples
    info_states = torch.as_tensor(
        np.stack([sample.info_state for sample in samples]),
        dtype=torch.float32,
        device=self._device,
    )
    actions = torch.as_tensor(
        [sample.action for sample in samples],
        dtype=torch.long,
        device=self._device,
    )
    legal_actions_mask = torch.as_tensor(
        np.stack([sample.legal_actions_mask for sample in samples]),
        dtype=torch.bool,
        device=self._device,
    )
    old_log_probs = torch.as_tensor(
        [sample.log_prob for sample in samples],
        dtype=torch.float32,
        device=self._device,
    )
    advantages = torch.as_tensor(
        [sample.advantage for sample in samples],
        dtype=torch.float32,
        device=self._device,
    )
    returns = torch.as_tensor(
        [sample.return_ for sample in samples],
        dtype=torch.float32,
        device=self._device,
    )
    if len(samples) > 1:
      advantages = (advantages - advantages.mean()) / (
          advantages.std(unbiased=False) + 1e-8
      )

    indices = np.arange(len(samples))
    losses = []
    metrics = []
    for _ in range(self._ppo_epochs):
      np.random.shuffle(indices)
      for start in range(0, len(samples), self._batch_size):
        batch_indices = indices[start : start + self._batch_size]
        batch = torch.as_tensor(
            batch_indices, dtype=torch.long, device=self._device
        )
        distribution = self._distribution(
            self._policy_network,
            info_states[batch],
            legal_actions_mask[batch],
        )
        new_log_probs = distribution.log_prob(actions[batch])
        ratio = torch.exp(new_log_probs - old_log_probs[batch])
        batch_advantages = advantages[batch]
        clipped_ratio = torch.clamp(
            ratio, 1.0 - self._clip_epsilon, 1.0 + self._clip_epsilon
        )
        policy_loss = -torch.minimum(
            ratio * batch_advantages, clipped_ratio * batch_advantages
        ).mean()

        values = self._value_network(info_states[batch]).squeeze(-1)
        value_loss = 0.5 * F.mse_loss(values, returns[batch])
        entropy = distribution.entropy().mean()

        with torch.no_grad():
          magnet_distribution = self._distribution(
              self._magnet_network,
              info_states[batch],
              legal_actions_mask[batch],
          )
          magnet_probs = magnet_distribution.probs
        # Categorical normalizes ``logits`` and exposes log probabilities here.
        new_log_probs_all = distribution.logits
        magnet_kl = torch.sum(
            magnet_probs * (torch.log(magnet_probs + 1e-8) - new_log_probs_all),
            dim=-1,
        ).mean()

        loss = (
            policy_loss
            + value_loss
            + self._magnet_coefficient * magnet_kl
            - self._entropy_coefficient * entropy
        )
        self._optimizer.zero_grad()
        loss.backward()
        if self._gradient_clipping is not None:
          nn.utils.clip_grad_norm_(
              list(self._policy_network.parameters())
              + list(self._value_network.parameters()),
              self._gradient_clipping,
          )
        self._optimizer.step()

        losses.append(loss.item())
        metrics.append({
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "magnet_kl": magnet_kl.item(),
        })

    self._training_samples.clear()
    self._num_updates += 1
    self._last_loss_value = float(np.mean(losses))
    self._last_metrics = {
        "loss": self._last_loss_value,
        **{
            key: float(np.mean([metric[key] for metric in metrics]))
            for key in metrics[0]
        },
    }
    if (
        self._auto_update_magnet
        and self._num_updates % self._magnet_update_period == 0
    ):
      self.update_magnet()

  def update_magnet(self) -> None:
    """Copies the current policy into the fixed KL-regularization policy."""
    self._magnet_network.load_state_dict(self._policy_network.state_dict())
    self._freeze_magnet()

  def step(
      self, time_step: rl_environment.TimeStep, is_evaluation: bool = False
  ) -> rl_agent.StepOutput | None:
    """Returns an action and records on-policy data during training."""
    if is_evaluation:
      if not self._can_act(time_step):
        return rl_agent.StepOutput(action=None, probs=[])
      return self._act(time_step, record_transition=False)

    self._step_counter += 1
    if self._pending_transition is not None and (
        time_step.last() or self._can_act(time_step)
    ):
      self._record_pending_transition(time_step)

    if time_step.last():
      self._finish_episode()
      return None
    if not self._can_act(time_step):
      return rl_agent.StepOutput(action=None, probs=[])
    return self._act(time_step, record_transition=True)

  @property
  def loss(self) -> float | None:
    return self._last_loss_value

  @property
  def metrics(self) -> dict:
    return dict(self._last_metrics)

  @property
  def step_counter(self) -> int:
    return self._step_counter

  @property
  def num_updates(self) -> int:
    return self._num_updates

  @property
  def policy_network(self) -> nn.Module:
    return self._policy_network

  @property
  def magnet_network(self) -> nn.Module:
    return self._magnet_network

  def save(self, data_path: pathlib.Path, save_optimiser: bool = True) -> None:
    """Saves policy, value, magnet, and optional optimizer state."""
    checkpoint = {
        "policy": self._policy_network.state_dict(),
        "value": self._value_network.state_dict(),
        "magnet": self._magnet_network.state_dict(),
        "step_counter": self._step_counter,
        "episode_counter": self._episode_counter,
        "num_updates": self._num_updates,
        "last_loss_value": self._last_loss_value,
    }
    if save_optimiser:
      checkpoint["optimizer"] = self._optimizer.state_dict()
    torch.save(checkpoint, data_path)

  def load(self, data_path: pathlib.Path, load_optimiser: bool = True) -> None:
    """Loads policy, value, magnet, and optional optimizer state."""
    checkpoint = torch.load(
        data_path, weights_only=True, map_location=self._device
    )
    self._policy_network.load_state_dict(checkpoint["policy"])
    self._value_network.load_state_dict(checkpoint["value"])
    self._magnet_network.load_state_dict(checkpoint["magnet"])
    self._freeze_magnet()
    if load_optimiser and "optimizer" in checkpoint:
      self._optimizer.load_state_dict(checkpoint["optimizer"])
    self._step_counter = checkpoint["step_counter"]
    self._episode_counter = checkpoint["episode_counter"]
    self._num_updates = checkpoint["num_updates"]
    self._last_loss_value = checkpoint["last_loss_value"]
