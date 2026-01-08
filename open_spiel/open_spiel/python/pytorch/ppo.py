# Copyright 2022 DeepMind Technologies Limited
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

"""An implementation of PPO.

Note: code adapted (with permission) from
https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py and
https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo_atari.py.

Currently only supports the single-agent case.
"""

import time

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions.categorical import Categorical

from open_spiel.python.rl_agent import StepOutput

INVALID_ACTION_PENALTY = -1e6


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
  torch.nn.init.orthogonal_(layer.weight, std)
  torch.nn.init.constant_(layer.bias, bias_const)
  return layer


class CategoricalMasked(Categorical):
  """A masked categorical."""

  # pylint: disable=dangerous-default-value
  def __init__(self,
               probs=None,
               logits=None,
               validate_args=None,
               masks=[],
               mask_value=None):
    logits = torch.where(masks.bool(), logits, mask_value)
    super(CategoricalMasked, self).__init__(probs, logits, validate_args)


class PPOAgent(nn.Module):
  """A PPO agent module."""

  def __init__(self, num_actions, observation_shape, device):
    super().__init__()
    self.critic = nn.Sequential(
        layer_init(nn.Linear(np.array(observation_shape).prod(), 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 1), std=1.0),
    )
    self.actor = nn.Sequential(
        layer_init(nn.Linear(np.array(observation_shape).prod(), 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, num_actions), std=0.01),
    )
    self.device = device
    self.num_actions = num_actions
    self.register_buffer("mask_value", torch.tensor(INVALID_ACTION_PENALTY))

  def get_value(self, x):
    return self.critic(x)

  def get_action_and_value(self, x, legal_actions_mask=None, action=None):
    if legal_actions_mask is None:
      legal_actions_mask = torch.ones((len(x), self.num_actions)).bool()

    logits = self.actor(x)
    probs = CategoricalMasked(
        logits=logits, masks=legal_actions_mask, mask_value=self.mask_value)
    if action is None:
      action = probs.sample()
    return action, probs.log_prob(action), probs.entropy(), self.critic(
        x), probs.probs


class PPOAtariAgent(nn.Module):
  """A PPO Atari agent module."""

  def __init__(self, num_actions, observation_shape, device):
    super(PPOAtariAgent, self).__init__()
    # Note: this network is intended for atari games, taken from
    # https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo_atari.py
    self.network = nn.Sequential(
        layer_init(nn.Conv2d(4, 32, 8, stride=4)),
        nn.ReLU(),
        layer_init(nn.Conv2d(32, 64, 4, stride=2)),
        nn.ReLU(),
        layer_init(nn.Conv2d(64, 64, 3, stride=1)),
        nn.ReLU(),
        nn.Flatten(),
        layer_init(nn.Linear(64 * 7 * 7, 512)),
        nn.ReLU(),
    )
    self.actor = layer_init(nn.Linear(512, num_actions), std=0.01)
    self.critic = layer_init(nn.Linear(512, 1), std=1)
    self.num_actions = num_actions
    self.device = device
    self.register_buffer("mask_value", torch.tensor(INVALID_ACTION_PENALTY))

  def get_value(self, x):
    return self.critic(self.network(x / 255.0))

  def get_action_and_value(self, x, legal_actions_mask=None, action=None):
    if legal_actions_mask is None:
      legal_actions_mask = torch.ones((len(x), self.num_actions)).bool()

    hidden = self.network(x / 255.0)
    logits = self.actor(hidden)
    probs = CategoricalMasked(
        logits=logits, masks=legal_actions_mask, mask_value=self.mask_value)

    if action is None:
      action = probs.sample()
    return action, probs.log_prob(action), probs.entropy(), self.critic(
        hidden), probs.probs


def legal_actions_to_mask(legal_actions_list, num_actions):
  """Converts a list of legal actions to a mask.

  The mask has size num actions with a 1 in a legal positions.

  Args:
    legal_actions_list: the list of legal actions
    num_actions: number of actions (width of mask)

  Returns:
    legal actions mask.
  """
  legal_actions_mask = torch.zeros((len(legal_actions_list), num_actions),
                                   dtype=torch.bool)
  for i, legal_actions in enumerate(legal_actions_list):
    legal_actions_mask[i, legal_actions] = 1
  return legal_actions_mask


class PPO(nn.Module):
  """PPO Agent implementation in PyTorch.

  See open_spiel/python/examples/ppo_example.py for an usage example.

  Note that PPO runs multiple environments concurrently on each step (see
  open_spiel/python/vector_env.py). In practice, this tends to improve PPO's
  performance. The number of parallel environments is controlled by the
  num_envs argument.
  """

  def __init__(
      self,
      input_shape,
      num_actions,
      num_players,
      player_id=0,
      num_envs=1,
      steps_per_batch=128,
      num_minibatches=4,
      update_epochs=4,
      learning_rate=2.5e-4,
      gae=True,
      gamma=0.99,
      gae_lambda=0.95,
      normalize_advantages=True,
      clip_coef=0.2,
      clip_vloss=True,
      entropy_coef=0.01,
      value_coef=0.5,
      max_grad_norm=0.5,
      target_kl=None,
      device="cpu",
      writer=None,  # Tensorboard SummaryWriter
      agent_fn=PPOAtariAgent,
  ):
    super().__init__()

    self.input_shape = input_shape
    self.num_actions = num_actions
    self.num_players = num_players
    self.player_id = player_id
    self.device = device

    # Training settings
    self.num_envs = num_envs
    self.steps_per_batch = steps_per_batch
    self.batch_size = self.num_envs * self.steps_per_batch
    self.num_minibatches = num_minibatches
    self.minibatch_size = self.batch_size // self.num_minibatches
    self.update_epochs = update_epochs
    self.learning_rate = learning_rate

    # Loss function
    self.gae = gae
    self.gamma = gamma
    self.gae_lambda = gae_lambda
    self.normalize_advantages = normalize_advantages
    self.clip_coef = clip_coef
    self.clip_vloss = clip_vloss
    self.entropy_coef = entropy_coef
    self.value_coef = value_coef
    self.max_grad_norm = max_grad_norm
    self.target_kl = target_kl

    # Logging
    self.writer = writer

    # Initialize networks
    self.network = agent_fn(self.num_actions, self.input_shape,
                            device).to(device)
    self.optimizer = optim.Adam(
        self.parameters(), lr=self.learning_rate, eps=1e-5)

    # Initialize training buffers
    self.legal_actions_mask = torch.zeros(
        (self.steps_per_batch, self.num_envs, self.num_actions),
        dtype=torch.bool).to(device)
    self.obs = torch.zeros((self.steps_per_batch, self.num_envs) +
                           self.input_shape).to(device)
    self.actions = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)
    self.logprobs = torch.zeros(
        (self.steps_per_batch, self.num_envs)).to(device)
    self.rewards = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)
    self.dones = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)
    self.values = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)

    # Initialize counters
    self.cur_batch_idx = 0
    self.total_steps_done = 0
    self.updates_done = 0
    self.start_time = time.time()

  def get_value(self, x):
    return self.network.get_value(x)

  def get_action_and_value(self, x, legal_actions_mask=None, action=None):
    return self.network.get_action_and_value(x, legal_actions_mask, action)

  def step(self, time_step, is_evaluation=False):
    if is_evaluation:
      with torch.no_grad():
        legal_actions_mask = legal_actions_to_mask([
            ts.observations["legal_actions"][self.player_id] for ts in time_step
        ], self.num_actions).to(self.device)
        obs = torch.Tensor(
            np.array([
                np.reshape(ts.observations["info_state"][self.player_id],
                           self.input_shape) for ts in time_step
            ])).to(self.device)
        action, _, _, value, probs = self.get_action_and_value(
            obs, legal_actions_mask=legal_actions_mask)
        return [
            StepOutput(action=a.item(), probs=p)
            for (a, p) in zip(action, probs)
        ]
    else:
      with torch.no_grad():
        # act
        obs = torch.Tensor(
            np.array([
                np.reshape(ts.observations["info_state"][self.player_id],
                           self.input_shape) for ts in time_step
            ])).to(self.device)
        legal_actions_mask = legal_actions_to_mask([
            ts.observations["legal_actions"][self.player_id] for ts in time_step
        ], self.num_actions).to(self.device)
        action, logprob, _, value, probs = self.get_action_and_value(
            obs, legal_actions_mask=legal_actions_mask)

        # store
        self.legal_actions_mask[self.cur_batch_idx] = legal_actions_mask
        self.obs[self.cur_batch_idx] = obs
        self.actions[self.cur_batch_idx] = action
        self.logprobs[self.cur_batch_idx] = logprob
        self.values[self.cur_batch_idx] = value.flatten()

        agent_output = [
            StepOutput(action=a.item(), probs=p)
            for (a, p) in zip(action, probs)
        ]
        return agent_output

  def post_step(self, reward, done):
    self.rewards[self.cur_batch_idx] = torch.tensor(reward).to(
        self.device).view(-1)
    self.dones[self.cur_batch_idx] = torch.tensor(done).to(self.device).view(-1)

    self.total_steps_done += self.num_envs
    self.cur_batch_idx += 1

  def learn(self, time_step):
    next_obs = torch.Tensor(
        np.array([
            np.reshape(ts.observations["info_state"][self.player_id],
                       self.input_shape) for ts in time_step
        ])).to(self.device)

    # bootstrap value if not done
    with torch.no_grad():
      next_value = self.get_value(next_obs).reshape(1, -1)
      if self.gae:
        advantages = torch.zeros_like(self.rewards).to(self.device)
        lastgaelam = 0
        for t in reversed(range(self.steps_per_batch)):
          nextvalues = next_value if t == self.steps_per_batch - 1 else self.values[
              t + 1]
          nextnonterminal = 1.0 - self.dones[t]
          delta = self.rewards[
              t] + self.gamma * nextvalues * nextnonterminal - self.values[t]
          advantages[
              t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + self.values
      else:
        returns = torch.zeros_like(self.rewards).to(self.device)
        for t in reversed(range(self.steps_per_batch)):
          next_return = next_value if t == self.steps_per_batch - 1 else returns[
              t + 1]
          nextnonterminal = 1.0 - self.dones[t]
          returns[
              t] = self.rewards[t] + self.gamma * nextnonterminal * next_return
        advantages = returns - self.values

    # flatten the batch
    b_legal_actions_mask = self.legal_actions_mask.reshape(
        (-1, self.num_actions))
    b_obs = self.obs.reshape((-1,) + self.input_shape)
    b_logprobs = self.logprobs.reshape(-1)
    b_actions = self.actions.reshape(-1)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = self.values.reshape(-1)

    # Optimizing the policy and value network
    b_inds = np.arange(self.batch_size)
    clipfracs = []
    for _ in range(self.update_epochs):
      np.random.shuffle(b_inds)
      for start in range(0, self.batch_size, self.minibatch_size):
        end = start + self.minibatch_size
        mb_inds = b_inds[start:end]

        _, newlogprob, entropy, newvalue, _ = self.get_action_and_value(
            b_obs[mb_inds],
            legal_actions_mask=b_legal_actions_mask[mb_inds],
            action=b_actions.long()[mb_inds])
        logratio = newlogprob - b_logprobs[mb_inds]
        ratio = logratio.exp()

        with torch.no_grad():
          # calculate approx_kl http://joschu.net/blog/kl-approx.html
          old_approx_kl = (-logratio).mean()
          approx_kl = ((ratio - 1) - logratio).mean()
          clipfracs += [
              ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
          ]

        mb_advantages = b_advantages[mb_inds]
        if self.normalize_advantages:
          mb_advantages = (mb_advantages - mb_advantages.mean()) / (
              mb_advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef,
                                                1 + self.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalue = newvalue.view(-1)
        if self.clip_vloss:
          v_loss_unclipped = (newvalue - b_returns[mb_inds])**2
          v_clipped = b_values[mb_inds] + torch.clamp(
              newvalue - b_values[mb_inds],
              -self.clip_coef,
              self.clip_coef,
          )
          v_loss_clipped = (v_clipped - b_returns[mb_inds])**2
          v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
          v_loss = 0.5 * v_loss_max.mean()
        else:
          v_loss = 0.5 * ((newvalue - b_returns[mb_inds])**2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - self.entropy_coef * entropy_loss + v_loss * self.value_coef

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

      if self.target_kl is not None:
        if approx_kl > self.target_kl:
          break

    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true -
                                                         y_pred) / var_y

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    if self.writer is not None:
      self.writer.add_scalar("charts/learning_rate",
                             self.optimizer.param_groups[0]["lr"],
                             self.total_steps_done)
      self.writer.add_scalar("losses/value_loss", v_loss.item(),
                             self.total_steps_done)
      self.writer.add_scalar("losses/policy_loss", pg_loss.item(),
                             self.total_steps_done)
      self.writer.add_scalar("losses/entropy", entropy_loss.item(),
                             self.total_steps_done)
      self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(),
                             self.total_steps_done)
      self.writer.add_scalar("losses/approx_kl", approx_kl.item(),
                             self.total_steps_done)
      self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs),
                             self.total_steps_done)
      self.writer.add_scalar("losses/explained_variance", explained_var,
                             self.total_steps_done)
      self.writer.add_scalar(
          "charts/SPS",
          int(self.total_steps_done / (time.time() - self.start_time)),
          self.total_steps_done)

    # Update counters
    self.updates_done += 1
    self.cur_batch_idx = 0

  def anneal_learning_rate(self, update, num_total_updates):
    # Annealing the rate
    frac = 1.0 - (update / num_total_updates)
    if frac <= 0:
      raise ValueError("Annealing learning rate to <= 0")
    lrnow = frac * self.learning_rate
    self.optimizer.param_groups[0]["lr"] = lrnow
