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
"""Mean field proximal policy optimaztion algorithm.

Reference:

    Algumaei, Talal, et al. "Regularization of the policy updates for
    stabilizing
    Mean Field Games." Pacific-Asia Conference on Knowledge Discovery and Data
    Mining. Cham: Springer Nature Switzerland, 2023. Available at:
    https://link.springer.com/chapter/10.1007/978-3-031-33377-4_28
"""

# pylint: disable=consider-using-from-import
# pylint: disable=g-importing-member

import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F

from open_spiel.python import policy as policy_std
from open_spiel.python.mfg import value
from open_spiel.python.mfg.algorithms import best_response_value
from open_spiel.python.mfg.algorithms import policy_value
from open_spiel.python.mfg.algorithms.nash_conv import NashConv


class NashC(NashConv):
  """Mainly used to calculate the exploitability."""

  def __init__(self, game, distrib, pi_value, root_state=None):
    self._game = game
    if root_state is None:
      self._root_states = game.new_initial_states()
    else:
      self._root_states = [root_state]
    self._distrib = distrib
    self._pi_value = pi_value
    self._br_value = best_response_value.BestResponse(
        self._game,
        self._distrib,
        value.TabularValueFunction(self._game),
        root_state=root_state,
    )


class Agent(nn.Module):
  """Mainly used to calculate the exploitability."""

  def __init__(self, info_state_size, num_actions):
    super(Agent, self).__init__()
    self.num_actions = num_actions
    self.info_state_size = info_state_size
    self.critic = nn.Sequential(
        self.layer_init(nn.Linear(info_state_size, 128)),
        nn.Tanh(),
        self.layer_init(nn.Linear(128, 128)),
        nn.Tanh(),
        self.layer_init(nn.Linear(128, 1)),
    )
    self.actor = nn.Sequential(
        self.layer_init(nn.Linear(info_state_size, 128)),
        nn.Tanh(),
        self.layer_init(nn.Linear(128, 128)),
        nn.Tanh(),
        self.layer_init(nn.Linear(128, num_actions)),
    )

  def layer_init(self, layer, bias_const=0.0):
    """Used to initalize layers."""
    nn.init.xavier_normal_(layer.weight)
    nn.init.constant_(layer.bias, bias_const)
    return layer

  def get_value(self, x):
    """Get the value of the state."""
    return self.critic(x)

  def get_action_and_value(self, x, action=None):
    """Get the action and value of the state."""
    logits = self.actor(x)
    probs = Categorical(logits=logits)
    if action is None:
      action = probs.sample()
    return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class Policy(policy_std.Policy):
  """Required obeject to work with OpenSpiel.

  Used in updating the distribution using the policy nd in calculating the
  nash-convergance.
  """

  def __init__(self, game, agent, player_ids, device):
    super().__init__(game, player_ids)
    self.agent = agent
    self.device = device

  def action_probabilities(self, state, player_id=None):
    """Calculate the action probabilities of the state."""
    obs = torch.Tensor(state.observation_tensor()).to(self.device)
    legal_actions = state.legal_actions()
    logits = self.agent.actor(obs).detach().cpu()
    legat_logits = np.array([logits[action] for action in legal_actions])
    probs = np.exp(legat_logits - legat_logits.max())
    probs /= probs.sum(axis=0)

    # returns a dict with actions as keys and their probabilities as values
    return {
        action: probs[legal_actions.index(action)] for action in legal_actions
    }


def rollout(env, iter_agent, eps_agent, num_epsiodes, steps, device):
  """Generates num_epsiodes rollouts."""
  info_state = torch.zeros((steps, iter_agent.info_state_size), device=device)
  actions = torch.zeros((steps,), device=device)
  logprobs = torch.zeros((steps,), device=device)
  rewards = torch.zeros((steps,), device=device)
  dones = torch.zeros((steps,), device=device)
  values = torch.zeros((steps,), device=device)
  entropies = torch.zeros((steps,), device=device)
  t_actions = torch.zeros((steps,), device=device)
  t_logprobs = torch.zeros((steps,), device=device)

  step = 0
  for _ in range(num_epsiodes):
    time_step = env.reset()
    while not time_step.last():
      obs = time_step.observations["info_state"][0]
      obs = torch.Tensor(obs).to(device)
      info_state[step] = obs
      with torch.no_grad():
        t_action, t_logprob, _, _ = iter_agent.get_action_and_value(obs)
        action, logprob, entropy, ivalue = eps_agent.get_action_and_value(obs)

      time_step = env.step([action.item()])

      # iteration policy data
      t_logprobs[step] = t_logprob
      t_actions[step] = t_action

      # episode policy data
      logprobs[step] = logprob
      dones[step] = time_step.last()
      entropies[step] = entropy
      values[step] = ivalue
      actions[step] = action
      rewards[step] = torch.Tensor(time_step.rewards).to(device)
      step += 1

  history = {
      "info_state": info_state,
      "actions": actions,
      "logprobs": logprobs,
      "rewards": rewards,
      "dones": dones,
      "values": values,
      "entropies": entropies,
      "t_actions": t_actions,
      "t_logprobs": t_logprobs,
  }
  return history


def calculate_advantage(gamma, norm, rewards, values, dones, device):
  """Function used to calculate the Generalized Advantage estimate."""
  with torch.no_grad():
    next_done = dones[-1]
    next_value = values[-1]
    steps = len(values)
    returns = torch.zeros_like(rewards).to(device)
    for t in reversed(range(steps)):
      if t == steps - 1:
        nextnonterminal = 1.0 - next_done
        next_return = next_value
      else:
        nextnonterminal = 1.0 - dones[t + 1]
        next_return = returns[t + 1]
      returns[t] = rewards[t] + gamma * nextnonterminal * next_return

    advantages = returns - values

  if norm:
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

  return advantages, returns


def learn(
    history,
    optimizer_actor,
    optimize_critic,
    agent,
    num_minibatches=5,
    update_epochs=5,
    itr_eps=0.05,
    eps_eps=0.2,
    alpha=0.5,
    ent_coef=0.01,
    max_grad_norm=5,
):
  """Update the agent network (actor and critic)."""
  v_loss = None
  batch_size = history["actions"].shape[0]
  b_inds = np.arange(batch_size)
  mini_batch_size = batch_size // num_minibatches
  # get batch indices
  np.random.shuffle(b_inds)
  for _ in range(update_epochs):
    for start in range(0, batch_size, mini_batch_size):
      end = start + mini_batch_size
      mb_inds = b_inds[start:end]
      # for each update epoch shuffle the batch indices
      # generate the new logprobs, entropy and value then calculate the ratio
      b_obs = history["info_state"][mb_inds]
      b_advantages = history["advantages"][mb_inds]

      # Get the data under the episode policy (representative agent current
      # policy)
      _, newlogprob, entropy, new_value = agent.get_action_and_value(
          b_obs, history["actions"][mb_inds]
      )
      logratio = newlogprob - history["logprobs"][mb_inds]
      ratio = torch.exp(logratio)

      # Get the data under the iteration policy (the population policy)
      _, t_newlogprob, _, _ = agent.get_action_and_value(
          b_obs, history["t_actions"][mb_inds]
      )
      t_logratio = t_newlogprob - history["t_logprobs"][mb_inds]
      t_ratio = torch.exp(t_logratio)

      # iteration update PPO
      t_pg_loss1 = b_advantages * t_ratio
      t_pg_loss2 = b_advantages * torch.clamp(t_ratio, 1 - itr_eps, 1 + itr_eps)

      # episodic update PPO
      pg_loss1 = b_advantages * ratio
      pg_loss2 = b_advantages * torch.clamp(ratio, 1 - eps_eps, 1 + eps_eps)

      # Calculate the loss using our loss function
      pg_loss = (
          -alpha * torch.min(pg_loss1, pg_loss2).mean()
          - (1 - alpha) * torch.min(t_pg_loss1, t_pg_loss2).mean()
      )
      v_loss = F.smooth_l1_loss(
          new_value.reshape(-1), history["returns"][mb_inds]
      ).mean()
      entropy_loss = entropy.mean()
      loss = pg_loss - ent_coef * entropy_loss

      # Actor update
      optimizer_actor.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(agent.actor.parameters(), max_grad_norm)
      optimizer_actor.step()

      # Critic update
      optimize_critic.zero_grad()
      v_loss.backward()
      nn.utils.clip_grad_norm_(agent.critic.parameters(), max_grad_norm)
      optimize_critic.step()

  assert v_loss is not None
  return v_loss


def calculate_explotability(game, distrib, policy):
  """This function is used to log the results to tensor board."""
  initial_states = game.new_initial_states()
  pi_value = policy_value.PolicyValue(
      game, distrib, policy, value.TabularValueFunction(game)
  )
  m = {
      f"ppo_br/{state}": pi_value.eval_state(state) for state in initial_states
  }
  nashc = NashC(game, distrib, pi_value).nash_conv()
  m["nash_conv_ppo"] = nashc

  return m
