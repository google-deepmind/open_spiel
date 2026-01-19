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

"""DQN agent implemented in PyTorch."""
from typing import Iterable, NamedTuple, Callable
import functools
import pathlib
import enum

import numpy as np
import tree as np_tree
import torch
from torch import nn
import torch.nn.functional as F

from open_spiel.python import rl_agent

ILLEGAL_ACTION_LOGITS_PENALTY = torch.finfo(torch.float).min

class Transition(NamedTuple):
  """Data structure for the Replay buffer."""
  info_state: np.ndarray
  action: np.ndarray
  reward: np.ndarray 
  next_info_state: np.ndarray
  is_final_step: np.ndarray
  legal_actions_mask: np.ndarray

class ReplayBuffer:
  """Generic Replay memory for DQN."""   
  def __init__(self, capacity: np.ndarray, experience: Transition) -> None:
    self.capacity = capacity
    self.experience = experience
    self.entry_index = np.array(0)

  def __len__(self) -> int:
    return min(self.entry_index, self.capacity).astype(int)

  @classmethod
  def init(cls, capacity: int, experience: Transition) -> "ReplayBuffer":
    # Initialize buffer by replicating the structure of the experience
    experience_ = np_tree.map_structure(
      lambda x: np.empty((capacity, *x.shape), dtype=x.dtype),
      experience
    )
    return cls(np.array(capacity), experience_)

  def append(
    self, 
    experience: Transition, 
  ) -> None:
    """Potentially adds `experience` to the replay buffer.
    Args:
      experience: data to be added to the replay buffer.
      
    Returns:
      None as the method updated the buffer in-place.
    """

    index = self.entry_index % self.capacity

    def _inplace(arr, idx, val):
      arr[idx] = val

    np_tree.map_structure(
      lambda buf_leaf, exp_leaf: _inplace(buf_leaf, index, exp_leaf),
      self.experience, 
      experience,
    )

    self.entry_index += 1
  
  def sample(self, num_samples: int) ->  Transition:
    """Returns `num_samples` uniformly sampled from the buffer.
    
    Args:
      num_samples: `int`, number of samples to draw.
    Returns:
      An iterable over `num_samples` random elements of the buffer.
    Raises:
      ValueError: If there are less than `num_samples` elements in the buffer.
    """
    max_size = len(self)
    if max_size < num_samples:
      raise ValueError(f"{num_samples} elements could not be sampled from size {max_size}")

    indices = np.random.choice(max_size, size=(num_samples,), replace=False)

    return np_tree.map_structure(
      lambda data: data[indices],
      self.experience
    )  
  
def set_seed(seed):
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MLP(nn.Module):
  """A simple network built from nn.linear layers."""

  def __init__(
      self,
      input_size: int,
      hidden_sizes: Iterable[int],
      output_size: int,
      final_activation: nn.Module = None,
      seed: int = 42
    ) -> None:
    """Create the MLP.
    Args:
      input_size: (int) number of inputs.
      hidden_sizes: (list) sizes (number of units) of each hidden layer.
      output_size: (int) number of outputs.
      final_activation: (nn.Module) final activation of the network.
        Defaults to None.
    """

    super().__init__()
    set_seed(seed)
    layers_ = []

    def _create_linear_block(in_features, out_features):
      return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.ReLU()
      )

    # Input and Hidden layers
    for size in hidden_sizes:
      layers_.append(_create_linear_block(input_size, size))
      input_size = size
    # Output layer
    layers_.append(nn.Linear(input_size, output_size))
    if final_activation:
      layers_.append(final_activation)
    self.model = nn.Sequential(*layers_)
    
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.model(x)

class Loss(enum.StrEnum):
  MSE="mse"
  HUBER="huber"

class Optimiser(enum.StrEnum):
  SGD="sgd"
  RMSPROP="rmsprop"
  ADAM="adam"

class EpsilonDecaySchedule(enum.StrEnum):
  LINEAR="linear"
  EXP="exp" 

# EPSILON DECAY SCHEDULES
def exponential_schedule(
    start_e: float, end_e: float, duration: float
  ) -> Callable:
  def _call(t: int) -> float:
    decay_steps = min(t, duration)
    return end_e + (start_e - end_e) * np.exp(-1. * decay_steps / duration)
  return _call

def linear_schedule(start_e: float, end_e: float, duration: int) -> Callable:
  slope = (end_e - start_e) / duration
  def _call(t: int) -> float:
    return max(slope * t + start_e, end_e)
  return _call

class DQN(rl_agent.AbstractAgent):
  """DQN Agent implementation in PyTorch.

  See open_spiel/python/examples/breakthrough_dqn.py for an usage example.
  """

  def __init__(
    self,
    player_id: int,
    state_representation_size: tuple[int, ...],
    num_actions: int,
    hidden_layers_sizes: Iterable[int] = (128,),
    batch_size: int=128,
    replay_buffer_class: Callable=ReplayBuffer,
    replay_buffer_capacity: int=10000,
    learning_rate: float=0.01,
    update_target_network_every: int=1000,
    weight_update_coeff: float = .995,
    learn_every: int=10,
    discount_factor: float=1.0,
    min_buffer_size_to_learn: int=1000,
    epsilon_start: float=1.0,
    epsilon_end: float =0.1,
    epsilon_decay_duration: int=int(1e6),
    epsilon_decay_schedule_str: EpsilonDecaySchedule = "exp",
    optimizer_str: Optimiser="sgd",
    loss_str: Loss="mse",
    huber_loss_parameter: float=1.0,
    seed: int = 42,
    gradient_clipping: float | None=None,
    device: str = "cpu"
  ) -> None:
    """Initialize the DQN agent."""

    # This call to locals() is used to store every argument used to initialise
    # the class instance, so it can be copied with no hyperparameter change.
    self._kwargs = locals()
    # Some type checks
    assert isinstance(player_id, int | None)
    assert isinstance(seed, int | None)
    assert isinstance(num_actions, int)
    assert isinstance(batch_size, int)
    assert isinstance(replay_buffer_capacity, int)
    assert isinstance(learn_every, int)
    assert isinstance(min_buffer_size_to_learn, int)
    assert isinstance(discount_factor, float)

    self.player_id = player_id
    self._num_actions = num_actions
    if isinstance(hidden_layers_sizes, int):
      hidden_layers_sizes = [hidden_layers_sizes]
    self._layer_sizes = hidden_layers_sizes
    self._batch_size = batch_size
    self._update_target_network_every = update_target_network_every
    self._learn_every = learn_every
    self._min_buffer_size_to_learn = min_buffer_size_to_learn
    self._discount_factor = discount_factor
    assert discount_factor >=0 and discount_factor <= 1

    self._device = torch.device(device)

    self._replay_buffer_class = replay_buffer_class

    assert hasattr(replay_buffer_class, "init")
    assert hasattr(replay_buffer_class, "append")
    assert hasattr(replay_buffer_class, "sample")

    self._replay_buffer_capacity = int(replay_buffer_capacity)
    self._replay_buffer = None
    
    self._tau = weight_update_coeff
    assert self._tau >=0 and self._tau < 1

    self._prev_timestep = None
    self._prev_action = None

    # Step counter to keep track of learning, eps decay and target network.
    self._iteration = 0

    # Keep track of the last training loss achieved in an update step.
    self._last_loss_value = None

    # Create the Q-network instances
    self._q_network = MLP(
      state_representation_size, 
      self._layer_sizes,
      num_actions,
      None, # outputs = raw logits
      seed=seed
    ).to(self._device)

    self._target_q_network = MLP(
      state_representation_size, 
      self._layer_sizes,
      num_actions,
      None, # outputs = raw logits
      seed=seed
    ).to(self._device)

    self._target_q_network.load_state_dict(self._q_network.state_dict())

    if loss_str == Loss.MSE:
      self.loss_class = F.mse_loss
    elif loss_str == Loss.HUBER:
      self.loss_class = functools.partial(
        F.smooth_l1_loss, beta=huber_loss_parameter
      )
    else:
      raise ValueError("Not implemented, choose from 'mse', 'huber'.")
    
    assert (epsilon_start > 0 and epsilon_end >= 0) and (epsilon_start >= epsilon_end)
    if epsilon_decay_schedule_str == EpsilonDecaySchedule.EXP:
      self.epsilon_schedule = exponential_schedule(
        epsilon_start, epsilon_end, epsilon_decay_duration
      )
    elif epsilon_decay_schedule_str == EpsilonDecaySchedule.LINEAR:
      self.epsilon_schedule = linear_schedule(
        epsilon_start, epsilon_end, epsilon_decay_duration
      )
    else:
      raise ValueError("Not implemented, choose from 'linear', 'exp'.")

    self._gradient_norm_clipping = gradient_clipping

    if optimizer_str == Optimiser.ADAM:
      self._optimizer = torch.optim.Adam(
        self._q_network.parameters(), lr=learning_rate)
    elif optimizer_str == Optimiser.RMSPROP:
      self._optimizer = torch.optim.RMSprop(
        self._q_network.parameters(), lr=learning_rate)
    elif optimizer_str == Optimiser.SGD:
      self._optimizer = torch.optim.SGD(
        self._q_network.parameters(), lr=learning_rate)
    else:
      raise ValueError("Not implemented, choose from 'adam', 'rmsprop', and 'sgd'.")
      
  @torch.inference_mode
  def select_action(self, time_step, greedy: bool = True) -> rl_agent.StepOutput:
    # Act step: don't act at terminal info states or if its not our turn.
    """Returns the action to be taken.

    Args:
      time_step: an instance of rl_environment.TimeStep.
      greedy: if the action needs to be greedy, not sampled

    Returns:
      A `rl_agent.StepOutput` containing the action probs and chosen action.
    """
    action = None
    probs = []
    if (not time_step.last()) and (
        time_step.is_simultaneous_move() or
        self.player_id == time_step.current_player()
      ):
      info_state = time_step.observations["info_state"][self.player_id]
      legal_actions = time_step.observations["legal_actions"][self.player_id]
      epsilon = self.epsilon_schedule(self._iteration) if not greedy else 0.0
      action, probs = self._act_epsilon_greedy(info_state, legal_actions, epsilon)
  
    return rl_agent.StepOutput(action=action, probs=probs)
  
  def act_epsilon_greedy(self, info_state, legal_actions, epsilon):
    return self._act_epsilon_greedy(info_state, legal_actions, epsilon)

  def step(self, time_step, is_evaluation: bool = False):
    """Returns the action to be taken and updates the Q-network if needed.

    Args:
      time_step: an instance of rl_environment.TimeStep.

    Returns:
      A `rl_agent.StepOutput` containing the action probs and chosen action.
    """
    if is_evaluation:
      return self.select_action(time_step, True)
    
    # Act step: don't act at terminal info states or if its not our turn.
    action, probs = self.select_action(time_step, False)

    self._iteration += 1

    if self._iteration % self._learn_every == 0:
      self._last_loss_value = self.learn()
      if self._last_loss_value != self._last_loss_value:
        return

    if self._iteration % self._update_target_network_every == 0:
      self._copy_weights(self._tau)

    if self._prev_timestep is not None and self._prev_action is not None:
      # We may omit record adding here if it's done elsewhere.
      self.add_transition(self._prev_timestep, self._prev_action, time_step)

    if time_step.last():  # prepare for the next episode.
      self._prev_timestep = None
      self._prev_action = None
      return None
    else:
      self._prev_timestep = time_step
      self._prev_action = action

    return rl_agent.StepOutput(action=action, probs=probs)

  def add_transition(self, prev_time_step, prev_action, time_step):
    """Adds the new transition using `time_step` to the replay buffer.

    Adds the transition from `self._prev_timestep` to `time_step` by
    `self._prev_action`.

    Args:
      prev_time_step: prev ts, an instance of rl_environment.TimeStep.
      prev_action: int, action taken at `prev_time_step`.
      time_step: current ts, an instance of rl_environment.TimeStep.
    """
    assert prev_time_step is not None
    legal_actions = time_step.observations["legal_actions"][self.player_id]
    legal_actions_mask = np.zeros(self._num_actions)
    legal_actions_mask[legal_actions] = 1.0
    transition = Transition(
        info_state=np.asarray(prev_time_step.observations["info_state"][self.player_id], dtype=np.float32),
        action=np.asarray(prev_action, dtype=int),
        reward=np.asarray(time_step.rewards[self.player_id], dtype=np.float32),
        next_info_state=np.asarray(time_step.observations["info_state"][self.player_id], dtype=np.float32),
        is_final_step=np.asarray(time_step.last(), dtype=bool),
        legal_actions_mask=np.asarray(legal_actions_mask, dtype=bool)
      )
    if self._replay_buffer is None:
      self._replay_buffer = self._replay_buffer_class.init(
        self._replay_buffer_capacity, transition)
      
    self._replay_buffer.append(transition)

  def _act_epsilon_greedy(self, info_state, legal_actions, epsilon):
    """Returns a valid epsilon-greedy action and valid action probs.

    Action probabilities are given by a softmax over legal q-values.

    Args:
      info_state: hashable representation of the information state.
      legal_actions: list of legal actions at `info_state`.
      epsilon: float, probability of taking an exploratory action.

    Returns:
      A valid epsilon-greedy action and valid action probabilities.
    """
    probs = np.zeros(self._num_actions)
    if np.random.rand() < epsilon:
      action = np.random.choice(legal_actions)
      probs[legal_actions] = 1.0 / len(legal_actions)
    else:
      info_state = torch.tensor(np.asarray(info_state).reshape(1, -1), device=self._device, dtype=torch.float32)
      q_values = self._q_network(info_state).detach().squeeze(0)
      legal_q_values = q_values[legal_actions]
      action = legal_actions[torch.argmax(legal_q_values)]
      probs[action] = 1.0

    probs = probs / probs.sum() 
    return action, probs
  
  def learn(self):
    """Compute the loss on sampled transitions and perform a Q-network update.

    If there are not enough elements in the buffer, no loss is computed and
    `None` is returned instead.

    Returns:
      The average loss obtained on this batch of transitions or `None`.
    """

    if (
      len(self._replay_buffer) < self._batch_size 
      or len(self._replay_buffer) < self._min_buffer_size_to_learn
    ):
      # return None because it's nothing to learn from
      return None

    transitions = self._replay_buffer.sample(self._batch_size)

    info_states = torch.tensor(transitions.info_state, device=self._device, dtype=torch.float32)
    actions = torch.tensor(transitions.action , device=self._device, dtype=torch.long)
    rewards = torch.tensor(transitions.reward, device=self._device, dtype=torch.float32)
    next_info_states = torch.tensor(transitions.next_info_state, device=self._device, dtype=torch.float32)
    are_final_steps = torch.tensor(transitions.is_final_step, device=self._device, dtype=torch.bool)
    legal_actions_mask = torch.tensor(transitions.legal_actions_mask, device=self._device, dtype=torch.bool)

    self._q_values = self._q_network(info_states)
    self._target_q_values = self._target_q_network(next_info_states).detach()

    illegal_actions_mask = torch.logical_not(legal_actions_mask)
    legal_target_q_values = self._target_q_values.masked_fill(
        illegal_actions_mask, ILLEGAL_ACTION_LOGITS_PENALTY
    )
    max_next_q = torch.max(legal_target_q_values, dim=1)[0]

    target = (
      rewards + torch.logical_not(are_final_steps) * self._discount_factor * max_next_q).detach()
    
    predictions = self._q_values.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)

    loss = self.loss_class(predictions, target)

    self._optimizer.zero_grad()
    loss.backward()

    if self._gradient_norm_clipping is not None:
      nn.utils.clip_grad_norm_(
        self._q_network.parameters(), self._gradient_norm_clipping
      )
    self._optimizer.step()

    return loss.item()

  @property
  def q_values(self):
    return self._q_values

  @property
  def replay_buffer(self):
    return self._replay_buffer

  @property
  def loss(self):
    return self._last_loss_value

  @property
  def prev_timestep(self):
    return self._prev_timestep

  @property
  def prev_action(self):
    return self._prev_action

  @property
  def step_counter(self):
    return self._iteration
  
  def _copy_weights(self, tau: float) -> None:
    """Soft update of the target network's weights.
      θ′ ← τ θ + (1 - τ )θ′

    Args:
        tau (float): main network parameters' weight.
    """
    #Ugly formatting
    for target_network_param, q_network_param in zip(
      self._target_q_network.parameters(), self._q_network.parameters()):

      target_network_param.data.copy_(
        tau * q_network_param.data + (1.0 - tau) * target_network_param.data
      )

  def save(self, data_path: pathlib.Path, save_optimiser: bool=True) -> None:
    """Save checkpoint/trained model and optimizer.

    Args:
      data_path: pathlib.Path for saving model. It can be relative or absolute but the
        filename should be included. For example: q_network.pt or
        /path/to/q_network.pt
      save_optimiser: bool whether to save the optimiser or not.
        Defaults to True.
    """
    checkpoint = { 
      "iteration": self._iteration,
      "last_loss_value": self._last_loss_value,
      "model": self._q_network.state_dict(),
    }
    if save_optimiser:
      checkpoint.update({"optimiser": self._optimizer.state_dict()})
    torch.save(checkpoint, data_path)


  def load(self, data_path: pathlib.Path, load_optimiser: bool=True) -> None:
    """Load checkpoint/trained model and optimizer.

    Args:
      data_path: pathlib.Path for loading model. It can be relative or absolute but the
        filename should be included. For example: q_network.pt or
        /path/to/q_network.pt
      load_optimiser: bool, whether to learn the optimiser's state of not. 
        Defaults to True.
    """
    checkpoint = torch.load(data_path, weights_only=True, map_location=self._device)
    self._q_network.load_state_dict(checkpoint["model_state_dict"])
    self._target_q_network.load_state_dict(checkpoint["model_state_dict"])
    
    if load_optimiser:
      self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    self._iteration = checkpoint["iteration"]
    self._last_loss_value = checkpoint["last_loss_value"]



