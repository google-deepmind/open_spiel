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

"""Neural Fictitious Self-Play (NFSP) agent implemented in PyTorch.

See the paper https://arxiv.org/abs/1603.01121 for more details.
"""

from typing import NamedTuple, Iterable
from enum import Enum, StrEnum
from pathlib import Path
import contextlib

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import tree as np_tree

from open_spiel.python import rl_agent
from open_spiel.python.pytorch import dqn

ILLEGAL_ACTION_LOGITS_PENALTY = torch.finfo(torch.float).min

def set_seed(seed):
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Transition(NamedTuple):
  info_state: np.ndarray
  action_probs: np.ndarray 
  legal_actions_mask: np.ndarray

class MODE(Enum):
  best_response=0
  average_policy=1


class Optimiser(StrEnum):
  SGD="sgd"
  ADAM="adam"
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
      input_size: (int) number of inputs
      hidden_sizes: (list) sizes (number of units) of each hidden layer
      output_size: (int) number of outputs
      final_activation: (nn.Module) an activation for the final later, defaults to None
      seed: (int) a random seed
    """

    super().__init__()
    set_seed(seed)

    _layers = []

    def _create_linear_block(in_features, out_features):
      return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.ReLU()
      )

    # Input and Hidden layers
    for size in hidden_sizes:
      _layers.append(_create_linear_block(input_size, size))
      input_size = size
    # Output layer
    _layers.append(nn.LayerNorm(input_size))
    _layers.append(nn.Linear(input_size, output_size))
    if final_activation:
      _layers.append(final_activation)
    self.model = nn.Sequential(*_layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.model(x)

class ReservoirBuffer:
  """Allows uniform sampling over a stream of data.
    See https://en.wikipedia.org/wiki/Reservoir_sampling for more details.
  """   
  def __init__(self, capacity: int, experience: Transition) -> None:
    self.capacity = capacity
    self.experience = experience
    self.add_calls = np.array(0)

  def __len__(self) -> int:
    return min(self.add_calls.item(), self.capacity.item())

  @classmethod
  def init(cls, capacity: int, experience: Transition) -> "ReservoirBuffer":
    # Initialize buffer by replicating the structure of the experience
    _experience = np_tree.map_structure(
      lambda x: np.empty((capacity, *x.shape), dtype=x.dtype),
      experience
    )
    return cls(np.array(capacity), _experience)

  def append(self, experience: Transition) -> None:
    """Potentially adds `experience` to the reservoir buffer.
    Args:
      experience: data to be added to the reservoir buffer.
      
    Returns:
      None as the method updated the buffer in-place
    """
    # Determine the insertion index
    # Note: count + 1 because the current item is the (count+1)-th item
    idx = np.random.randint(0, self.add_calls + 1)

    # 2. Logic: 
    # If buffer is not full, we always add at 'count'.
    # If buffer is full, we replace at 'idx' ONLY IF idx < capacity.
    is_full = self.add_calls >= self.capacity
    write_idx = np.where(is_full, idx, self.add_calls)
    should_update = write_idx < self.capacity

    def _inplace(arr, idx, val):
      arr[idx] = val

    if should_update:
      np_tree.map_structure(
        lambda buf_leaf, exp_leaf: _inplace(buf_leaf, write_idx, exp_leaf),
        self.experience, 
        experience,
      )
    self.add_calls += 1

  def sample(self, num_samples: int) -> Transition:
    """Returns `num_samples` uniformly sampled from the buffer.
    
    Args:
      num_samples: `int`, number of samples to draw.
    Returns:
      An iterable over `num_samples` random elements of the buffer.
    Raises:
      ValueError: If there are less than `num_samples` elements in the buffer
    """
    max_size = len(self)
    if max_size < num_samples:
      raise ValueError("{} elements could not be sampled from size {}".format(num_samples, max_size))

    indices = np.random.choice(max_size, size=(num_samples,), replace=False)

    return np_tree.map_structure(
      lambda data: data[indices],
      self.experience
    )

  def shuffle(self) -> None:
    """Shuffling the reservoir buffer along the batch axis
    """
    np_tree.map_structure(lambda x: np.random.shuffle(x[:len(self)]), self.experience)   

class NFSP(rl_agent.AbstractAgent):
  """NFSP Agent implementation in JAX.

  See open_spiel/python/examples/kuhn_nfsp.py for an usage example.
  """

  def __init__(
    self,
    player_id: int,
    state_representation_size: tuple[int, ...],
    num_actions: int,
    hidden_layers_sizes: Iterable[int],
    reservoir_buffer_capacity: int,
    anticipatory_param: float,
    replay_buffer_class: object = ReservoirBuffer,
    batch_size: int=128,
    rl_learning_rate: float=0.01,
    sl_learning_rate: float=0.01,
    min_buffer_size_to_learn: int=1000,
    learn_every: int=64,
    optimizer_str="sgd",
    gradient_clipping: float = None,
    seed: int = 42,
    **kwargs
  ) -> None:
    """Initialize the `NFSP` agent."""
    self.player_id = player_id
    self._num_actions = num_actions
    self._layer_sizes = hidden_layers_sizes
    self._batch_size = batch_size
    self._learn_every = learn_every
    self._anticipatory_param = anticipatory_param
    self._min_buffer_size_to_learn = min_buffer_size_to_learn

    self._replay_buffer_class = replay_buffer_class
    self._device = torch.device(kwargs.get("device", "cpu"))

    assert hasattr(replay_buffer_class, "init")
    assert hasattr(replay_buffer_class, "append")
    assert hasattr(replay_buffer_class, "sample")

    self._reservoir_buffer_capacity = int(reservoir_buffer_capacity)
    self._reservoir_buffer = None
    self._prev_timestep = None
    self._prev_action = None

    # Step counter to keep track of learning.
    self._iteration = 0

    # Inner RL agent
    kwargs.update({
        "batch_size": batch_size,
        "learning_rate": rl_learning_rate,
        "learn_every": learn_every,
        "min_buffer_size_to_learn": min_buffer_size_to_learn,
        "optimizer_str": optimizer_str,
    })
    self._rl_agent = dqn.DQN(
      player_id, 
      state_representation_size,
      num_actions, 
      hidden_layers_sizes, 
      **kwargs
    )

    # Keep track of the last training loss achieved in an update step.
    self._last_rl_loss_value = lambda: self._rl_agent.loss
    self._sl_loss_fn = F.cross_entropy
    self._last_sl_loss_value = None

    # Average policy network.
    self._avg_network = MLP(
      state_representation_size, 
      self._layer_sizes, 
      num_actions, 
      seed=seed+1
    ).to(self._device)

    set_seed(seed=seed+2)

    self._gradient_norm_clipping = gradient_clipping

    if optimizer_str == Optimiser.ADAM:
      self._optimizer = torch.optim.Adam(
        self._avg_network.parameters(), lr=sl_learning_rate)
    elif optimizer_str == Optimiser.SGD:
      self._optimizer = torch.optim.SGD(
        self._avg_network.parameters(), lr=sl_learning_rate)
    else:
      raise ValueError("Not implemented, choose from 'adam' and 'sgd'.")
       
    self._sample_episode_policy()

  @contextlib.contextmanager
  def temp_mode_as(self, mode):
    """Context manager to temporarily overwrite the mode."""
    previous_mode = self._mode
    self._mode = mode
    yield
    self._mode = previous_mode
    
  @property
  def step_counter(self) -> int:
    return self._iteration
  
  def _sample_episode_policy(self) -> None:
    if np.random.uniform() < self._anticipatory_param:
      self._mode = MODE.best_response
    else:
      self._mode = MODE.average_policy

  def _act(self, info_state: np.ndarray, legal_actions: np.ndarray):
    action_values = self._avg_network(
      torch.FloatTensor(info_state, device=self._device).unsqueeze(0)
    ).squeeze(0)
    # Remove illegal actions, normalize probs
    probs = torch.where(
      legal_actions, 
      action_values, 
      torch.full_like(action_values, ILLEGAL_ACTION_LOGITS_PENALTY)
    )
    probs = F.softmax(probs, dim=-1).detach().cpu().numpy()
    action = np.random.choice(np.arange(len(probs)), p=probs)
    return action_values, action, probs

  @property
  def mode(self):
    return self._mode

  @property
  def loss(self):
    return (self._last_sl_loss_value, self._last_rl_loss_value())

  def step(self, time_step, is_evaluation=False):
    """Returns the action to be taken and updates the Q-networks if needed.

    Args:
      time_step: an instance of rl_environment.TimeStep.
      is_evaluation: bool, whether this is a training or evaluation call.

    Returns:
      A `rl_agent.StepOutput` containing the action probs and chosen action.
    """
    if self._mode == MODE.best_response:
      agent_output = self._rl_agent.step(time_step, is_evaluation)
      if not is_evaluation and not time_step.last():
        self._add_transition(time_step, agent_output)

    elif self._mode == MODE.average_policy:
      # Act step: don't act at terminal info states.
      if not time_step.last():
        info_state = time_step.observations["info_state"][self.player_id]
        legal_actions = time_step.observations["legal_actions"][self.player_id]
        action_values, action, probs = self._act(
          np.asarray(info_state), 
          F.one_hot(
            torch.LongTensor(legal_actions, device=self._device), self._num_actions
          ).sum(0).to(torch.bool)
        )
        self._last_action_values = action_values

        agent_output = rl_agent.StepOutput(action=action, probs=probs)

      if self._prev_timestep and not is_evaluation:
        self._rl_agent._add_transition(self._prev_timestep, self._prev_action, time_step)
    else:
      raise ValueError("Invalid mode ({})".format(self._mode))

    if not is_evaluation:
      self._iteration += 1

      if self._iteration % self._learn_every == 0 and self._reservoir_buffer:
        self._last_sl_loss_value = self._learn()
        # If learn step not triggered by rl policy, learn.
        if self._mode == MODE.average_policy:
          self._rl_agent.learn()

      # Prepare for the next episode.
      if time_step.last():
        self._sample_episode_policy()
        self._prev_timestep = None
        self._prev_action = None
        return
      else:
        self._prev_timestep = time_step
        self._prev_action = agent_output.action
    return agent_output

  def _add_transition(self, time_step, agent_output: rl_agent.StepOutput) -> None:
    """Adds the new transition using `time_step` to the reservoir buffer.

    Transitions are in the form (time_step, agent_output.probs, legal_mask).

    Args:
      time_step: an instance of rl_environment.TimeStep.
      agent_output: an instance of rl_agent.StepOutput.
    """
    legal_actions = time_step.observations["legal_actions"][self.player_id]
    legal_actions_mask = np.zeros(self._num_actions)
    legal_actions_mask[legal_actions] = 1.0
    transition = Transition(
      info_state=np.asarray(time_step.observations["info_state"][self.player_id], dtype=np.float32),
      action_probs=np.asarray(agent_output.probs, dtype=np.float32),
      legal_actions_mask=np.asarray(legal_actions_mask, dtype=bool)
    )
    
    if self._reservoir_buffer is None:
      self._reservoir_buffer = self._replay_buffer_class.init(
        self._reservoir_buffer_capacity, transition)

    self._reservoir_buffer.append(transition)


  def _learn(self) -> None | float:
    """Compute the loss on sampled transitions and perform a avg-network update.

    If there are not enough elements in the buffer, no loss is computed and
    `None` is returned instead.

    Returns:
      The average loss obtained on this batch of transitions or `None`.
    """

    if (len(self._reservoir_buffer) < self._batch_size or
      len(self._reservoir_buffer) < self._min_buffer_size_to_learn):
      return None

    transitions = self._reservoir_buffer.sample(self._batch_size)

    info_states = torch.FloatTensor(transitions.info_state, device=self._device)
    action_probs = torch.FloatTensor(transitions.action_probs, device=self._device).detach()
    legal_actions_mask = torch.BoolTensor(transitions.legal_actions_mask, device=self._device)

    avg_actions_logits = self._avg_network(info_states)

    avg_actions_logits = torch.where(
      legal_actions_mask,
      avg_actions_logits,
      torch.full_like(avg_actions_logits, ILLEGAL_ACTION_LOGITS_PENALTY)
    )

    loss = self._sl_loss_fn(avg_actions_logits, action_probs)
    self._optimizer.zero_grad()
    loss.backward()

    if self._gradient_norm_clipping is not None:
      nn.utils.clip_grad_norm_(
        self._avg_network.parameters(), self._gradient_norm_clipping
      )
    self._optimizer.step()

    return loss.item()

  def save(self, checkpoint_dir: Path, save_optimiser: bool = True) -> None:
    """Saves the average policy network and the inner RL agent's q-network.

    Args:
      checkpoint_dir (pathlib.Path): directory from which checkpoints will be restored.
      save_optimiser (bool, optional): whether save only the optimiser (if it's been saved) 
        or just the network's weights. Defaults to True.
    """
    self._rl_agent.save(checkpoint_dir / "avg_network", save_optimiser)
    checkpoint = { 
      "iteration": self._iteration,
      "last_loss_value": self._last_sl_loss_value,
      "model": self._avg_network.state_dict(),
    }
    if save_optimiser:
      checkpoint.update({"optimiser": self._optimizer.state_dict()})
    torch.save(checkpoint, checkpoint_dir)


  def restore(self, checkpoint_dir: Path, load_optimiser: bool = True) -> None:
    """Restores the average policy network and the inner RL agent's q-network.

    Args:
      checkpoint_dir (pathlib.Path): directory from which checkpoints will be restored.
      load_optimiser (bool, optional): whether load only the optimiser (if it's been saved) 
        or just the network's weights. Defaults to True.
    """
    self._rl_agent.load(checkpoint_dir / "avg_network", load_optimiser)

    checkpoint = torch.load(checkpoint_dir, weights_only=True, map_location=self._device)
    self._avg_network.load_state_dict(checkpoint['model_state_dict'])

    if load_optimiser:
      self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    self._iteration = checkpoint["iteration"]
    self._last_sl_loss_value = checkpoint["last_loss_value"]
    self._last_rl_loss_value = self._rl_agent._last_loss_value

