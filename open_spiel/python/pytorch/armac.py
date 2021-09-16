import collections
import torch as T
import torch.nn as nn
import torch.optim as O
import torch.nn.functional as F
import numpy as np
import numpy.random as rn
import pyspiel
from tqdm import tqdm
from open_spiel.python import policy
from copy import deepcopy

class ReservoirBuffer(object):
  """Allows uniform sampling over a stream of data.

  This class supports the storage of arbitrary elements, such as observation
  tensors, integer actions, etc.
  See https://en.wikipedia.org/wiki/Reservoir_sampling for more details.
  """

  def __init__(self, reservoir_buffer_capacity):
    self._reservoir_buffer_capacity = reservoir_buffer_capacity
    self._data = []
    self._add_calls = 0

  def add(self, element):
    """Potentially adds `element` to the reservoir buffer.

    Args:
      element: data to be added to the reservoir buffer.
    """
    if len(self._data) < self._reservoir_buffer_capacity:
      self._data.append(element)
    else:
      idx = rn.randint(0, self._add_calls + 1)
      if idx < self._reservoir_buffer_capacity:
        self._data[idx] = element
    self._add_calls += 1

  def sample(self, num_samples):
    """Returns `num_samples` uniformly sampled from the buffer.

    Args:
      num_samples: `int`, number of samples to draw.

    Returns:
      An iterable over `num_samples` random elements of the buffer.
    Raises:
      ValueError: If there are less than `num_samples` elements in the buffer
    """
    if len(self._data) < num_samples:
      raise ValueError("{} elements could not be sampled from size {}".format(
        num_samples, len(self._data)))
    return rn.choice(self._data, num_samples)

  def clear(self):
    self._data = []
    self._add_calls = 0

  def __len__(self):
    return len(self._data)

  def __iter__(self):
    return iter(self._data)

def state_to_matrix(state, device):
  t = T.tensor(state).to(device)
  if t.ndim == 1:
    return t.unsqueeze(0)
  else:
    return t

def action_to_matrix(action, device):
  t = T.LongTensor(action).to(device)
  if t.ndim == 1:
    return t.unsqueeze(0)
  else:
    return t

class ArmacNeuralNetwork(nn.Module):
  def __init__(self, information_state_size, num_distinct_actions, observation_generalization_size, hidden):
    super(ArmacNeuralNetwork, self).__init__()
    # observation layers of all the heads
    self.observation1 = nn.Linear(information_state_size, observation_generalization_size)
    self.observation2 = nn.Linear(information_state_size, observation_generalization_size)
    # first layer of history value head
    self.l1 = nn.Linear(observation_generalization_size * 2, hidden)
    # first layer of accumulative regrets head and average policy head
    self.l2s = [nn.Linear(observation_generalization_size, hidden) for _ in range(2)]
    # second layer of all three heads
    self.l3s = [nn.Linear(hidden, num_distinct_actions) for _ in range(3)]
    # to store which device on
    self.dummy = nn.Parameter(T.empty(0))

  def get_history_values(self, full_history_state, legal_actions):
    state1 = state_to_matrix(full_history_state[0], self.dummy.device)
    state2 = state_to_matrix(full_history_state[1], self.dummy.device)
    legal_actions = action_to_matrix(legal_actions, self.dummy.device)
    temp = T.cat([self.observation1(state1), self.observation2(state2)], 1)
    temp = self.l3s[0](F.sigmoid(self.l1(temp)))
    return T.gather(temp, 1, legal_actions)

  def get_single_history_value(self, full_history_state, legal_actions, action):
    action = [legal_actions.index(action)]
    action = action_to_matrix(action, self.dummy.device)
    return T.gather(self.get_history_values(full_history_state, legal_actions), 1, action)

  def get_max_history_value(self, full_history_state, legal_actions):
    return self.get_history_values(full_history_state, legal_actions).max(dim=1, keepdims=True).values

  def get_accumulative_regrets(self, state, legal_actions):
    state = state_to_matrix(state, self.dummy.device)
    legal_actions = action_to_matrix(legal_actions, self.dummy.device)
    temp = self.l3s[1](F.sigmoid(self.l2s[0](F.sigmoid(self.observation1(state)))))
    return T.gather(temp, 1, legal_actions)

  def get_average_strategy(self, state, legal_actions):
    state = state_to_matrix(state, self.dummy.device)
    legal_actions = action_to_matrix(legal_actions, self.dummy.device)
    temp = F.relu(self.l3s[2](F.sigmoid(self.l2s[1](F.sigmoid(self.observation1(state))))))
    temp = T.gather(temp, 1, legal_actions)
    if temp.sum() == .0:
      return temp  # won't create a uniform distribution vector, otherwise the gradient will miss
    else:
      return temp / temp.sum(1, keepdims=True)

  def get_history_values_without_grad(self, full_history_state, legal_actions):
    with T.no_grad():
      return self.get_history_values(full_history_state, legal_actions)

  def get_strategy_without_grad(self, state, legal_actions):
    with T.no_grad():
      temp = F.relu(self.get_accumulative_regrets(state, legal_actions)).to('cpu')
      if temp.sum() == .0:
        return temp # won't create a uniform distribution vector, otherwise the gradient will miss
      else:
        return temp / temp.sum(1, keepdims=True)

  def get_average_strategy_without_grad(self, state, legal_actions):
    with T.no_grad():
      return self.get_average_strategy(state, legal_actions).to('cpu')

#TRJ = collections.namedtuple('Record', 'history state action legal_actions player_seat regrets retrospective_policy')
TRJ = collections.namedtuple('Record', 'history state action legal_actions current_player regrets retrospective_policy')
class Trajectory:
  def __init__(self, player_seat, reward=None, capacity=100):
    #self.player = player
    self.player_seat = player_seat
    self.reward = reward
    self.buffer = []

  def add(self, history, state, action, legal_actions, acting_player, regrets, retrospective_policy):
    temp = TRJ(history, state, action, legal_actions, acting_player, regrets, retrospective_policy)
    self.buffer.append(temp)

  def __iter__(self):
    return iter(self.buffer)

  def __str__(self):
    return f'Trajectory : player {self.player_seat} reward {self.reward} length {len(self.buffer)}'

class ArmacSolver(policy.Policy):
  def __init__(self,
               game,
               device: str = 'cuda' if T.cuda.is_available() else 'cpu',
               num_learning: int = 64,
               learning_rate: float = 1e-4,
               batch_size: int = 1024,
               memory_capacity: int = int(1e6),
               reset_parameters: bool = False):
    all_players = list(range(game.num_players()))
    super(ArmacSolver, self).__init__(game, all_players)
    self._game = game
    self._device = device
    self._root_node = game.new_initial_state()
    self._feature_size = len(self._root_node.information_state_tensor(0))
    self._history_size = self._feature_size + 8
    self._num_actions = game.num_distinct_actions()
    self._num_learning = num_learning
    self._num_players = game.num_players()
    # This class will first focus on two-player zero-sum poker games
    assert(self._num_players == 2)
    self._batch_size = batch_size
    self._memory_capacity = memory_capacity

    self._retrospective_networks = []
    self._epoch_buffer = ReservoirBuffer(memory_capacity)
    self._network = ArmacNeuralNetwork(self._feature_size, self._num_actions, 240, 120)
    self._past_network = None
    self._optimizer = O.Adam(self._network.parameters(), lr=learning_rate)

    self._loss_MSE = nn.MSELoss().to(device)
    self._loss_div = nn.KLDivLoss().to(device)

    if reset_parameters:
        self.__class__.reset_parameters(self._network)

    self._player_seat = 1
    self._counter = 0

  @staticmethod
  def reset_parameters(network):
    for par in network.state_dict().values():
      par.fill_(.0)

  def _load_past_network(self):
    if len(self._retrospective_networks) == 0:
      self._past_network = self._network
    else:
      self._past_network = self._retrospective_networks[rn.randint(0, len(self._retrospective_networks))]

  def solve(self, num_iterations, num_traversals):
    for _ in range(num_iterations):
      self._epoch_buffer.clear()
      for _ in range(num_traversals):
        self._player_seat = (self._player_seat + 1) % self._num_players
        trajectory_buffer = Trajectory(self._player_seat, capacity=num_traversals)
        self._load_past_network()
        terminal_state = self._traverse_game_tree(self._root_node,
                                                  self._player_seat,
                                                  trajectory_buffer)
        reward = terminal_state.returns()[self._player_seat]
        trajectory_buffer.reward = reward
        self._epoch_buffer.add(trajectory_buffer)

      for _ in range(self._num_learning):
        self._learning()

      self._retrospective_networks.append(deepcopy(self._network).to('cpu'))

  def _learning(self):
    batch_trajectories = self._epoch_buffer.sample(self._batch_size)
    for trajectory in batch_trajectories:
      reward = T.FloatTensor([trajectory.reward]).to(self._device).unsqueeze(0)
      player_seat = trajectory.player_seat
      for idx in range(len(trajectory.buffer)):
        record = trajectory.buffer[idx]
        next_record = trajectory.buffer[idx+1]  if (idx+1) < len(trajectory.buffer) else None

        full_history = record.history
        state = record.state
        next_state = next_record.state if next_record else None
        next_full_history = next_record.history if next_record else None
        action = record.action
        legal_actions = record.legal_actions
        next_legal_actions = next_record.legal_actions if next_record else None
        current_player = record.current_player
        retrospective_policy = T.FloatTensor(record.retrospective_policy).to(self._device)

        self._optimizer.zero_grad()
        # history state value loss for all
        if next_state:
          l1 = self._loss_MSE(self._network.get_single_history_value(full_history, legal_actions, action),
                              self._network.get_max_history_value(next_full_history, next_legal_actions))
        else:
          l1 = self._loss_MSE(self._network.get_single_history_value(full_history, legal_actions, action),
                              reward)

        # accumulative regrets loss for player i
        if current_player == player_seat:
          regrets = T.FloatTensor(record.regrets).to(self._device)
          l2 = self._loss_MSE(self._network.get_accumulative_regrets(state, legal_actions), regrets) + l1
          l2.backward()
          self._optimizer.step()
        # average policy loss for player j
        else:
          l3 = self._loss_div(self._network.get_average_strategy(state, legal_actions), retrospective_policy) + l1
          l3.backward()
          self._optimizer.step()

  def _traverse_game_tree(self, state, player_seat, buffer):
    if state.is_terminal():
      return state

    elif state.is_chance_node():
      action = rn.choice([i[0] for i in state.chance_outcomes()])
      return self._traverse_game_tree(state.child(action), player_seat, buffer)

    else:
      # sample action to traverse game tree for both players
      actor_network = self._network if self._player_seat == state.current_player() else self._past_network
      information_state = state.information_state_tensor(state.current_player()) # not a torch tensor
      legal_actions = state.legal_actions()
      strategy = actor_network.get_strategy_without_grad(information_state, legal_actions)
      probabilities = np.array(strategy).squeeze()
      if probabilities.sum() == .0:
        probabilities = np.ones(len(legal_actions)) / len(legal_actions)
      action = rn.choice(legal_actions, 1, p=probabilities)
      past_strategy = self._past_network.get_strategy_without_grad(information_state, legal_actions)
      full_history_state = [state.information_state_tensor(0), state.information_state_tensor(1)]
      if state.current_player() == player_seat:
        past_history_values = self._past_network.get_history_values_without_grad(full_history_state, legal_actions)
        regrets = past_history_values - (past_strategy * past_history_values).sum(1, keepdims=True)
        buffer.add(full_history_state, information_state, action, legal_actions, state.current_player(), regrets, past_strategy)
      else:
        buffer.add(full_history_state, information_state, action, legal_actions, state.current_player(), None, past_strategy)

      return self._traverse_game_tree(state.child(action), player_seat, buffer)

  def action_probabilities(self, state, player_id=None):
    legal_actions = state.legal_actions()
    strategy = self._network.get_average_strategy_without_grad(state.information_state_tensor(state.current_player()), legal_actions)
    probabilities = np.array(strategy).squeeze()
    if probabilities.sum() == .0:
      probabilities = np.ones(len(legal_actions)) / len(legal_actions)
    return {legal_actions[i]: probabilities[i] for i in range(len(legal_actions))}


def main():
  current = 0
  iterations = list(range(100, 100100, 100))
  rn.seed(1234)
  G = pyspiel.load_game('leduc_poker')
  armac = ArmacSolver(G,
                  num_learning=50,
                  batch_size=32)
  f = open('armac_result.txt', 'wt')
  # for testing
#  iterations = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
#  iterations = [x//10 for x in iterations]

  with tqdm(total=iterations[-1]) as pbar:
    for iteration_ in iterations:
      armac.solve(num_iterations=(iteration_ - current), num_traversals = 64)
      conv = pyspiel.nash_conv(G, policy.python_policy_to_pyspiel_policy(policy.tabular_policy_from_callable(G, armac.action_probabilities)))
      f.write(f'{iteration_} {conv}\n')
      f.flush()
      pbar.update(iteration_ - current)
      current = iteration_

  f.close()


if __name__ == '__main__':
  main()