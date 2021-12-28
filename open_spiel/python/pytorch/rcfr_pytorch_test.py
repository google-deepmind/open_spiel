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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
# Note: this import needs to come before Tensorflow to fix a malloc error.
import pyspiel  # pylint: disable=g-bad-import-order

from open_spiel.python.pytorch import rcfr

_GAME = pyspiel.load_game('kuhn_poker')
_BOOLEANS = [False, True]

_BATCH_SIZE = 12


def _new_model():
  return rcfr.DeepRcfrModel(
      _GAME,
      num_hidden_layers=1,
      num_hidden_units=13,
      num_hidden_factors=1,
      use_skip_connections=True)


class RcfrTest(parameterized.TestCase, TestCase):

  def setUp(self):
    # pylint: disable=useless-super-delegation
    super(RcfrTest, self).setUp()

  def assertListAlmostEqual(self, list1, list2, delta=1e-06):
    self.assertEqual(len(list1), len(list2))
    for a, b in zip(list1, list2):
      self.assertAlmostEqual(a, b, delta=delta)

  def test_with_one_hot_action_features_single_state_vector(self):
    information_state_features = [1., 2., 3.]
    features = rcfr.with_one_hot_action_features(
        information_state_features,
        legal_actions=[0, 1],
        num_distinct_actions=3)
    self.assertEqual([
        [1., 2., 3., 1., 0., 0.],
        [1., 2., 3., 0., 1., 0.],
    ], features)

    features = rcfr.with_one_hot_action_features(
        information_state_features,
        legal_actions=[1, 2],
        num_distinct_actions=3)
    self.assertEqual([
        [1., 2., 3., 0., 1., 0.],
        [1., 2., 3., 0., 0., 1.],
    ], features)

  def test_sequence_features(self):
    state = _GAME.new_initial_state()
    while state.is_chance_node():
      state.apply_action(state.legal_actions()[0])
    assert len(state.legal_actions()) == 2
    features = rcfr.sequence_features(state, 3)

    x = state.information_state_tensor()
    self.assertEqual([x + [1, 0, 0], x + [0, 1, 0]], features)

  def test_num_features(self):
    assert rcfr.num_features(_GAME) == 13

  def test_root_state_wrapper_num_sequences(self):
    root_state_wrapper = rcfr.RootStateWrapper(_GAME.new_initial_state())
    assert root_state_wrapper.num_player_sequences[0] == 12
    assert root_state_wrapper.num_player_sequences[1] == 12

  def test_root_state_wrapper_sequence_indices(self):
    root_state_wrapper = rcfr.RootStateWrapper(_GAME.new_initial_state())
    self.assertEqual(
        {
            # Info state string -> initial sequence index map for player 1.
            '0': 0,
            '0pb': 2,
            '1': 4,
            '1pb': 6,
            '2': 8,
            '2pb': 10,
            # Info state string -> initial sequence index map for player 2.
            '1p': 0,
            '1b': 2,
            '2p': 4,
            '2b': 6,
            '0p': 8,
            '0b': 10,
        },
        root_state_wrapper.info_state_to_sequence_idx)

  def test_root_state_wrapper_sequence_features(self):
    root_state_wrapper = rcfr.RootStateWrapper(_GAME.new_initial_state())

    p1_info_state_features = [
        [1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0.],
        [1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0.],
        [1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 1., 1., 0., 0., 1., 0., 0.],
    ]
    p2_info_state_features = [
        [0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0.],
        [0., 1., 0., 1., 0., 0., 1., 0., 0., 0., 0.],
        [0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0.],
        [0., 1., 1., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0.],
    ]
    action_features = [[1., 0.], [0., 1.]]
    expected_p1_sequence_features = [
        p1_info_state_features[0] + action_features[0],
        p1_info_state_features[0] + action_features[1],
        p1_info_state_features[1] + action_features[0],
        p1_info_state_features[1] + action_features[1],
        p1_info_state_features[2] + action_features[0],
        p1_info_state_features[2] + action_features[1],
        p1_info_state_features[3] + action_features[0],
        p1_info_state_features[3] + action_features[1],
        p1_info_state_features[4] + action_features[0],
        p1_info_state_features[4] + action_features[1],
        p1_info_state_features[5] + action_features[0],
        p1_info_state_features[5] + action_features[1],
    ]
    expected_p2_sequence_features = [
        p2_info_state_features[0] + action_features[0],
        p2_info_state_features[0] + action_features[1],
        p2_info_state_features[1] + action_features[0],
        p2_info_state_features[1] + action_features[1],
        p2_info_state_features[2] + action_features[0],
        p2_info_state_features[2] + action_features[1],
        p2_info_state_features[3] + action_features[0],
        p2_info_state_features[3] + action_features[1],
        p2_info_state_features[4] + action_features[0],
        p2_info_state_features[4] + action_features[1],
        p2_info_state_features[5] + action_features[0],
        p2_info_state_features[5] + action_features[1],
    ]
    expected_sequence_features = [
        expected_p1_sequence_features, expected_p2_sequence_features
    ]

    self.assertEqual(expected_sequence_features,
                     root_state_wrapper.sequence_features)

  def test_root_state_wrapper_sequence_terminal_values(self):
    root_state_wrapper = rcfr.RootStateWrapper(_GAME.new_initial_state())

    expected_terminal_values = {}
    no_call_histories_p1_win = [
        '2, 0, 0, 0', '2, 0, 1, 0', '0, 1, 1, 0', '1, 2, 1, 0', '1, 0, 1, 0',
        '1, 0, 0, 0', '2, 1, 1, 0', '2, 1, 0, 0', '0, 2, 1, 0'
    ]
    for h in no_call_histories_p1_win:
      expected_terminal_values[h] = [1., -1.]

    no_call_histories_p2_win = [
        '0, 2, 0, 1, 0', '0, 1, 0, 0', '0, 1, 0, 1, 0', '0, 2, 0, 0',
        '1, 2, 0, 0', '2, 0, 0, 1, 0', '1, 2, 0, 1, 0', '2, 1, 0, 1, 0',
        '1, 0, 0, 1, 0'
    ]
    for h in no_call_histories_p2_win:
      expected_terminal_values[h] = [-1., 1.]

    call_histories_p1_win = [
        '1, 0, 1, 1', '2, 1, 1, 1', '2, 1, 0, 1, 1', '2, 0, 0, 1, 1',
        '1, 0, 0, 1, 1', '2, 0, 1, 1'
    ]
    for h in call_histories_p1_win:
      expected_terminal_values[h] = [2., -2.]

    call_histories_p2_win = [
        '0, 2, 0, 1, 1', '0, 1, 0, 1, 1', '0, 1, 1, 1', '1, 2, 1, 1',
        '1, 2, 0, 1, 1', '0, 2, 1, 1'
    ]
    for h in call_histories_p2_win:
      expected_terminal_values[h] = [-2., 2.]

    self.assertEqual(
        expected_terminal_values,
        {k: v.tolist() for k, v in root_state_wrapper.terminal_values.items()})

  def test_normalized_by_sum(self):
    self.assertListAlmostEqual(
        rcfr.normalized_by_sum([1., 2., 3., 4.]), [0.1, 0.2, 0.3, 0.4])

  def test_counterfactual_regrets_and_reach_weights_value_error(self):
    root = rcfr.RootStateWrapper(_GAME.new_initial_state())

    # Initialize arbitrary weights to generate an arbitrary profile.
    sequence_weights1_with_a_missing_sequence = [
        0.4967141530112327,
        0.0,
        0.6476885381006925,
        1.5230298564080254,
        0.0,
        0.0,
        1.5792128155073915,
        0.7674347291529088,
        0.0,
        0.5425600435859647,
        0.0,
        # 0.0,
    ]
    # Ensure this player's policy is fully mixed so that each of player 1's
    # information states are reached.
    sequence_weights2 = [
        0.24196227156603412,
        0.1,
        0.1,
        0.1,
        0.1,
        0.3142473325952739,
        0.1,
        0.1,
        1.465648768921554,
        0.1,
        0.06752820468792384,
        0.1,
    ]

    with self.assertRaises(ValueError):
      root.counterfactual_regrets_and_reach_weights(
          0, 1, sequence_weights1_with_a_missing_sequence, sequence_weights2)

  def test_counterfactual_regrets_and_reach_weights(self):
    root = rcfr.RootStateWrapper(_GAME.new_initial_state())

    # Initialize arbitrary weights to generate an arbitrary profile.
    sequence_weights1 = [
        0.4967141530112327,
        0.0,
        0.6476885381006925,
        1.5230298564080254,
        0.0,
        0.0,
        1.5792128155073915,
        0.7674347291529088,
        0.0,
        0.5425600435859647,
        0.0,
        0.0,
    ]
    sequence_weights2 = [
        0.24196227156603412,
        0.0,
        0.0,
        0.0,
        0.0,
        0.3142473325952739,
        0.0,
        0.0,
        1.465648768921554,
        0.0,
        0.06752820468792384,
        0.0,
    ]

    # These expected regrets and sequence weights were computed for the given
    # sequence weights.
    expected_regrets_given_sequence_weights = [
        0.,
        0.283604,
        0.116937,
        -0.049729,
        -0.06892,
        0.06892,
        0.054506,
        -0.112161,
        -0.083333,
        0.,
        0.,
        0.,
    ]
    expected_reach_weights_given_sequence_weights = [
        2.,
        0.,
        1.,
        1.,
        0.,
        2.,
        1.,
        1.,
        2.,
        0.,
        2.,
        0.,
    ]

    regrets, weights = root.counterfactual_regrets_and_reach_weights(
        0, 1, sequence_weights1, sequence_weights2)

    self.assertListAlmostEqual(
        regrets,
        expected_regrets_given_sequence_weights)
    self.assertListAlmostEqual(
        weights,
        expected_reach_weights_given_sequence_weights)

  def test_all_states(self):
    states = rcfr.all_states(
        _GAME.new_initial_state(),
        depth_limit=-1,
        include_terminals=False,
        include_chance_states=False)
    self.assertLen(list(states), 24)

    states = rcfr.all_states(
        _GAME.new_initial_state(),
        depth_limit=-1,
        include_terminals=True,
        include_chance_states=False)
    self.assertLen(list(states), 54)

    states = rcfr.all_states(
        _GAME.new_initial_state(),
        depth_limit=-1,
        include_terminals=False,
        include_chance_states=True)
    self.assertLen(list(states), 28)

    states = rcfr.all_states(
        _GAME.new_initial_state(),
        depth_limit=-1,
        include_terminals=True,
        include_chance_states=True)
    self.assertLen(list(states), 58)

  def test_sequence_weights_to_tabular_profile(self):
    root = rcfr.RootStateWrapper(_GAME.new_initial_state())

    def policy_fn(state):
      """Generates a policy profile by treating sequence indices as weights."""
      info_state = state.information_state_string()
      sequence_offset = root.info_state_to_sequence_idx[info_state]
      num_actions = len(state.legal_actions())
      return rcfr.normalized_by_sum(
          list(range(sequence_offset, sequence_offset + num_actions)))

    profile = rcfr.sequence_weights_to_tabular_profile(root.root, policy_fn)

    expected_profile = {
        # Player 1
        '0': [(0, 0.), (1, 1.)],  # Sequences 0 and 1 (sums to 1)
        '0pb': [(0, 0.4), (1, 0.6)],  # Sequences 2 and 3 (sums to 5)
        # Sequences 4 and 5 (sums to 9)
        '1': [(0, 0.44444444444444442), (1, 0.55555555555555558)],
        # Sequences 6 and 7 (sums to 13)
        '1pb': [(0, 0.46153846153846156), (1, 0.53846153846153844)],
        # Sequences 8 and 9 (sums to 17)
        '2': [(0, 0.47058823529411764), (1, 0.52941176470588236)],
        # Sequences 10 and 11 (sums to 21)
        '2pb': [(0, 0.47619047619047616), (1, 0.52380952380952384)],

        # Player 2
        '1p': [(0, 0.), (1, 1.)],  # Sequences 0 and 1 (sums to 1)
        '1b': [(0, 0.4), (1, 0.6)],  # Sequences 2 and 3 (sums to 5)
        # Sequences 4 and 5 (sums to 9)
        '2p': [(0, 0.44444444444444442), (1, 0.55555555555555558)],
        # Sequences 6 and 7 (sums to 13)
        '2b': [(0, 0.46153846153846156), (1, 0.53846153846153844)],
        # Sequences 8 and 9 (sums to 17)
        '0p': [(0, 0.47058823529411764), (1, 0.52941176470588236)],
        # Sequences 10 and 11 (sums to 21)
        '0b': [(0, 0.47619047619047616), (1, 0.52380952380952384)],
    }
    self.assertAlmostEqual(profile, expected_profile, delta=1e-06)

  def test_cfr(self):
    root = rcfr.RootStateWrapper(_GAME.new_initial_state())
    num_half_iterations = 6

    cumulative_regrets = [np.zeros(n) for n in root.num_player_sequences]
    cumulative_reach_weights = [np.zeros(n) for n in root.num_player_sequences]

    average_profile = root.sequence_weights_to_tabular_profile(
        cumulative_reach_weights)
    # parameterized.TestCase
    self.assertGreater(pyspiel.nash_conv(_GAME, average_profile), 0.91)

    regret_player = 0
    for _ in range(num_half_iterations):
      reach_weights_player = 1 if regret_player == 0 else 0

      regrets, reach = root.counterfactual_regrets_and_reach_weights(
          regret_player, reach_weights_player, *rcfr.relu(cumulative_regrets))

      cumulative_regrets[regret_player] += regrets
      cumulative_reach_weights[reach_weights_player] += reach

      regret_player = reach_weights_player

    average_profile = root.sequence_weights_to_tabular_profile(
        cumulative_reach_weights)
    self.assertLess(pyspiel.nash_conv(_GAME, average_profile), 0.27)

  def test_rcfr_functions(self):
    models = [_new_model() for _ in range(_GAME.num_players())]
    root = rcfr.RootStateWrapper(_GAME.new_initial_state())

    num_half_iterations = 4
    num_epochs = 100

    cumulative_regrets = [np.zeros(n) for n in root.num_player_sequences]
    cumulative_reach_weights = [np.zeros(n) for n in root.num_player_sequences]

    average_profile = root.sequence_weights_to_tabular_profile(
        cumulative_reach_weights)
    self.assertGreater(pyspiel.nash_conv(_GAME, average_profile), 0.91)

    regret_player = 0
    sequence_weights = [
        model(root.sequence_features[player]).detach().numpy()
        for player, model in enumerate(models)
    ]

    for _ in range(num_half_iterations):
      reach_weights_player = 1 if regret_player == 0 else 0

      sequence_weights[reach_weights_player] = models[reach_weights_player](
          root.sequence_features[reach_weights_player]).detach().numpy()

      regrets, seq_probs = root.counterfactual_regrets_and_reach_weights(
          regret_player, reach_weights_player, *sequence_weights)

      cumulative_regrets[regret_player] += regrets
      cumulative_reach_weights[reach_weights_player] += seq_probs

      data = torch.utils.data.TensorDataset(
          root.sequence_features[regret_player],
          torch.unsqueeze(
              torch.Tensor(cumulative_regrets[regret_player]), axis=1))
      data = torch.utils.data.DataLoader(
          data, batch_size=_BATCH_SIZE, shuffle=True)

      loss_fn = nn.SmoothL1Loss()
      optimizer = torch.optim.Adam(
          models[regret_player].parameters(), lr=0.005, amsgrad=True)
      for _ in range(num_epochs):
        for x, y in data:
          optimizer.zero_grad()
          output = models[regret_player](x)
          loss = loss_fn(output, y)
          loss.backward()
          optimizer.step()

      regret_player = reach_weights_player

    average_profile = root.sequence_weights_to_tabular_profile(
        cumulative_reach_weights)
    self.assertLess(pyspiel.nash_conv(_GAME, average_profile), 0.91)

  @parameterized.parameters(list(itertools.product(_BOOLEANS, _BOOLEANS)))
  def test_rcfr(self, bootstrap, truncate_negative):
    num_epochs = 100
    num_iterations = 2
    models = [_new_model() for _ in range(_GAME.num_players())]

    patient = rcfr.RcfrSolver(
        _GAME, models, bootstrap=bootstrap, truncate_negative=truncate_negative)

    def _train(model, data):
      data = torch.utils.data.DataLoader(
          data, batch_size=_BATCH_SIZE, shuffle=True)

      loss_fn = nn.SmoothL1Loss()
      optimizer = torch.optim.Adam(model.parameters(), lr=0.005, amsgrad=True)
      for _ in range(num_epochs):
        for x, y in data:
          optimizer.zero_grad()
          output = model(x)
          loss = loss_fn(output, y)
          loss.backward()
          optimizer.step()

    average_policy = patient.average_policy()
    self.assertGreater(pyspiel.nash_conv(_GAME, average_policy), 0.91)

    for _ in range(num_iterations):
      patient.evaluate_and_update_policy(_train)

    average_policy = patient.average_policy()
    self.assertLess(pyspiel.nash_conv(_GAME, average_policy), 0.91)

  def test_reservior_buffer_insert(self):
    buffer_size = 10
    patient = rcfr.ReservoirBuffer(buffer_size)

    x_buffer = []
    for i in range(buffer_size):
      patient.insert(i)
      x_buffer.append(i)
      assert patient.num_elements == len(x_buffer)
      self.assertEqual(x_buffer, patient.buffer)

    assert patient.num_available_spaces() == 0

    for i in range(buffer_size):
      patient.insert(buffer_size + i)
      assert patient.num_elements == buffer_size

  def test_reservior_buffer_insert_all(self):
    buffer_size = 10
    patient = rcfr.ReservoirBuffer(buffer_size)

    x_buffer = list(range(buffer_size))
    patient.insert_all(x_buffer)
    assert patient.num_elements == buffer_size
    self.assertEqual(x_buffer, patient.buffer)

    assert patient.num_available_spaces() == 0

    x_buffer = list(range(buffer_size, 2 * buffer_size))
    patient.insert_all(x_buffer)
    assert patient.num_elements == buffer_size

  def test_rcfr_with_buffer(self):
    buffer_size = 12
    num_epochs = 100
    num_iterations = 2
    models = [_new_model() for _ in range(_GAME.num_players())]

    patient = rcfr.ReservoirRcfrSolver(_GAME, models, buffer_size=buffer_size)

    def _train(model, data):
      data = torch.utils.data.DataLoader(
          data, batch_size=_BATCH_SIZE, shuffle=True)

      loss_fn = nn.SmoothL1Loss()
      optimizer = torch.optim.Adam(model.parameters(), lr=0.005, amsgrad=True)
      for _ in range(num_epochs):
        for x, y in data:
          optimizer.zero_grad()
          output = model(x)
          loss = loss_fn(output, y)
          loss.backward()
          optimizer.step()

    average_policy = patient.average_policy()
    self.assertGreater(pyspiel.nash_conv(_GAME, average_policy), 0.91)

    for _ in range(num_iterations):
      patient.evaluate_and_update_policy(_train)

    average_policy = patient.average_policy()
    self.assertLess(pyspiel.nash_conv(_GAME, average_policy), 0.91)


if __name__ == '__main__':
  absltest.main()
