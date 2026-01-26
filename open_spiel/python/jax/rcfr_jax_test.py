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

import functools
import itertools

from absl.testing import absltest
from absl.testing import parameterized

import chex
import flax.nnx as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyspiel  # pylint: disable=g-bad-import-order

from open_spiel.python.jax import rcfr

_GAME = pyspiel.load_game("kuhn_poker")
_BOOLEANS = [False, True]

_BATCH_SIZE = 12
SEED = 24984617


def _new_model() -> rcfr.DeepRcfrModel:
  return rcfr.DeepRcfrModel(
      _GAME,
      num_hidden_layers=1,
      num_hidden_units=13,
      num_hidden_factors=1,
      use_skip_connections=True,
  )


@functools.partial(jax.jit, static_argnames=("graphdef",))
def jax_train_step(
    graphdef: nn.GraphDef, state: nn.State, x: chex.Array, y: chex.Array
) -> tuple[chex.Numeric, nn.State]:
  """Train step in pure jax."""

  model, optimizer = nn.merge(graphdef, state, copy=True)

  def loss_fn(model):
    y_pred = model(x)
    return optax.hinge_loss(y_pred, y).mean()

  loss, grads = nn.value_and_grad(loss_fn)(model)
  optimizer.update(model, grads)
  state = nn.state((model, optimizer))
  return loss, state


@nn.vmap(in_axes=(None, 0), out_axes=0)
def forward(model: rcfr.DeepRcfrModel, x: chex.Array) -> chex.Array:
  """Batched call for the flax.nnx model."""
  return model(x)


class RcfrTest(parameterized.TestCase, absltest.TestCase):

  def setUp(self):
    # pylint: disable=useless-super-delegation
    super(RcfrTest, self).setUp()

  def assertListAlmostEqual(self, list1, list2, delta=1e-06):
    self.assertEqual(len(list1), len(list2))
    for a, b in zip(list1, list2):
      self.assertAlmostEqual(a, b, delta=delta)

  def test_with_one_hot_action_features_single_state_vector(self):
    information_state_features = [1.0, 2.0, 3.0]
    features = rcfr.sequence_features(
        information_state_features, legal_actions=[0, 1], num_distinct_actions=3
    )
    np.testing.assert_array_equal([1.0, 2.0, 3.0, 1.0, 0.0, 0.0], features[0])
    np.testing.assert_array_equal([1.0, 2.0, 3.0, 0.0, 1.0, 0.0], features[1])

    features = rcfr.sequence_features(
        information_state_features, legal_actions=[1, 2], num_distinct_actions=3
    )
    np.testing.assert_array_equal([1.0, 2.0, 3.0, 0.0, 1.0, 0.0], features[0])
    np.testing.assert_array_equal([1.0, 2.0, 3.0, 0.0, 0.0, 1.0], features[1])

  def test_root_state_wrapper_num_sequences(self):
    root_state_wrapper = rcfr.RootStateWrapper(_GAME.new_initial_state(), _GAME)
    assert root_state_wrapper.num_player_sequences[0] == 12
    assert root_state_wrapper.num_player_sequences[1] == 12

  def test_root_state_wrapper_sequence_indices(self):
    root_state_wrapper = rcfr.RootStateWrapper(_GAME.new_initial_state(), _GAME)
    self.assertEqual(
        {
            # Info state string -> initial sequence index map for player 1.
            "0": 0,
            "0pb": 2,
            "1": 4,
            "1pb": 6,
            "2": 8,
            "2pb": 10,
            # Info state string -> initial sequence index map for player 2.
            "1p": 0,
            "1b": 2,
            "2p": 4,
            "2b": 6,
            "0p": 8,
            "0b": 10,
        },
        root_state_wrapper.info_state_to_sequence_idx,
    )

  def test_root_state_wrapper_sequence_features(self):
    root_state_wrapper = rcfr.RootStateWrapper(_GAME.new_initial_state(), _GAME)

    p1_info_state_features = [
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    ]
    p2_info_state_features = [
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    ]
    action_features = [[1.0, 0.0], [0.0, 1.0]]
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
    np.testing.assert_array_equal(
        expected_p1_sequence_features, root_state_wrapper.sequence_features[0]
    )
    np.testing.assert_array_equal(
        expected_p2_sequence_features, root_state_wrapper.sequence_features[1]
    )

  def test_root_state_wrapper_sequence_terminal_values(self):
    root_state_wrapper = rcfr.RootStateWrapper(_GAME.new_initial_state(), _GAME)

    expected_terminal_values = {}
    no_call_histories_p1_win = [
        "2, 0, 0, 0",
        "2, 0, 1, 0",
        "0, 1, 1, 0",
        "1, 2, 1, 0",
        "1, 0, 1, 0",
        "1, 0, 0, 0",
        "2, 1, 1, 0",
        "2, 1, 0, 0",
        "0, 2, 1, 0",
    ]
    for h in no_call_histories_p1_win:
      expected_terminal_values[h] = [1.0, -1.0]

    no_call_histories_p2_win = [
        "0, 2, 0, 1, 0",
        "0, 1, 0, 0",
        "0, 1, 0, 1, 0",
        "0, 2, 0, 0",
        "1, 2, 0, 0",
        "2, 0, 0, 1, 0",
        "1, 2, 0, 1, 0",
        "2, 1, 0, 1, 0",
        "1, 0, 0, 1, 0",
    ]
    for h in no_call_histories_p2_win:
      expected_terminal_values[h] = [-1.0, 1.0]

    call_histories_p1_win = [
        "1, 0, 1, 1",
        "2, 1, 1, 1",
        "2, 1, 0, 1, 1",
        "2, 0, 0, 1, 1",
        "1, 0, 0, 1, 1",
        "2, 0, 1, 1",
    ]
    for h in call_histories_p1_win:
      expected_terminal_values[h] = [2.0, -2.0]

    call_histories_p2_win = [
        "0, 2, 0, 1, 1",
        "0, 1, 0, 1, 1",
        "0, 1, 1, 1",
        "1, 2, 1, 1",
        "1, 2, 0, 1, 1",
        "0, 2, 1, 1",
    ]
    for h in call_histories_p2_win:
      expected_terminal_values[h] = [-2.0, 2.0]

    self.assertEqual(
        expected_terminal_values,
        {k: v.tolist() for k, v in root_state_wrapper.terminal_values.items()},
    )

  def test_normalized_by_sum(self):
    self.assertListAlmostEqual(
        rcfr.normalised_by_sum(jnp.array([1.0, 2.0, 3.0, 4.0])),
        jnp.array([0.1, 0.2, 0.3, 0.4]),
    )

  def test_counterfactual_regrets_and_reach_weights_value_error(self):
    root = rcfr.RootStateWrapper(_GAME.new_initial_state(), _GAME)

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
          0, 1, sequence_weights1_with_a_missing_sequence, sequence_weights2
      )

  def test_counterfactual_regrets_and_reach_weights(self):
    root = rcfr.RootStateWrapper(_GAME.new_initial_state(), _GAME)

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
        0.0,
        0.283604,
        0.116937,
        -0.049729,
        -0.06892,
        0.06892,
        0.054506,
        -0.112161,
        -0.083333,
        0.0,
        0.0,
        0.0,
    ]
    expected_reach_weights_given_sequence_weights = [
        2.0,
        0.0,
        1.0,
        1.0,
        0.0,
        2.0,
        1.0,
        1.0,
        2.0,
        0.0,
        2.0,
        0.0,
    ]

    regrets, weights = root.counterfactual_regrets_and_reach_weights(
        0, 1, sequence_weights1, sequence_weights2
    )

    self.assertListAlmostEqual(regrets, expected_regrets_given_sequence_weights)
    self.assertListAlmostEqual(
        weights, expected_reach_weights_given_sequence_weights
    )

  def test_all_states(self):
    states = rcfr.all_states(
        _GAME.new_initial_state(),
        depth_limit=-1,
        include_terminals=False,
        include_chance_states=False,
    )
    self.assertLen(list(states), 24)

    states = rcfr.all_states(
        _GAME.new_initial_state(),
        depth_limit=-1,
        include_terminals=True,
        include_chance_states=False,
    )
    self.assertLen(list(states), 54)

    states = rcfr.all_states(
        _GAME.new_initial_state(),
        depth_limit=-1,
        include_terminals=False,
        include_chance_states=True,
    )
    self.assertLen(list(states), 28)

    states = rcfr.all_states(
        _GAME.new_initial_state(),
        depth_limit=-1,
        include_terminals=True,
        include_chance_states=True,
    )
    self.assertLen(list(states), 58)

  def test_sequence_weights_to_tabular_profile(self):
    root = rcfr.RootStateWrapper(_GAME.new_initial_state(), _GAME)

    def policy_fn(state):
      """Generates a policy profile by treating sequence indices as weights."""
      info_state = state.information_state_string()
      sequence_offset = root.info_state_to_sequence_idx[info_state]
      num_actions = len(state.legal_actions())
      return rcfr.normalised_by_sum(
          jnp.array(list(range(sequence_offset, sequence_offset + num_actions)))
      )

    profile = rcfr.sequence_weights_to_tabular_profile(root.root, policy_fn)

    expected_profile = {
        # Player 1
        "0": [(0, 0.0), (1, 1.0)],  # Sequences 0 and 1 (sums to 1)
        "0pb": [(0, 0.4), (1, 0.6)],  # Sequences 2 and 3 (sums to 5)
        # Sequences 4 and 5 (sums to 9)
        "1": [(0, 0.44444444444444442), (1, 0.55555555555555558)],
        # Sequences 6 and 7 (sums to 13)
        "1pb": [(0, 0.46153846153846156), (1, 0.53846153846153844)],
        # Sequences 8 and 9 (sums to 17)
        "2": [(0, 0.47058823529411764), (1, 0.52941176470588236)],
        # Sequences 10 and 11 (sums to 21)
        "2pb": [(0, 0.47619047619047616), (1, 0.52380952380952384)],
        # Player 2
        "1p": [(0, 0.0), (1, 1.0)],  # Sequences 0 and 1 (sums to 1)
        "1b": [(0, 0.4), (1, 0.6)],  # Sequences 2 and 3 (sums to 5)
        # Sequences 4 and 5 (sums to 9)
        "2p": [(0, 0.44444444444444442), (1, 0.55555555555555558)],
        # Sequences 6 and 7 (sums to 13)
        "2b": [(0, 0.46153846153846156), (1, 0.53846153846153844)],
        # Sequences 8 and 9 (sums to 17)
        "0p": [(0, 0.47058823529411764), (1, 0.52941176470588236)],
        # Sequences 10 and 11 (sums to 21)
        "0b": [(0, 0.47619047619047616), (1, 0.52380952380952384)],
    }
    self.assertAlmostEqual(profile, expected_profile, delta=1e-06)

  def test_cfr(self):
    root = rcfr.RootStateWrapper(_GAME.new_initial_state(), _GAME)
    num_half_iterations = 6

    cumulative_regrets = [np.zeros(n) for n in root.num_player_sequences]
    cumulative_reach_weights = [np.zeros(n) for n in root.num_player_sequences]

    average_profile = root.sequence_weights_to_tabular_profile(
        cumulative_reach_weights
    )
    # parameterized.TestCase
    self.assertGreater(pyspiel.nash_conv(_GAME, average_profile), 0.91)

    regret_player = 0
    for _ in range(num_half_iterations):
      reach_weights_player = 1 if regret_player == 0 else 0

      regrets, reach = root.counterfactual_regrets_and_reach_weights(
          regret_player,
          reach_weights_player,
          *rcfr.relu(jnp.asarray(cumulative_regrets)),
      )

      cumulative_regrets[regret_player] += regrets
      cumulative_reach_weights[reach_weights_player] += reach

      regret_player = reach_weights_player

    average_profile = root.sequence_weights_to_tabular_profile(
        cumulative_reach_weights
    )
    self.assertLess(pyspiel.nash_conv(_GAME, average_profile), 0.27)

  def test_rcfr_functions(self):
    models = [_new_model() for _ in range(_GAME.num_players())]
    root = rcfr.RootStateWrapper(_GAME.new_initial_state(), _GAME)

    num_half_iterations = 4
    num_epochs = 20

    cumulative_regrets = [jnp.zeros(n) for n in root.num_player_sequences]
    cumulative_reach_weights = [jnp.zeros(n) for n in root.num_player_sequences]

    average_profile = root.sequence_weights_to_tabular_profile(
        cumulative_reach_weights
    )
    self.assertGreater(pyspiel.nash_conv(_GAME, average_profile), 0.91)

    regret_player = 0
    sequence_weights = [
        model(root.sequence_features[player])
        for player, model in enumerate(models)
    ]

    rng = jax.random.key(SEED)

    for _ in range(num_half_iterations):
      reach_weights_player = 1 if regret_player == 0 else 0

      sequence_weights[reach_weights_player] = models[reach_weights_player](
          root.sequence_features[reach_weights_player]
      )

      regrets, seq_probs = root.counterfactual_regrets_and_reach_weights(
          regret_player, reach_weights_player, *sequence_weights
      )
      cumulative_regrets[regret_player] += regrets
      cumulative_reach_weights[reach_weights_player] += seq_probs

      data = (
          jnp.asarray(root.sequence_features[regret_player]),
          jnp.asarray(cumulative_regrets[regret_player]),
      )

      rng_, rng = jax.random.split(rng)

      # pylint: disable=cell-var-from-loop

      num_batches = len(data[0]) // _BATCH_SIZE
      data = jax.tree.map(
          lambda x: jax.random.permutation(rng_, x, axis=0).reshape(
              num_batches, _BATCH_SIZE, -1
          ),
          data,
      )

      optimizer = nn.Optimizer(
          models[regret_player], optax.adam(learning_rate=0.005), wrt=nn.Param
      )
      graphdef, state = nn.split((models[regret_player], optimizer))

      for _ in range(num_epochs):
        for x, y in zip(*data):
          _, state = jax_train_step(graphdef, state, x, y.squeeze(-1))

      # refreshing
      nn.update((models[regret_player], optimizer), state)
      regret_player = reach_weights_player

    average_profile = root.sequence_weights_to_tabular_profile(
        cumulative_reach_weights
    )
    self.assertLess(pyspiel.nash_conv(_GAME, average_profile), 0.91)

  @parameterized.parameters(list(itertools.product(_BOOLEANS, _BOOLEANS)))
  def test_rcfr(self, bootstrap, truncate_negative):
    num_epochs = 100
    num_iterations = 10
    models = [_new_model() for _ in range(_GAME.num_players())]

    rng = jax.random.key(SEED)

    patient = rcfr.RcfrSolver(
        _GAME, models, bootstrap=bootstrap, truncate_negative=truncate_negative
    )

    # pylint: disable=g-bare-generic
    def _train(model: nn.Module, data: tuple) -> None:
      data_, rng = data
      optimizer = nn.Optimizer(
          model, optax.adam(learning_rate=0.05), wrt=nn.Param
      )
      graphdef, state = nn.split((model, optimizer))

      num_batches = len(data_[0]) // _BATCH_SIZE
      data_ = jax.tree.map(
          lambda x: jax.random.permutation(rng, x, axis=0).reshape(
              num_batches, _BATCH_SIZE, -1
          ),
          data_,
      )

      for _ in range(num_epochs):
        for x, y in zip(*data_):
          _, state = jax_train_step(graphdef, state, x, y.squeeze(-1))

      nn.update((model, optimizer), state)
      return

    average_policy = patient.average_policy()
    self.assertGreater(pyspiel.nash_conv(_GAME, average_policy), 0.91)

    for _ in range(num_iterations):
      rng, rng_ = jax.random.split(rng)
      patient.evaluate_and_update_policy(_train, rng_)

    average_policy = patient.average_policy()
    self.assertLess(pyspiel.nash_conv(_GAME, average_policy), 0.92)

  def test_rcfr_with_buffer(self):
    buffer_size = 12
    num_epochs = 100
    num_iterations = 2
    models = [_new_model() for _ in range(_GAME.num_players())]

    rng = jax.random.key(SEED)

    patient = rcfr.ReservoirRcfrSolver(_GAME, models, buffer_size=buffer_size)

    # pylint: disable=g-bare-generic
    def _train(model: nn.Module, data: tuple) -> None:
      data_, rng = data
      optimizer = nn.Optimizer(
          model, optax.adam(learning_rate=0.005), wrt=nn.Param
      )
      graphdef, state = nn.split((model, optimizer))

      num_batches = len(data_[0]) // _BATCH_SIZE
      data_ = jax.tree.map(
          lambda x: jax.random.permutation(rng, x, axis=0).reshape(
              num_batches, _BATCH_SIZE, -1
          ),
          data_,
      )

      for _ in range(num_epochs):
        for x, y in zip(*data_):
          _, state = jax_train_step(graphdef, state, x, y.squeeze(-1))

      nn.update((model, optimizer), state)
      return

    average_policy = patient.average_policy()
    self.assertGreater(pyspiel.nash_conv(_GAME, average_policy), 0.91)

    for _ in range(num_iterations):
      rng, rng_ = jax.random.split(rng)
      patient.evaluate_and_update_policy(_train, rng_)
      _average_policy = patient.average_policy()
      print(pyspiel.nash_conv(_GAME, _average_policy), 0.91)

    average_policy = patient.average_policy()
    self.assertLess(pyspiel.nash_conv(_GAME, average_policy), 0.91)


if __name__ == "__main__":
  absltest.main()
