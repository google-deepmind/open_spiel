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

# Lint as: python3
"""Train a policy net on Hearts actions based given a dataset of trajectories.

Trajectories from the Hearts bot Xinxin can be generated using
open_spiel/bots/xinxin/xinxin_game_generator.cc.
"""

import os
import pickle
from typing import Any, Tuple

from absl import app
from absl import flags

import haiku as hk
import jax
from jax import numpy as jnp
import numpy as np
import optax

import pyspiel

OptState = Any
Params = Any

FLAGS = flags.FLAGS
GAME = pyspiel.load_game('hearts')
NUM_CARDS = 52
NUM_ACTIONS = NUM_CARDS
NUM_PLAYERS = 4
TOP_K_ACTIONS = 5  # How many alternative actions to display
DEFAULT_LAYER_SIZES = [1024, 1024, 1024, 1024]

flags.DEFINE_integer('iterations', 100000, 'Number of iterations')
flags.DEFINE_string('data_path', None, 'Location for data')
flags.DEFINE_integer('eval_every', 10000, 'How often to evaluate the policy')
flags.DEFINE_integer('num_examples', 3,
                     'How many examples to print per evaluation')
flags.DEFINE_integer('train_batch', 128, 'Batch size for training step')
flags.DEFINE_integer('eval_batch', 10000, 'Batch size when evaluating')
flags.DEFINE_float('step_size', 1e-4, 'Step size for training')
flags.DEFINE_list('hidden_layer_sizes', None,
                  'Number of hidden units and layers in the network')
flags.DEFINE_integer('rng_seed', 42, 'Seed for initial network weights')
flags.DEFINE_string('save_path', None, 'Location for saved networks')
flags.DEFINE_string('checkpoint_file', None,
                    'Provides weights and optimzer state to resume training')


def _trajectory(line: str):
  """Returns parsed action trajectory."""
  actions = [int(x) for x in line.split(' ')]
  return tuple(actions)


def make_dataset(file: str):
  """Creates dataset as a generator of single examples."""
  lines = [line for line in open(file)]
  while True:
    np.random.shuffle(lines)
    for line in lines:
      trajectory = _trajectory(line)
      # skip pass_dir and deal actions
      action_index = np.random.randint(NUM_CARDS + 1, len(trajectory))
      state = GAME.new_initial_state()
      for action in trajectory[:action_index]:
        state.apply_action(action)
      yield (state.information_state_tensor(), trajectory[action_index])


def batch(dataset, batch_size: int):
  """Creates a batched dataset from a one-at-a-time dataset."""
  observations = np.zeros([batch_size] + GAME.information_state_tensor_shape(),
                          np.float32)
  labels = np.zeros(batch_size, dtype=np.int32)
  while True:
    for batch_index in range(batch_size):
      observations[batch_index], labels[batch_index] = next(dataset)
    yield observations, labels


def one_hot(x, k):
  """Returns a one-hot encoding of `x` of size `k`."""
  return jnp.array(x[..., jnp.newaxis] == jnp.arange(k), dtype=np.float32)


def net_fn(x):
  """Haiku module for our network."""
  layers = []
  for layer_size in FLAGS.hidden_layer_sizes:
    layers.append(hk.Linear(int(layer_size)))
    layers.append(jax.nn.relu)
  layers.append(hk.Linear(NUM_ACTIONS))
  layers.append(jax.nn.log_softmax)
  net = hk.Sequential(layers)
  return net(x)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if FLAGS.hidden_layer_sizes is None:
    # Cannot pass default arguments as lists due to style requirements, so we
    # override it here if they are not set.
    FLAGS.hidden_layer_sizes = DEFAULT_LAYER_SIZES

  # Make the network.
  net = hk.without_apply_rng(hk.transform(net_fn))

  # Make the optimiser.
  opt = optax.adam(FLAGS.step_size)

  @jax.jit
  def loss(
      params: Params,
      inputs: np.ndarray,
      targets: np.ndarray,
  ) -> jnp.DeviceArray:
    """Cross-entropy loss."""
    assert targets.dtype == np.int32
    log_probs = net.apply(params, inputs)
    return -jnp.mean(one_hot(targets, NUM_ACTIONS) * log_probs)

  @jax.jit
  def accuracy(
      params: Params,
      inputs: np.ndarray,
      targets: np.ndarray,
  ) -> jnp.DeviceArray:
    """Classification accuracy."""
    predictions = net.apply(params, inputs)
    return jnp.mean(jnp.argmax(predictions, axis=-1) == targets)

  @jax.jit
  def update(
      params: Params,
      opt_state: OptState,
      inputs: np.ndarray,
      targets: np.ndarray,
  ) -> Tuple[Params, OptState]:
    """Learning rule (stochastic gradient descent)."""
    _, gradient = jax.value_and_grad(loss)(params, inputs, targets)
    updates, opt_state = opt.update(gradient, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state

  def output_samples(params: Params, max_samples: int):
    """Output some cases where the policy disagrees with the dataset action."""
    if max_samples == 0:
      return
    count = 0
    with open(os.path.join(FLAGS.data_path, 'test.txt')) as f:
      lines = list(f)
    np.random.shuffle(lines)
    for line in lines:
      state = GAME.new_initial_state()
      actions = _trajectory(line)
      for action in actions:
        if not state.is_chance_node():
          observation = np.array(state.information_state_tensor(), np.float32)
          policy = np.exp(net.apply(params, observation))
          probs_actions = [(p, a) for a, p in enumerate(policy)]
          pred = max(probs_actions)[1]
          if pred != action:
            print(state)
            for p, a in reversed(sorted(probs_actions)[-TOP_K_ACTIONS:]):
              print('{:7} {:.2f}'.format(state.action_to_string(a), p))
            print('Ground truth {}\n'.format(state.action_to_string(action)))
            count += 1
            break
        state.apply_action(action)
      if count >= max_samples:
        return

  # Store what we need to rebuild the Haiku net.
  if FLAGS.save_path:
    filename = os.path.join(FLAGS.save_path, 'layers.txt')
    with open(filename, 'w') as layer_def_file:
      for s in FLAGS.hidden_layer_sizes:
        layer_def_file.write(f'{s} ')
      layer_def_file.write('\n')

  # Make datasets.
  if FLAGS.data_path is None:
    raise app.UsageError(
        'Please generate your own supervised training data and supply the local'
        'location as --data_path')
  train = batch(
      make_dataset(os.path.join(FLAGS.data_path, 'train.txt')),
      FLAGS.train_batch)
  test = batch(
      make_dataset(os.path.join(FLAGS.data_path, 'test.txt')), FLAGS.eval_batch)

  # Initialize network and optimiser.
  if FLAGS.checkpoint_file:
    with open(FLAGS.checkpoint_file, 'rb') as pkl_file:
      params, opt_state = pickle.load(pkl_file)
  else:
    rng = jax.random.PRNGKey(FLAGS.rng_seed)  # seed used for network weights
    inputs, unused_targets = next(train)
    params = net.init(rng, inputs)
    opt_state = opt.init(params)

  # Train/eval loop.
  for step in range(FLAGS.iterations):
    # Do SGD on a batch of training examples.
    inputs, targets = next(train)
    params, opt_state = update(params, opt_state, inputs, targets)

    # Periodically evaluate classification accuracy on the test set.
    if (1 + step) % FLAGS.eval_every == 0:
      inputs, targets = next(test)
      test_accuracy = accuracy(params, inputs, targets)
      print(f'After {1+step} steps, test accuracy: {test_accuracy}.')
      if FLAGS.save_path:
        filename = os.path.join(FLAGS.save_path, f'checkpoint-{1 + step}.pkl')
        with open(filename, 'wb') as pkl_file:
          pickle.dump((params, opt_state), pkl_file)
      output_samples(params, FLAGS.num_examples)


if __name__ == '__main__':
  app.run(main)
