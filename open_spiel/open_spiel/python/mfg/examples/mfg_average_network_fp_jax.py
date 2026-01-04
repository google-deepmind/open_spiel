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
"""Runs Deep Average-network Fictitious Play with DQN agents."""

import os
from typing import Sequence

from absl import flags
import jax

from open_spiel.python import policy as policy_std
from open_spiel.python import rl_environment
from open_spiel.python.jax import dqn
from open_spiel.python.mfg import utils
from open_spiel.python.mfg.algorithms import average_network_fictitious_play
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms import nash_conv
from open_spiel.python.mfg.algorithms import policy_value
from open_spiel.python.mfg.games import factory
from open_spiel.python.utils import app
from open_spiel.python.utils import metrics
from open_spiel.python.utils import training

_GAME_NAME = flags.DEFINE_string('game_name', 'mfg_crowd_modelling_2d',
                                 'Name of the game.')
_ENV_SETTING = flags.DEFINE_string(
    'env_setting', 'mfg_crowd_modelling_2d',
    'Name of the game settings. If None, the game name will be used.')
_LOGDIR = flags.DEFINE_string(
    'logdir', None,
    'Logging dir to use for TF summary files. If None, the metrics will only '
    'be logged to stderr.')
_LOG_DISTRIBUTION = flags.DEFINE_bool('log_distribution', False,
                                      'Enables logging of the distribution.')
_NUM_ITERATIONS = flags.DEFINE_integer('num_iterations', 100,
                                       'Number of iterations.')
_EVAL_EVERY = flags.DEFINE_integer(
    'eval_every', 200, 'Episode frequency at which the agents are evaluated.')

# Flags for best response RL (DQN) agent.
# Training options.
_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size', 128, 'Number of transitions to sample at each learning step.')
_LEARN_EVERY = flags.DEFINE_integer(
    'learn_every', 40, 'Number of steps between learning updates.')
_NUM_DQN_EPISODES_PER_ITERATION = flags.DEFINE_integer(
    'num_dqn_episodes_per_iteration', 3000,
    'Number of DQN training episodes for each iteration.')
_EPSILON_DECAY_DURATION = flags.DEFINE_integer(
    'epsilon_decay_duration', int(20e6),
    'Number of game steps over which epsilon is decayed.')
_EPSILON_START = flags.DEFINE_float('epsilon_start', 0.1,
                                    'Starting exploration parameter.')
_EPSILON_END = flags.DEFINE_float('epsilon_end', 0.1,
                                  'Final exploration parameter.')
_DISCOUNT_FACTOR = flags.DEFINE_float('discount_factor', 1.0,
                                      'Discount factor for future rewards.')
_SEED = flags.DEFINE_integer('seed', 42, 'Training seed.')
# Network options.
_HIDDEN_LAYERS_SIZES = flags.DEFINE_list(
    'hidden_layers_sizes', ['128', '128'],
    'Number of hidden units in the Q-net.')
_UPDATE_TARGET_NETWORK_EVERY = flags.DEFINE_integer(
    'update_target_network_every', 200,
    'Number of steps between DQN target network updates.')
# Replay buffer options.
_REPLAY_BUFFER_CAPACITY = flags.DEFINE_integer('replay_buffer_capacity', 5000,
                                               'Size of the replay buffer.')
_MIN_BUFFER_SIZE_TO_LEARN = flags.DEFINE_integer(
    'min_buffer_size_to_learn', 200,
    'Number of samples in buffer before learning begins.')
# Loss and optimizer options.
_OPTIMIZER = flags.DEFINE_enum('optimizer', 'adam', ['sgd', 'adam'],
                               'Optimizer.')
_LEARNING_RATE = flags.DEFINE_float('learning_rate', 0.001,
                                    'Learning rate for inner rl agent.')
_LOSS = flags.DEFINE_enum('loss', 'mse', ['mse', 'huber'], 'Loss function.')
_HUBER_LOSS_PARAMETER = flags.DEFINE_float('huber_loss_parameter', 1.0,
                                           'Parameter for Huber loss.')
_GRADIENT_CLIPPING = flags.DEFINE_float('gradient_clipping', 40,
                                        'Value to clip the gradient to.')

# Flags for average policy RL agent.
# Training options.
_AVG_POL_BATCH_SIZE = flags.DEFINE_integer(
    'avg_pol_batch_size', 128,
    'Number of transitions to sample at each learning step.')
_AVG_POL_NUM_TRAINING_STEPS_PER_ITERATION = flags.DEFINE_integer(
    'avg_pol_num_training_steps_per_iteration', 2000,
    'Number of steps for average policy at each FP iteration.')
_AVG_POL_NUM_EPISODES_PER_ITERATION = flags.DEFINE_integer(
    'avg_pol_num_episodes_per_iteration', 100,
    'Number of samples to store at each FP iteration.')
# Network options.
_AVG_POL_HIDDEN_LAYERS_SIZES = flags.DEFINE_list(
    'avg_pol_hidden_layers_sizes', ['128', '128'],
    'Number of hidden units in the avg-net and Q-net.')
# Reservoir buffer options.
_AVG_POL_RESERVOIR_BUFFER_CAPACITY = flags.DEFINE_integer(
    'avg_pol_reservoir_buffer_capacity', 100000000,
    'Size of the reservoir buffer.')
_AVG_POL_MIN_BUFFER_SIZE_TO_LEARN = flags.DEFINE_integer(
    'avg_pol_min_buffer_size_to_learn', 100,
    'Number of samples in buffer before learning begins.')
# Loss and optimizer options.
_AVG_POL_OPTIMIZER = flags.DEFINE_enum('avg_pol_optimizer', 'sgd',
                                       ['sgd', 'adam'], 'Optimizer.')
_AVG_POL_LEARNING_RATE = flags.DEFINE_float(
    'avg_pol_learning_rate', 0.01, 'Learning rate for inner rl agent.')
_AVG_GRADIENT_CLIPPING = flags.DEFINE_float('avg_gradient_clipping', 100,
                                            'Value to clip the gradient to.')
_AVG_POL_TAU = flags.DEFINE_float('avg_pol_tau', 10.0,
                                  'Temperature for softmax in policy.')


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  game = factory.create_game_with_setting(_GAME_NAME.value, _ENV_SETTING.value)
  num_players = game.num_players()

  # Create the environments with uniform initial policy.
  uniform_policy = policy_std.UniformRandomPolicy(game)
  uniform_dist = distribution.DistributionPolicy(game, uniform_policy)

  envs = [
      rl_environment.Environment(
          game, mfg_distribution=uniform_dist, mfg_population=p)
      for p in range(num_players)
  ]

  env = envs[0]
  info_state_size = env.observation_spec()['info_state'][0]
  num_actions = env.action_spec()['num_actions']

  # Best response policy agents.
  kwargs_dqn = {
      'batch_size': _BATCH_SIZE.value,
      'discount_factor': _DISCOUNT_FACTOR.value,
      'epsilon_decay_duration': _EPSILON_DECAY_DURATION.value,
      'epsilon_end': _EPSILON_END.value,
      'epsilon_start': _EPSILON_START.value,
      'gradient_clipping': _GRADIENT_CLIPPING.value,
      'hidden_layers_sizes': [int(l) for l in _HIDDEN_LAYERS_SIZES.value],
      'huber_loss_parameter': _HUBER_LOSS_PARAMETER.value,
      'learn_every': _LEARN_EVERY.value,
      'learning_rate': _LEARNING_RATE.value,
      'loss_str': _LOSS.value,
      'min_buffer_size_to_learn': _MIN_BUFFER_SIZE_TO_LEARN.value,
      'optimizer_str': _OPTIMIZER.value,
      'replay_buffer_capacity': _REPLAY_BUFFER_CAPACITY.value,
      'seed': _SEED.value,
      'update_target_network_every': _UPDATE_TARGET_NETWORK_EVERY.value,
  }
  br_rl_agents = [
      dqn.DQN(p, info_state_size, num_actions, **kwargs_dqn)
      for p in range(num_players)
  ]

  num_training_steps_per_iteration = (
      _AVG_POL_NUM_TRAINING_STEPS_PER_ITERATION.value)

  # Metrics writer will also log the metrics to stderr.
  just_logging = _LOGDIR.value is None or jax.host_id() > 0
  writer = metrics.create_default_writer(
      _LOGDIR.value, just_logging=just_logging)

  def logging_fn(it, step, vals):
    writer.write_scalars(it * num_training_steps_per_iteration + step, vals)

  # Average policy agents.
  kwargs_avg = {
      'batch_size': _AVG_POL_BATCH_SIZE.value,
      'hidden_layers_sizes': [
          int(l) for l in _AVG_POL_HIDDEN_LAYERS_SIZES.value
      ],
      'reservoir_buffer_capacity': _AVG_POL_RESERVOIR_BUFFER_CAPACITY.value,
      'learning_rate': _AVG_POL_LEARNING_RATE.value,
      'min_buffer_size_to_learn': _AVG_POL_MIN_BUFFER_SIZE_TO_LEARN.value,
      'optimizer_str': _AVG_POL_OPTIMIZER.value,
      'gradient_clipping': _AVG_GRADIENT_CLIPPING.value,
      'seed': _SEED.value,
      'tau': _AVG_POL_TAU.value
  }
  fp = average_network_fictitious_play.AverageNetworkFictitiousPlay(
      game,
      envs,
      br_rl_agents,
      _AVG_POL_NUM_EPISODES_PER_ITERATION.value,
      num_training_steps_per_iteration,
      eval_every=_EVAL_EVERY.value,
      logging_fn=logging_fn,
      **kwargs_avg)

  def log_metrics(it):
    """Logs the training metrics for each iteration."""
    initial_states = game.new_initial_states()
    distrib = distribution.DistributionPolicy(game, fp.policy)
    pi_value = policy_value.PolicyValue(game, distrib, fp.policy)
    m = {
        f'best_response/{state}': pi_value.eval_state(state)
        for state in initial_states
    }
    m.update({
        f'br_agent{i}/loss': agent.loss for i, agent in enumerate(br_rl_agents)
    })
    nash_conv_fp = nash_conv.NashConv(game, fp.policy)
    m['nash_conv_fp'] = nash_conv_fp.nash_conv()
    logging_fn(it, 0, m)

    # Also save the distribution.
    if _LOG_DISTRIBUTION.value and not just_logging:
      filename = os.path.join(_LOGDIR.value, f'distribution_{it}.pkl')
      utils.save_parametric_distribution(nash_conv_fp.distribution, filename)

  for it in range(_NUM_ITERATIONS.value):
    # Train the RL agent to learn a best response.
    training.run_episodes(
        envs,
        br_rl_agents,
        num_episodes=_NUM_DQN_EPISODES_PER_ITERATION.value,
        is_evaluation=False)

    # Run an iteration of average-network fictitious play and log the metrics.
    fp.iteration()
    log_metrics(it + 1)

  # Make sure all values were written.
  writer.flush()


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  app.run(main)
