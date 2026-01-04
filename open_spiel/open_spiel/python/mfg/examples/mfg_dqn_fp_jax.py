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
"""Deep Fictitious Play using DQN agents trained on an MFG."""

from absl import flags
import jax

from open_spiel.python import policy
from open_spiel.python import rl_agent_policy
from open_spiel.python import rl_environment
from open_spiel.python.jax import dqn
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms import fictitious_play
from open_spiel.python.mfg.algorithms import nash_conv
from open_spiel.python.mfg.algorithms import policy_value
from open_spiel.python.mfg.games import factory
from open_spiel.python.utils import app
from open_spiel.python.utils import metrics

FLAGS = flags.FLAGS

flags.DEFINE_string("game_name", "python_mfg_predator_prey",
                    "Name of the game.")
flags.DEFINE_string(
    "env_setting", None,
    "Name of the game settings. If None, the game name will be used.")
flags.DEFINE_integer("num_iterations", 100,
                     "Number of fictitious play iterations.")
flags.DEFINE_integer("num_episodes_per_iteration", 1000,
                     "Number of training eepisodes for each iteration.")
flags.DEFINE_list("hidden_layers_sizes", [
    128,
    128,
], "Number of hidden units in the avg-net and Q-net.")
flags.DEFINE_integer("replay_buffer_capacity", int(2e5),
                     "Size of the replay buffer.")
flags.DEFINE_integer("min_buffer_size_to_learn", 1000,
                     "Number of samples in buffer before learning begins.")
flags.DEFINE_integer("batch_size", 128,
                     "Number of transitions to sample at each learning step.")
flags.DEFINE_integer("learn_every", 64,
                     "Number of steps between learning updates.")
flags.DEFINE_float("rl_learning_rate", 0.01,
                   "Learning rate for inner rl agent.")
flags.DEFINE_string("optimizer_str", "sgd",
                    "Optimizer, choose from 'adam', 'sgd'.")
flags.DEFINE_string("loss_str", "mse",
                    "Loss function, choose from 'mse', 'huber'.")
flags.DEFINE_integer("update_target_network_every", 400,
                     "Number of steps between DQN target network updates.")
flags.DEFINE_float("discount_factor", 1.0,
                   "Discount factor for future rewards.")
flags.DEFINE_integer("epsilon_decay_duration", int(20e6),
                     "Number of game steps over which epsilon is decayed.")
flags.DEFINE_float("epsilon_start", 0.1, "Starting exploration parameter.")
flags.DEFINE_float("epsilon_end", 0.1, "Final exploration parameter.")
flags.DEFINE_bool("use_checkpoints", False, "Save/load neural network weights.")
flags.DEFINE_string("checkpoint_dir", "/tmp/dqn_test",
                    "Directory to save/load the agent.")
flags.DEFINE_string(
    "logdir", None,
    "Logging dir to use for TF summary files. If None, the metrics will only "
    "be logged to stderr.")


def main(unused_argv):
  game = factory.create_game_with_setting(FLAGS.game_name, FLAGS.env_setting)
  uniform_policy = policy.UniformRandomPolicy(game)
  mfg_dist = distribution.DistributionPolicy(game, uniform_policy)

  envs = [
      rl_environment.Environment(
          game, mfg_distribution=mfg_dist, mfg_population=p)
      for p in range(game.num_players())
  ]
  info_state_size = envs[0].observation_spec()["info_state"][0]
  num_actions = envs[0].action_spec()["num_actions"]

  hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
  kwargs = {
      "replay_buffer_capacity": FLAGS.replay_buffer_capacity,
      "min_buffer_size_to_learn": FLAGS.min_buffer_size_to_learn,
      "batch_size": FLAGS.batch_size,
      "learn_every": FLAGS.learn_every,
      "learning_rate": FLAGS.rl_learning_rate,
      "optimizer_str": FLAGS.optimizer_str,
      "loss_str": FLAGS.loss_str,
      "update_target_network_every": FLAGS.update_target_network_every,
      "discount_factor": FLAGS.discount_factor,
      "epsilon_decay_duration": FLAGS.epsilon_decay_duration,
      "epsilon_start": FLAGS.epsilon_start,
      "epsilon_end": FLAGS.epsilon_end,
  }

  # pylint: disable=g-complex-comprehension
  agents = [
      dqn.DQN(idx, info_state_size, num_actions, hidden_layers_sizes, **kwargs)
      for idx in range(game.num_players())
  ]
  joint_avg_policy = rl_agent_policy.JointRLAgentPolicy(
      game, {idx: agent for idx, agent in enumerate(agents)},
      envs[0].use_observation)

  if FLAGS.use_checkpoints:
    for agent in agents:
      if agent.has_checkpoint(FLAGS.checkpoint_dir):
        agent.restore(FLAGS.checkpoint_dir)

  # Metrics writer will also log the metrics to stderr.
  just_logging = FLAGS.logdir is None or jax.host_id() > 0
  writer = metrics.create_default_writer(
      logdir=FLAGS.logdir, just_logging=just_logging)

  # Save the parameters.
  writer.write_hparams(kwargs)

  fp = fictitious_play.FictitiousPlay(game)
  num_episodes_per_iteration = FLAGS.num_episodes_per_iteration

  def log_metrics(it, episode=0):
    initial_states = game.new_initial_states()
    fp_policy = fp.get_policy()
    distrib = distribution.DistributionPolicy(game, fp_policy)
    pi_value = policy_value.PolicyValue(game, distrib, fp_policy)
    m = {
        f"dqn_br/{state}": pi_value.eval_state(state)
        for state in initial_states
    }
    # Loss will be None at the beginning.
    if agents[0].loss is not None:
      m.update({
          f"agent{i}/loss": float(agent.loss) for i, agent in enumerate(agents)
      })
    nash_conv_fp = nash_conv.NashConv(game, fp_policy).nash_conv()
    m["nash_conv_fp"] = nash_conv_fp
    # We log using the total number of episode steps so that runs with different
    # training regimes are comparable.
    writer.write_scalars(it * num_episodes_per_iteration + episode, m)

  log_metrics(0)
  for it in range(FLAGS.num_iterations):
    # Update the Fictitious Play policy.
    fp.iteration(br_policy=joint_avg_policy)

    # Update the distribution of the environments.
    distrib = distribution.DistributionPolicy(game, fp.get_policy())
    for env in envs:
      env.update_mfg_distribution(distrib)

    # Train the RL agent to learn a best response.
    for _ in range(num_episodes_per_iteration):
      for p in range(game.num_players()):
        time_step = envs[p].reset()
        while not time_step.last():
          agent_output = agents[p].step(time_step)
          action_list = [agent_output.action]
          time_step = envs[p].step(action_list)

        # Episode is over, step all agents with final info state.
        agents[p].step(time_step)

    # Check point the agents.
    if FLAGS.use_checkpoints:
      for agent in agents:
        agent.save(FLAGS.checkpoint_dir)

    # Log the final metrics.
    log_metrics(it + 1)

  # Make sure all values were written.
  writer.flush()


if __name__ == "__main__":
  app.run(main)
