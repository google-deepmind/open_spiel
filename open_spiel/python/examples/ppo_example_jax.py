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

"""Self-play PPO training on OpenSpiel games using JAX.

A single PPO agent controls all players, learning a strategy via self-play.
Supports both sequential (e.g. kuhn_poker) and simultaneous-move games
(e.g. matrix_pd). Periodically evaluates exploitability as a convergence
measure and stores metrics for plotting.

Example usage:
    python open_spiel/python/examples/ppo_example_jax.py --game=kuhn_poker
    python open_spiel/python/examples/ppo_example_jax.py --game=leduc_poker
    python open_spiel/python/examples/ppo_example_jax.py --game=matrix_pd
"""

from absl import app
from absl import flags
from absl import logging
import numpy as np

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.jax import ppo
from open_spiel.python.jax import ppo_utils
from open_spiel.python.rl_agent_policy import JointRLAgentPolicy

FLAGS = flags.FLAGS

flags.DEFINE_string("game", "kuhn_poker", "Name of the OpenSpiel game.")
flags.DEFINE_integer("num_iterations", 1000,
                     "Number of training iterations.")
flags.DEFINE_integer("episodes_per_batch", 128,
                     "Number of self-play episodes per training batch.")
flags.DEFINE_integer("eval_every", 50,
                     "Evaluate every N training iterations.")
flags.DEFINE_integer("num_eval_episodes", 1000,
                     "Number of episodes for evaluation.")

# PPO hyperparameters.
flags.DEFINE_float("learning_rate", 2.5e-4, "Adam learning rate.")
flags.DEFINE_float("gamma", 1.0,
                   "Discount factor (1.0 for short episodic games).")
flags.DEFINE_float("gae_lambda", 0.95, "GAE lambda parameter.")
flags.DEFINE_float("clip_coef", 0.2, "PPO clipping coefficient.")
flags.DEFINE_float("entropy_coef", 0.01, "Entropy bonus coefficient.")
flags.DEFINE_float("value_coef", 0.5, "Value loss coefficient.")
flags.DEFINE_float("max_grad_norm", 0.5, "Max gradient norm for clipping.")
flags.DEFINE_integer("update_epochs", 4, "PPO update epochs per batch.")
flags.DEFINE_integer("num_minibatches", 4,
                     "Number of minibatches per update epoch.")
flags.DEFINE_list("hidden_sizes", ["64", "64"],
                  "Hidden layer sizes for the actor-critic network.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("plot_path", None,
                    "If set, save training curve plot to this file path.")


def run_episode(env, agent, num_players, is_evaluation=False):
  """Run a single self-play episode, handling both sequential and simultaneous.

  Args:
    env: the OpenSpiel rl_environment.Environment.
    agent: the PPO agent.
    num_players: number of players in the game.
    is_evaluation: if True, no data is collected.

  Returns:
    Per-player rewards as a list.
  """
  time_step = env.reset()
  while not time_step.last():
    if time_step.is_simultaneous_move():
      actions = []
      for pid in range(num_players):
        output = agent.step(time_step, player_id=pid,
                            is_evaluation=is_evaluation)
        actions.append(output.action)
      time_step = env.step(actions)
    else:
      output = agent.step(time_step, is_evaluation=is_evaluation)
      time_step = env.step([output.action])
    if not is_evaluation:
      agent.post_step(time_step)
  agent.step(time_step, is_evaluation=is_evaluation)
  return list(time_step.rewards)


def evaluate_returns(env, agent, num_episodes, num_players):
  """Run evaluation episodes and return per-player average returns."""
  total_returns = np.zeros(num_players)
  for _ in range(num_episodes):
    rewards = run_episode(env, agent, num_players, is_evaluation=True)
    for pid in range(num_players):
      total_returns[pid] += rewards[pid]
  return total_returns / num_episodes


def main(_):
  env = rl_environment.Environment(FLAGS.game)
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]
  num_players = env.num_players

  logging.info("Game: %s", FLAGS.game)
  logging.info("Info state size: %d, Num actions: %d, Num players: %d",
               info_state_size, num_actions, num_players)

  hidden_sizes = [int(s) for s in FLAGS.hidden_sizes]

  agent = ppo.PPO(
      player_id=0,
      info_state_size=info_state_size,
      num_actions=num_actions,
      hidden_sizes=hidden_sizes,
      learning_rate=FLAGS.learning_rate,
      gamma=FLAGS.gamma,
      gae_lambda=FLAGS.gae_lambda,
      clip_coef=FLAGS.clip_coef,
      entropy_coef=FLAGS.entropy_coef,
      value_coef=FLAGS.value_coef,
      max_grad_norm=FLAGS.max_grad_norm,
      update_epochs=FLAGS.update_epochs,
      num_minibatches=FLAGS.num_minibatches,
      seed=FLAGS.seed,
  )

  joint_policy = JointRLAgentPolicy(
      env.game,
      {pid: agent for pid in range(num_players)},
      use_observation=env.use_observation,
  )

  tracker = ppo_utils.TrainingMetrics()

  for iteration in range(1, FLAGS.num_iterations + 1):
    for _ in range(FLAGS.episodes_per_batch):
      run_episode(env, agent, num_players)

    batch_steps = agent.buffer_size
    metrics = agent.learn()
    tracker.record_train(iteration, metrics)

    if iteration % FLAGS.eval_every == 0:
      avg_returns = evaluate_returns(
          env, agent, FLAGS.num_eval_episodes, num_players)
      expl = exploitability.exploitability(env.game, joint_policy)
      tracker.record_eval(iteration, expl, avg_returns)

      logging.info("-" * 60)
      logging.info("Iteration %d  |  batch_steps=%d", iteration, batch_steps)
      logging.info(
          "  policy_loss=%.6f  value_loss=%.6f  entropy=%.6f  approx_kl=%.6f",
          metrics.get("policy_loss", 0),
          metrics.get("value_loss", 0),
          metrics.get("entropy", 0),
          metrics.get("approx_kl", 0))
      returns_str = "  ".join(
          f"player_{p}={avg_returns[p]:.4f}" for p in range(num_players))
      logging.info("  Avg returns: %s", returns_str)
      logging.info("  Exploitability: %.6f", expl)

  logging.info("Training complete.")

  if FLAGS.plot_path:
    ppo_utils.plot_training_curves(tracker, FLAGS.game, FLAGS.plot_path)
    logging.info("Plot saved to %s", FLAGS.plot_path)


if __name__ == "__main__":
  app.run(main)
