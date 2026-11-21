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

"""Benchmark PPO self-play across multiple OpenSpiel games.

Runs PPO training on kuhn_poker, leduc_poker, and matrix_pd, collects metrics,
and produces a comparison plot.

Example usage:
    python open_spiel/python/examples/ppo_benchmark_jax.py
    python open_spiel/python/examples/ppo_benchmark_jax.py --num_iterations=500
"""

import os

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

flags.DEFINE_integer("num_iterations", 300,
                     "Number of training iterations per game.")
flags.DEFINE_integer("episodes_per_batch", 128,
                     "Self-play episodes per training batch.")
flags.DEFINE_integer("eval_every", 25,
                     "Evaluate every N iterations.")
flags.DEFINE_integer("num_eval_episodes", 500,
                     "Episodes per evaluation.")
flags.DEFINE_float("entropy_coef", 0.05,
                   "Entropy coefficient (higher helps mixing in poker).")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("output_dir", "/tmp/ppo_benchmark",
                    "Directory for saving plots.")

BENCHMARK_GAMES = ["kuhn_poker", "leduc_poker", "matrix_pd"]


def run_episode(env, agent, num_players):
  """Run a single self-play episode, collecting training data."""
  time_step = env.reset()
  while not time_step.last():
    if time_step.is_simultaneous_move():
      actions = []
      for pid in range(num_players):
        output = agent.step(time_step, player_id=pid)
        actions.append(output.action)
      time_step = env.step(actions)
    else:
      output = agent.step(time_step)
      time_step = env.step([output.action])
    agent.post_step(time_step)
  agent.step(time_step)
  return list(time_step.rewards)


def evaluate_returns(env, agent, num_episodes, num_players):
  """Run evaluation episodes and return per-player average returns."""
  total_returns = np.zeros(num_players)
  for _ in range(num_episodes):
    time_step = env.reset()
    while not time_step.last():
      if time_step.is_simultaneous_move():
        actions = []
        for pid in range(num_players):
          out = agent.step(time_step, player_id=pid, is_evaluation=True)
          actions.append(out.action)
        time_step = env.step(actions)
      else:
        out = agent.step(time_step, is_evaluation=True)
        time_step = env.step([out.action])
    for pid in range(num_players):
      total_returns[pid] += time_step.rewards[pid]
  return total_returns / num_episodes


def train_game(game_name):
  """Train PPO on a single game, returning metrics."""
  logging.info("=" * 60)
  logging.info("Benchmarking: %s", game_name)
  logging.info("=" * 60)

  env = rl_environment.Environment(game_name)
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]
  num_players = env.num_players

  agent = ppo.PPO(
      player_id=0,
      info_state_size=info_state_size,
      num_actions=num_actions,
      gamma=1.0,
      entropy_coef=FLAGS.entropy_coef,
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
    metrics = agent.learn()
    tracker.record_train(iteration, metrics)

    if iteration % FLAGS.eval_every == 0:
      avg_returns = evaluate_returns(
          env, agent, FLAGS.num_eval_episodes, num_players)
      expl = exploitability.exploitability(env.game, joint_policy)
      tracker.record_eval(iteration, expl, avg_returns)

      logging.info("  [%s] iter=%d  expl=%.4f  entropy=%.4f  kl=%.4f",
                   game_name, iteration, expl,
                   metrics.get("entropy", 0), metrics.get("approx_kl", 0))

  return tracker


def plot_benchmark(all_metrics, output_dir):
  """Generate a combined benchmark comparison plot."""
  # pylint: disable=g-import-not-at-top
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig, axes = plt.subplots(1, 3, figsize=(15, 4))
  fig.suptitle("PPO Self-Play Benchmark", fontsize=14)

  for i, (game, tracker) in enumerate(all_metrics.items()):
    ax = axes[i]
    if tracker.exploitability:
      ax.plot(tracker.eval_iterations, tracker.exploitability, "o-",
              markersize=3)
    ax.set_title(game)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Exploitability")
    ax.grid(True, alpha=0.3)

  plt.tight_layout()
  path = os.path.join(output_dir, "ppo_benchmark.png")
  plt.savefig(path, dpi=150, bbox_inches="tight")
  plt.close(fig)
  logging.info("Benchmark plot saved to %s", path)

  for game, tracker in all_metrics.items():
    game_path = os.path.join(output_dir, f"ppo_{game}.png")
    ppo_utils.plot_training_curves(tracker, game, save_path=game_path)
    logging.info("Training curves saved to %s", game_path)


def main(_):
  os.makedirs(FLAGS.output_dir, exist_ok=True)

  all_metrics = {}
  for game in BENCHMARK_GAMES:
    all_metrics[game] = train_game(game)

  plot_benchmark(all_metrics, FLAGS.output_dir)

  logging.info("-" * 60)
  logging.info("Final results:")
  for game, tracker in all_metrics.items():
    if tracker.exploitability:
      logging.info("  %s: final_expl=%.4f", game, tracker.exploitability[-1])

  logging.info("Benchmark complete. Plots in %s", FLAGS.output_dir)


if __name__ == "__main__":
  app.run(main)
