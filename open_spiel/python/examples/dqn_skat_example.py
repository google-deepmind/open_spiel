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

"""DQN agents trained on Skat by independent Q-learning."""

import random

from absl import app
from absl import flags
from absl import logging
import numpy as np

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent
from open_spiel.python.jax import dqn

FLAGS = flags.FLAGS

# Training parameters
flags.DEFINE_string(
    "checkpoint_dir", "/tmp/skat_dqn/", "Directory to save/load the agent."
)
flags.DEFINE_integer("num_train_episodes", int(1e6),
                     "Number of training episodes.")
flags.DEFINE_integer(
    "eval_every", 1000,
    "Episode frequency at which the DQN agents are evaluated.")
flags.DEFINE_integer(
    "num_eval_games", 1000, "How many games to play during each evaluation."
)

# DQN model hyper-parameters
flags.DEFINE_list("hidden_layers_sizes", [64, 64],
                  "Number of hidden units in the Q-Network MLP.")
flags.DEFINE_integer("replay_buffer_capacity", int(1e5),
                     "Size of the replay buffer.")
flags.DEFINE_integer("batch_size", 32,
                     "Number of transitions to sample at each learning step.")
flags.DEFINE_bool(
    "randomize_positions",
    True,
    "Randomize the position of each agent before every game.",
)
flags.DEFINE_bool("use_checkpoints", False, "Save/load neural network weights.")


def eval_against_random_bots(env, trained_agents, random_agents, num_episodes):
  """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
  num_players = len(trained_agents)
  sum_episode_rewards = np.zeros(num_players)
  for player_pos in range(num_players):
    for _ in range(num_episodes):
      cur_agents = random_agents[:]
      if FLAGS.randomize_positions:
        eval_player_pos = random.randrange(num_players)
      else:
        eval_player_pos = player_pos
      cur_agents[eval_player_pos] = trained_agents[player_pos]
      cur_agents[eval_player_pos].player_id = eval_player_pos
      time_step = env.reset()
      episode_rewards = 0
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        agent_output = cur_agents[player_id].step(time_step, is_evaluation=True)
        action_list = [agent_output.action]
        time_step = env.step(action_list)
        episode_rewards += time_step.rewards[eval_player_pos]
      sum_episode_rewards[player_pos] += episode_rewards
  return sum_episode_rewards / num_episodes


def main(_):
  game = "skat"
  num_players = 3

  env_configs = {}
  env = rl_environment.Environment(game, **env_configs)
  observation_tensor_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  # random agents for evaluation
  random_agents = [
      random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
      for idx in range(num_players)
  ]

  hidden_layers_sizes = [int(s) for s in FLAGS.hidden_layers_sizes]
  # pylint: disable=g-complex-comprehension
  agents = [
      dqn.DQN(
          player_id=idx,
          state_representation_size=observation_tensor_size,
          num_actions=num_actions,
          hidden_layers_sizes=hidden_layers_sizes,
          replay_buffer_capacity=FLAGS.replay_buffer_capacity,
          batch_size=FLAGS.batch_size,
      )
      for idx in range(num_players)
  ]

  for ep in range(FLAGS.num_train_episodes):
    if (ep + 1) % FLAGS.eval_every == 0:
      r_mean = eval_against_random_bots(
          env, agents, random_agents, FLAGS.num_eval_games
      )
      logging.info("[%s] Mean episode rewards %s", ep + 1, r_mean)
      # If you want saving, uncomment
      if FLAGS.use_checkpoints:
        for i in range(num_players):
          agents[i].save(FLAGS.checkpoint_dir)

    time_step = env.reset()
    # Randomize position.
    if FLAGS.randomize_positions:
      positions = random.sample(range(len(agents)), len(agents))
    while not time_step.last():
      player_id = time_step.observations["current_player"]
      if FLAGS.randomize_positions:
        position = positions[player_id]
        agents[position].player_id = player_id
      else:
        position = player_id
      agent_output = agents[position].step(time_step)
      action_list = [agent_output.action]
      time_step = env.step(action_list)

    # Episode is over, step all agents with final info state.
    for agent in agents:
      agent.step(time_step)


if __name__ == "__main__":
  app.run(main)
