# Copyright 2026 DeepMind Technologies Limited
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

"""Train NashPG agents on Kuhn poker and report exploitability."""

from absl import app
from absl import flags
from absl import logging

from open_spiel.python import rl_agent_policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.pytorch import nash_pg


FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "num_train_episodes", 10000, "Number of self-play episodes."
)
flags.DEFINE_integer(
    "eval_every",
    1000,
    "Episode frequency at which exploitability is evaluated.",
)
flags.DEFINE_list(
    "hidden_layers_sizes",
    [64, 64],
    "Number of hidden units in the policy and value networks.",
)
flags.DEFINE_integer(
    "batch_size", 128, "Maximum number of samples in a PPO minibatch."
)
flags.DEFINE_float("learning_rate", 3e-4, "Adam learning rate.")
flags.DEFINE_float(
    "magnet_coefficient",
    0.01,
    "KL penalty coefficient toward the magnet policy.",
)


def main(_):
  env = rl_environment.Environment("kuhn_poker")
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]
  agents = [
      nash_pg.NashPG(
          player_id=player_id,
          state_representation_size=info_state_size,
          num_actions=num_actions,
          hidden_layers_sizes=[int(size) for size in FLAGS.hidden_layers_sizes],
          batch_size=FLAGS.batch_size,
          learning_rate=FLAGS.learning_rate,
          magnet_coefficient=FLAGS.magnet_coefficient,
          seed=42 + player_id,
      )
      for player_id in range(env.num_players)
  ]
  joint_policy = rl_agent_policy.JointRLAgentPolicy(
      env.game,
      {player_id: agents[player_id] for player_id in range(env.num_players)},
      use_observation=False,
  )

  for episode in range(FLAGS.num_train_episodes):
    if (episode + 1) % FLAGS.eval_every == 0:
      current_exploitability = exploitability.exploitability(
          env.game, joint_policy
      )
      logging.info(
          "[%s] exploitability = %.6f, losses = %s",
          episode + 1,
          current_exploitability,
          [agent.loss for agent in agents],
      )

    time_step = env.reset()
    while not time_step.last():
      player_id = time_step.observations["current_player"]
      output = agents[player_id].step(time_step)
      time_step = env.step([output.action])

    for agent in agents:
      agent.step(time_step)


if __name__ == "__main__":
  app.run(main)
