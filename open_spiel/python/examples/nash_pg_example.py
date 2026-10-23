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

"""NashPG self-play example."""

from absl import app
from absl import flags
from absl import logging

from open_spiel.python import rl_agent_policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.pytorch import nash_pg

FLAGS = flags.FLAGS

flags.DEFINE_string("game_name", "kuhn_poker", "Game to play.")
flags.DEFINE_integer("num_episodes", 500_000, "Number of training episodes.")
flags.DEFINE_integer("eval_every", 10_000, "Episodes between evaluations.")
flags.DEFINE_integer("seed", 0, "Random seed.")

# NashPG hyperparameters (constructor defaults).
flags.DEFINE_list("hidden_layers_sizes", ["128", "128"], "Policy/value MLP.")
flags.DEFINE_integer("batch_size", 1024, "Own decisions per update.")
flags.DEFINE_float("learning_rate", 1e-3, "Adam learning rate.")
flags.DEFINE_float("magnet_coef", 0.2, "KL-to-magnet penalty weight.")
flags.DEFINE_integer("magnet_update_period", 40, "Updates between refreshes.")
flags.DEFINE_float("entropy_coef", 0.1, "Entropy bonus weight.")


def main(_):
  env = rl_environment.Environment(FLAGS.game_name)
  env.seed(FLAGS.seed)

  agents = [
      nash_pg.NashPG(
          player_id,
          env.observation_spec()["info_state"][0],
          env.action_spec()["num_actions"],
          hidden_layers_sizes=[int(x) for x in FLAGS.hidden_layers_sizes],
          batch_size=FLAGS.batch_size,
          learning_rate=FLAGS.learning_rate,
          magnet_coef=FLAGS.magnet_coef,
          magnet_update_period=FLAGS.magnet_update_period,
          entropy_coef=FLAGS.entropy_coef,
          seed=FLAGS.seed + player_id,
      )
      for player_id in range(env.num_players)
  ]
  joint_policy = rl_agent_policy.JointRLAgentPolicy(
      env.game,
      {p: agents[p] for p in range(env.num_players)},
      use_observation=env.use_observation,
  )

  for episode in range(FLAGS.num_episodes):
    if (episode + 1) % FLAGS.eval_every == 0:
      expl = exploitability.exploitability(env.game, joint_policy)
      logging.info(
          "[episode %d] exploitability %.5f | losses %s",
          episode + 1,
          expl,
          [a.loss for a in agents],
      )

    time_step = env.reset()
    while not time_step.last():
      if time_step.is_simultaneous_move():
        actions = [agent.step(time_step).action for agent in agents]
      else:
        player_id = time_step.observations["current_player"]
        actions = [agents[player_id].step(time_step).action]
        # For sequential game with dense rewards, replace the line above with:
        # outs = [agent.step(time_step) for agent in agents]
        # actions = [outs[player_id].action]
      time_step = env.step(actions)
    for agent in agents:
      agent.step(time_step)


if __name__ == "__main__":
  app.run(main)
