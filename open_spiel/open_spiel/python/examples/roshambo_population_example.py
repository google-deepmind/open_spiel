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

"""Simple example of using the Roshambo population.

Note: the Roshambo bots are an optional dependency and excluded by default.
To enable Roshambo bots, set OPEN_SPIEL_BUILD_WITH_ROSHAMBO to ON when building.
See
https://github.com/deepmind/open_spiel/blob/master/docs/install.md#configuring-conditional-dependencies
for details.
"""

import re
from absl import app
from absl import flags
import numpy as np

from open_spiel.python import games  # pylint: disable=unused-import
from open_spiel.python import rl_agent
from open_spiel.python import rl_environment
import pyspiel

FLAGS = flags.FLAGS

# See open_spiel/data/paper_data/pbe_rrps for the bot table from the RRPS paper:
# https://arxiv.org/abs/2303.03196
flags.DEFINE_string("bot_table_file", None,
                    "The file containing the bot entries.")

flags.DEFINE_integer("player0_pop_id", 0, "Population member ID for player 0")
flags.DEFINE_integer("player1_pop_id", 1, "Population member ID for player 1")
flags.DEFINE_integer("seed", 0, "Seed to use for RNG")
flags.DEFINE_integer("env_recall", 1,
                     "Number of recent steps to include in observation")


class BotAgent(rl_agent.AbstractAgent):
  """Agent class that wraps a bot.

  Note, the environment must include the OpenSpiel state in its observations,
  which means it must have been created with use_full_state=True.
  """

  def __init__(self, num_actions, bot, name="bot_agent"):
    assert num_actions > 0
    self._bot = bot
    self._num_actions = num_actions

  def restart(self):
    self._bot.restart()

  def step(self, time_step, is_evaluation=False):
    # If it is the end of the episode, don't select an action.
    if time_step.last():
      return

    _, state = pyspiel.deserialize_game_and_state(
        time_step.observations["serialized_state"])

    action = self._bot.step(state)
    probs = np.zeros(self._num_actions)
    probs[action] = 1.0

    return rl_agent.StepOutput(action=action, probs=probs)


def eval_agents(env, agents, num_players, num_episodes):
  """Evaluate the agent."""
  sum_episode_rewards = np.zeros(num_players)
  for ep in range(num_episodes):
    for agent in agents:
      # Bots need to be restarted at the start of the episode.
      if hasattr(agent, "restart"):
        agent.restart()
    time_step = env.reset()
    episode_rewards = np.zeros(num_players)
    while not time_step.last():
      agents_output = [
          agent.step(time_step, is_evaluation=True) for agent in agents
      ]
      action_list = [agent_output.action for agent_output in agents_output]
      time_step = env.step(action_list)
      episode_rewards += time_step.rewards
    sum_episode_rewards += episode_rewards
    print(f"Finished episode {ep}, "
          + f"avg returns: {sum_episode_rewards / num_episodes}")

  return sum_episode_rewards / num_episodes


def print_roshambo_bot_names_and_ids(roshambo_bot_names):
  print("Roshambo bot population:")
  for i in range(len(roshambo_bot_names)):
    print(f"{i}: {roshambo_bot_names[i]}")


def create_roshambo_bot_agent(player_id, num_actions, bot_names, pop_id):
  name = bot_names[pop_id]
  # Creates an OpenSpiel bot with the default number of throws
  # (pyspiel.ROSHAMBO_NUM_THROWS). To create one for a different number of
  # throws per episode, add the number as the third argument here.
  bot = pyspiel.make_roshambo_bot(player_id, name)
  return BotAgent(num_actions, bot, name=name)


def analyze_bot_table(filename):
  """Do some analysis on the payoff cross-table."""
  print(f"Opening bot table file: {filename}")
  bot_table_file = open(filename, "r")
  table = np.zeros(shape=(pyspiel.ROSHAMBO_NUM_BOTS,
                          pyspiel.ROSHAMBO_NUM_BOTS), dtype=np.float64)
  print("Parsing file...")
  values = {}
  bot_names_map = {}
  for line in bot_table_file:
    line = line.strip()
    # ('driftbot', 'driftbot', -0.571)
    myre = re.compile(r"\'(.*)\', \'(.*)\', (.*)\)")
    match_obj = myre.search(line)
    row_agent, col_agent, value = match_obj.groups()
    values[f"{row_agent},{col_agent}"] = value
    bot_names_map[row_agent] = True
  bot_names_list = list(bot_names_map.keys())
  bot_names_list.sort()
  print(len(bot_names_list))
  assert len(bot_names_list) == pyspiel.ROSHAMBO_NUM_BOTS
  print(bot_names_list)
  for i in range(pyspiel.ROSHAMBO_NUM_BOTS):
    for j in range(pyspiel.ROSHAMBO_NUM_BOTS):
      key = f"{bot_names_list[i]},{bot_names_list[j]}"
      assert key in values
      table[i][j] = float(values[key])
  print("Population returns:")
  pop_returns = np.zeros(pyspiel.ROSHAMBO_NUM_BOTS)
  pop_aggregate = np.zeros(pyspiel.ROSHAMBO_NUM_BOTS)
  for i in range(pyspiel.ROSHAMBO_NUM_BOTS):
    pop_eval = 0
    for j in range(pyspiel.ROSHAMBO_NUM_BOTS):
      pop_eval += table[i][j]
    pop_eval /= pyspiel.ROSHAMBO_NUM_BOTS
    # print(f"  {bot_names_list[i]}: {pop_eval}")
    pop_returns[i] = pop_eval
    pop_aggregate[i] += pop_eval
    print(f"  {pop_eval},")
  print("Population exploitabilities: ")
  pop_expls = np.zeros(pyspiel.ROSHAMBO_NUM_BOTS)
  avg_pop_expl = 0
  for i in range(pyspiel.ROSHAMBO_NUM_BOTS):
    pop_expl = -float(pyspiel.ROSHAMBO_NUM_THROWS)
    for j in range(pyspiel.ROSHAMBO_NUM_BOTS):
      pop_expl = max(pop_expl, -table[i][j])
    avg_pop_expl += pop_expl
    pop_expls[i] = pop_expl
    pop_aggregate[i] -= pop_expl
    print(f"  {pop_expl},")
  avg_pop_expl /= pyspiel.ROSHAMBO_NUM_BOTS
  print(f"Avg within-pop expl: {avg_pop_expl}")
  print("Aggregate: ")
  indices = np.argsort(pop_aggregate)
  for i in range(pyspiel.ROSHAMBO_NUM_BOTS):
    idx = indices[pyspiel.ROSHAMBO_NUM_BOTS - i - 1]
    print(f"  {i+1} & \\textsc{{{bot_names_list[idx]}}} & " +
          f" ${pop_returns[idx]:0.3f}$ " +
          f"& ${pop_expls[idx]:0.3f}$ & ${pop_aggregate[idx]:0.3f}$ \\\\")
  print("Dominance:")
  for i in range(pyspiel.ROSHAMBO_NUM_BOTS):
    for j in range(pyspiel.ROSHAMBO_NUM_BOTS):
      if np.all(np.greater(table[i], table[j])):
        print(f"{bot_names_list[i]} dominates {bot_names_list[j]}")


def main(_):
  np.random.seed(FLAGS.seed)

  if FLAGS.bot_table_file is not None:
    analyze_bot_table(FLAGS.bot_table_file)
    return

  # Note that the include_full_state variable has to be enabled because the
  # BotAgent needs access to the full state.
  env = rl_environment.Environment(
      "repeated_game(stage_game=matrix_rps(),num_repetitions=" +
      f"{pyspiel.ROSHAMBO_NUM_THROWS}," +
      f"recall={FLAGS.env_recall})",
      include_full_state=True)
  num_players = 2
  num_actions = env.action_spec()["num_actions"]
  # Learning agents might need this:
  # info_state_size = env.observation_spec()["info_state"][0]

  print("Loading population...")
  pop_size = pyspiel.ROSHAMBO_NUM_BOTS
  print(f"Population size: {pop_size}")
  roshambo_bot_names = pyspiel.roshambo_bot_names()
  roshambo_bot_names.sort()
  print_roshambo_bot_names_and_ids(roshambo_bot_names)

  bot_id = 0
  roshambo_bot_ids = {}
  for name in roshambo_bot_names:
    roshambo_bot_ids[name] = bot_id
    bot_id += 1

  # Create two bot agents
  agents = [
      create_roshambo_bot_agent(0, num_actions, roshambo_bot_names,
                                FLAGS.player0_pop_id),
      create_roshambo_bot_agent(1, num_actions, roshambo_bot_names,
                                FLAGS.player1_pop_id)
  ]

  print("Starting eval run.")
  print(f"Player 0 is (pop_id {FLAGS.player0_pop_id}: " +
        f"{roshambo_bot_names[FLAGS.player0_pop_id]})")
  print(f"Player 1 is (pop_id {FLAGS.player1_pop_id}: " +
        f"{roshambo_bot_names[FLAGS.player1_pop_id]})")
  avg_eval_returns = eval_agents(env, agents, num_players, 100)
  print(avg_eval_returns)


if __name__ == "__main__":
  app.run(main)
