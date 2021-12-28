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
"""Provides tools to evaluate bots against specific scenarios."""

import dataclasses
from typing import Text, List
from absl import logging


@dataclasses.dataclass
class Scenario(object):
  name: Text
  init_actions: List[Text]
  expected_action_str: Text
  expected_prob: float
  player_id: int


CATCH_SCENARIOS = [
    Scenario("Ball in column 1, chooses left.", [
        "Initialized ball to 0", "LEFT", "STAY", "STAY", "STAY", "STAY", "STAY",
        "STAY", "STAY"
    ], "LEFT", 1., 0),
    Scenario("Ball in column 2, chooses left.", [
        "Initialized ball to 1", "STAY", "STAY", "STAY", "STAY", "STAY", "STAY",
        "STAY", "STAY"
    ], "LEFT", 1., 0),
    Scenario("Ball in column 3, chooses left.", [
        "Initialized ball to 2", "RIGHT", "STAY", "STAY", "STAY", "STAY",
        "STAY", "STAY", "STAY"
    ], "LEFT", 1., 0),
]

SCENARIOS = {
    "catch": CATCH_SCENARIOS,
}


def get_default_scenarios(game_name):
  """Loads the default scenarios for a given game.

  Args:
    game_name: The game to load scenarios for.

  Returns:
    A List[Scenario] detailing the scenarios for that game.
  """
  return SCENARIOS[game_name]


def play_bot_in_scenarios(game, bots, scenarios=None):
  """Plays a bot against a number of scenarios.

  Args:
    game: The game the bot is playing.
    bots: A list of length game.num_players() of pyspiel.Bots (or equivalent).
      Must implement the apply_action and step methods.
    scenarios: The scenarios we evaluate the bot in. A List[Scenario].

  Returns:
    A dict mapping scenarios to their scores (with an additional "mean_score"
    field containing the mean score across all scenarios).
    The average score across all scenarios.
  """
  if scenarios is None:
    scenarios = get_default_scenarios(game.get_type().short_name)

  results = []
  total_score = 0
  for scenario in scenarios:
    state = game.new_initial_state()
    bot = bots[scenario.player_id]
    bot.restart()
    for action_str in scenario.init_actions:
      action = state.string_to_action(action_str)
      if state.current_player() == scenario.player_id:
        bot.force_action(state, action)
      state.apply_action(action)
    actions_and_probs, _ = bot.step(state)
    expected_action = state.string_to_action(scenario.expected_action_str)
    for action, prob in actions_and_probs:
      if action == expected_action:
        actual_prob = prob
        break
    score = 1 - abs(actual_prob - scenario.expected_prob)
    results.append((scenario.name, score, scenario.expected_action_str,
                    scenario.expected_prob, actual_prob))
    total_score += score

  if scenarios:
    total_score /= len(scenarios)
  logging.info("Average score across all scenarios: %.4f.", total_score)
  results_dict = {}
  for name, score, expected_action, expected_prob, actual_prob in results:
    logging.info("************************************************************")
    logging.info("Scenario: '%s'. Score: %.4f.", name, score)
    logging.info("Expected action %s with probability %.4f but assigned %.4f.",
                 expected_action, expected_prob, actual_prob)
    logging.info("***************************")
    results_dict["scenario_score: " + name] = score
  results_dict["mean_score"] = total_score
  return results_dict
