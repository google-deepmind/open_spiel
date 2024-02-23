# Copyright 2023 DeepMind Technologies Limited
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

"""A pyspiel config for meta-generated meeting schedule negotiation games.
"""

import collections

from ml_collections import config_dict

from open_spiel.python.games.chat_games.envs.base_envs import schedule_meeting_with_dow_info as env_schedule_meeting_with_dow_info
from open_spiel.python.games.chat_games.envs.observations import summary
from open_spiel.python.games.chat_games.envs.observations import utils as obs_utils
from open_spiel.python.games.chat_games.envs.payoffs import schedule_meeting as payoffs_schedule_meeting
from open_spiel.python.games.chat_games.envs.scenarios.domains import schedule_meeting as scenario_schedule_meeting
from open_spiel.python.games.chat_games.envs.scenarios.players import names as names_schedule_meeting


def get_config():
  """Get configuration for chat game."""
  config = config_dict.ConfigDict()

  num_players = 2

  observations = [
      obs_utils.Observation(summary.PREFIX, summary.POSTFIX)
      for _ in range(num_players)
  ]

  header = env_schedule_meeting_with_dow_info.HEADER

  payoffs = [payoffs_schedule_meeting.PAYOFF]

  examples_names = names_schedule_meeting.NAMES

  given_prompt_actions = collections.OrderedDict()
  days = ['Monday',
          'Tuesday',
          'Wednesday',
          'Thursday',
          'Friday',
          'Saturday',
          'Sunday']
  given_prompt_actions[header.action_keys[0]] = days + ['any']
  num_days = len(days) + 1

  examples_private_info = collections.OrderedDict()
  examples_private_info['ooo_days'] = [scenario_schedule_meeting.OOO_A,
                                       scenario_schedule_meeting.OOO_B]
  examples_private_info['day_prefs'] = [scenario_schedule_meeting.DAY_PREFS_A,
                                        scenario_schedule_meeting.DAY_PREFS_B]

  scenario_a = env_schedule_meeting_with_dow_info.Scenario(
      scenario_schedule_meeting.SCENARIO_A,
      'Bob',
      'Suzy',
      scenario_schedule_meeting.OOO_A,
      scenario_schedule_meeting.DAY_PREFS_A,
      'Thursday')
  scenario_b = env_schedule_meeting_with_dow_info.Scenario(
      scenario_schedule_meeting.SCENARIO_B,
      'Jill',
      'George',
      scenario_schedule_meeting.OOO_B,
      scenario_schedule_meeting.DAY_PREFS_B,
      'Friday')

  examples_scenarios = [scenario_a, scenario_b]

  llm_termination_prompt = scenario_schedule_meeting.LLM_TERMINATION_PROMPT

  params = {'num_distinct_actions': num_players * num_days,
            'num_llm_seeds': 2,
            'num_players': num_players,
            'min_utility': min([float(p.min) for p in payoffs]),
            'max_utility': max([float(p.max) for p in payoffs]),
            'num_max_replies': 1}

  config.params = params

  config.game = config_dict.ConfigDict()
  config.game.observations = observations
  config.game.header = header
  config.game.payoffs = payoffs
  config.game.given_prompt_actions = given_prompt_actions
  config.game.num_names = 10
  config.game.num_prompt_actions = (num_days,)
  config.game.num_private_info = (3, 3)
  config.game.examples_names = examples_names
  config.game.examples_private_info = examples_private_info
  config.game.examples_scenarios = examples_scenarios
  config.game.llm_list_suffix = 'Output: '
  config.game.llm_termination_prompt = llm_termination_prompt

  return config
