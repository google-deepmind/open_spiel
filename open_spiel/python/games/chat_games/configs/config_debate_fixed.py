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

"""A pyspiel config for a fixed debate.
"""

import collections

from ml_collections import config_dict

from open_spiel.python.games.chat_games.envs.base_envs import debate_with_style_info as env_debate_with_style_info
from open_spiel.python.games.chat_games.envs.observations import summary_debate
from open_spiel.python.games.chat_games.envs.observations import utils as obs_utils
from open_spiel.python.games.chat_games.envs.payoffs import debate as payoffs_debate
from open_spiel.python.games.chat_games.envs.scenarios.actions import arguments
from open_spiel.python.games.chat_games.envs.scenarios.domains import debate as scenario_debate


def get_config():
  """Get configuration for chat game."""
  config = config_dict.ConfigDict()

  num_players = 2

  observations = [
      obs_utils.Observation(summary_debate.PREFIX, summary_debate.POSTFIX)
      for _ in range(num_players)
  ]

  header = env_debate_with_style_info.HEADER

  payoffs = [payoffs_debate.PAYOFF]

  given_prompt_actions = collections.OrderedDict()
  given_prompt_actions[header.action_keys[0]] = arguments.STYLES + ['any']
  num_styles = len(arguments.STYLES) + 1

  given_private_info = collections.OrderedDict()
  given_private_info['info'] = ['Argue for the topic statement.',
                                'Argue against the topic statement.']
  given_private_info['topic'] = [scenario_debate.TOPIC_B,
                                 scenario_debate.TOPIC_B]

  initial_scenario = env_debate_with_style_info.Scenario(
      '',
      'Bob',
      'Alice',
      'logos',
      scenario_debate.TOPIC_B,
      'Argue for the topic statement.')

  llm_termination_prompt = scenario_debate.LLM_TERMINATION_PROMPT

  params = {'num_distinct_actions': num_players * num_styles,
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
  config.game.num_private_info = (2, 2)
  config.game.given_names = ['Bob', 'Alice']
  config.game.given_private_info = given_private_info
  config.game.initial_scenario = initial_scenario
  config.game.llm_list_suffix = 'Output: '
  config.game.llm_termination_prompt = llm_termination_prompt

  return config
