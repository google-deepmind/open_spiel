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

"""A pyspiel config for meta-generated fruit trading games.
"""

import collections

from ml_collections import config_dict

from open_spiel.python.games.chat_games.envs.base_envs import trade_fruit_with_tone_info as env_trade_fruit_with_tone_info
from open_spiel.python.games.chat_games.envs.observations import summary
from open_spiel.python.games.chat_games.envs.observations import utils as obs_utils
from open_spiel.python.games.chat_games.envs.payoffs import trade_fruit as payoffs_trade_fruit
from open_spiel.python.games.chat_games.envs.scenarios.domains import trade_fruit as scenario_trade_fruit


def get_config():
  """Get configuration for chat game."""
  config = config_dict.ConfigDict()

  num_players = 2

  observations = [
      obs_utils.Observation(summary.PREFIX, summary.POSTFIX)
      for _ in range(num_players)
  ]

  header = env_trade_fruit_with_tone_info.HEADER

  payoffs = [payoffs_trade_fruit.PAYOFF]

  given_prompt_actions = collections.OrderedDict()
  tones = ['calm',
           'assertive',
           'submissive',
           'any']
  given_prompt_actions[header.action_keys[0]] = tones
  num_tones = len(tones)

  given_private_info = collections.OrderedDict()
  given_private_info['fruit_endowment'] = [scenario_trade_fruit.ENDOWMENT_A,
                                           scenario_trade_fruit.ENDOWMENT_B]
  given_private_info['fruit_valuations'] = [scenario_trade_fruit.VALUATION_A,
                                            scenario_trade_fruit.VALUATION_B]

  scenario_a = env_trade_fruit_with_tone_info.Scenario(
      scenario_trade_fruit.SCENARIO_A,
      'Bob',
      'Suzy',
      scenario_trade_fruit.ENDOWMENT_A,
      scenario_trade_fruit.VALUATION_A,
      'calm')

  llm_termination_prompt = scenario_trade_fruit.LLM_TERMINATION_PROMPT

  params = {'num_distinct_actions': num_players * num_tones,
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
  config.game.given_names = ['Bob', 'Suzy']
  config.game.given_private_info = given_private_info
  config.game.initial_scenario = scenario_a
  config.game.llm_list_suffix = 'Output: '
  config.game.llm_termination_prompt = llm_termination_prompt

  return config
