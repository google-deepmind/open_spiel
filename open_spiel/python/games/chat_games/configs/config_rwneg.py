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

"""A pyspiel config for meta-generated real-world negotiation games.
"""

import collections

from ml_collections import config_dict

from open_spiel.python.games.chat_games.envs.base_envs import email_with_tone
from open_spiel.python.games.chat_games.envs.observations import summary
from open_spiel.python.games.chat_games.envs.observations import utils as obs_utils
from open_spiel.python.games.chat_games.envs.payoffs import sentiment
from open_spiel.python.games.chat_games.envs.scenarios.actions import tones
from open_spiel.python.games.chat_games.envs.scenarios.domains import real_world_negotiations as rwn
from open_spiel.python.games.chat_games.envs.scenarios.players import names
from open_spiel.python.games.chat_games.envs.termination import utils as term_utils


def get_config():
  """Get configuration for chat game."""
  config = config_dict.ConfigDict()

  num_players = 3

  observations = [
      obs_utils.Observation(summary.PREFIX, summary.POSTFIX)
      for _ in range(num_players)
  ]

  scenario_a = email_with_tone.Scenario(rwn.SCENARIO_A, 'Alice', 'Bob')
  scenario_b = email_with_tone.Scenario(rwn.SCENARIO_B, 'Joel', 'Gene')
  scenario_c = email_with_tone.Scenario(rwn.SCENARIO_C, 'George', 'Jill')
  examples_scenarios = [scenario_a,
                        scenario_b,
                        scenario_c]

  header = email_with_tone.HEADER

  payoffs = [sentiment.PAYOFF]

  examples_names = names.NAMES

  examples_prompt_actions = collections.OrderedDict()
  examples_prompt_actions[header.action_keys[0]] = tones.TONES
  num_tones = 3

  query = ('Read the following message. Does it appear that ' +
           'the relevant parties have agreed on a deal? ' +
           'After reading the message, respond Yes or No. ' +
           'Here is the message:\n\n{msg}\n\n')
  llm_termination_prompt = term_utils.Termination(query, '', '')

  params = {'num_distinct_actions': num_players * num_tones,
            'num_llm_seeds': 2,
            'num_players': num_players,
            'min_utility': min([float(p.min) for p in payoffs]),
            'max_utility': max([float(p.max) for p in payoffs]),
            'num_max_replies': 2}

  config.params = params

  config.game = config_dict.ConfigDict()
  config.game.observations = observations
  config.game.header = header
  config.game.payoffs = payoffs
  config.game.num_names = 10
  config.game.num_prompt_actions = (num_tones,)
  config.game.num_private_info = (3,)
  config.game.examples_names = examples_names
  config.game.examples_prompt_actions = examples_prompt_actions
  config.game.examples_scenarios = examples_scenarios
  config.game.llm_list_suffix = 'Output: '
  config.game.llm_termination_prompt = llm_termination_prompt

  return config
