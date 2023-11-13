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

"""A dm_env config for testing a given fixed game with prompt actions.
"""

import collections

from ml_collections import config_dict

from open_spiel.python.games.chat_games.envs.base_envs import email_with_tone
from open_spiel.python.games.chat_games.envs.observations import utils as obs_utils
from open_spiel.python.games.chat_games.envs.payoffs import sentiment
from open_spiel.python.games.chat_games.envs.termination import utils as term_utils
from open_spiel.python.games.chat_games.envs.utils import text as text_utils


def get_config():
  """Get configuration for chat game."""
  config = config_dict.ConfigDict()

  observations = [obs_utils.Observation(),
                  obs_utils.Observation()]

  header = email_with_tone.HEADER

  payoffs = [sentiment.PAYOFF,
             sentiment.PAYOFF]

  given_names = ['Bob',
                 'Suzy']
  num_players = len(given_names)

  given_llm_seeds = [12345]

  given_prompt_actions = collections.OrderedDict()
  tones = ['Happy',
           'Sad',
           'Angry',
           'Calm']
  given_prompt_actions[header.action_keys[0]] = tones
  num_tones = len(tones)

  # Vacuous message
  message = '\n\n'.join(text_utils.wrap(
      ['Hi {receiver},', 'I hope you are well,', 'Best,', '{sender}']
      ))
  initial_scenario = email_with_tone.Scenario(message, 'Bob', 'Suzy', 'Calm')

  query = ('Read the following message. Does it appear that ' +
           'the relevant parties have agreed on a deal? ' +
           'After reading the message, respond Yes or No. ' +
           'Here is the message:\n\n{msg}\n\n')
  llm_termination_prompt = term_utils.Termination(query, '', '')

  params = {'num_distinct_actions': num_players * num_tones,
            'num_llm_seeds': 1,
            'num_players': num_players,
            'min_utility': min([float(p.min) for p in payoffs]),
            'max_utility': max([float(p.max) for p in payoffs]),
            'num_max_replies': 2}

  config.params = params

  config.game = config_dict.ConfigDict()
  config.game.observations = observations
  config.game.header = header
  config.game.payoffs = payoffs
  config.game.given_names = given_names
  config.game.given_llm_seeds = given_llm_seeds
  config.game.given_prompt_actions = given_prompt_actions
  config.game.initial_scenario = initial_scenario
  config.game.llm_list_suffix = 'Output: '
  config.game.llm_termination_prompt = llm_termination_prompt

  return config
