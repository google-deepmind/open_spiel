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

"""Examples of schedule negotations -- useful for generating more examples.
"""

from open_spiel.python.games.chat_games.envs.observations import summary
from open_spiel.python.games.chat_games.envs.termination import utils as term_utils
from open_spiel.python.games.chat_games.envs.utils import text

# Scenario A
OOO_LIST_A = ['monday: false',
              'tuesday: true',
              'wednesday: true',
              'thursday: false',
              'friday: false',
              'saturday: true',
              'sunday: false']
OOO_A = '\n'.join(text.wrap(OOO_LIST_A))

DAY_PREFS_LIST_A = ['monday: 10',
                    'tuesday: 5',
                    'wednesday: 15',
                    'thursday: 3',
                    'friday: 2',
                    'saturday: 1',
                    'sunday: 1'
                    ]
DAY_PREFS_A = '\n'.join(text.wrap(DAY_PREFS_LIST_A))

SCENARIO_A_LIST = ['Hi {receiver},',
                   'I would like to propose meeting on thursday.',
                   'Would you like to meet with me then?',
                   'Best,', '{sender}']
SCENARIO_A = '\n\n'.join(text.wrap(SCENARIO_A_LIST))

# Scenario B
OOO_LIST_B = ['monday: true',
              'tuesday: false',
              'wednesday: true',
              'thursday: false',
              'friday: false',
              'saturday: true',
              'sunday: false']
OOO_B = '\n'.join(text.wrap(OOO_LIST_B))

DAY_PREFS_LIST_B = ['monday: 5',
                    'tuesday: 5',
                    'wednesday: 5',
                    'thursday: 1',
                    'friday: 1',
                    'saturday: 1',
                    'sunday: 1'
                    ]
DAY_PREFS_B = '\n'.join(text.wrap(DAY_PREFS_LIST_B))

SCENARIO_B_LIST = ['Hi {receiver},',
                   'I strongly urge you to meet me on friday when I am in ' +
                   'the office.',
                   'what do you say?',
                   'Best,', '{sender}']
SCENARIO_B = '\n\n'.join(text.wrap(SCENARIO_B_LIST))

query = ('Read the following summary of a dialgoue between two parties ' +
         'attempting to reach an agreement. Have the players reached an ' +
         'agreement? If a meeting time has been accepted or the players ' +
         'cannot come to an agreement, respond Yes. Otherwise, if the ' +
         'players are still discussing terms, respond No.' +
         'Here is the dialogue:\n\n{msg}\n\n' + '&' *50 +
         '\n\nHave all parties agreed on a meeting time?'
         '\nResponse: ')

LLM_TERMINATION_PROMPT = term_utils.Termination(query,
                                                summary.PREFIX,
                                                summary.POSTFIX)
