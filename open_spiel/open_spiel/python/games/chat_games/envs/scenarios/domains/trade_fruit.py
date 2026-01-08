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

"""Examples of fruit trading scenarios -- useful for generating more examples.
"""

from open_spiel.python.games.chat_games.envs.observations import summary
from open_spiel.python.games.chat_games.envs.termination import utils as term_utils
from open_spiel.python.games.chat_games.envs.utils import text

# Scenario A
SCENARIO_A_LIST = ['Hi {receiver},',
                   'I would like to trade you 1 banana for 1 apple.',
                   'Would you like to trade with me?',
                   'Best,', '{sender}']
SCENARIO_A = '\n\n'.join(text.wrap(SCENARIO_A_LIST))

ENDOWMENT_A_LIST = ['apple: 1', 'banana: 2', 'blueberry: 0', 'kiwi: 0']
ENDOWMENT_A = '\n'.join(text.wrap(ENDOWMENT_A_LIST))

VALUATION_A_LIST = ['apple: 10',
                    'banana: 5',
                    'blueberry: 1',
                    'kiwi: 3']
VALUATION_A = '\n'.join(text.wrap(VALUATION_A_LIST))

# Scenario B
SCENARIO_B_LIST = ['Hi {receiver},',
                   'I would like to trade you 3 blueberries for 1 banana.',
                   'Would you like to trade with me?',
                   'Best,', '{sender}']
SCENARIO_B = '\n\n'.join(text.wrap(SCENARIO_A_LIST))

ENDOWMENT_B_LIST = ['apple: 0', 'banana: 0', 'blueberry: 5', 'kiwi: 3']
ENDOWMENT_B = '\n'.join(text.wrap(ENDOWMENT_B_LIST))

VALUATION_B_LIST = ['apple: 8',
                    'banana: 7',
                    'blueberry: 2',
                    'kiwi: 2']
VALUATION_B = '\n'.join(text.wrap(VALUATION_B_LIST))

query = ('Read the following summary of a dialgoue between two parties ' +
         'attempting to reach a trade agreement. Have the players reached a ' +
         'trade agreement? If a trade has been accepted or the players cannot' +
         ' come to an agreement, respond Yes. Otherwise, if the players are ' +
         'still discussing terms, respond No.' +
         'Here is the dialogue:\n\n{msg}\n\n' + '&' *50 +
         'Response: ')

LLM_TERMINATION_PROMPT = term_utils.Termination(query,
                                                summary.PREFIX,
                                                summary.POSTFIX)
