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

"""Examples of debates -- useful for generating more examples.
"""

from open_spiel.python.games.chat_games.envs.utils import text

# Scenario A
SCENARIO_A_LIST = ['Tom Brady is the GOAT and coach Bill Bellichick ' +
                   'is a genius']
SCENARIO_A = '\n\n'.join(text.wrap(SCENARIO_A_LIST))

TOPIC_A = 'The New England Patriots are the best NFL team in 2023.'

INFO_A = ''

# Scenario B
SCENARIO_B_LIST = ['Breakfast is the most important meal of the day.']
SCENARIO_B = '\n\n'.join(text.wrap(SCENARIO_B_LIST))

TOPIC_B = 'Breakfast is the most important meal of the day.'

INFO_B = ''

LLM_TERMINATION_PROMPT = None
