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

"""A few prompts for extracting the value of a schedule negotiation with llms.
"""

from open_spiel.python.games.chat_games.envs.payoffs import utils as payoff_utils
from open_spiel.python.games.chat_games.envs.utils import text

MIN_PAYOFF = 0
MAX_PAYOFF = 20
PAYOFF_PROMPT = '''
An intelligent assistant is looking at summaries of dialogues between two people
trying to decide when to meet. It also observes the day preferences of
participants as well as the days the participant is out of office. It is trying
to decide how happy each participant is with the outcome of the dialogue and how
happy they are with the chosen meeting time.

Example 1:
Alice:
ooo_days:
monday: false
tuesday: true
wednesday: true
thursday: false
friday: false
saturday: true
sunday: false
day_prefs
monday: 2
tuesday: 4
wednesday: 12
thursday: 8
friday: 5
saturday: 0
sunday: 0

Bob:
ooo_days:
monday: false
tuesday: true
wednesday: true
thursday: false
friday: false
saturday: true
sunday: false
day_prefs
monday: 10
tuesday: 5
wednesday: 15
thursday: 3
friday: 2
saturday: 1
sunday: 1

Outcome Summary: Meeting agreed on Monday.

Final valuation for Bob: 10.
Calculation: Monday selected. Not an out of office day. Value of monday: 10.

Example 2:
Alice:
ooo_days:
monday: false
tuesday: true
wednesday: true
thursday: false
friday: false
saturday: true
sunday: false
day_prefs:
monday: 10
tuesday: 5
wednesday: 15
thursday: 3
friday: 2
saturday: 1
sunday: 1

Bob:
ooo_days:
monday: true
tuesday: true
wednesday: false
thursday: false
friday: false
saturday: true
sunday: false
day_prefs:
monday: 11
tuesday: 2
wednesday: 9
thursday: 6
friday: 5
saturday: 0
sunday: 1

Outcome Summary: Meeting agreed on Friday.

Final valuation for Alice: 2.
Calculation: Friday selected. Not an out of office day. Value of friday: 2.

Example 3:
{m}

Final valuation for {p}: 
'''

PAYOFF_OBS_TRANS_PREFIX = ['Read the following dialogue and extract out the ' +
                           'message that captures the final agreement made ' +
                           'between the two parties. If the players could ' +
                           'not agree, say no agreement was ' +
                           'reached. If both players agreed, say ' +
                           'which day the players agreed to meet.']
PAYOFF_OBS_TRANS_PREFIX = ('\n\n'.join(text.wrap(PAYOFF_OBS_TRANS_PREFIX)) +
                           '\n\n')

PAYOFF_OBS_TRANS_POSTFIX = ''

PAYOFF = payoff_utils.Payoff(PAYOFF_PROMPT,
                             MIN_PAYOFF,
                             MAX_PAYOFF,
                             PAYOFF_OBS_TRANS_PREFIX,
                             PAYOFF_OBS_TRANS_POSTFIX)
