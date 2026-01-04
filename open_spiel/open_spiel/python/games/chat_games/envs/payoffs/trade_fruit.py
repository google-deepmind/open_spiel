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

"""A few prompts for extracting the value of a fruit trade with llms.
"""

from open_spiel.python.games.chat_games.envs.payoffs import utils as payoff_utils
from open_spiel.python.games.chat_games.envs.utils import text

MIN_PAYOFF = -20
MAX_PAYOFF = 20
PAYOFF_PROMPT_a = [
    'You are an assistant designed to calculate the values of trades ' +
    'in a fruit trading game. Determine the value of the fruits the player ' +
    'is receiving in the trade. Then determine the value of the fruits the ' +
    'player is giving up through the trade. Subtract the value the player ' +
    'gives away from the value the player receives. Here is an example ' +
    'enclosed by "&".']

PAYOFF_PROMPT_b = [
    '&' * 50,
    'To calculate the trade value, we first calculate the value of ' +
    'the fruit Bob receives in the trade. Bob receives 3 kiwis worth 3 each. ' +
    'Therefore Bob receives a value of 9 in the trade.',
    'Receives: 9',
    'Now we calculate the value of the fruits Bob gives up in the trade. ' +
    'Bob gives up1 banana which is worth 5, therefore, Bob gives up a value ' +
    'of 5 in the trade.',
    'Gives: 5',
    'Subtracting the value Bob gives away from the value Bob receives, we ' +
    'find 9 - 5 = 4.',
    'Calculation: Receives - Gives = 9 - 5 = 4.',
    'Value for Bob: 4.',
    '&' * 50,
    'Now calculate the value of the trade made in the following message.',
    '{m}',
    '&' * 50,
    'Trade calculation for {p} ONLY: ']

PAYOFF_PROMPT = ('\n\n'.join(text.wrap(PAYOFF_PROMPT_a)) + '\n\n' + '&' * 50 +
                 '\n\nBob offered to give up 1 banana for 3 kiwis. Alice ' +
                 'agreed to the trade.\n\n' +
                 '\n\n'.join(text.wrap(PAYOFF_PROMPT_b)))

PAYOFF_OBS_TRANS_PREFIX = ['Read the following dialogue between two parties ' +
                           'attempting to reach a trade agreement. If the ' +
                           'dialogue ends with someone asking a question or ' +
                           'making a couterproposal, an agreement has not ' +
                           'been reached. If the dialogue ends with someone ' +
                           'saying they accept the trade, an agreement has ' +
                           'been reached. Report how much of each fruit each ' +
                           'player gave and received in the tradeby stating ' +
                           'the players names followed by a list of the ' +
                           'fruits the gave up and then a list of the fruits ' +
                           'they received in this format:',
                           'Player [Name]: Receives x Gives y',
                           'Player [Name]: Receives y Gives x',
                           'Example 1:',
                           'Dialogue:',
                           'Bob offered to give up 1 banana for 3 kiwis. ' +
                           'Alice agreed to the trade.',
                           'Player Bob: Receives 3 kiwis Gives 1 banana',
                           'Player Suzy: Receives 1 banana Gives 3 kiwis',
                           'Example 2:',
                           'Dialogue:',
                           'Alice offered to give up 1 banana for 3 kiwis. ' +
                           'George does not want to trade.',
                           'Player Bob: Receives 0 kiwi Gives 0 banana',
                           'Player Suzy: Receives 0 banana Gives 0 kiwi',
                           'Dialogue:']
PAYOFF_OBS_TRANS_PREFIX = ('\n\n'.join(text.wrap(PAYOFF_OBS_TRANS_PREFIX)) +
                           '\n\n')

PAYOFF_OBS_TRANS_POSTFIX = ''

PAYOFF = payoff_utils.Payoff(PAYOFF_PROMPT,
                             MIN_PAYOFF,
                             MAX_PAYOFF,
                             PAYOFF_OBS_TRANS_PREFIX,
                             PAYOFF_OBS_TRANS_POSTFIX)
