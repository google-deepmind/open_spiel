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

"""A few prompts for extracting the winner of a debate.
"""

from open_spiel.python.games.chat_games.envs.observations import summary_debate
from open_spiel.python.games.chat_games.envs.payoffs import utils as payoff_utils
from open_spiel.python.games.chat_games.envs.utils import text

MIN_PAYOFF = 0
MAX_PAYOFF = 1
PAYOFF_PROMPT_a = [
    'You are an assistant designed to determine the winner of a debate. ' +
    'Read the arguments by each side in the debate. Then determine if ' +
    'the player won the debate. Here is an example enclosed by "&".']

PAYOFF_PROMPT_b = [
    # '&' * 50,
    'Bob argues that Tom Brady is the GOAT and coach Bill Bellichick is a ' +
    'genius.',
    'Alice argues that Tom Brady is no longer on the Patriots and the ' +
    'Patriots were 8-9 last season. They lost more games than they won. They ' +
    'are no longer the powerhouse they used to be.',
    'Bob makes an argument based on stale information. Alice acutely points ' +
    'this out and provides more current evidence that supports the negative ' +
    'of Bobs argument. Therefore, Bob loses the debate.',
    'Value for Bob: 0.',
    'Value for Alice: 1.',
    '&' * 50,
    'Now determine the winner of the following debate.',
    '{m}',
    '%' * 50,
    'Payoff for {p} ONLY: ']

PAYOFF_PROMPT = ('\n\n'.join(text.wrap(PAYOFF_PROMPT_a)) + '\n\n' + '&' * 50 +
                 '\n\nDebate Topic: The New England Patriots are the best ' +
                 'NFL team in 2023.\n\n' +
                 '\n\n'.join(text.wrap(PAYOFF_PROMPT_b)))

PAYOFF_OBS_TRANS_PREFIX = summary_debate.PREFIX

PAYOFF_OBS_TRANS_POSTFIX = summary_debate.POSTFIX

PAYOFF = payoff_utils.Payoff(PAYOFF_PROMPT,
                             MIN_PAYOFF,
                             MAX_PAYOFF,
                             PAYOFF_OBS_TRANS_PREFIX,
                             PAYOFF_OBS_TRANS_POSTFIX)
