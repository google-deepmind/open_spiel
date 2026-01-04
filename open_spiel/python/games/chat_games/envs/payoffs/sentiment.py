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

"""A few prompts for performing sentiment analysis with llms.
"""

from open_spiel.python.games.chat_games.envs.payoffs import utils as payoff_utils
from open_spiel.python.games.chat_games.envs.utils import text

# maybe sentiment score should look at summarized dialogue and predict how
# happy each participant is with the state of things
MIN_PAYOFF = -10
MAX_PAYOFF = 10
PAYOFF_PROMPT = [
    'You are an assistant designed to analyze sentiment from speech data. ' +
    'Users will paste in a string of text and you will respond with an ' +
    'assessment of the response of the speaker. Rate on a scale of 1-10 ' +
    '(10 being highest). Explain why this rating was given. Here are a few ' +
    'examples.',
    '#' * 20,
    'message: Hi Bob, I am sorry you think the rent increase is unfair, but ' +
    'the economy has forced my hand. If I cannot afford it, you will have to ' +
    'leave.',
    f'sentiment_score for Bob: 3/{MAX_PAYOFF}.',
    'message: Hi Bob, you are right. The rent is expensive and it is not fair' +
    '_to raise it that high. I will compromise and only raise it by 3 percent.',
    f'sentiment_score for Bob: 7/{MAX_PAYOFF}.',
    'message: Hi Alice, I think that rent increase is really unfair. I think ' +
    'you are trying to take advantage of me and I do not appreciate it.',
    f'sentiment_score for Alice: 3/{MAX_PAYOFF}.',
    'message: Hi Alice, the rent is expensive but it is worth it and I am ' +
    'willing to pay you a higher rent.',
    f'sentiment_score for Alice: 8/{MAX_PAYOFF}.',
    '#' * 20,
    'Now provide a rating for the following message.',
    'message: {m}',
    'sentiment score for {p}: ']
PAYOFF_PROMPT = '\n\n'.join(text.wrap(PAYOFF_PROMPT))

PAYOFF_OBS_TRANS_PREFIX = ''
PAYOFF_OBS_TRANS_POSTFIX = ''

PAYOFF = payoff_utils.Payoff(PAYOFF_PROMPT,
                             MIN_PAYOFF,
                             MAX_PAYOFF,
                             PAYOFF_OBS_TRANS_PREFIX,
                             PAYOFF_OBS_TRANS_POSTFIX)
