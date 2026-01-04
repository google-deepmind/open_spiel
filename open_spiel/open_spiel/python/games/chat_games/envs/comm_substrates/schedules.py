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

"""A communication format (substrate) for setting schedules.
"""

from open_spiel.python.games.chat_games.envs.utils import text


CHAR_OPT = '%'
CHAR_MSG = '#'
BLOCK_LEN = 28

SPECIAL_CHARS = (CHAR_OPT, CHAR_MSG)
BLOCK_OPT = CHAR_OPT * BLOCK_LEN
BLOCK_MSG = CHAR_MSG * BLOCK_LEN

PLAIN = ('\n\n' + BLOCK_MSG + '\n' +
         'Schedule Proposal Message:\n' +
         'from: {sender}\n' +
         'to: {receiver}\n' +
         BLOCK_MSG + '\n\n')

W_OPTS_PREFIX = '\n\n' + BLOCK_OPT + '\n\n'


def strip_msg(msg: str, terminal_str: str = '') -> str:
  return text.strip_msg(msg, BLOCK_MSG, BLOCK_OPT, terminal_str)
