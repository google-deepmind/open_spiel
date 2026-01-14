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

"""A base environment for emails with tone actions.
"""

import dataclasses

from open_spiel.python.games.chat_games.envs.comm_substrates import emails
from open_spiel.python.games.chat_games.envs.utils import header


action_keys = tuple(['tone'])
action_defaults = tuple(['calm'])

w_opts = (emails.W_OPTS_PREFIX +
          'Tone: Use a {tone} tone.' +
          emails.PLAIN)

HEADER = header.Header(emails.PLAIN,
                       w_opts,
                       emails.strip_msg,
                       emails.SPECIAL_CHARS,
                       action_keys)


@dataclasses.dataclass(frozen=True)
class Scenario(header.BaseScenario):
  tone: str = 'calm'
