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

"""A base environment for debate with style actions (logos) and private info.
"""

import dataclasses

from open_spiel.python.games.chat_games.envs.comm_substrates import debates
from open_spiel.python.games.chat_games.envs.utils import header


action_keys = tuple(['style'])
action_defaults = tuple(['logos'])
info_keys = tuple(['info', 'topic'])
info_defaults = tuple(['NA', 'NA'])

w_opts = (debates.W_OPTS_PREFIX +
          'Debate Topic: {topic}\n' +
          'Position: {info}\n' +
          'Style: Make a {style} style argument.' +
          debates.PLAIN)

context = '''You are an intelligent assistant in a debate with another debater.
 The debate topic is given. The goal is to provide arguments that support your
 position as well as arguments against your opponents position. An argument style
is also given. Attempt to craft your arguments according to this given style.

Here are some useful definitions of argument styles:

- logos appeals to the audiences reason, building up logical arguments.

- ethos appeals to the speakers status or authority, making the audience more
likely to trust them.

- pathos appeals to the emotions, trying to make the audience feel angry or
sympathetic, for example.

Try to construct a strong argument to support your position.
'''

HEADER = header.Header(debates.PLAIN,
                       w_opts,
                       debates.strip_msg,
                       debates.SPECIAL_CHARS,
                       action_keys,
                       info_keys,
                       context)


@dataclasses.dataclass(frozen=True)
class Scenario(header.BaseScenario):
  style: str = 'logos'
  topic: str = 'NA'
  info: str = 'NA'
