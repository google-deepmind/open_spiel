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

"""Base utils for constructing agent dialogue message headers.
"""

import dataclasses
import string

from typing import Callable, Tuple


@dataclasses.dataclass(frozen=True)
class BaseScenario:
  msg: str
  sender: str
  receiver: str


@dataclasses.dataclass(frozen=True)
class Header:
  plain: str
  w_opts: str
  strip_msg: Callable[[str, str], str]
  special_chars: Tuple[str, ...]
  action_keys: Tuple[str, ...] = tuple([])
  info_keys: Tuple[str, ...] = tuple([])
  context: str = ''


def plain_header_is_valid(header: Header) -> bool:
  plain = header.plain
  keys = [t[1] for t in string.Formatter().parse(plain) if t[1] is not None]
  return 'sender' in keys and 'receiver' in keys
