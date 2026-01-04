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

"""Utils for parsing and constructing message strings.
"""

import textwrap

from typing import List, Tuple


def strip_msg(text: str,
              block_msg: str,
              block_opt: str,
              terminal_str: str = '') -> str:
  """Strip email message (with header) from text block, i.e., [ (A) - (B) ).

  Assumes messages adhere to the following format:
  BLOCK_OPT
  <-- action & info -->
  BLOCK_MSG (A)
  <-- e.g., sender/receiver -->
  BLOCK_MSG
  <-- e.g., message -->
  BLOCK_OPT (B)

  Args:
    text: str
    block_msg: str, string of characters delineating the message
    block_opt: str, string of characters demarking the start of
      the options (actions and info)
    terminal_str: str (optional), indicates the end of a message if block_opt
      is not found. this will be included in the stripped output.
  Returns:
    stripped_text: str
  """
  ctr = 0
  right_ptr = 0
  left_ptr = text.find(block_msg)
  if left_ptr == -1:
    return ''
  while ctr < 2:
    block_idx = text[right_ptr:].find(block_msg)
    if block_idx == -1:
      return ''
    right_ptr += block_idx + len(block_msg)
    ctr += 1
  block_idx = text[right_ptr:].find(block_opt)
  if block_idx != -1:  # if find block_opt return message ending at (B)
    right_ptr += block_idx
  else:
    if terminal_str:  # if no block_opt, return message ending at terminal_str
      block_idx = text[right_ptr:].find(terminal_str)
      if block_idx != -1:
        right_ptr += block_idx + len(terminal_str)
      else:  # if no terminal_str, return message to end of text string
        right_ptr = len(text)
  return text[left_ptr:right_ptr]


def first_special_char(text: str,
                       max_idx: int,
                       special_chars: Tuple[str, ...]) -> int:
  first_special_chars = [max_idx]
  for char in special_chars:
    idx = text.find(char)
    if idx < 0:
      first_special_chars.append(max_idx)
    else:
      first_special_chars.append(idx)
  return min(first_special_chars)


def retrieve_special_char_block(text: str,
                                special_chars: Tuple[str, ...] = ('*',),
                                useless_chars: Tuple[str, ...] = (' ', '\n')):
  for char in special_chars:
    text = text.strip(char)
  idx_end = first_special_char(text, len(text), special_chars)
  text = text[:idx_end]
  for char in useless_chars:
    text = text.strip(char)
  return text


def retrieve_alpha_block(text: str) -> str:
  """Return the first instance of a contiguous alpha(not numeric) substring."""
  first_alpha_char = next(filter(str.isalpha, text), -1)
  if first_alpha_char == -1:
    return ''
  start = text.find(first_alpha_char)
  sliced = text[start:]
  last_alpha_char = next(filter(lambda s: not str.isalpha(s), sliced), -1)
  if last_alpha_char == -1:
    return sliced
  finish = sliced.find(last_alpha_char)
  return text[start:start + finish]


def retrieve_numeric_block(text: str) -> str:
  """Return the first instance of a contiguous numeric(not alpha) substring."""
  first_numeric_char = next(filter(str.isnumeric, text), -1)
  if first_numeric_char == -1:
    return ''
  start = text.find(first_numeric_char)
  sliced = text[start:]
  last_numeric_char = next(filter(lambda s: not str.isnumeric(s), sliced), -1)
  if start > 0 and text[start - 1] == '-':
    start -= 1
    sliced = text[start:]
  if last_numeric_char == -1:
    return sliced
  finish = sliced.find(last_numeric_char)
  return text[start:start + finish]


def wrap(message: List[str]) -> List[str]:
  """Given a list of strings, returns a list of them `wrapped` (paragraphs).

  Args:
    message: list of strings
  Returns:
    wrapped: list of strings with each string `wrapped` so that each line only
      contains (default) 70 characters
  """
  wrapped = []
  for sub_msg in message:
    sub_msg_wrapped = textwrap.wrap(sub_msg)
    if len(sub_msg_wrapped) > 1:
      sub_msg_wrapped = ['\n'.join(sub_msg_wrapped)]
    wrapped.extend(sub_msg_wrapped)
  return wrapped
