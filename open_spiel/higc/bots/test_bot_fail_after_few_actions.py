# Copyright 2019 DeepMind Technologies Limited
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

"""Tests if a bot fails after a few actions.

A bot that picks the first action from the list for the first two rounds,
and then exists with an exception.
Used only for tests.
"""

import sys

game_name = input()
play_as = int(input())
print("ready")

while True:
  print("start")
  num_actions = 0
  while True:
    message = input()
    if message == "tournament over":
      print("tournament over")
      sys.exit(0)
    if message.startswith("match over"):
      print("match over")
      break
    public_buf, private_buf, *legal_actions = message.split(" ")
    if legal_actions:
      num_actions += 1
      print(legal_actions[-1])
    else:
      print("ponder")

    if num_actions > 2:
      raise RuntimeError
