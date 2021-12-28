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

"""A bot that picks the first action from the list. Used only for tests."""

import base64
import sys
from open_spiel.python.observation import make_observation
import pyspiel


game_name = input()
play_as = int(input())
game = pyspiel.load_game(game_name)
public_observation = make_observation(
    game,
    pyspiel.IIGObservationType(
        perfect_recall=False,
        public_info=True,
        private_info=pyspiel.PrivateInfoType.NONE))
private_observation = make_observation(
    game,
    pyspiel.IIGObservationType(
        perfect_recall=False,
        public_info=False,
        private_info=pyspiel.PrivateInfoType.SINGLE_PLAYER))
print("ready")

while True:
  print("start")
  while True:
    message = input()
    if message == "tournament over":
      print("tournament over")
      sys.exit(0)
    if message.startswith("match over"):
      print("match over")
      break
    public_buf, private_buf, *legal_actions = message.split(" ")
    public_observation.decompress(base64.b64decode(public_buf))
    private_observation.decompress(base64.b64decode(private_buf))
    if legal_actions:
      print(legal_actions[0])
    else:
      print("ponder")
