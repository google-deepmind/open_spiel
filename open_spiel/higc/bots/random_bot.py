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

"""A simple random bot."""

import base64
import sys
import numpy as np
from open_spiel.python.observation import make_observation
import pyspiel

# Example implementation of the random bot for the HIG competition.
# The bot must strictly follow the communication protocol via stdin/stdout,
# but it can print any message to stderr for debugging.

# Read the current setup.
game_name = input()
play_as = int(input())

print(game_name, play_as, file=sys.stderr)  # For debugging purposes.

# Load the provided game.
game = pyspiel.load_game(game_name)

# Observations will be received later from the referee.
# The referee factors the observation into public (common knowledge across all
# players) and private parts.
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

# Now there is 5 secs warm-up time that could be used for loading relevant
# supplementary data. All data can be read/written from persistent /data
# directory mounted from an external storage.
print("ready")

# Loop per match. This loop will end when referee instructs the player to do so.
while True:

  # Acknowledge the match started.
  print("start")

  # This is just a placeholder for other implementations -- we do not use
  # state in random agent, as it receives list of actions it can pick from.
  state = game.new_initial_state()

  while True:  # Loop per state in match.
    message = input()  # Read message from the referee.
    print(message, file=sys.stderr)  # For debugging purposes.

    if message == "tournament over":
      # The tournament is now over: there is 60 sec shutdown time
      # available for processing tournament results by the agent,
      # for example to update supplementary data.
      print("tournament over")
      sys.exit(0)

    if message.startswith("match over"):
      # The full message has format "game over 123"
      # where 123 is the final float reward received by this bot.
      #
      # Note that this message does not necessarily mean the match
      # reached a terminal state: if opponent crashed / violated
      # rules, the match will be over as well.
      print("match over")
      break

    # Regular message: a public and private observation followed by
    # a list of legal actions (if the bot should be acting).
    public_buf, private_buf, *legal_actions = message.split(" ")
    public_observation.decompress(base64.b64decode(public_buf))
    private_observation.decompress(base64.b64decode(private_buf))

    if legal_actions:
      # There is time limit of 5 secs.
      print(np.random.choice(legal_actions))
    else:
      # Pondering phase, i.e. thinking when the bot is not acting.
      # The time limit is always at least 0.2s, but can be longer,
      # up to 5s, depending on how long the opponent thinks.
      print("ponder")  # This bot does not ponder.

  assert message.startswith("match over")
  score = int(message.split(" ")[-1])
  print("score:", score, file=sys.stderr)
