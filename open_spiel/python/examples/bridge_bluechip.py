# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as python3
r"""Two BlueChip bridge bots bid with each other.

The bot_cmd FLAG should contain a command-line to launch an external bot, e.g.
`Wbridge5 Autoconnect {port}`.

"""

import os
import re
import socket
import subprocess
import uuid

from absl import app
from absl import flags
from absl import logging
import numpy as np

from open_spiel.python.bots import bluechip_bridge
import pyspiel

FLAGS = flags.FLAGS
flags.DEFINE_float("timeout_secs", 60, "Seconds to wait for bot to respond")
flags.DEFINE_integer("num_deals", 10, "How many deals to play")
flags.DEFINE_string(
    "bot_cmd", None,
    "Command to launch the external bot; must include {port} which will be "
    "replaced by the port number to attach to.")
flags.DEFINE_string("output_path", None, "Directory to write output to.")


def _run_once(state, bots):
  """Plays bots with each other, returns final state."""
  while state.is_chance_node():
    outcomes, probs = zip(*state.chance_outcomes())
    state.apply_action(np.random.choice(outcomes, p=probs))
  while not state.is_terminal():
    logging.info(state)
    state.apply_action(bots[state.current_player()].step(state))
  return state


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  filename = os.path.join(FLAGS.output_path, str(uuid.uuid4()) + ".txt")
  file = open(filename, "w")
  game = pyspiel.load_game("bridge(use_double_dummy_result=false)")
  bots = [
      bluechip_bridge.BlueChipBridgeBot(game, player,
                                        _WBridge5Client(FLAGS.bot_cmd, player))
      for player in range(4)
  ]
  for i_deal in range(FLAGS.num_deals):
    state = _run_once(game.new_initial_state(), bots)
    logging.info("Deal #%d; final state:\n%s", i_deal, state)
    file.write(state.history_str() + "\n")
  file.close()
  logging.info("Written %d deals to file %s", FLAGS.num_deals, filename)


class _WBridge5Client(object):
  """Manages the connection to a WBridge5 bot."""

  def __init__(self, command, player_id):
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.sock.bind(("", 0))
    self.port = self.sock.getsockname()[1]
    self.sock.listen(1)
    self.process = None
    self.command = command.format(port=self.port)
    self.player_id = player_id
    self.line = ""

  def start(self):
    if self.process is not None:
      self.process.kill()
    self.process = subprocess.Popen(self.command.split(" "))
    self.conn, self.addr = self.sock.accept()

  def read_line(self):
    """Read a single line from the external program."""
    while "\n" not in self.line:
      self.conn.settimeout(FLAGS.timeout_secs)
      data = self.conn.recv(1024)
      if not data:
        raise EOFError("Connection {} closed".format(self.player_id))
      self.line += data.decode("ascii")
    line, self.line = self.line.split("\n", maxsplit=1)
    line = re.sub(r"\s+", " ", line).strip()
    logging.info("Recv %d %s", self.player_id, line)
    return line

  def send_line(self, line):
    logging.info("Send %d %s", self.player_id, line)
    self.conn.send((line + "\r\n").encode("ascii"))


if __name__ == "__main__":
  app.run(main)
