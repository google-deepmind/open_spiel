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

# Lint as python3
r"""Two BlueChip bridge bots bid with each other.

The bot_cmd FLAG should contain a command-line to launch an external bot, e.g.
`Wbridge5 Autoconnect {port}`.

"""
# pylint: enable=line-too-long

import re
import socket
import subprocess

from absl import app
from absl import flags
import numpy as np

from open_spiel.python.bots import bluechip_bridge_uncontested_bidding
import pyspiel

FLAGS = flags.FLAGS
flags.DEFINE_float("timeout_secs", 60, "Seconds to wait for bot to respond")
flags.DEFINE_integer("rng_seed", 1234, "Seed to use to generate hands")
flags.DEFINE_integer("num_deals", 10, "How many deals to play")
flags.DEFINE_string(
    "bot_cmd", None,
    "Command to launch the external bot; must include {port} which will be "
    "replaced by the port number to attach to.")


def _run_once(state, bots):
  """Plays bots with each other, returns terminal utility for each player."""
  for bot in bots:
    bot.restart_at(state)
  while not state.is_terminal():
    if state.is_chance_node():
      outcomes, probs = zip(*state.chance_outcomes())
      state.apply_action(np.random.choice(outcomes, p=probs))
    else:
      state.apply_action(bots[state.current_player()].step(state)[1])
  return state


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  game = pyspiel.load_game("bridge_uncontested_bidding", {
      "relative_scoring": True,
      "rng_seed": FLAGS.rng_seed,
  })
  bots = [
      bluechip_bridge_uncontested_bidding.BlueChipBridgeBot(
          game, 0, _WBridge5Client(FLAGS.bot_cmd)),
      bluechip_bridge_uncontested_bidding.BlueChipBridgeBot(
          game, 1, _WBridge5Client(FLAGS.bot_cmd)),
  ]
  results = []

  for i_deal in range(FLAGS.num_deals):
    state = _run_once(game.new_initial_state(), bots)
    print("Deal #{}; final state:\n{}".format(i_deal, state))
    results.append(state.returns())

  stats = np.array(results)
  mean = np.mean(stats, axis=0)
  stderr = np.std(stats, axis=0, ddof=1) / np.sqrt(FLAGS.num_deals)
  print(u"Absolute score: {:+.1f}\u00b1{:.1f}".format(mean[0], stderr[0]))
  print(u"Relative score: {:+.1f}\u00b1{:.1f}".format(mean[1], stderr[1]))


class _WBridge5Client(object):
  """Manages the connection to a WBridge5 bot."""

  def __init__(self, command):
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.sock.bind(("", 0))
    self.port = self.sock.getsockname()[1]
    self.sock.listen(1)
    self.process = None
    self.command = command.format(port=self.port)

  def start(self):
    if self.process is not None:
      self.process.kill()
    self.process = subprocess.Popen(self.command.split(" "))
    self.conn, self.addr = self.sock.accept()

  def read_line(self):
    line = ""
    while True:
      self.conn.settimeout(FLAGS.timeout_secs)
      data = self.conn.recv(1024)
      if not data:
        raise EOFError("Connection closed")
      line += data.decode("ascii")
      if line.endswith("\n"):
        return re.sub(r"\s+", " ", line).strip()

  def send_line(self, line):
    self.conn.send((line + "\r\n").encode("ascii"))


if __name__ == "__main__":
  app.run(main)
