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

"""A bot that uses an external agent over the Go Text Protocol."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import subprocess
import time

import pyspiel


class CommandError(Exception):
  """An error message returned from the GTP bot."""


class GTPBot(pyspiel.Bot):
  """A bot that uses an external agent over GTP to get the action to play.

  The Go Text Protocol, GTP, is a text based protocol for communication with
  computer Go programs (https://www.lysator.liu.se/~gunnar/gtp/). It has also
  been adopted by agents in other games including Hex and Havannah. If you need
  to configure your agent in some specific way (eg time/resource limits), you
  can use `gtp_cmd` to send raw commands to it.
  """

  def __init__(self, game, exec_path, player_colors=("b", "w"),
               suppress_stderr=True):
    """Create a Bot that runs an external binary using GTP.

    Args:
      game: A Game object to pull the configuration (boardsize)
      exec_path: A string or list to be passed to popen to launch the binary.
      player_colors: A list or tuple of names to be passed to gtp's `play`
          command to tell it which player made the move.
      suppress_stderr: Whether to suppress stderr from the binary.
    """
    pyspiel.Bot.__init__(self)
    self._process = subprocess.Popen(
        exec_path, bufsize=0, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=(subprocess.DEVNULL if suppress_stderr else None))

    self._game = game
    params = game.get_parameters()
    if "board_size" in params:
      self.gtp_cmd("boardsize", str(params["board_size"]))

    if len(player_colors) != game.num_players():
      raise ValueError(
          ("player_colors has the wrong number of players for this game. Got "
           "{}, expected {}").format(len(player_colors), game.num_players()))
    self._player_colors = player_colors

  def __del__(self):
    self.close()

  def close(self):
    """Tell the game to quit and wait for it to do so, killing eventually."""
    # We support closing an already closed instance, as __del__ will be called
    # when the object is deleted, thus closing a potentially already closed obj.
    # The hasattr is in case Popen raises and self._process doesn't exist.
    if hasattr(self, "_process") and self._process is not None:
      if self.running:
        try:
          self.gtp_cmd("quit")
        except (CommandError, IOError):
          pass
        self._process.stdin.close()
        self._process.stdout.close()
        _shutdown_proc(self._process, 3)
      self._process = None

  def gtp_cmd(self, *args):
    """Send commands directly to the game, and get the response as a string."""
    cmd = " ".join([str(a) for a in args]).encode()
    self._process.stdin.write(cmd + b"\n")
    response = ""
    while True:
      line = self._process.stdout.readline().decode()
      if not line:
        raise IOError("Engine closed the connection.")
      if line == "\n":
        if response:
          break  # A blank line signifies end of response.
        else:
          continue  # Ignore leading newlines, possibly left from prev response.
      response += line
    if response.startswith("="):
      return response[1:].strip()
    else:
      raise CommandError(response[1:].strip())

  def inform_action(self, state, player_id, action):
    """Let the bot know of the other agent's actions."""
    self.gtp_cmd("play", self._player_colors[player_id],
                 state.action_to_string(action))

  def step(self, state):
    """Returns the selected action and steps the internal state forward."""
    return state.string_to_action(self.gtp_cmd(
        "genmove", self._player_colors[state.current_player()]))

  def restart(self):
    self.gtp_cmd("clear_board")

  def restart_at(self, state):
    self.restart()
    new_state = self._game.new_initial_state()
    for action in state.history():
      self.inform_action(new_state, new_state.current_player(),
                         new_state.action_to_string(action))
      new_state.apply_action(action)

  @property
  def name(self):
    """The name reported by the agent."""
    return self.gtp_cmd("name")

  @property
  def version(self):
    """The version reported by the agent."""
    return self.gtp_cmd("version")

  @property
  def running(self):
    """Whether the agent binary is still running."""
    # poll returns None if it's running, otherwise the exit code.
    return self._process and (self._process.poll() is None)

  @property
  def pid(self):
    """The pid of the agent binary."""
    return self._process.pid if self.running else None


def _shutdown_proc(p, timeout):
  """Waits for a proc to shut down; terminates or kills it after `timeout`."""
  freq = 10  # how often to check per second
  for _ in range(1 + timeout * freq):
    p.terminate()
    ret = p.poll()
    if ret is not None:
      return ret
    time.sleep(1 / freq)
  p.kill()
  return p.wait()
