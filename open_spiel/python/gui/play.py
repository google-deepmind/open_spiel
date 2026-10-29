# Copyright 2026 DeepMind Technologies Limited
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


"""Play a game against a bot in one step.

  python -m open_spiel.python.gui.play --game gin_rummy \\
      --bot MODULE:FACTORY --checkpoint /path/to/run@30000

starts the reference bot server (`remote_bot_server`) serving `--bot` on a
free port of this machine, with the checkpoint in the environment
variables CHECKPOINT_ENVS (both names a bot's factory may read), or in
`--checkpoint-env` alone; waits until its `/health` answers; starts the GUI
server; opens a session of the game whose seat `--bot-seat` is a Remote bot
seat at that URL (state `--state`, its `checkpoint` setting the checkpoint,
autoplay on) and every other seat human; and prints the address to play it
at. Ctrl-C, SIGTERM or SIGHUP (closing the terminal) stops both servers, as
does any error, at any point of the launch or after it.

The bot runs in its own process: this module imports neither the bot nor
anything it needs. `--bot random` plays a uniformly random legal action, and
`--bot open_spiel.python.gui.stand_in_checkpoint_bot:make_bot` a stand-in
for a trained checkpoint, so the launch can be tried without one.
"""

import argparse
from collections.abc import Callable, Mapping, Sequence
import dataclasses
import http.client
import json
import os
import re
import signal
import sys
import threading
import types
from typing import Any, TextIO
import urllib.parse
import webbrowser

import pyspiel

from open_spiel.python.gui import child_servers
from open_spiel.python.gui import remote_bot
from open_spiel.python.gui import remote_bot_server
from open_spiel.python.gui import static

# The variable the bot server's factory reads its checkpoint from, unless
# `--checkpoint-env` names another.
CHECKPOINT_ENV = remote_bot_server.CHECKPOINT_ENV
# The variables the bot server is given the checkpoint in, unless
# `--checkpoint-env` names one: the generic CHECKPOINT_ENV, and the one
# `open_spiel.python.scale.serve.remote_bot` reads.
CHECKPOINT_ENVS = (CHECKPOINT_ENV, "OPEN_SPIEL_SCALE_REMOTE_BOT_CHECKPOINT")
# Seconds; the GUI server's move budget, and so the most the bot seat waits
# for a move. A checkpoint named only in the seat (its Checkpoint field)
# loads and warms up at its first move, which takes about 16 s for a trained
# gin rummy bot: over the GUI server's own default of 10 s.
BOT_MOVE_BUDGET_SECONDS = 60.0

# What each server prints once it listens; the first group is its URL.
_BOT_READY = re.compile(r"^Serving the .* bot at (http://\S+)")
_GUI_READY = re.compile(r"^Serving OpenSpiel games at (http://\S+)")

# Seconds. The GUI server registers every game first.
GUI_START_SECONDS = 120.0
# Seconds each HTTP request of the launcher may take.
_REQUEST_SECONDS = 30.0
# The signals that stop the launcher and both servers. SIGHUP is closing the
# terminal: the servers run in their own sessions, so it does not reach them.
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


LaunchError = child_servers.LaunchError


@dataclasses.dataclass(frozen=True)
class Options:
  """What to launch.

  Attributes:
    game: The game, as a game string such as `gin_rummy` or
      `gin_rummy(oklahoma=true)`.
    bot: The bot server's `--bot`: random, mcts or MODULE:FACTORY.
    checkpoint: The checkpoint the bot plays, or empty for none.
    checkpoint_envs: The environment variables the bot server is given the
      checkpoint in.
    bot_seat: The seat the bot plays.
    state: What the bot is sent: seat_view or full_state.
    host: The address the GUI server listens on.
    port: The GUI server's port; 0 picks a free one.
    client_dir: The built browser client.
    move_budget_seconds: The GUI server's move budget.
    bot_start_seconds: How long the bot server may take to start (a bot that
      loads and warms up a checkpoint takes a while).
    gui_flags: Further flags for the GUI server.
    open_browser: Whether to open the address in a browser too.
    python: The Python interpreter both servers run on.
  """

  game: str
  bot: str
  checkpoint: str = ""
  checkpoint_envs: tuple[str, ...] = CHECKPOINT_ENVS
  bot_seat: int = 1
  state: str = remote_bot.FULL_STATE
  host: str = "127.0.0.1"
  port: int = 8080
  client_dir: str = str(static.DEFAULT_ROOT)
  move_budget_seconds: float = BOT_MOVE_BUDGET_SECONDS
  bot_start_seconds: float = 900.0
  gui_flags: tuple[str, ...] = ()
  open_browser: bool = False
  python: str = sys.executable


def parse_game(text: str) -> tuple[str, dict[str, Any]]:
  """Returns the short name and parameters of the game string `text`.

  Raises:
    LaunchError: If `text` is not a game string of a registered game.
  """
  try:
    parameters = dict(pyspiel.game_parameters_from_string(text))
  except pyspiel.SpielError as e:
    raise LaunchError(f"--game {text!r} is not a game string: {e}") from e
  name = str(parameters.pop("name", ""))
  if name not in pyspiel.registered_names():
    raise LaunchError(f"--game {text!r} names no registered game.")
  return name, parameters


def _request(
    method: str, url: str, body: Mapping[str, Any] | None = None
) -> Any:
  """Sends a JSON request to a server on this machine; returns its JSON reply.

  Raises:
    LaunchError: If the server cannot be reached, or the reply is not 2xx
      JSON.
  """
  parts = urllib.parse.urlsplit(url)
  target = parts.path + (f"?{parts.query}" if parts.query else "")
  headers = {"Accept": "application/json"}
  data = None
  if body is not None:
    data = json.dumps(body).encode()
    headers["Content-Type"] = "application/json"
  connection = http.client.HTTPConnection(
      parts.hostname or "", parts.port, timeout=_REQUEST_SECONDS
  )
  try:
    connection.request(method, target, body=data, headers=headers)
    response = connection.getresponse()
    status, raw = response.status, response.read()
  except (OSError, http.client.HTTPException) as e:
    raise LaunchError(f"{method} {url} failed: {e}") from e
  finally:
    connection.close()
  if not 200 <= status < 300:
    raise LaunchError(
        f"{method} {url} answered HTTP {status}: "
        + raw[:500].decode(errors="replace")
    )
  try:
    return json.loads(raw)
  except (json.JSONDecodeError, UnicodeDecodeError) as e:
    raise LaunchError(f"{method} {url} answered with no JSON.") from e


def bot_unready(bot_url: str) -> str | None:
  """Returns why the bot server at `bot_url` is not ready, or None if it is.

  Ready is its `/health` answering `{"ok": true}`, as the reference server
  does once its bot is loaded.
  """
  try:
    health = _request("GET", bot_url + remote_bot.HEALTH_PATH)
  except LaunchError as e:
    return str(e)
  if not isinstance(health, dict) or health.get("ok") is not True:
    return f"The bot server's /health answered {health!r}."
  return None


def seat_config(options: Options, bot_url: str) -> dict[str, Any]:
  """Returns the Remote bot seat's settings.

  The seat waits for each move as long as the move budget allows
  (`timeout_seconds` 0), so a checkpoint the bot loads at its first move
  (BOT_MOVE_BUDGET_SECONDS) is not cut short.
  """
  return {
      "url": bot_url,
      "state": options.state,
      "checkpoint": options.checkpoint,
      "timeout_seconds": 0.0,
  }


def session_body(
    options: Options, bot_url: str, num_players: int
) -> dict[str, Any]:
  """Returns the body that opens the session, its bot seat a Remote bot."""
  name, parameters = parse_game(options.game)
  seats = [
      {"controller": "human", "config": {}, "autoplay": False}
      for _ in range(num_players)
  ]
  seats[options.bot_seat] = {
      "controller": remote_bot.NAME,
      "config": seat_config(options, bot_url),
      "autoplay": True,
  }
  return {
      "game": name,
      "parameters": parameters,
      "seats": seats,
      "chance_policy": "auto",
      "seed": None,
  }


@dataclasses.dataclass
class Running:
  """A launched bot server, GUI server and session.

  Attributes:
    bot_url: The bot server's URL.
    gui_url: The GUI server's URL, without a trailing slash.
    session: The session's id.
    play_url: The address to play the session at, as the first human seat.
  """

  bot_url: str
  gui_url: str
  session: str
  play_url: str
  _bot: child_servers.ChildServer
  _gui: child_servers.ChildServer | None

  def exited(self) -> str | None:
    """Returns the label of a server that has exited, or None."""
    for child in (self._bot, self._gui):
      if child is not None and child.process.poll() is not None:
        return child.label
    return None

  def stop(self) -> None:
    """Stops the GUI server, then the bot server."""
    child_servers.stop_all((self._gui, self._bot))


def launch(
    options: Options,
    out: TextIO = sys.stderr,
    started: Callable[[Running], object] | None = None,
) -> Running:
  """Starts the bot server and the GUI server, and opens the session.

  Every server started is stopped again if a later step fails or is
  interrupted.

  Args:
    options: What to launch.
    out: Where both servers' output is copied.
    started: Called with the `Running` before it is returned. A caller that
      stops the servers on a signal hands its handler the `Running` here: if
      `started` raises, or a signal interrupts it, the servers are stopped,
      so no signal finds them running and in no one's hands.

  Raises:
    LaunchError: If the options are invalid, a server does not start, or
      the session cannot be opened.
  """
  name, _ = parse_game(options.game)
  game = pyspiel.load_game(options.game)
  num_players = game.num_players()
  if not 0 <= options.bot_seat < num_players:
    raise LaunchError(
        f"--bot-seat {options.bot_seat} is not a seat of {name}, which has "
        f"{num_players}."
    )
  if num_players < 2:
    raise LaunchError(f"{name} has one seat, so no one plays against the bot.")
  if options.state not in remote_bot.STATE_MODES:
    raise LaunchError(
        f"--state {options.state!r} is not seat_view or full_state."
    )
  env = dict(os.environ)
  if options.checkpoint:
    for variable in options.checkpoint_envs:
      env[variable] = options.checkpoint
  bot: child_servers.ChildServer | None = None
  gui: child_servers.ChildServer | None = None
  try:
    bot = child_servers.ChildServer(
        "bot",
        [
            options.python,
            "-m",
            "open_spiel.python.gui.remote_bot_server",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--bot",
            options.bot,
        ],
        env,
        _BOT_READY,
        out,
    )
    bot_url = bot.wait_ready(options.bot_start_seconds)
    reason = bot_unready(bot_url)
    if reason is not None:
      raise LaunchError(reason)
    gui = child_servers.ChildServer(
        "gui",
        [
            options.python,
            "-m",
            "open_spiel.python.gui",
            "--host",
            options.host,
            "--port",
            str(options.port),
            "--client-dir",
            options.client_dir,
            "--move-budget-seconds",
            f"{options.move_budget_seconds:g}",
            *options.gui_flags,
        ],
        os.environ,
        _GUI_READY,
        out,
    )
    gui_url = gui.wait_ready(GUI_START_SECONDS)
    # The API is asked on loopback, which the GUI server always accepts.
    port = urllib.parse.urlsplit(gui_url).port
    local_api = f"http://127.0.0.1:{port}/api"
    frame = _request(
        "POST",
        f"{local_api}/sessions?viewer=oracle",
        session_body(options, bot_url, num_players),
    )
    session = str(frame["session"]["id"])
    seat = frame["seats"][options.bot_seat]
    if seat.get("controller") != remote_bot.NAME:
      raise LaunchError(f"The bot seat is {seat!r}, not a Remote bot.")
    if seat.get("failure"):
      raise LaunchError(f"The bot seat failed: {seat['failure']}.")
    human = next(i for i in range(num_players) if i != options.bot_seat)
    query = urllib.parse.urlencode({"session": session, "viewer": human})
    running = Running(
        bot_url=bot_url,
        gui_url=gui_url,
        session=session,
        play_url=f"{gui_url}/game/{urllib.parse.quote(name)}?{query}",
        _bot=bot,
        _gui=gui,
    )
    if started is not None:
      started(running)
    return running
  except BaseException:
    child_servers.stop_all((gui, bot))
    raise


def parse_args(argv: Sequence[str] | None = None) -> Options:
  """Parses the command line; flags after `--` go to the GUI server."""
  argv = list(sys.argv[1:] if argv is None else argv)
  gui_flags: list[str] = []
  if "--" in argv:
    split = argv.index("--")
    argv, gui_flags = argv[:split], argv[split + 1 :]
  parser = argparse.ArgumentParser(
      prog="python -m open_spiel.python.gui.play",
      description=(
          "Play a game against a bot in one step: start a bot server and "
          "the GUI server, and open a session with a Remote bot seat. Flags "
          "after -- go to the GUI server."
      ),
  )
  parser.add_argument(
      "--game", required=True, help="The game string, such as gin_rummy."
  )
  parser.add_argument(
      "--bot",
      required=True,
      help=(
          "The bot server's --bot: random, mcts, or MODULE:FACTORY, a "
          "function returning a bot (REMOTE_BOTS.md)."
      ),
  )
  parser.add_argument(
      "--checkpoint",
      default="",
      help=(
          "The checkpoint the bot plays, <location>@<step>: a directory, "
          "or any path or URL the bot reads (a cloud bucket, say); given to "
          "the bot server in its environment and set as the seat's "
          "checkpoint."
      ),
  )
  parser.add_argument(
      "--checkpoint-env",
      default=None,
      metavar="VAR",
      help=(
          "The one variable the bot reads its checkpoint from. Default: "
          f"set all of {', '.join(CHECKPOINT_ENVS)}."
      ),
  )
  parser.add_argument(
      "--bot-seat", type=int, default=1, help="The bot's seat. Default: 1."
  )
  parser.add_argument(
      "--state",
      default=remote_bot.FULL_STATE,
      choices=remote_bot.STATE_MODES,
      help="What the bot is sent. Default: %(default)s.",
  )
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument(
      "--port", type=int, default=8080, help="The GUI's port; 0 picks one."
  )
  parser.add_argument("--client-dir", default=str(static.DEFAULT_ROOT))
  parser.add_argument(
      "--move-budget-seconds",
      type=float,
      default=BOT_MOVE_BUDGET_SECONDS,
      help=(
          "How long the bot may take per move (its first move with a new "
          "checkpoint loads and warms it up, about 16 s for a trained gin "
          "rummy bot). Default: %(default)g."
      ),
  )
  parser.add_argument(
      "--bot-start-seconds",
      type=float,
      default=900.0,
      help=(
          "How long the bot server may take to load its bot and listen. "
          "Default: %(default)g."
      ),
  )
  parser.add_argument(
      "--open",
      action="store_true",
      help="Open the address in a browser too.",
  )
  args = parser.parse_args(argv)
  options = Options(
      game=args.game,
      bot=args.bot,
      checkpoint=args.checkpoint.strip(),
      checkpoint_envs=(
          CHECKPOINT_ENVS
          if args.checkpoint_env is None
          else (args.checkpoint_env,)
      ),
      bot_seat=args.bot_seat,
      state=args.state,
      host=args.host,
      port=args.port,
      client_dir=args.client_dir,
      move_budget_seconds=args.move_budget_seconds,
      bot_start_seconds=args.bot_start_seconds,
      gui_flags=tuple(gui_flags),
      open_browser=args.open,
  )
  return options


def serve(
    running: Running,
    stopping: threading.Event,
    opener: Callable[[str], object] | None = None,
) -> int:
  """Prints the address, then waits for a stop or a server exiting.

  Both servers are stopped however it ends: a stop, a server exiting, or an
  error (printing to a closed stdout, say, or the opener failing).

  Returns:
    0 when stopped, 1 when a server exited by itself.
  """
  try:
    print(f"Play {running.play_url}", flush=True)
    if opener is not None:
      opener(running.play_url)
    while not stopping.wait(0.5):
      exited = running.exited()
      if exited is not None:
        print(f"The {exited} server exited; stopping.", file=sys.stderr)
        return 1
    return 0
  finally:
    running.stop()


class _Stopper:
  """The launcher's handler of `STOP_SIGNALS`, installed before the launch.

  The first signal sets `stopping`. Until the launch has handed over its
  `Running` (`started`), it also raises KeyboardInterrupt, which unwinds the
  launch, and `launch` stops what it started; after, `serve` sees `stopping`
  and stops both servers. The hand-over is one assignment, so every signal
  lands on one side of it. A later signal does nothing, so it cannot cut a
  stop short.
  """

  def __init__(self) -> None:
    self.stopping = threading.Event()
    self._serving = False

  def started(self, unused_running: Running) -> None:
    """Called by `launch` with the servers; from here a signal ends `serve`."""
    self._serving = True

  def __call__(self, signum: int, unused_frame: types.FrameType | None) -> None:
    if self.stopping.is_set():
      return
    self.stopping.set()
    name = signal.Signals(signum).name
    if not self._serving:
      raise KeyboardInterrupt(name)
    print(f"Received {name}; stopping.", file=sys.stderr, flush=True)


def main(argv: Sequence[str] | None = None) -> None:
  """Launches, serves until interrupted, then stops both servers."""
  options = parse_args(argv)
  opener = webbrowser.open if options.open_browser else None
  stopper = _Stopper()
  for signum in STOP_SIGNALS:
    signal.signal(signum, stopper)
  try:
    running = launch(options, started=stopper.started)
  except LaunchError as e:
    raise SystemExit(f"error: {e}") from e
  except KeyboardInterrupt:
    raise SystemExit("Interrupted; both servers stopped.") from None
  raise SystemExit(serve(running, stopper.stopping, opener))


if __name__ == "__main__":
  main()
