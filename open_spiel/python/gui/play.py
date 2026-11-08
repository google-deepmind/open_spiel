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
`--checkpoint-env` alone; waits until its `/health` answers, and stops,
naming the `--state` to rerun with, if its `/info` does not take `--state`;
starts the GUI server; opens a session of the game whose bot seats are
Remote bot seats at that URL (each one's `checkpoint` setting the
checkpoint, autoplay on) and every other seat human; and prints the address
to play it at, as the human seat. The seat's state is `--state` if given;
otherwise the one state mode the bot server's `/info` names, if it names
one, and seat_view if it takes both. When the bot is sent the full state of
a game that hides information, the launcher prints one line saying the bot
sees the whole state.

The bot seats are every seat but `--human-seat` (seat 0 by default), so
`--bot` fills all three other seats of a four-player game such as hearts or
euchre; `--bot-seat`, which may be repeated, names them instead, every other
seat human. `--seat-bot SEAT=MODULE:FACTORY` and
`--seat-checkpoint SEAT=CHECKPOINT` give one seat a bot or checkpoint of its
own; each different bot and checkpoint runs in a bot server of its own, and
seats with the same bot and checkpoint share one, which is then asked for
each of their moves (a bot that keeps state between requests keys it on the
request's `seat`, or is given `--server-per-seat`).
Ctrl-C, SIGTERM or SIGHUP (closing the terminal) stops both servers, as does
any error, at any point of the launch or after it.

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
from open_spiel.python.gui import information
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
class SeatBot:
  """The bot a seat plays.

  Attributes:
    bot: The bot server's `--bot`: random, mcts or MODULE:FACTORY.
    checkpoint: The checkpoint the bot plays, or empty for none.
  """

  bot: str
  checkpoint: str = ""


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
    bot_seat: One seat the bot plays, as `bot_seats=(bot_seat,)` does; kept
      for callers that seat one bot.
    bot_seats: The seats the bots play, every other seat human. Empty (the
      default, with no `bot_seat`) is every seat but the human seat.
    human_seat: The seat the printed address opens as; None (the default) is
      seat 0, or the first seat that is not a bot seat.
    seat_bots: The seats that play a bot or checkpoint other than `bot` and
      `checkpoint`, each a bot seat.
    server_per_seat: Whether every bot seat gets a bot server of its own,
      for a bot that keeps state between requests and cannot key it on the
      request's seat; otherwise the seats with the same bot and checkpoint
      share one.
    state: What the bot is sent: seat_view, what its seat may see; or
      full_state, the whole state, for a bot that needs it. None (the
      default) is the bot server's one state mode if it declares one, and
      seat_view if it takes both.
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
  bot_seat: int | None = None
  bot_seats: tuple[int, ...] = ()
  human_seat: int | None = None
  seat_bots: Mapping[int, SeatBot] = dataclasses.field(default_factory=dict)
  server_per_seat: bool = False
  state: str | None = None
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


@dataclasses.dataclass(frozen=True)
class Seating:
  """Who plays each seat.

  Attributes:
    bots: Every bot seat, in seat order, and the bot it plays.
    human: The seat the printed address opens as, a human seat.
    server_per_seat: Whether every bot seat has a bot server of its own,
      rather than one shared by the seats with the same bot.
  """

  bots: Mapping[int, SeatBot]
  human: int
  server_per_seat: bool = False

  def server_of(self, seat: int) -> tuple[int | None, SeatBot]:
    """Returns the key of the bot server the bot seat `seat` plays on.

    That is its bot, with the seat too when every seat has its own server.
    """
    return (seat if self.server_per_seat else None, self.bots[seat])

  def servers(self) -> list[tuple[int | None, SeatBot]]:
    """Returns the keys of the bot servers to start, in seat order."""
    return list(dict.fromkeys(self.server_of(seat) for seat in self.bots))


def seating(options: Options, num_players: int) -> Seating:
  """Returns who plays each seat of the game `options.game`.

  Raises:
    LaunchError: If a seat named is not a seat of the game, the human seat
      is a bot seat, or every seat is a bot seat.
  """
  name, _ = parse_game(options.game)
  named = list(options.bot_seats)
  if options.bot_seat is not None:
    named.append(options.bot_seat)
  checks = [("--bot-seat", seat) for seat in named]
  checks += [("--seat-bot", seat) for seat in options.seat_bots]
  if options.human_seat is not None:
    checks.append(("--human-seat", options.human_seat))
  for flag, seat in checks:
    if not 0 <= seat < num_players:
      raise LaunchError(
          f"{flag} {seat} is not a seat of {name}, which has {num_players}."
      )
  if named:
    seats = set(named)
  else:
    human = 0 if options.human_seat is None else options.human_seat
    if options.human_seat is None and human in options.seat_bots:
      raise LaunchError(
          f"--seat-bot {human} is the human seat ({human}); pass "
          "--human-seat to sit at another."
      )
    seats = set(range(num_players)) - {human}
  seats |= set(options.seat_bots)
  if options.human_seat in seats:
    raise LaunchError(f"--human-seat {options.human_seat} is a bot seat.")
  humans = [seat for seat in range(num_players) if seat not in seats]
  if not humans:
    raise LaunchError(
        f"Every seat of {name} is a bot seat, so no one plays against the bots."
    )
  default = SeatBot(options.bot, options.checkpoint)
  return Seating(
      bots={
          seat: options.seat_bots.get(seat, default) for seat in sorted(seats)
      },
      human=humans[0] if options.human_seat is None else options.human_seat,
      server_per_seat=options.server_per_seat,
  )


def full_state_note(
    options: Options, game: pyspiel.Game, chosen: bool = False
) -> str | None:
  """Returns the line saying the bot sees what its seat may not, or None.

  That is when the bot is sent the full state of a game that hides
  information (another seat's cards, say); None otherwise.

  Args:
    options: What is launched, its `state` the state the bot is sent.
    game: The game.
    chosen: Whether `options.state` was not given but is the one state mode
      the bot server takes; the line then says so.
  """
  if options.state != remote_bot.FULL_STATE:
    return None
  if not information.hides_information(game.get_type()):
    return None
  how = " (the only state this bot server takes)" if chosen else ""
  return (
      f"Note: --state full_state{how} sends the bot the whole state of "
      f"{game.get_type().short_name}, which hides information, so it sees "
      "what its seat may not (the other seats' private cards, say); "
      "--state seat_view sends only what its seat may see."
  )


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


def resolve_state(bot_url: str, state: str | None) -> str:
  """Returns the state the bot server at `bot_url` is sent.

  That is `state` if given, else the one state mode the server's `/info`
  names (a checkpoint bot that replays the game takes only full_state, say),
  else seat_view.

  Raises:
    LaunchError: If `/info` names no state modes, or not `state`. The error
      names the `--state` that fixes it: otherwise the session would fail on
      creation.
  """
  info = _request("GET", bot_url + "/info")
  modes = info.get("state_modes") if isinstance(info, dict) else None
  if (
      not isinstance(modes, list)
      or not modes
      or not set(modes) <= set(remote_bot.STATE_MODES)
  ):
    raise LaunchError(
        "The bot server's /info names no state_modes from seat_view and "
        f"full_state: {info!r}."
    )
  if state is None:
    return modes[0] if len(modes) == 1 else remote_bot.SEAT_VIEW
  if state in modes:
    return state
  raise LaunchError(
      f"The bot server takes {', '.join(modes)}, not {state}: "
      f"rerun with --state {modes[0]}."
  )


def seat_config(options: Options, bot_url: str) -> dict[str, Any]:
  """Returns the Remote bot seat's settings.

  The seat waits for each move as long as the move budget allows
  (`timeout_seconds` 0), so a checkpoint the bot loads at its first move
  (BOT_MOVE_BUDGET_SECONDS) is not cut short.

  Raises:
    ValueError: If `options.state` is None, not yet resolved against the bot
      server (`resolve_state`).
  """
  if options.state is None:
    raise ValueError("The seat's state is not resolved; see resolve_state.")
  return {
      "url": bot_url,
      "state": options.state,
      "checkpoint": options.checkpoint,
      "timeout_seconds": 0.0,
  }


def session_body(
    options: Options,
    bot_url: str | Mapping[int, str],
    num_players: int,
    states: Mapping[int, str] | None = None,
) -> dict[str, Any]:
  """Returns the body that opens the session, its bot seats Remote bots.

  Args:
    options: What is launched.
    bot_url: The bot server's URL, or each bot seat's.
    num_players: The game's number of seats.
    states: Each bot seat's state, if not `options.state`.
  """
  name, parameters = parse_game(options.game)
  who = seating(options, num_players)
  seats = [
      {"controller": "human", "config": {}, "autoplay": False}
      for _ in range(num_players)
  ]
  for seat, bot in who.bots.items():
    url = bot_url if isinstance(bot_url, str) else bot_url[seat]
    state = options.state if states is None else states[seat]
    seats[seat] = {
        "controller": remote_bot.NAME,
        "config": seat_config(
            dataclasses.replace(
                options, checkpoint=bot.checkpoint, state=state
            ),
            url,
        ),
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
  """Launched bot servers, a GUI server and a session.

  Attributes:
    bot_url: The first bot seat's bot server's URL.
    gui_url: The GUI server's URL, without a trailing slash.
    session: The session's id.
    play_url: The address to play the session at, as the human seat.
    bot_urls: Each bot seat's bot server's URL, in seat order.
  """

  bot_url: str
  gui_url: str
  session: str
  play_url: str
  _bots: tuple[child_servers.ChildServer, ...]
  _gui: child_servers.ChildServer | None
  bot_urls: Mapping[int, str] = dataclasses.field(default_factory=dict)

  def exited(self) -> str | None:
    """Returns the label of a server that has exited, or None."""
    for child in (*self._bots, self._gui):
      if child is not None and child.process.poll() is not None:
        return child.label
    return None

  def stop(self) -> None:
    """Stops the GUI server, then the bot servers."""
    child_servers.stop_all((self._gui, *self._bots))


def _start_bot(
    options: Options,
    bot: SeatBot,
    label: str,
    out: TextIO,
    started: list[child_servers.ChildServer],
) -> str:
  """Starts a bot server serving `bot`; returns its URL once it is healthy.

  The server is appended to `started` as soon as it runs, so the caller
  stops it whatever happens next.
  """
  env = dict(os.environ)
  if bot.checkpoint:
    for variable in options.checkpoint_envs:
      env[variable] = bot.checkpoint
  child = child_servers.ChildServer(
      label,
      [
          options.python,
          "-m",
          "open_spiel.python.gui.remote_bot_server",
          "--host",
          "127.0.0.1",
          "--port",
          "0",
          "--bot",
          bot.bot,
      ],
      env,
      _BOT_READY,
      out,
  )
  started.append(child)
  url = child.wait_ready(options.bot_start_seconds)
  reason = bot_unready(url)
  if reason is not None:
    raise LaunchError(reason)
  return url


def _start_bots(
    options: Options,
    who: Seating,
    game: pyspiel.Game,
    out: TextIO,
    started: list[child_servers.ChildServer],
) -> tuple[dict[int, str], dict[int, str]]:
  """Starts the bot servers `who` needs; returns each bot seat's URL and state.

  Each server is appended to `started` as soon as it runs. The full-state
  note is printed once, however many servers it applies to.
  """
  urls: dict[tuple[int | None, SeatBot], str] = {}
  server_states: dict[tuple[int | None, SeatBot], str] = {}
  printed: set[str] = set()
  for i, served in enumerate(who.servers()):
    label = "bot" if i == 0 else f"bot {i + 1}"
    url = _start_bot(options, served[1], label, out, started)
    state = resolve_state(url, options.state)
    note = full_state_note(
        dataclasses.replace(options, state=state),
        game,
        chosen=options.state is None,
    )
    if note is not None and note not in printed:
      printed.add(note)
      print(note, file=out, flush=True)
    urls[served] = url
    server_states[served] = state
  keys = {seat: who.server_of(seat) for seat in who.bots}
  bot_urls = {seat: urls[key] for seat, key in keys.items()}
  states = {seat: server_states[key] for seat, key in keys.items()}
  return bot_urls, states


def launch(
    options: Options,
    out: TextIO = sys.stderr,
    started: Callable[[Running], object] | None = None,
) -> Running:
  """Starts the bot servers and the GUI server, and opens the session.

  Every bot seat's bot (`seating`) runs in a bot server; seats with the same
  bot and checkpoint share one. Every server started is stopped again if a
  later step fails or is interrupted.

  Args:
    options: What to launch.
    out: Where every server's output is copied.
    started: Called with the `Running` before it is returned. A caller that
      stops the servers on a signal hands its handler the `Running` here: if
      `started` raises, or a signal interrupts it, the servers are stopped,
      so no signal finds them running and in no one's hands.

  Raises:
    LaunchError: If the options are invalid, the GUI's port is taken (found
      before a bot server starts), a server does not start, or the
      session cannot be opened.
  """
  name, _ = parse_game(options.game)
  game = pyspiel.load_game(options.game)
  num_players = game.num_players()
  if num_players < 2:
    raise LaunchError(f"{name} has one seat, so no one plays against the bot.")
  who = seating(options, num_players)
  if options.state is not None and options.state not in remote_bot.STATE_MODES:
    raise LaunchError(
        f"--state {options.state!r} is not seat_view or full_state."
    )
  reason = child_servers.port_unavailable(options.host, options.port)
  if reason is not None:
    raise LaunchError(reason)
  bots: list[child_servers.ChildServer] = []
  gui: child_servers.ChildServer | None = None
  try:
    bot_urls, states = _start_bots(options, who, game, out, bots)
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
        session_body(options, bot_urls, num_players, states),
    )
    session = str(frame["session"]["id"])
    for seat_number in who.bots:
      seat = frame["seats"][seat_number]
      if seat.get("controller") != remote_bot.NAME:
        raise LaunchError(
            f"The bot seat {seat_number} is {seat!r}, not a Remote bot."
        )
      if seat.get("failure"):
        raise LaunchError(
            f"The bot seat {seat_number} failed: {seat['failure']}."
        )
    query = urllib.parse.urlencode({"session": session, "viewer": who.human})
    running = Running(
        bot_url=next(iter(bot_urls.values())),
        gui_url=gui_url,
        session=session,
        play_url=f"{gui_url}/game/{urllib.parse.quote(name)}?{query}",
        _bots=tuple(bots),
        _gui=gui,
        bot_urls=bot_urls,
    )
    if started is not None:
      started(running)
    return running
  except BaseException:
    child_servers.stop_all((gui, *bots))
    raise


def _seat_values(
    parser: argparse.ArgumentParser, flag: str, values: Sequence[str]
) -> dict[int, str]:
  """Returns `flag`'s SEAT=VALUE arguments as a map from seat to value.

  Exits through `parser.error` if one is not SEAT=VALUE with SEAT a number
  and VALUE not empty, or a seat is named twice.
  """
  result: dict[int, str] = {}
  for value in values:
    seat, _, text = value.partition("=")
    if not seat.strip().isdigit() or not text.strip():
      parser.error(f"{flag} {value!r} is not SEAT=VALUE.")
    if int(seat) in result:
      parser.error(f"{flag} names seat {int(seat)} twice.")
    result[int(seat)] = text.strip()
  return result


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
      "--bot-seat",
      type=int,
      action="append",
      default=None,
      metavar="SEAT",
      help=(
          "A seat the bot plays; repeat it for several, every other seat "
          "human. Default: every seat but --human-seat."
      ),
  )
  parser.add_argument(
      "--human-seat",
      type=int,
      default=None,
      metavar="SEAT",
      help=(
          "The seat you play, which the printed address opens as. Default: "
          "seat 0, or the first seat that is not a --bot-seat."
      ),
  )
  parser.add_argument(
      "--seat-bot",
      action="append",
      default=[],
      metavar="SEAT=BOT",
      help=(
          "A bot seat's own bot server --bot, instead of --bot, with no "
          "checkpoint unless --seat-checkpoint gives one; may be repeated."
      ),
  )
  parser.add_argument(
      "--seat-checkpoint",
      action="append",
      default=[],
      metavar="SEAT=CHECKPOINT",
      help=(
          "A bot seat's own checkpoint, instead of --checkpoint; may be "
          "repeated."
      ),
  )
  parser.add_argument(
      "--state",
      default=None,
      choices=remote_bot.STATE_MODES,
      help=(
          "What the bot is sent: seat_view, what its seat may see; or "
          "full_state, the whole state, for a bot that needs it (in a game "
          "that hides information, the bot then sees what its seat may "
          "not). Default: the one state the bot server takes, if it takes "
          "one, else seat_view."
      ),
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
  parser.add_argument(
      "--server-per-seat",
      action="store_true",
      help=(
          "Give every bot seat a bot server of its own, for a bot that keeps "
          "state between requests and cannot key it on the request's seat. "
          "Default: seats with the same bot and checkpoint share one."
      ),
  )
  args = parser.parse_args(argv)
  bots = _seat_values(parser, "--seat-bot", args.seat_bot)
  checkpoints = _seat_values(parser, "--seat-checkpoint", args.seat_checkpoint)
  seat_bots = {
      seat: SeatBot(
          bots.get(seat, args.bot),
          checkpoints.get(
              seat, "" if seat in bots else args.checkpoint.strip()
          ),
      )
      for seat in sorted(set(bots) | set(checkpoints))
  }
  options = Options(
      game=args.game,
      bot=args.bot,
      checkpoint=args.checkpoint.strip(),
      checkpoint_envs=(
          CHECKPOINT_ENVS
          if args.checkpoint_env is None
          else (args.checkpoint_env,)
      ),
      bot_seats=tuple(args.bot_seat or ()),
      human_seat=args.human_seat,
      seat_bots=seat_bots,
      server_per_seat=args.server_per_seat,
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
