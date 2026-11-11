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


"""The command line: `python -m open_spiel.python.gui`."""

import argparse
from collections.abc import Sequence
import faulthandler
import ipaddress
import math
import os
import pathlib
import signal
import threading
import types

from absl import logging

from open_spiel.python.gui import api
from open_spiel.python.gui import builds
from open_spiel.python.gui import contract
from open_spiel.python.gui import controllers
from open_spiel.python.gui import health_report
from open_spiel.python.gui import loader
from open_spiel.python.gui import mcts_controller
from open_spiel.python.gui import mcts_search
from open_spiel.python.gui import plugins
from open_spiel.python.gui import remote_bot
from open_spiel.python.gui import server as server_lib
from open_spiel.python.gui import static
from open_spiel.python.gui import store

# Bind addresses that listen on every interface, loopback included.
_WILDCARD_HOSTS = frozenset({"0.0.0.0", "::"})

# What the GUI is for. The --help text, the log at start and the READMEs and
# docs all carry these words.
DISCLAIMER = (
    "The OpenSpiel GUI is intended for personal research use on your own "
    "machine. It is not designed or reviewed as a production or secured "
    "system: do not expose it to untrusted networks or users."
)


def _seconds(value: str) -> float:
  """Parses a positive, finite number of seconds, for argparse."""
  try:
    seconds = float(value)
  except ValueError as e:
    raise argparse.ArgumentTypeError(f"{value!r} is not a number") from e
  if not (math.isfinite(seconds) and seconds > 0):
    raise argparse.ArgumentTypeError(f"{value!r} is not a positive number")
  return seconds


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  """Parses the command line.

  Args:
    argv: The arguments after the program name; the process's when None.

  Returns:
    The parsed flags.
  """
  parser = argparse.ArgumentParser(
      prog="python -m open_spiel.python.gui",
      description="Play OpenSpiel games in a browser.",
      epilog=DISCLAIMER,
  )
  parser.add_argument(
      "--host",
      default="127.0.0.1",
      help=(
          "The address to listen on. The server has no authentication and "
          "serves every player's private information to the oracle viewer, "
          "so listen beyond loopback only on a trusted network."
      ),
  )
  parser.add_argument(
      "--port",
      type=int,
      default=8080,
      help="The port to listen on; 0 picks a free port.",
  )
  parser.add_argument(
      "--allow-host",
      action="append",
      default=[],
      metavar="NAME",
      help=(
          "A further host name that requests may address the server by, "
          "such as the machine's name on the network. Repeatable."
      ),
  )
  parser.add_argument(
      "--client-dir",
      type=pathlib.Path,
      default=static.DEFAULT_ROOT,
      help="The directory of the built browser client.",
  )
  parser.add_argument(
      "--controllers",
      action="append",
      default=[],
      metavar="MODULE[,MODULE...]",
      help=(
          "Controller plugins to load before serving: modules, each with a "
          "function register_into(controllers) that registers its bots with "
          f"controllers.register. {plugins.AUTO!r} loads every plugin that "
          "an installed distribution declares in the entry-point group "
          f"{plugins.ENTRY_POINT_GROUP!r}. Repeatable. Without this flag no "
          "plugin is loaded. A plugin that cannot be imported, has no "
          "register_into, or fails to register stops the server starting."
      ),
  )
  parser.add_argument(
      "--move-budget-seconds",
      type=_seconds,
      default=controllers.DEFAULT_MOVE_BUDGET_SECONDS,
      metavar="SECONDS",
      help=(
          "How long a controller may take to choose a move. One that takes "
          "longer fails, and its seat pauses. Default: %(default)g."
      ),
  )
  parser.add_argument(
      "--build-budget-seconds",
      type=_seconds,
      default=builds.DEFAULT_BUDGET_SECONDS,
      metavar="SECONDS",
      help=(
          "How long a controller may take to be built, as a bot loads and "
          "warms up a trained model. Its seat shows it preparing meanwhile; "
          "one that takes longer leaves its seat unavailable, and the "
          "session paused. Default: %(default)g."
      ),
  )
  parser.add_argument(
      "--remote-bot-host",
      action="append",
      default=[],
      metavar="NAME",
      help=(
          "A host, beyond this machine, that a Remote bot seat's URL may "
          "name; '*' admits any host. Repeatable. Without it the server "
          "posts positions only to bot servers on this machine."
      ),
  )
  parser.add_argument(
      "--remote-bot-token-env",
      action="append",
      default=[],
      metavar="HOST[:PORT]=VAR",
      help=(
          "Send the token held in the environment variable VAR as "
          "'Authorization: Bearer <token>' to the remote bot server at HOST "
          "(and PORT, if given), and to no other. Repeatable. A token is "
          "never part of a seat's settings, which every viewer sees."
      ),
  )
  parser.add_argument(
      "--exclude-games",
      action="append",
      default=[],
      metavar="GAME[,GAME...]",
      help=(
          "Registered games to leave out of the catalogue, and to refuse "
          "sessions of. Repeatable. A name this build does not register is "
          "ignored."
      ),
  )
  return parser.parse_args(argv)


def remote_bot_tokens(values: Sequence[str]) -> dict[str, str]:
  """Returns the tokens `--remote-bot-token-env` names, by host[:port].

  Raises:
    SystemExit: If a value is not HOST[:PORT]=VAR, or VAR is not set or is
      blank.
  """
  tokens = {}
  for value in values:
    where, sep, variable = value.partition("=")
    if not sep or not where.strip() or not variable.strip():
      raise SystemExit(
          f"--remote-bot-token-env {value!r} is not HOST[:PORT]=VAR."
      )
    token = os.environ.get(variable.strip(), "").strip()
    if not token:
      raise SystemExit(
          f"--remote-bot-token-env names {variable.strip()}, which is not set "
          "or blank."
      )
    tokens[where.strip()] = token
  return tokens


def excluded_games(values: Sequence[str]) -> tuple[str, ...]:
  """Returns the game names `--exclude-games` gives, each comma separated."""
  return tuple(
      name.strip()
      for value in values
      for name in value.split(",")
      if name.strip()
  )


def load_plugins(values: Sequence[str]) -> tuple[str, ...]:
  """Loads the controller plugins that `--controllers` names.

  Args:
    values: The `--controllers` values, each comma separated.

  Returns:
    The names of the controllers the plugins registered.

  Raises:
    plugins.PluginError: If a plugin fails to load or register.
  """
  return plugins.load(
      [name for value in values for name in plugins.parse(value)]
  )


def allowed_hosts(host: str, extra: Sequence[str]) -> frozenset[str]:
  """Returns the host names requests may address the server by.

  Args:
    host: The address the server listens on.
    extra: Further host names to allow.

  Returns:
    The loopback names, `host` unless it is a wildcard, and `extra`, in lower
    case.
  """
  names = set(server_lib.LOOPBACK_HOSTS)
  if host not in _WILDCARD_HOSTS:
    names.add(server_lib.host_name(host))
  names.update(server_lib.host_name(name) for name in extra)
  return frozenset(names)


def is_loopback(host: str) -> bool:
  """Returns whether `host` reaches only this machine.

  Args:
    host: The address the server listens on, or a host name.

  Returns:
    True for `localhost` and loopback addresses; False for a wildcard, any
    other address, and any other host name, which may resolve elsewhere.
  """
  if host.lower() == "localhost":
    return True
  try:
    return ipaddress.ip_address(host).is_loopback
  except ValueError:
    return False


def log_disclaimer(host: str) -> None:
  """Logs the disclaimer once, as a warning when `host` is not loopback."""
  if is_loopback(host):
    logging.info("%s", DISCLAIMER)
    return
  logging.warning(
      "Listening on %s, which other machines can reach. %s The server has "
      "no authentication, and the oracle viewer sees every player's private "
      "information.",
      host,
      DISCLAIMER,
  )


def configure_mcts() -> None:
  """Sets MCTS's limits by how the served pyspiel's C++ MCTS bounds a search.

  Probes once (`mcts_search.limits`) and logs what it found.
  """
  probe = mcts_search.limits()
  controllers.configure_mcts(probe.limits)
  maximum = mcts_controller.SIMULATIONS[probe.limits].maximum
  logging.info(
      "C++ MCTS limits: %s (the probe's search, given %d simulations and "
      "%g s, ran %d simulations in %.1f ms); max_simulations is at most %d%s.",
      probe.limits.value,
      mcts_search.PROBE_SIMULATIONS,
      mcts_search.PROBE_SECONDS,
      probe.simulations,
      probe.seconds * 1000,
      maximum,
      (
          ", with a wall-clock backstop inside the move budget"
          if probe.limits == contract.MctsLimits.SIMULATIONS_OR_CLOCK
          else ", with no wall-clock time"
      ),
  )


def enable_crash_stacks() -> None:
  """Makes a crash of the server print every thread's Python stack.

  A fault in native code (a segmentation fault, an abort) kills the process
  before Python can log anything; with this, stderr shows where each thread
  was. Already enabled by `python -X faulthandler` or PYTHONFAULTHANDLER, it
  is left as it is.
  """
  if not faulthandler.is_enabled():
    faulthandler.enable(all_threads=True)


def main(argv: Sequence[str] | None = None) -> None:
  """Serves until interrupted or terminated, then shuts down cleanly."""
  args = parse_args(argv)
  logging.set_verbosity(logging.INFO)
  enable_crash_stacks()
  log_disclaimer(args.host)
  controllers.set_move_budget(args.move_budget_seconds)
  builds.set_budget_seconds(args.build_budget_seconds)
  # Before the server: the catalogue and the health report read the
  # registry once. A plugin that fails stops the start here.
  configure_mcts()
  remote_bot.configure(
      hosts=args.remote_bot_host,
      tokens=remote_bot_tokens(args.remote_bot_token_env),
  )
  if args.remote_bot_host:
    logging.info(
        "Remote bot seats may post to this machine and to %s.",
        ", ".join(args.remote_bot_host),
    )
  if args.exclude_games:
    left_out = loader.exclude(excluded_games(args.exclude_games))
    logging.info(
        "Games left out of the catalogue: %s.", ", ".join(left_out) or "none"
    )
  if args.controllers:
    load_plugins(args.controllers)
  server = server_lib.Server(
      (args.host, args.port),
      api.Api(
          store.SessionStore(),
          catalogue=loader.build_catalogue,
          health=health_report.build,
      ),
      static.ClientFiles(args.client_dir),
      allowed_hosts(args.host, args.allow_host),
  )

  def stop(signum: int, unused_frame: types.FrameType | None) -> None:
    """Shuts the server down on a signal."""
    logging.info("Received %s; shutting down.", signal.Signals(signum).name)
    # `shutdown` waits for `serve_forever`, which runs on this thread.
    threading.Thread(target=server.shutdown).start()

  signal.signal(signal.SIGINT, stop)
  signal.signal(signal.SIGTERM, stop)
  with server:
    if not server.client.built:
      logging.warning(
          "The client is not built in %s; serving the API only.",
          server.client.root,
      )
    # Flushed, so the line is not held back when stdout is a pipe or a file.
    print(
        f"Serving OpenSpiel games at {_display_url(server, args.host)}",
        flush=True,
    )
    server.serve_forever()


def _display_url(server: server_lib.Server, host: str) -> str:
  """Returns a URL to open the server at; loopback for a wildcard bind."""
  if host in _WILDCARD_HOSTS:
    return f"http://localhost:{server.server_address[1]}/"
  return server.url
