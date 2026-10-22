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
import ipaddress
import pathlib
import signal
import threading
import types

from absl import logging

from open_spiel.python.gui import api
from open_spiel.python.gui import health_report
from open_spiel.python.gui import loader
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
  return parser.parse_args(argv)


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


def main(argv: Sequence[str] | None = None) -> None:
  """Serves until interrupted or terminated, then shuts down cleanly."""
  args = parse_args(argv)
  logging.set_verbosity(logging.INFO)
  log_disclaimer(args.host)
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
