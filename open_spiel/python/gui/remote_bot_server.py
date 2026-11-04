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


"""A reference bot server for the GUI's Remote bot seats.

  python -m open_spiel.python.gui.remote_bot_server --port 9001 --bot random

serves a bot over the remote bot protocol (REMOTE_BOTS.md); a seat given
the Remote bot controller with the URL http://localhost:9001 then plays it.
`--bot` is `random`, `mcts`, or `MODULE:FACTORY`: a function that takes no
arguments and returns a bot, so that any bot (a trained checkpoint, say) is
served without the GUI knowing of it. A bot is an object with:

  state_modes: the requests it takes, a subset of ("seat_view",
    "full_state").
  games: the short names of the games it plays, or None for any.
  step(mode, request): the action to play, as an int, or a dict
    {"action": int, "notice": str | None}. `mode` is "seat_view" or
    "full_state", and `request` is the request's JSON body. A ValueError
    is answered as a refusal (HTTP 422), with its message.

A factory that plays a checkpoint reads it, by convention, from the
environment variable CHECKPOINT_ENV, and a request's `config.checkpoint`
(a Remote bot seat's `checkpoint` setting) may name another.

The server listens on 127.0.0.1 unless told otherwise. With `--token-env
VAR` it answers only requests carrying `Authorization: Bearer <token>`, the
token held in the environment variable VAR, and 401 to any other; it refuses
to listen beyond loopback without one.
"""

import argparse
from collections.abc import Sequence
import hmac
import http
import http.server
import json
import os
import random
from typing import Any, Protocol

from absl import logging
import numpy as np

from open_spiel.python.algorithms import mcts
from open_spiel.python.gui import plugins
from open_spiel.python.gui import remote_bot
import pyspiel

SEAT_VIEW = "seat_view"
FULL_STATE = "full_state"

# Each request path, and the state mode it carries.
_PATHS = {"/spiel/observe": SEAT_VIEW, "/spiel/step": FULL_STATE}

# The environment variable a bot's factory reads the checkpoint it plays
# from, by convention: `play --checkpoint` sets it and
# `stand_in_checkpoint_bot` reads it.
CHECKPOINT_ENV = "OPEN_SPIEL_REMOTE_BOT_CHECKPOINT"

# The largest request body read.
MAX_BODY_BYTES = 16 << 20

# MCTS's simulations per move, unless a request's config gives
# `max_simulations`.
DEFAULT_MCTS_SIMULATIONS = 200


class Bot(Protocol):
  """What the server serves."""

  @property
  def state_modes(self) -> Sequence[str]:
    """The requests it takes: "seat_view", "full_state" or both."""

  @property
  def games(self) -> Sequence[str] | None:
    """The short names of the games it plays, or None for any."""

  def step(self, mode: str, request: dict[str, Any]) -> int | dict[str, Any]:
    """Returns the action to play, or {"action", "notice"}."""


def legal_actions(mode: str, request: dict[str, Any]) -> list[int]:
  """Returns the seat's legal actions, from either kind of request."""
  if mode == SEAT_VIEW:
    return list(request["observation"]["legal_actions"])
  return list(request["legal_actions"])


def seed_of(mode: str, request: dict[str, Any]) -> int:
  """Returns the request's per-move seed."""
  if mode == SEAT_VIEW:
    return int(request["observation"]["seed"])
  return int(request["seed"])


class RandomBot:
  """Plays a uniformly random legal action, from the request's seed."""

  state_modes = (SEAT_VIEW, FULL_STATE)
  games = None

  def step(self, mode: str, request: dict[str, Any]) -> int:
    """Returns one of the legal actions, drawn uniformly."""
    rng = random.Random(seed_of(mode, request))
    return rng.choice(legal_actions(mode, request))


class MctsBot:
  """Plays OpenSpiel's Python MCTS on the full state.

  The state is rebuilt from the game and the history, so it needs
  `full_state`. A request's config may give `max_simulations`.
  """

  state_modes = (FULL_STATE,)
  games = None

  def step(self, mode: str, request: dict[str, Any]) -> int:
    """Returns MCTS's choice at the request's state."""
    if mode != FULL_STATE:
      raise ValueError("MCTS needs the full state.")
    game = pyspiel.load_game(request["game"])
    state = game.new_initial_state()
    for action in request["history"]:
      state.apply_action(int(action))
    if state.current_player() != request["seat"]:
      raise ValueError(
          f"Seat {request['seat']} is not to move after the history."
      )
    simulations = int(
        request.get("config", {}).get(
            "max_simulations", DEFAULT_MCTS_SIMULATIONS
        )
    )
    if not 1 <= simulations <= 100_000:
      raise ValueError(f"max_simulations {simulations} is not 1..100000.")
    rng = np.random.RandomState(  # pylint: disable=no-member
        seed_of(mode, request) % (2**32)
    )
    bot = mcts.MCTSBot(
        game,
        uct_c=2.0,
        max_simulations=simulations,
        evaluator=mcts.RandomRolloutEvaluator(1, rng),
        random_state=rng,
    )
    return int(bot.step(state))


def load_bot(name: str) -> Bot:
  """Returns the bot `--bot` names: random, mcts or MODULE:FACTORY.

  Raises:
    ValueError: If `name` is none of these, or the factory's bot does not
      declare its state modes.
    plugins.PluginError: If the factory's module cannot be imported.
  """
  if name == "random":
    return RandomBot()
  if name == "mcts":
    return MctsBot()
  module_name, sep, factory_name = name.partition(":")
  if not sep or not module_name or not factory_name:
    raise ValueError(f"--bot {name!r} is not random, mcts or MODULE:FACTORY.")
  factory = getattr(
      plugins.import_named(module_name, "The bot's module"), factory_name
  )
  bot: Bot = factory()
  modes = tuple(getattr(bot, "state_modes", ()))
  if not modes or not set(modes) <= {SEAT_VIEW, FULL_STATE}:
    raise ValueError(
        f"{name}'s bot declares state_modes {modes!r}, not a non-empty subset "
        f"of {SEAT_VIEW!r} and {FULL_STATE!r}."
    )
  if not callable(getattr(bot, "step", None)):
    raise ValueError(f"{name}'s bot has no step(mode, request).")
  return bot


def reply_of(result: object, notice: str | None) -> dict[str, Any]:
  """Returns the reply body for a bot's result.

  Args:
    result: What the bot's `step` returned.
    notice: The server's `--notice`, added when the bot gives none.

  Raises:
    TypeError: If `result` is neither an int nor a dict with an int
      `action` or a non-empty `actions`.
  """
  if isinstance(result, bool) or not isinstance(result, (int, dict)):
    raise TypeError(f"The bot returned {result!r}, not an action.")
  if isinstance(result, int):
    reply: dict[str, Any] = {"action": result}
  else:
    action = result.get("action")
    line = result.get("actions")
    if (isinstance(action, bool) or not isinstance(action, int)) and not (
        isinstance(line, list) and line
    ):
      raise TypeError(
          f"The bot returned {result!r}, with no int action or actions."
      )
    reply = dict(result)
  if notice is not None and reply.get("notice") is None:
    reply["notice"] = notice
  return reply


class Server(http.server.ThreadingHTTPServer):
  """Serves one bot over the remote bot protocol."""

  daemon_threads = True

  def __init__(
      self,
      address: tuple[str, int],
      bot: Bot,
      *,
      name: str,
      notice: str | None = None,
      token: str | None = None,
  ) -> None:
    """Initializes the server.

    Args:
      address: The host and port to listen on.
      bot: The bot to serve.
      name: The bot's name, for /health and /info.
      notice: A notice added to every move the bot gives none for.
      token: The bearer token every request must carry; None for none.

    Raises:
      ValueError: If `address` is beyond loopback and there is no token.
    """
    if token is None and not remote_bot.is_loopback(address[0].lower()):
      raise ValueError(
          f"The bot server would listen on {address[0]}, beyond this "
          "machine, with no token; give --token-env."
      )
    super().__init__(address, _Handler)
    self.bot = bot
    self.name = name
    self.notice = notice
    self.token = token

  @property
  def url(self) -> str:
    """The URL to give a Remote bot seat."""
    host, port = self.server_address[:2]
    return f"http://{host!s}:{port}"


class _Handler(http.server.BaseHTTPRequestHandler):
  """Answers the protocol's requests."""

  server: Server
  protocol_version = "HTTP/1.1"

  def log_message(self, format: str, *args: object) -> None:
    """Logs a request through absl."""
    logging.info("%s %s", self.address_string(), format % args)

  def _send(self, status: int, body: dict[str, Any]) -> None:
    """Sends a JSON reply."""
    data = json.dumps(body).encode()
    self.send_response(status)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(data)))
    self.end_headers()
    self.wfile.write(data)

  def _authorized(self) -> bool:
    """Whether the request carries the server's token; answers 401 if not."""
    token = self.server.token
    if token is None:
      return True
    given = self.headers.get("Authorization", "")
    if hmac.compare_digest(given.encode(), f"Bearer {token}".encode()):
      return True
    self._send(
        http.HTTPStatus.UNAUTHORIZED,
        {"error": "a bearer token is required", "reason": "unauthorized"},
    )
    return False

  def do_GET(self) -> None:  # pylint: disable=invalid-name
    """Answers /health and /info."""
    if not self._authorized():
      return
    bot = self.server.bot
    if self.path == "/health":
      self._send(http.HTTPStatus.OK, {"ok": True, "name": self.server.name})
    elif self.path == "/info":
      games = getattr(bot, "games", None)
      self._send(
          http.HTTPStatus.OK,
          {
              "name": self.server.name,
              "protocol": 1,
              "state_modes": list(bot.state_modes),
              "games": None if games is None else list(games),
          },
      )
    else:
      self._send(http.HTTPStatus.NOT_FOUND, {"error": f"no {self.path}"})

  def do_POST(self) -> None:  # pylint: disable=invalid-name
    """Answers a move request."""
    if not self._authorized():
      return
    mode = _PATHS.get(self.path)
    if mode is None:
      self._send(http.HTTPStatus.NOT_FOUND, {"error": f"no {self.path}"})
      return
    length = int(self.headers.get("Content-Length", "0"))
    if not 0 < length <= MAX_BODY_BYTES:
      self._send(
          http.HTTPStatus.BAD_REQUEST,
          {"error": "bad body length", "reason": "bad_request"},
      )
      return
    try:
      request = json.loads(self.rfile.read(length))
    except json.JSONDecodeError:
      self._send(
          http.HTTPStatus.BAD_REQUEST,
          {"error": "the body is not JSON", "reason": "bad_request"},
      )
      return
    bot = self.server.bot
    if mode not in bot.state_modes:
      self._send(
          http.HTTPStatus.CONFLICT,
          {
              "error": f"this bot takes {', '.join(bot.state_modes)}",
              "reason": "state_mode",
          },
      )
      return
    try:
      result = bot.step(mode, request)
    except ValueError as e:
      self._send(
          http.HTTPStatus.UNPROCESSABLE_ENTITY,
          {"error": str(e), "reason": "bot_refused"},
      )
      return
    except Exception:  # pylint: disable=broad-exception-caught
      logging.exception("The bot failed on %s.", self.path)
      self._send(
          http.HTTPStatus.INTERNAL_SERVER_ERROR,
          {"error": "the bot failed; see the bot server's log"},
      )
      return
    self._send(http.HTTPStatus.OK, reply_of(result, self.server.notice))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  """Parses the command line."""
  parser = argparse.ArgumentParser(
      prog="python -m open_spiel.python.gui.remote_bot_server",
      description="Serve a bot to the OpenSpiel GUI's Remote bot seats.",
  )
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=9001)
  parser.add_argument(
      "--bot",
      default="random",
      help="random, mcts, or MODULE:FACTORY (a function returning a bot).",
  )
  parser.add_argument(
      "--token-env",
      metavar="VAR",
      help=(
          "Answer only requests carrying 'Authorization: Bearer <token>', the "
          "token in the environment variable VAR (the GUI server sends it "
          "with --remote-bot-token-env). Required to listen beyond loopback."
      ),
  )
  parser.add_argument(
      "--notice",
      default=None,
      help=(
          "A notice to send with every move, which the GUI shows to the "
          "seat's own viewer and to omniscient viewers; for trying it out."
      ),
  )
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
  """Serves until interrupted."""
  args = parse_args(argv)
  logging.set_verbosity(logging.INFO)
  token = None
  if args.token_env is not None:
    token = os.environ.get(args.token_env, "").strip()
    if not token:
      raise SystemExit(f"--token-env names {args.token_env}, which is unset.")
  try:
    bot = load_bot(args.bot)
    server = Server(
        (args.host, args.port),
        bot,
        name=args.bot,
        notice=args.notice,
        token=token,
    )
  except (ValueError, plugins.PluginError) as e:
    # A mistake on the command line is one line, not a traceback.
    raise SystemExit(f"error: {e}") from e
  with server:
    print(f"Serving the {args.bot} bot at {server.url}", flush=True)
    try:
      server.serve_forever()
    except KeyboardInterrupt:
      logging.info("Interrupted; shutting down.")


if __name__ == "__main__":
  main()
