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


"""The remote bot: a seat played by a bot server, over HTTP.

A seat given the `remote` controller sends each position to a bot server at
the seat's `url`, which answers with the action to play. The bot server runs
anywhere: on this machine at another port, or elsewhere. What it is sent is
the seat's `state` setting:

  seat_view   POST {url}/spiel/observe with what the seat may see: its
              information state string, its legal actions and a seed,
              nothing else, which a strict seat-view bot accepts; on
              request (`extras`, `tensors`) also its observation string,
              its legal actions' labels, the history as the seat sees it,
              and its tensors.
  full_state  POST {url}/spiel/step with the full state (`state.serialize()`)
              and the whole history.

A seat's settings go to every viewer, so every viewer can see which a seat
is sent. REMOTE_BOTS.md documents the protocol, and
`remote_bot_server` is a reference bot server that wraps any bot.

The server posts only to loopback addresses unless it is started with
`--remote-bot-host` (`configure`), so a viewer who sets a seat's URL cannot
make it post a game's state to an arbitrary host.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import ipaddress
import json
import urllib.parse

from open_spiel.python.gui import contract
from open_spiel.python.gui import controller_registry
from open_spiel.python.gui import controller_settings
from open_spiel.python.gui import information
from open_spiel.python.gui import move_budget
from open_spiel.python.gui import plies
from open_spiel.python.gui import remote_bot_config
from open_spiel.python.gui import remote_bot_http
import pyspiel

NAME = "remote"

# The `state` setting's values: what the bot server is sent.
SEAT_VIEW = "seat_view"
FULL_STATE = "full_state"
STATE_MODES = (SEAT_VIEW, FULL_STATE)

# The protocol version every request carries, in a header.
PROTOCOL = remote_bot_http.PROTOCOL

# A seat-view request's keys and its observation's, with `extras` and
# `tensors` off: exactly what a strict seat-view bot accepts (one that
# refuses a request with any other key).
SEAT_VIEW_KEYS = ("game", "observation", "config", "game_encoding")
SEAT_VIEW_OBSERVATION_KEYS = (
    "information_state_string",
    "seat",
    "legal_actions",
    "seed",
)
# The observation's keys `extras` adds, and those `tensors` adds.
EXTRAS_KEYS = ("observation_string", "legal_action_labels", "history")
TENSORS_KEYS = ("observation_tensor", "information_state_tensor")

# The bot server's routes, below the seat's `url`.
OBSERVE_PATH = "/spiel/observe"
STEP_PATH = "/spiel/step"
HEALTH_PATH = "/health"
INFO_PATH = "/info"

# How long the build waits for `/health` and `/info`.
CHECK_TIMEOUT_SECONDS = 10.0

# How much of the move budget is left for the server, after the bot's reply.
_MARGIN_SECONDS = 0.5

# The ways a remote bot fails, as its seat's failure names them.
RemoteBotRefusedError = remote_bot_http.RemoteBotRefusedError
RemoteBotUnreachableError = remote_bot_http.RemoteBotUnreachableError
RemoteBotReplyError = remote_bot_http.RemoteBotReplyError


@dataclasses.dataclass
class _Configured:
  """Where remote bots may be, and their tokens (`configure`)."""

  hosts: frozenset[str] = frozenset()
  tokens: Mapping[str, str] = dataclasses.field(default_factory=dict)


_CONFIGURED = _Configured()


def configure(
    *, hosts: Sequence[str] = (), tokens: Mapping[str, str] | None = None
) -> None:
  """Sets where remote bots may be, and the bearer tokens sent to them.

  Call it before serving.

  Args:
    hosts: Host names, beyond loopback, that a seat's URL may name; `*`
      admits any host.
    tokens: Bearer tokens by `host` or `host:port`. A request to a bot
      server at that host (and port, if given) carries
      `Authorization: Bearer <token>`; a request anywhere else carries none,
      so a seat pointed at another listener never receives it. A token is
      never part of a seat's settings, which every viewer sees.

  Raises:
    ValueError: If a host name is blank or `*` is given a token, or a token
      is blank.
  """
  for host in hosts:
    if not host.strip():
      raise ValueError("A remote bot host is blank.")
  scoped = {}
  for where, token in (tokens or {}).items():
    where = where.strip().lower()
    if not where or where == "*":
      raise ValueError(
          "A remote bot token is for one host (or host:port), not "
          f"{where or 'a blank one'!r}."
      )
    if not token.strip():
      raise ValueError(f"The remote bot token for {where} is blank.")
    scoped[where] = token
  _CONFIGURED.hosts = frozenset(host.strip().lower() for host in hosts)
  _CONFIGURED.tokens = scoped


def token_for(url: str) -> str | None:
  """Returns the token for the bot server at `url`, or None if it has none."""
  parts = urllib.parse.urlsplit(url)
  host = (parts.hostname or "").lower()
  port = parts.port or (443 if parts.scheme == "https" else 80)
  tokens = _CONFIGURED.tokens
  return tokens.get(f"{host}:{port}", tokens.get(host))


def allowed_hosts() -> frozenset[str]:
  """Returns the host names beyond loopback a seat's URL may name."""
  return _CONFIGURED.hosts


def is_loopback(host: str) -> bool:
  """Whether `host` names this machine: `localhost` or a loopback address."""
  if host == "localhost":
    return True
  try:
    return ipaddress.ip_address(host).is_loopback
  except ValueError:
    return False


def url_refusal(url: str) -> str | None:
  """Returns why the server may not post to `url`, or None if it may."""
  try:
    parts = urllib.parse.urlsplit(url)
    host = (parts.hostname or "").lower()
    port = parts.port  # Raises on a port that is not a number.
  except ValueError:
    return f"The remote bot URL {controller_settings.shown(url)} is invalid."
  if parts.scheme not in ("http", "https") or not host:
    return (
        f"The remote bot URL {controller_settings.shown(url)} is not an "
        "http or https URL with a host."
    )
  if port == 0:
    return "The remote bot URL names port 0."
  if parts.query or parts.fragment or parts.username or parts.password:
    return (
        "The remote bot URL takes no query, fragment or credentials; give the "
        "bot server's base URL."
    )
  if is_loopback(host) or "*" in allowed_hosts() or host in allowed_hosts():
    return None
  return (
      "The server posts only to this machine and to the hosts it is started "
      f"with (--remote-bot-host), and {host} is not one."
  )


def _settings() -> tuple[controller_settings.Setting, ...]:
  """The remote bot's settings."""
  return (
      controller_settings.Setting(
          "url",
          str,
          controller_settings.REQUIRED,
          "The bot server's base URL, such as http://localhost:9001.",
          min_length=1,
          max_length=2000,
          pattern="^https?://",
      ),
      controller_settings.Setting(
          "state",
          str,
          SEAT_VIEW,
          "What the bot is sent: seat_view, what its seat may see; or "
          "full_state, the whole state, for a bot that needs it.",
          choices=STATE_MODES,
      ),
      controller_settings.Setting(
          "checkpoint",
          str,
          "",
          "The checkpoint the bot plays, such as /path/to/run@30000, sent "
          "with every request as `config.checkpoint`; empty, the bot server's "
          "own. Only a bot that reads `config.checkpoint` uses it.",
          max_length=2000,
      ),
      controller_settings.Setting(
          "config",
          str,
          "{}",
          "A JSON object passed to the bot with every request, as `config`.",
          max_length=4000,
      ),
      controller_settings.Setting(
          "extras",
          bool,
          False,
          "Whether a seat_view request also carries the seat's observation "
          "string, its legal actions' labels and the history as it sees it. "
          "Off, it carries only the information state string, the legal "
          "actions and a seed, which is all a strict seat-view bot accepts.",
      ),
      controller_settings.Setting(
          "tensors",
          bool,
          False,
          "Whether a seat_view request also carries the seat's observation "
          "and information state tensors, where the game provides them.",
      ),
      controller_settings.Setting(
          "timeout_seconds",
          float,
          0.0,
          "How long to wait for each reply, capped by the move budget; 0 "
          "waits as long as the budget allows.",
          minimum=0.0,
      ),
  )


def _provides_seat_view(game: pyspiel.Game) -> bool:
  """Whether the game gives a seat something to see beyond its actions."""
  game_type = game.get_type()
  return (
      not information.hides_information(game_type)
      or game_type.provides_observation_string
      or _provides_information_state_string(game_type)
  )


def _provides_information_state_string(game_type: pyspiel.GameType) -> bool:
  """Whether the game gives a seat an information state it may see."""
  return (
      game_type.provides_information_state_string
      and game_type.short_name not in information.HIDDEN_DESPITE_DECLARATION
  )


def _information_state_string(state: pyspiel.State, seat: int) -> str | None:
  """The seat's information state string, or None if the game gives none.

  Some games give one without declaring it (gin rummy's, the seat's own
  action-observation history, is one), and a seat-view bot may rely on it,
  so an undeclared one is sent when the game gives it.
  """
  game_type = state.get_game().get_type()
  if game_type.short_name in information.HIDDEN_DESPITE_DECLARATION:
    return None
  if game_type.provides_information_state_string:
    return str(state.information_state_string(seat))
  try:
    return str(state.information_state_string(seat))
  except pyspiel.SpielError:
    return None


def config_refusal(
    game: pyspiel.Game, config: Mapping[str, contract.ParameterValue]
) -> str | None:
  """Returns why a remote bot with `config` cannot play `game`, or None."""
  reason = url_refusal(str(config["url"]))
  if reason is not None:
    return reason
  try:
    remote_bot_config.seat_bot_config(config)
  except ValueError as e:
    return str(e)
  if config["state"] == SEAT_VIEW and not _provides_seat_view(game):
    return (
        f"{game.get_type().short_name} gives a seat no observation or "
        "information state to send; use full_state."
    )
  return None


def reads_full_state(config: Mapping[str, contract.ParameterValue]) -> bool:
  """Whether a remote bot with `config` is sent the full state."""
  return config["state"] == FULL_STATE


def _request_json(
    url: str, body: bytes | None, timeout: float
) -> tuple[int, contract.JsonValue]:
  """Sends a request to a bot server, with the configured token."""
  return remote_bot_http.send(url, body, timeout, token_for(url))


def _seat_history(
    state: pyspiel.State, seat: int, hides: bool
) -> list[dict[str, contract.JsonValue]]:
  """The history as `seat` sees it, as a frame for the seat shows it.

  The seat's own actions are given in full; any other ply, a chance outcome
  included, only as its public label, in a game that hides information.
  """
  _, played = plies.rebuild(state.get_game(), state.history())
  history: list[dict[str, contract.JsonValue]] = []
  for ply in played:
    if not hides or (ply.seat is not None and ply.seat == seat):
      history.append({
          "seat": ply.seat,
          "action": ply.action,
          "label": ply.label,
          "public_label": None,
      })
    else:
      history.append({
          "seat": ply.seat,
          "action": None,
          "label": None,
          "public_label": ply.public_label,
      })
  return history


class RemoteBot:
  """Plays a seat by asking a bot server for each move.

  Attributes:
    last_notice: The bot server's notice about its latest choice, or None.
  """

  def __init__(
      self,
      game: pyspiel.Game,
      *,
      url: str,
      state_mode: str,
      bot_config: dict[str, contract.JsonValue],
      extras: bool,
      tensors: bool,
      timeout_seconds: float,
      seed: int,
  ) -> None:
    if state_mode not in STATE_MODES:
      raise ValueError(f"Unknown state mode {state_mode!r}.")
    self._game = game
    self._url = url.rstrip("/")
    self._state_mode = state_mode
    self._bot_config = bot_config
    self._extras = extras
    self._tensors = tensors
    self._timeout_seconds = timeout_seconds
    self._seed = seed
    self.last_notice: str | None = None

  def check(self) -> None:
    """Checks the bot server answers, and can play this game this way.

    `GET /health` must answer 2xx. `GET /info` is optional: when it answers
    2xx with `games` or `state_modes`, a game or a state mode it does not
    list is refused.

    Raises:
      controller_registry.UnavailableError: If the bot server says it cannot
        play the game, or be sent the seat's state mode.
      RemoteBotUnreachableError: If `/health` cannot be reached or does not
        answer 2xx.
      RemoteBotReplyError: If either answers with a body that is not JSON,
        or `/info` with one that is not the protocol's.
    """
    status, _ = _request_json(
        self._url + HEALTH_PATH, None, CHECK_TIMEOUT_SECONDS
    )
    if not 200 <= status < 300:
      raise RemoteBotUnreachableError(
          f"The bot server's {HEALTH_PATH} answered HTTP {status}."
      )
    try:
      status, info = _request_json(
          self._url + INFO_PATH, None, CHECK_TIMEOUT_SECONDS
      )
    except RemoteBotUnreachableError:
      return  # `/info` is optional.
    if not 200 <= status < 300:
      return
    if not isinstance(info, dict):
      raise RemoteBotReplyError(
          f"The bot server's {INFO_PATH} is not an object."
      )
    name = self._game.get_type().short_name
    games, modes = info.get("games"), info.get("state_modes")
    if games is not None:
      if not isinstance(games, list):
        raise RemoteBotReplyError(f"{INFO_PATH} `games` is not a list.")
      if name not in games and str(self._game) not in games:
        raise controller_registry.UnavailableError(
            f"The bot server plays {', '.join(map(str, games)) or 'no game'}, "
            f"not {name}."
        )
    if modes is not None:
      if not isinstance(modes, list):
        raise RemoteBotReplyError(f"{INFO_PATH} `state_modes` is not a list.")
      if self._state_mode not in modes:
        raise controller_registry.UnavailableError(
            f"The bot server takes {', '.join(map(str, modes)) or 'nothing'}, "
            f"not {self._state_mode}."
        )

  def _timeout(self) -> float:
    """How long to wait for a reply: within the move budget."""
    budget = move_budget.seconds()
    limit = (
        budget - _MARGIN_SECONDS
        if budget > 2 * _MARGIN_SECONDS
        else (budget / 2)
    )
    if self._timeout_seconds > 0:
      return min(self._timeout_seconds, limit)
    return limit

  def request(self, state: pyspiel.State, seat: int) -> tuple[str, bytes]:
    """Returns the path and JSON body of the request for `seat` at `state`."""
    game_type = self._game.get_type()
    legal = [int(action) for action in state.legal_actions(seat)]
    seed = controller_registry.choice_seed(self._seed, state)
    encoding = {"num_distinct_actions": self._game.num_distinct_actions()}
    body: dict[str, contract.JsonValue]
    if self._state_mode == FULL_STATE:
      path = STEP_PATH
      body = {
          "game": str(self._game),
          "state": state.serialize(),
          "history": [int(action) for action in state.history()],
          "seat": seat,
          "legal_actions": legal,
          "seed": seed,
          "game_encoding": encoding,
          "config": self._bot_config,
      }
    else:
      path = OBSERVE_PATH
      observation: dict[str, contract.JsonValue] = {
          "information_state_string": _information_state_string(state, seat),
          "seat": seat,
          "legal_actions": legal,
          "seed": seed,
      }
      if self._extras:
        observation["observation_string"] = (
            str(state.observation_string(seat))
            if game_type.provides_observation_string
            else None
        )
        observation["legal_action_labels"] = [
            str(state.action_to_string(seat, action)) for action in legal
        ]
        observation["history"] = _seat_history(
            state, seat, information.hides_information(game_type)
        )
      if self._tensors:
        observation["observation_tensor"] = (
            [float(x) for x in state.observation_tensor(seat)]
            if game_type.provides_observation_tensor
            else None
        )
        observation["information_state_tensor"] = (
            [float(x) for x in state.information_state_tensor(seat)]
            if game_type.provides_information_state_tensor
            and game_type.short_name
            not in information.HIDDEN_DESPITE_DECLARATION
            else None
        )
      body = {
          "game": str(self._game),
          "observation": observation,
          "config": self._bot_config,
          "game_encoding": encoding,
      }
    return path, json.dumps(body).encode()

  def choose_action(self, state: pyspiel.State, seat: int) -> int:
    """Asks the bot server for `seat`'s action at `state`.

    Raises:
      RemoteBotRefusedError: If the bot server refuses.
      RemoteBotUnreachableError: If it cannot be reached, or does not answer
        within the timeout.
      RemoteBotReplyError: If its reply is not the protocol's, or its action
        is not legal.
    """
    self.last_notice = None
    path, body = self.request(state, seat)
    status, reply = _request_json(self._url + path, body, self._timeout())
    if status in remote_bot_http.REFUSAL_STATUSES:
      raise RemoteBotRefusedError(
          f"The bot server refused (HTTP {status}):"
          f" {remote_bot_http.refusal_text(reply)}."
      )
    if not 200 <= status < 300:
      raise RemoteBotUnreachableError(f"The bot server answered HTTP {status}.")
    action, self.last_notice = remote_bot_http.parse_reply(
        reply, state.legal_actions(seat)
    )
    return action


def _build(
    game: pyspiel.Game, config: Mapping[str, contract.ParameterValue], seed: int
) -> controller_registry.Controller:
  """Builds a remote bot, checking its bot server answers.

  Raises:
    controller_registry.UnavailableError: If the bot server cannot be
      reached, refuses, or says it cannot play the game this way. The
      message names the URL, which the seat's settings show every viewer,
      and what the server answered.
  """
  url = str(config["url"])
  bot = RemoteBot(
      game,
      url=url,
      state_mode=str(config["state"]),
      bot_config=remote_bot_config.seat_bot_config(config),
      extras=bool(config["extras"]),
      tensors=bool(config["tensors"]),
      timeout_seconds=float(config["timeout_seconds"]),
      seed=seed,
  )
  try:
    bot.check()
  except (RemoteBotUnreachableError, RemoteBotReplyError) as e:
    raise controller_registry.UnavailableError(
        f"The bot server at {url} is not ready: {e}"
    ) from e
  return bot


def kind() -> controller_registry.Kind:
  """Returns the remote bot as it is registered."""
  return controller_registry.Kind(
      "Remote bot",
      _settings(),
      # What it reads is its seat's `state` setting, not its kind's.
      reads_full_state=False,
      build=_build,
      config_refusal=config_refusal,
      config_reads_full_state=reads_full_state,
  )
