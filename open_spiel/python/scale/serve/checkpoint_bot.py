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

"""checkpoint_bot: a trained artifact as a bot that plays pyspiel states.

A game GUI or a match runner holds a pyspiel state and wants the move a
trained checkpoint makes for the seat to move:

  # step None: the latest step
  bot = checkpoint_bot("/ckpts/ttt", step=150)
  # ValueError unless the artifact plays `game`
  bot.check_serves(game)
  # compiles every history length (no move compiles)
  bot.warmup(range(bot.num_players))
  action = bot.choose_action(state, seat)
  # {legal action: probability} of that move
  bot.last_probs

The bot is a thin shell over PolicyServer (and so HostAgent): the move is
the move the artifact makes in evaluation, decided by the greedy rule
(deterministic=True: the first legal action of maximal logit) or by sampling
at `temperature` (deterministic=False). It replays the state's history into
the artifact's env and the policy sees only the SeatSteps of the seat it
plays, the env's view of that seat (its own private information, the public
actions, nothing of another seat's private information), as in evaluation.
Envs that hide moves refuse stateful policies (core.types.check_policy_
admitted), and so does the bot.

Rules differences: on a rare node the env's rules and the installed
OpenSpiel's disagree (for gin rummy, some sequential layoffs and an Oklahoma
knock card of 0). The bot never refuses such a move mid-hand: it plays the
policy renormalised over the actions both allow, or, when the env cannot
replay the hand at all, OpenSpiel's first legal action
(server.PolicyServer.decide, repair). It is never silent: each such move is
logged as a WARNING, counted in `stats` ("legal_mismatch", "diverged") and
described in `last_notice` (None after an ordinary move), for a GUI to show.
The GUI's controller contract (version 8) shows a controller's notice to
that seat's viewer and to omniscient viewers, and fails the seat on a notice
that is blank or over NOTICE_MAX_LENGTH characters, so `last_notice` is the
seat's notice (seat_notice): a legal-set notice names only the seat's own
legal actions and the move played; a divergence notice says the env could
not replay the hand and names the move played, while the replay's error,
which can name another seat's action or a chance outcome, stays in the log
and in the Reply; each is cut to NOTICE_MAX_LENGTH.

Randomness: the key of a decision is derived from `seed`, the seat and the
number of plies in the state's history (server.seat_key, then the Arena's
ply keys), so a sampling bot draws afresh at every move of a game, and
replaying the same game with the same seed reproduces every move.

`path` is a local directory, a gs:// URL or a public Cloud Storage HTTPS
URL (ckpt.public.ENDPOINT/<bucket>/<path>) holding the artifact's steps (a
public URL, and gs:// without Google credentials, is read anonymously:
ckpt.public); the step defaults to the largest step there.
"""

from __future__ import annotations

import itertools
from typing import Any, Iterable, Mapping, Optional

from absl import logging
import jax
import numpy as np

from open_spiel.python.scale.ckpt import artifact as artifact_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.serve import games
from open_spiel.python.scale.serve import server as server_lib

# The most combinations of a game's parameter values game_string tries when
# the artifact's variant is not the game's default one.
_MAX_VARIANTS = 4096
# Plies a warmup playout makes before it gives up on reaching a terminal.
_MAX_WARMUP_PLIES = 100_000

#: The longest notice contract 8 carries (the GUI's NOTICE_MAX_LENGTH).
NOTICE_MAX_LENGTH = 200
#: The notice of a move the env could not replay (decider "fallback"): what
#: the seat may see, the replay's error left to the log (module doc).
DIVERGED_NOTICE = (
    "engine/agent rules differed earlier in this hand; playing the first "
    "engine-legal move, {move}"
)


def clip_notice(notice: Optional[str]) -> Optional[str]:
  """`notice` as contract 8 takes it: non-blank, at most NOTICE_MAX_LENGTH.

  A blank notice is None; a longer one is cut, with a trailing "...".

  Args:
    notice: the bot's notice, or None.

  Returns:
    The notice a seat is shown, or None.
  """
  if notice is None or not notice.strip():
    return None
  if len(notice) <= NOTICE_MAX_LENGTH:
    return notice
  return notice[: NOTICE_MAX_LENGTH - 3] + "..."


def seat_notice(reply: Any, state: Any, seat: int) -> Optional[str]:
  """The notice a seat is shown for the server's `reply` (module doc).

  A legal-set notice names only the seat's own actions and is sent as it
  is; a divergence notice (decider "fallback") carries the replay's error,
  which can name what the seat may not see, so DIVERGED_NOTICE is sent in
  its place. Either is then clipped (clip_notice).

  Args:
    reply: the server's Reply (action, decider, notice).
    state: the state the move was chosen at.
    seat: the seat that moved.

  Returns:
    The reply's notice, or None.
  """
  notice = reply.notice
  if notice is not None and reply.decider == "fallback":
    notice = DIVERGED_NOTICE.format(
        move=state.action_to_string(seat, int(reply.action))
    )
  return clip_notice(notice)


def split_checkpoint(spec: str) -> tuple[str, Optional[int]]:
  """The (<dir or URL>, step) of "<dir or URL>@<step>".

  Args:
    spec: "<dir or URL>@<step>" or "<dir or URL>".

  Returns:
    (<dir or URL>, step), or (<dir or URL>, None) without a step.
  """
  head, sep, tail = spec.rpartition("@")
  if sep and tail.isdigit():
    return head, int(tail)
  return spec, None


def _format_value(v: Any) -> str:
  """A parameter value as pyspiel game strings spell it.

  Args:
    v: The value.

  Returns:
    "true"/"false" for a bool, else str(v).
  """
  if isinstance(v, bool):
    return "true" if v else "false"
  return str(v)


def format_game_string(name: str, params: Mapping[str, Any]) -> str:
  """Pyspiel's game string of `name` with `params` ("name(k=v,...)").

  Args:
    name: the pyspiel game name.
    params: the game parameters.

  Returns:
    The game string.
  """
  if not params:
    return name
  args = ",".join(f"{k}={_format_value(params[k])}" for k in sorted(params))
  return f"{name}({args})"


def installed_parameter_specification(name: str) -> dict[str, Any]:
  """{parameter: default} of the installed pyspiel's game `name`.

  Args:
    name: the pyspiel game name.

  Returns:
    The parameter specification.
  """
  # Lazy: pyspiel is optional (the serve package imports without it).
  import pyspiel  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  return dict(pyspiel.load_game(name).get_type().parameter_specification)


def game_string_of(
    manifest: types.Manifest,
    parameter_specification: Optional[Mapping[str, Any]] = None,
) -> str:
  """The pyspiel game string of the variant an artifact was trained on.

  The game's default variant (its short name) when the artifact plays it;
  otherwise the first combination of the game's finitely-valued served
  parameters whose env is the artifact's training env, written for the
  installed pyspiel (games.game_string_params: parameters at pyspiel's
  default are dropped).

  Args:
    manifest: the artifact's manifest.
    parameter_specification: {parameter: default} of the pyspiel game to
      write the string for; None reads the installed pyspiel's.

  Returns:
    The game string.

  Raises:
    ValueError: the game is not served, no pyspiel game string plays the
      artifact's training env, or the installed pyspiel's game lacks a
      parameter of that string.
  """
  name = manifest.game
  if name not in games.served_games():
    raise ValueError(
        f"no pyspiel game plays the artifact's game {name!r}; served: "
        f"{games.served_games()}"
    )
  if manifest.env_spec is None:
    return name
  trained = env_registry.EnvSpec.from_json(manifest.env_spec)
  if env_registry.covers(trained, games.env_spec_for(name, {})):
    return name
  allowed = env_registry.game(name).serving.pyspiel_params
  finite = sorted(
      (k, tuple(v)) for k, v in allowed.items() if v is not games.ANY
  )
  keys = [k for k, _ in finite]
  for n, values in enumerate(itertools.product(*(v for _, v in finite))):
    if n >= _MAX_VARIANTS:
      break
    params = dict(zip(keys, values))
    if games.env_spec_for(name, params) != trained:
      continue
    if parameter_specification is None:
      parameter_specification = installed_parameter_specification(name)
    return format_game_string(
        name, games.game_string_params(name, params, parameter_specification)
    )
  raise ValueError(
      f"no pyspiel game string of {name!r} plays the artifact's training env "
      f"{trained.to_json()}"
  )


class CheckpointBot:
  """One artifact playing pyspiel states (module doc).

  Build with checkpoint_bot.

  Attributes:
    path: the artifact's checkpoint dir.
    step: its step.
    rule: the DecisionRule it plays.
    seed: the seed of its sampling keys.
    manifest: the artifact's manifest.
    num_players: the players of its game.
    game_string: the game string for the installed pyspiel.
    last_probs: None, or the action probabilities of the last choose_action.
    last_notice: None, or the seat's notice about a rules difference the
      last move played through (module doc, "Rules differences").
    stats: {"legal_mismatch": n, "diverged": n}, this bot's moves that
      played through a rules difference (module doc).
  """

  def __init__(
      self,
      server: server_lib.PolicyServer,
      path: str,
      step: int,
      rule: types.DecisionRule,
      seed: int,
  ):
    self._server = server
    self.path = path
    self.step = int(step)
    self.rule = rule
    self.seed = int(seed)
    art = server.artifact(path, self.step)
    self.manifest = art.manifest
    self.num_players = int(art.policy.spec.num_players)
    self.game_string: str = game_string_of(art.manifest)
    self._game = None  # the pyspiel game last accepted by check_serves
    self.last_probs: Optional[dict[int, float]] = None
    self.last_notice: Optional[str] = None
    self.stats = {"legal_mismatch": 0, "diverged": 0}

  def __repr__(self) -> str:
    return (
        f"CheckpointBot({self.path}@{self.step}, game={self.game_string!r}, "
        f"rule={self.rule.kind}/{self.rule.temperature}, seed={self.seed})"
    )

  def check_serves(self, game: Any) -> None:
    """Checks that the artifact plays pyspiel `game` with its parameters.

    Args:
      game: the pyspiel game.

    Raises:
      ValueError: the artifact plays another game, the game's parameters are
        a variant its training env does not cover (or no env plays them), the
        env refuses the artifact's policy (a stateful policy in a game that
        hides moves), or the player counts differ.
    """
    name = game.get_type().short_name
    params = dict(game.get_parameters())
    if name != self.manifest.game:
      raise ValueError(
          f"the checkpoint plays {self.manifest.game!r}, not {name!r}"
      )
    if int(game.num_players()) != self.num_players:
      raise ValueError(
          f"the checkpoint plays {self.num_players} players; {name} has "
          f"{game.num_players()}"
      )
    try:
      for seat in range(self.num_players):
        self._server.agent(
            self.path, self.step, seat, name, params, rule=self.rule
        )
    except KeyError as e:  # games.env_for: no env plays the game
      raise ValueError(str(e)) from e
    self._game = game

  def _warmup_game(self) -> Any:
    """The pyspiel game the bot plays, loaded and checked once.

    Returns:
      The pyspiel game.
    """
    if self._game is None:
      # Lazy: pyspiel is optional (the serve package imports without it).
      import pyspiel  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

      game = pyspiel.load_game(self.game_string)
      self.check_serves(game)
    return self._game

  def warmup(
      self,
      seats: Iterable[int],
      *,
      games_per_seat: int = 1,
      max_plies: Optional[int] = None,
  ) -> None:
    """Loads the artifact and compiles its decisions for `seats`.

    For each seat, compiles every call a decision makes on a history of up
    to `max_plies` plies (default: the game's max_game_length() + 1, every
    decision node plus the terminal one): the replay's length buckets and
    the policy (serve.host_agent, "Compiles"), so no move of a game within
    that length compiles, whatever its length. Then plays `games_per_seat`
    playouts of the game (the game last accepted by check_serves, else
    game_string's) to the end, the bot moving for the seat and uniform
    random moves for the others, which runs the rest of the request path
    once. A stateful policy still compiles once per history length. When
    every seat's agent (shared through the PolicyServer) was already warmed up
    to max_plies, by an earlier bot in this process, nothing is compiled and
    no playout is played.

    Args:
      seats: the seats to compile.
      games_per_seat: the playouts per seat.
      max_plies: the longest history compiled (None: as above).

    Raises:
      ValueError: a seat out of range, max_plies < 1, or check_serves refuses
        the game.
      RuntimeError: a playout did not end.
    """
    game = self._warmup_game()
    seats = [int(s) for s in seats]
    for seat in seats:
      if not 0 <= seat < self.num_players:
        raise ValueError(f"seat {seat} is not in [0, {self.num_players})")
    if max_plies is None:
      max_plies = int(game.max_game_length()) + 1
    if int(max_plies) < 1:
      raise ValueError(f"max_plies must be >= 1, got {max_plies}")
    name, params = game.get_type().short_name, dict(game.get_parameters())
    warm = True  # every seat's agent was warmed before (an earlier bot)
    for seat in seats:
      agent = self._server.agent(
          self.path, self.step, seat, name, params, rule=self.rule
      )
      warm = warm and getattr(agent, "warmed_plies", 0) >= int(max_plies)
      agent.warmup(int(max_plies))
      # the request's key
      jax.block_until_ready(server_lib.seat_key(self.seed, seat))
    if warm and seats:
      # The PolicyServer's agents are shared: an earlier bot's warmup compiled
      # them and played its playouts through them, in this process.
      logging.info(
          "%r: seats %s already warm (histories up to %d plies)",
          self,
          seats,
          max_plies,
      )
      self.last_probs = None
      self.last_notice = None
      return
    rng = np.random.default_rng(self.seed)
    for seat in seats:
      for _ in range(int(games_per_seat)):
        state = game.new_initial_state()
        for _ in range(_MAX_WARMUP_PLIES):
          if state.is_terminal():
            break
          if state.is_chance_node():
            actions, probs = zip(*state.chance_outcomes())
            state.apply_action(
                int(rng.choice(actions, p=np.asarray(probs) / sum(probs)))
            )
            continue
          mover = int(state.current_player())
          if mover == seat:
            state.apply_action(self.choose_action(state, seat))
          else:
            state.apply_action(int(rng.choice(state.legal_actions())))
        else:
          raise RuntimeError(
              f"a warmup playout of {self.game_string} did not end in "
              f"{_MAX_WARMUP_PLIES} plies"
          )
    logging.info(
        "%r: warmed up seats %s (histories up to %d plies)",
        self,
        seats,
        max_plies,
    )
    self.last_probs = None
    self.last_notice = None

  def reply(self, state: Any, seat: int) -> server_lib.Reply:
    """The full Reply (action, probs, value, decider, artifact) of a decision.

    Sets last_probs to the decision's {legal action: probability}, and
    last_notice and stats (module doc, "Rules differences").

    Args:
      state: the pyspiel state.
      seat: the seat to move.

    Returns:
      The server's Reply.

    Raises:
      ValueError: `seat` is not to move.
    """
    if int(state.current_player()) != int(seat):
      raise ValueError(
          f"seat {seat} is not to move; player {state.current_player()} is"
      )
    reply = self._server.act(
        state,
        checkpoint_dir=self.path,
        step=self.step,
        seed=self.seed,
        rule=self.rule,
        repair=True,
    )
    self.last_probs = dict(reply.probs)
    self.last_notice = seat_notice(reply, state, seat)
    if reply.notice is not None:
      self.stats[
          "diverged" if reply.decider == "fallback" else "legal_mismatch"
      ] += 1
    return reply

  def choose_action(self, state: Any, seat: int) -> int:
    """The legal action the artifact plays for `seat`, who is to move.

    Sets last_probs to the decision's {legal action: probability}.

    Args:
      state: the pyspiel state.
      seat: the seat to move.

    Returns:
      The action.

    Raises:
      ValueError: `seat` is not to move, the state is terminal or a chance
        node, or the artifact does not play the state's game.
    """
    return int(self.reply(state, seat).action)


def checkpoint_bot(
    path: str,
    *,
    step: Optional[int] = None,
    deterministic: bool = True,
    temperature: float = 1.0,
    seed: int = 0,
    server: Optional[server_lib.PolicyServer] = None,
) -> CheckpointBot:
  """A bot playing the artifact at path@step (module doc).

  Args:
    path: the artifact's directory: local, gs:// or a public HTTPS URL.
    step: the artifact step; None takes the largest step under `path`.
    deterministic: True decides greedily (the first legal action of maximal
      logit); False samples from the policy's distribution at `temperature`.
    temperature: > 0, the sampling temperature (unused when deterministic).
    seed: seeds every sampled decision, with the seat and the ply.
    server: the PolicyServer to load and serve through (one shared server
      keeps one copy of an artifact for many bots); None makes a new one.

  Returns:
    The CheckpointBot.

  Raises:
    FileNotFoundError: no artifact at path (or at path@step).
    ValueError: a sampling temperature that is not positive and finite, or an
      artifact of a game no pyspiel game string plays.
  """
  # an entry point (package doc): it loads and serves artifacts
  x64.enable_x64()
  if deterministic:
    rule = types.ARGMAX
  else:
    t = float(temperature)
    if not (t > 0.0 and np.isfinite(t)):
      raise ValueError(
          "a sampling temperature must be positive and finite, got "
          f"{temperature}"
      )
    rule = types.DecisionRule("sample", t)
  if step is None:
    step = artifact_lib.latest_step(path)
  if server is None:
    server = server_lib.PolicyServer()
  return CheckpointBot(server, path.rstrip("/"), int(step), rule, int(seed))
