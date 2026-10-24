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

r"""Play two checkpoints against each other and save every game.

  python -m open_spiel.python.scale.examples.play_checkpoints \
      --a=/tmp/ttt@20 --b=uniform --games=1024 --out=/tmp/ttt_vs_uniform
  python -m open_spiel.python.scale.examples.play_checkpoints \
      --a=/tmp/ttt --b=/tmp/ttt --probs --games=8 --out=/tmp/ttt_self

What it does:
  * loads each side with the public loader (`ckpt.artifact.load_artifact`,
    which rebuilds the policy from its manifest and checks the params against
    the manifest's digest). A side is an artifact, `<dir>@<step>` (or `<dir>`
    for its latest step), or a built-in opponent: `uniform` (uniform random
    over the legal actions) or `arch:<name>`, a registered scripted policy of
    the game (for example `arch:gin_rummy.rule_bot`). At least one side is an
    artifact; the game and its rules come from the artifacts' manifests
    (their env spec). Two artifacts must record the same env spec and rules
    hash, or the script refuses;
  * plays --games games in seat-swapped pairs on duplicate deals: deal i is
    played twice with the seats swapped (a in seat 0, then in seat 1) and the
    same chance outcomes, through the Arena (`core.arena`), in batches, on
    every local device (CPU or GPU). The keys are the Evaluator's, so the
    payoffs equal `eval.match.head_to_head(env, a, b, games // 2, key)`;
  * prints a's mean payoff with its standard error, overall and per seat
    (the unit is the deal pair, so luck of the deal cancels within a pair);
  * writes every game to --out (the format below), and, when pyspiel is
    installed, replays every saved game in pyspiel and checks each move is
    legal and the returns are the same. Without pyspiel (the pure-JAX
    install) everything else runs and the script says the replay was
    skipped.

Decision rules: each artifact plays its manifest's default rule (what its
trainer recorded, usually greedy "argmax"), `uniform` samples and a scripted
policy plays greedily; --rule=argmax or --rule=sample overrides both sides.

The output format ("game_trajectories", version 1). --out holds two files:

  summary.json        {"format": "game_trajectories", "format_version": 1,
                       "game", "env_spec", "rules_sha", "pyspiel_params",
                       "entrants": {"a": {...}, "b": {...}} (spec, name,
                       rule, and for an artifact its path, step, run_id,
                       params_sha256), "games", "seed", "result": {mean,
                       se, lo, hi of a's payoff overall and per seat},
                       "num_records"}
  trajectories.jsonl  one JSON object per line, one line per game:
    "index":       the game's index (0 .. games - 1)
    "deal":        its deal pair (games 2i and 2i + 1 share deal i)
    "game_string": a pyspiel game string, pyspiel.load_game(game_string)
                   loads the game this record was played in; null when the
                   game has no pyspiel counterpart here or its env's
                   histories are not pyspiel's, and then "history" is in
                   the env's ids
    "seats":       ["a" or "b" for seat 0, the same for seat 1]
    "history":     the game's pyspiel action history: chance outcomes and
                   player actions interleaved, OpenSpiel action ids, so
                   state.apply_action(a) for each a from
                   game.new_initial_state() replays it exactly
    "actors":      per history entry, the seat that acted (-1 for chance)
    "returns":     [seat 0's return, seat 1's return]
    "env_history": the env's own raw history of the game (equal to
                   "history" whenever "game_string" is not null)
    "policy":      with --probs only: per history entry, null for chance,
                   else the acting policy's distribution over the legal
                   actions, [[action, probability], ...] (the softmax at the
                   rule's temperature; under "argmax" the action played is
                   the first most probable one)

`read_trajectories(out_dir)` reads it back and `replay_in_pyspiel(record)`
replays one record in pyspiel; they use only json and pyspiel.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import os
import time
from typing import Any, Iterator, Optional, Sequence

from absl import app
from absl import flags
from absl import logging
from etils import epath
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.ckpt import artifact
from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import arena_engine
from open_spiel.python.scale.core import chance
from open_spiel.python.scale.core import decide
from open_spiel.python.scale.core import host_arena
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.eval import match
from open_spiel.python.scale.eval import payoff
from open_spiel.python.scale.policies import registry as policy_registry
from open_spiel.python.scale.policies import routed
from open_spiel.python.scale.policies import uniform
from open_spiel.python.scale.serve import games as served

FORMAT = "game_trajectories"
FORMAT_VERSION = 1
SUMMARY = "summary.json"
TRAJECTORIES = "trajectories.jsonl"
UNIFORM = "uniform"
ARCH_PREFIX = "arch:"


def _define_flags() -> dict[str, flags.FlagHolder]:
  """The command line's flags, defined only when the module runs as a script.

  The module is also a library (read_trajectories, replay_in_pyspiel, run):
  importing it in a process that has another entry point's flags (the
  trainer's --seed) must not define a flag twice.

  Returns:
    The flag holders by name.
  """
  return dict(
      a=flags.DEFINE_string(
          "a",
          None,
          "side a: <artifact dir>@<step>, <artifact dir> (its"
          " latest step), uniform, or arch:<scripted policy name>",
      ),
      b=flags.DEFINE_string("b", None, "side b, as --a"),
      games=flags.DEFINE_integer(
          "games", 1024, "games to play, an even number (seat-swapped pairs)"
      ),
      seed=flags.DEFINE_integer(
          "seed", 0, "seed of the deals and of sampled play"
      ),
      out=flags.DEFINE_string(
          "out", None, "directory for summary.json and trajectories.jsonl"
      ),
      rule=flags.DEFINE_enum(
          "rule",
          "default",
          ["default", "argmax", "sample"],
          "decision rule of both sides (default: each side's own)",
      ),
      probs=flags.DEFINE_bool(
          "probs", False, "also save each decision's policy distribution"
      ),
      pyspiel_params=flags.DEFINE_string(
          "pyspiel_params",
          "",
          "pyspiel game parameters of the artifacts' variant, k=v,..."
          " (needed only when the env was not trained on pyspiel's defaults, "
          "e.g. oklahoma=true)",
      ),
      scripted_endgame=flags.DEFINE_bool(
          "scripted_endgame",
          False,
          "route the game's scripted final phase (GameSpec.endgame,"
          " e.g. gin_rummy's knock and layoff) around each artifact's policy",
      ),
      max_rows=flags.DEFINE_integer(
          "max_rows",
          0,
          "games per Arena batch (0: as many as fit a 1 GiB trajectory budget)",
      ),
      verify=flags.DEFINE_bool(
          "verify_pyspiel",
          True,
          "replay every saved game in pyspiel when pyspiel is installed",
      ),
  )


# -- the format: reading and replaying (json and pyspiel only) -----------------


def read_trajectories(
    out_dir: str,
) -> tuple[dict[str, Any], Iterator[dict[str, Any]]]:
  """Reads a directory this script wrote.

  Args:
    out_dir: The directory (the script's --out).

  Returns:
    (summary, records): summary.json's contents and an iterator over the
    trajectories.jsonl records, one dict per game.

  Raises:
    ValueError: the directory holds another format or version.
  """
  with open(os.path.join(out_dir, SUMMARY), encoding="utf-8") as f:
    summary = json.load(f)
  if (summary.get("format"), summary.get("format_version")) != (
      FORMAT,
      FORMAT_VERSION,
  ):
    raise ValueError(
        f"{out_dir}: format {summary.get('format')!r} version "
        f"{summary.get('format_version')!r}; this reader reads {FORMAT!r} "
        f"version {FORMAT_VERSION}"
    )

  def records() -> Iterator[dict[str, Any]]:
    """Yields the records of trajectories.jsonl in order.

    Yields:
      One dict per game.
    """
    with open(os.path.join(out_dir, TRAJECTORIES), encoding="utf-8") as f:
      for line in f:
        yield json.loads(line)

  return summary, records()


def replay_in_pyspiel(record: dict[str, Any], atol: float = 1e-4):
  """Replays a record in pyspiel.

  Args:
    record: One trajectories.jsonl record.
    atol: The largest allowed difference between pyspiel's returns and the
      record's.

  Returns:
    The terminal pyspiel state.

  Raises:
    ValueError: the record has no pyspiel game, a move is not legal where it
      was played, an actor differs, the game does not end where the record
      does, or the returns differ by more than `atol`.
  """
  # Lazy: pyspiel is an optional dependency, needed only to replay.
  import pyspiel  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  if record["game_string"] is None:
    raise ValueError(f"game {record['index']} has no pyspiel game string")
  state = pyspiel.load_game(record["game_string"]).new_initial_state()
  for i, (a, who) in enumerate(zip(record["history"], record["actors"])):
    if state.is_terminal():
      raise ValueError(
          f"game {record['index']}: pyspiel ended before entry {i}"
      )
    if state.is_chance_node():
      legal = [o for o, _ in state.chance_outcomes()]
      actor = -1
    else:
      legal = state.legal_actions()
      actor = state.current_player()
    if actor != who or a not in legal:
      raise ValueError(
          f"game {record['index']} entry {i}: action {a} by {who}; pyspiel has "
          f"actor {actor} with legal actions {legal}"
      )
    state.apply_action(a)
  if not state.is_terminal():
    raise ValueError(f"game {record['index']}: pyspiel's game has not ended")
  if not np.allclose(state.returns(), record["returns"], atol=atol, rtol=0):
    raise ValueError(
        f"game {record['index']}: pyspiel returns {state.returns()}, the "
        f"record has {record['returns']}"
    )
  return state


def game_string(name: str, params: dict[str, Any]) -> str:
  """Returns pyspiel's game string of `name` with `params`.

  Args:
    name: The pyspiel game name.
    params: The game parameters; empty for the defaults.

  Returns:
    "name(k=v,...)" with the keys sorted, or `name` when `params` is empty.
  """
  if not params:
    return name

  def fmt(v):
    """Formats one parameter value as pyspiel parses it.

    Args:
      v: The value.

    Returns:
      "true"/"false" for a bool, else str(v).
    """
    if isinstance(v, bool):
      return "true" if v else "false"
    return str(v)

  return f"{name}({','.join(f'{k}={fmt(params[k])}' for k in sorted(params))})"


def parse_params(text: str) -> dict[str, Any]:
  """Parses k=v,... into a dict of JSON scalars.

  Args:
    text: The --pyspiel_params flag value.

  Returns:
    {k: v}: true/false and numbers as JSON values, anything else as strings.

  Raises:
    app.UsageError: a part is not k=v.
  """
  out = {}
  for part in filter(None, (p.strip() for p in text.split(","))):
    if "=" not in part:
      raise app.UsageError(f"--pyspiel_params: {part!r} is not k=v")
    k, v = part.split("=", 1)
    try:
      out[k.strip()] = json.loads(v.strip())
    except ValueError:
      out[k.strip()] = v.strip()
  return out


def pyspiel_params_for(
    env, env_spec, params: dict[str, Any]
) -> Optional[dict[str, Any]]:
  """Returns the pyspiel parameters whose game the env's histories replay in.

  None when the game has no pyspiel counterpart here, or its env's raw
  histories differ from pyspiel's (GameSpec.pyspiel_history_to_raw is set).

  Args:
    env: The env the games are played in.
    env_spec: The artifacts' env spec.
    params: The --pyspiel_params parameters.

  Returns:
    The parameters, or None.

  Raises:
    ValueError: `params` play another env spec than the artifacts'.
  """
  if env_registry.game(env.name).pyspiel_history_to_raw is not None:
    return None
  try:
    spec = served.env_spec_for(env.name, params)
  except KeyError:
    return None
  if spec != env_spec:
    raise ValueError(
        f"{env.name}: pyspiel parameters {params} play {spec.to_json()}, the "
        f"artifacts {env_spec.to_json()}; pass --pyspiel_params for their "
        "variant"
    )
  return dict(params)


# -- sides ---------------------------------------------------------------------


@dataclasses.dataclass
class Side:
  """One side of the match: its entrant and how it was specified.

  Attributes:
    entrant: eval.match.Entrant.
    info: summary.json's record of the side.
    manifest: The artifact's Manifest, None for a built-in.
    spec: The side as given on the command line.
  """

  spec: str
  entrant: Any
  info: dict[str, Any]
  manifest: Any = None


def split_artifact(spec: str) -> tuple[str, Optional[int]]:
  """Splits an artifact spec into its directory and step.

  Args:
    spec: "<dir>@<step>" or "<dir>".

  Returns:
    (<dir>, step), or (<dir>, None) when no step is named.
  """
  head, sep, tail = spec.rpartition("@")
  if sep and tail.isdigit():
    return head, int(tail)
  return spec, None


def latest_step(path: str) -> int:
  """Returns the largest step under `path` that holds an artifact.

  Args:
    path: The artifact directory.

  Returns:
    The step.

  Raises:
    FileNotFoundError: no step under `path` holds an artifact.
  """
  root = epath.Path(artifact.norm_path(path))
  names = [c.name for c in root.iterdir()] if root.exists() else []
  steps = sorted(
      int(n) for n in names if n.isdigit() and artifact.exists(path, int(n))
  )
  if not steps:
    raise FileNotFoundError(
        f"no artifact step under {path}; name one as {path}@<step>"
    )
  return steps[-1]


def load_side(spec: str, rule: Optional[Any]) -> Side:
  """Loads and verifies the artifact side `spec`.

  Args:
    spec: "<dir>@<step>" or "<dir>" (its latest step).
    rule: The decision rule to play, or None for the manifest's default.

  Returns:
    The side.
  """
  path, step = split_artifact(spec)
  if step is None:
    step = latest_step(path)
  art = artifact.load_artifact(path, step, verify=True)
  m = art.manifest
  name = (
      f"{m.run_id}@{m.step}" if m.run_id is not None else f"{art.path}@{m.step}"
  )
  e = match.Entrant.from_artifact(art, rule=rule, name=name)
  info = dict(
      spec=spec,
      name=name,
      path=art.path,
      step=int(m.step),
      run_id=m.run_id,
      params_sha256=m.params_sha256,
      arch_name=m.arch_name,
      game=m.game,
      rule=dict(kind=e.rule.kind, temperature=e.rule.temperature),
  )
  return Side(spec, e, info, m)


def builtin_side(spec: str, env, rule: Optional[Any]) -> Side:
  """Returns the built-in side `spec` for `env`.

  Args:
    spec: "uniform" or a registered scripted policy (module doc).
    env: The env the games are played in.
    rule: The decision rule to play, or None for the side's default.

  Returns:
    The side.

  Raises:
    app.UsageError: `spec` names no registered scripted policy of the game.
  """
  if spec == UNIFORM:
    pol, default = (
        uniform.UniformPolicy(env.name, env.num_actions, env.num_players),
        types.SAMPLE,
    )
  else:
    arch = spec[len(ARCH_PREFIX) :]
    env_registry.load_extras(env.name)
    try:
      pol = policy_registry.build(arch, {})
    except (KeyError, TypeError) as e:
      mine = [
          n
          for n in policy_registry.registered()
          if n.startswith(f"{env.name}.")
      ]
      raise app.UsageError(
          f"{spec}: no scripted policy {arch!r} that builds without kwargs "
          f"({e}); {env.name}'s registered policies: {mine}"
      ) from e
    if pol.spec.game != env.name:
      raise app.UsageError(
          f"{spec} plays {pol.spec.game!r}, the artifacts {env.name!r}"
      )
    default = types.ARGMAX
  e = match.Entrant(spec, pol, pol.init_params(None), rule or default)
  return Side(
      spec,
      e,
      dict(
          spec=spec,
          name=spec,
          arch_name=pol.spec.arch_name,
          rule=dict(kind=e.rule.kind, temperature=e.rule.temperature),
      ),
  )


def match_env(sides: Sequence[Side]):
  """Returns the env every artifact side was trained on, checked to be one.

  Args:
    sides: The sides; at least one is an artifact.

  Returns:
    (env, env spec).

  Raises:
    ValueError: two artifacts record different env specs or rules hashes.
  """
  arts = [s for s in sides if s.manifest is not None]
  assert arts, "at least one side is an artifact"
  envs = [env_registry.training_env(s.manifest) for s in arts]
  specs = [env_registry.spec_of(e) for e in envs]
  for s, sp in zip(arts[1:], specs[1:]):
    if sp != specs[0]:
      raise ValueError(
          f"the artifacts play different games: {arts[0].spec} was trained on "
          f"{specs[0].to_json()}, {s.spec} on {sp.to_json()}"
      )
    if s.manifest.rules_sha != arts[0].manifest.rules_sha:
      raise ValueError(
          f"the artifacts record different rules: {arts[0].spec} has rules_sha "
          f"{arts[0].manifest.rules_sha}, {s.spec} has {s.manifest.rules_sha}"
      )
  env = envs[0]
  now = env_registry.rules_sha(env)
  if now != arts[0].manifest.rules_sha:
    # The hash covers the source of the game's module, so any edit to it moves
    # the hash: the play below is valid for the rules this code implements.
    logging.warning(
        "the artifacts record rules_sha %s, this code's %s is %s: the game's "
        "source changed since training",
        arts[0].manifest.rules_sha,
        env.name,
        now,
    )
    print(
        f"note: {env.name}'s source differs from the one the artifacts were "
        "trained on (rules_sha); the games are played under this code's rules",
        flush=True,
    )
  return env, specs[0]


def with_endgame(side: Side, env) -> Side:
  """Routes the game's scripted final phase around the side's policy.

  Args:
    side: The side; updated in place.
    env: The env the games are played in.

  Returns:
    `side`, with the routed policy and params.

  Raises:
    app.UsageError: the game has no scripted final phase.
  """
  endgame = env_registry.game(env.name).endgame
  if endgame is None:
    raise app.UsageError(
        f"--scripted_endgame: {env.name} has no scripted final phase"
    )
  pol, params = routed.routed_pair(
      side.entrant.policy, side.entrant.params, endgame
  )
  side.entrant = dataclasses.replace(side.entrant, policy=pol, params=params)
  side.info["scripted_endgame"] = endgame
  return side


# -- play ----------------------------------------------------------------------


@dataclasses.dataclass
class Played:
  """Every game's log: rows (i, r) for deal i, r = a's seat.

  Attributes:
    action: i32[rows, T].
    actor: i32[rows, T].
    deal_key: [rows] jax keys.
    done: bool[rows, T].
    probs: f64[rows, T, A] of the acting side, or None.
    rewards: f64[rows, T, P].
  """

  actor: np.ndarray
  action: np.ndarray
  rewards: np.ndarray
  done: np.ndarray
  deal_key: Any
  probs: Optional[np.ndarray]


def _chunk_rows(
    rows: int, num_steps: int, num_actions: int, unit: int, max_rows: int
) -> int:
  """Rows per Arena batch: a multiple of `unit`, within the memory budget.

  Args:
    rows: The rows wanted.
    num_steps: The plies.
    num_actions: The actions.
    unit: The row multiple.
    max_rows: The cap, or <= 0 for the memory budget.

  Returns:
    The rows per batch.
  """
  if max_rows <= 0:
    per_row = num_steps * (
        arena_engine.BYTES_PER_ACTION * num_actions + arena_engine.BYTES_FIXED
    )
    max_rows = max(1, arena_engine.LOG_BUDGET_BYTES // per_row)
  chunk = max(unit, max_rows - max_rows % unit)
  return min(chunk, -(-rows // unit) * unit)


def play(env, a, b, n_deals: int, key, probs: bool, max_rows: int) -> Played:
  """Plays a against b on n_deals seat-swapped duplicate deals (module doc).

  Args:
    env: The env.
    a: Side a's entrant.
    b: Side b's entrant.
    n_deals: The number of deals; each is played twice, seats swapped.
    key: The match key.
    probs: Whether to record the acting side's action distribution.
    max_rows: The Arena batch rows, or 0 to fit the log memory budget.

  Returns:
    Every game's log.

  Raises:
    RuntimeError: with `probs`, the two perspective passes played different
      games.
  """
  slots = [arena_lib.Slot(e.policy, 1, e.rule) for e in (a, b)]
  params = [
      arena_engine.stack_members([e.params]) if sl.in_graph else [e.params]
      for sl, e in zip(slots, (a, b))
  ]
  cls = (
      arena_lib.Arena
      if all(sl.in_graph for sl in slots)
      else host_arena.HostArena
  )
  passes = (0, 1) if probs else (0,)
  arenas = {
      p: cls(env, slots, perspective_slot=p, hands_per_match=1) for p in passes
  }
  tables = {
      p: arena_lib.rotated_table(n_deals, [0, 1], [0, 0], perspective=p)
      for p in passes
  }
  deal_key, policy_key = arena_engine.deal_keys(key, n_deals, True)
  rows, num_steps = 2 * n_deals, env.max_len + 1
  chunk = _chunk_rows(
      rows, num_steps, env.num_actions, 2 * len(jax.devices()), max_rows
  )
  out = dict(actor=[], action=[], rewards=[], done=[], probs=[])
  for start in range(0, rows, chunk):
    idx = np.arange(start, min(start + chunk, rows))
    full = np.concatenate([idx, np.full(chunk - idx.size, idx[0])])
    logs, prob = (
        {},
        np.zeros((chunk, num_steps, env.num_actions)) if probs else None,
    )
    for p in passes:
      table = arena_lib.SlotTable(*(np.asarray(x)[full] for x in tables[p]))
      _, batch, log = arenas[p].collect(
          params, table, deal_key[full], policy_key[full], num_steps
      )
      logs[p] = log
      if probs:
        valid = np.asarray(batch.valid)
        rule = (a, b)[p].rule
        _, pr = decide.decide(
            jnp.asarray(batch.behavior_logits)[valid],
            jnp.asarray(batch.steps.legal_mask)[valid],
            rule,
            jax.random.key(0),
        )
        prob[valid] = np.asarray(pr)
    if probs and not np.array_equal(
        np.asarray(logs[0].action), np.asarray(logs[1].action)
    ):
      raise RuntimeError(
          "the two perspective passes played different games (the policies' "
          "outputs differ with the batch layout); run without --probs"
      )
    log = logs[0]
    n = idx.size
    out["actor"].append(np.asarray(log.actor)[:n])
    out["action"].append(np.asarray(log.action)[:n])
    out["rewards"].append(np.asarray(log.rewards, np.float64)[:n])
    out["done"].append(np.asarray(log.done)[:n])
    if probs:
      out["probs"].append(prob[:n])
  cat = {k: np.concatenate(v) if v else None for k, v in out.items()}
  return Played(
      cat["actor"],
      cat["action"],
      cat["rewards"],
      cat["done"],
      deal_key,
      cat["probs"],
  )


def raw_histories(
    env, played: Played
) -> tuple[list[list[int]], list[list[int]], list[list[int]], np.ndarray]:
  """Replays each game's raw env history from its root, with chance outcomes.

  The Arena logs player actions only; its chance outcomes are functions of
  the deal key (core.arena: hand chance key, event n keyed by
  fold_in64(hand key, n), drawn by core.chance.sample_outcome). This replays
  each game's logged actions from the root with the same draws, and checks each
  action was legal for the logged actor and that the game ends where the log
  does.

  Args:
    env: The env.
    played: The games' logs.

  Returns:
    (raw histories, actors per entry, ply index per entry (-1 for chance),
    terminal returns f64[rows, P]).

  Raises:
    RuntimeError: a game did not replay from its deal key.
  """

  def build(h: int):
    """Builds the jitted, vmapped replay for histories of up to `h` entries.

    Args:
      h: The history buffer length.

    Returns:
      A function of (deal_key, actor, action, done), batched over games.
    """

    def one(deal_key, actor, action, done):
      """Replays one game.

      Args:
        deal_key: The game's deal key.
        actor: i32[T], the logged actor per step.
        action: i32[T], the logged action per step.
        done: bool[T], the logged terminal flag per step.

      Returns:
        (history, actors, plies, length, ok, terminal returns).
      """
      ck = arena_lib.hand_chance_key(
          deal_key[None], jnp.zeros((1,), jnp.int64)
      )[0]

      def resolve(c, t):
        """Plays out the chance nodes from the carry's state.

        Args:
          c: The replay carry.
          t: The ply; unused.

        Returns:
          The carry at the next player node or terminal state.
        """

        def cond(c):
          """Returns whether the carry's state is a live chance node.

          Args:
            c: The replay carry.

          Returns:
            bool[].
          """
          s = c[0]
          return env.is_chance(s) & ~env.terminal(s)

        def body(c):
          """Draws and applies one chance outcome, recording it.

          Args:
            c: The replay carry.

          Returns:
            The carry after the outcome.
          """
          s, n, buf, who, ply, ptr = c
          o = chance.sample_outcome(env, s, x64.fold_in64(ck, n))
          return (
              env.apply_chance(s, o),
              n + 1,
              buf.at[ptr].set(o, mode="drop"),
              who.at[ptr].set(-1, mode="drop"),
              ply.at[ptr].set(-1, mode="drop"),
              ptr + 1,
          )

        del t
        # as core.chance.resolve: no chance nodes
        if env.max_chance_outcomes == 0:
          return c
        return jax.lax.while_loop(cond, body, c)

      init = (
          env.root(),
          jnp.int64(0),
          jnp.full((h,), -1, jnp.int32),
          jnp.full((h,), -2, jnp.int32),
          jnp.full((h,), -1, jnp.int32),
          jnp.int64(0),
      )
      c = resolve(init, -1)

      def step(carry, x):
        """Applies one logged step and checks it.

        Args:
          carry: (replay carry, ended, ok).
          x: (t, actor, action, done) of the step.

        Returns:
          ((replay carry, ended, ok), None) for `jax.lax.scan`.
        """
        c, ended, ok = carry
        t, actor_t, action_t, done_t = x
        s, n, buf, who, ply, ptr = c
        act = ~ended & (actor_t >= 0)
        a_safe = jnp.maximum(action_t, 0)
        legal = env.legal_mask(s)[a_safe] & (env.actor(s) == actor_t)
        s2 = jax.tree_util.tree_map(
            lambda x2, x1: jnp.where(act, x2, x1), env.apply(s, a_safe), s
        )
        c2 = (
            s2,
            n,
            jnp.where(act, buf.at[ptr].set(action_t, mode="drop"), buf),
            jnp.where(act, who.at[ptr].set(actor_t, mode="drop"), who),
            jnp.where(act, ply.at[ptr].set(t, mode="drop"), ply),
            ptr + act.astype(jnp.int64),
        )
        c2 = resolve(c2, t)
        term = env.terminal(c2[0])
        ok = ok & (~act | (legal & (term == done_t)))
        return (c2, ended | (act & term), ok), None

      ts = jnp.arange(actor.shape[0], dtype=jnp.int32)
      (c, ended, ok), _ = jax.lax.scan(
          step,
          (c, jnp.bool_(False), jnp.bool_(True)),
          (ts, actor, action, done),
      )
      return c[2], c[3], c[4], c[5], ended & ok, env.returns(c[0])

    return jax.jit(jax.vmap(one))

  rows = played.actor.shape[0]
  h = 2 * (env.max_len + 1) + 64
  buf, who, ply, length, ok, ret = build(h)(
      played.deal_key, played.actor, played.action, played.done
  )
  if int(np.max(length)) > h:
    h = int(np.max(length))
    buf, who, ply, length, ok, ret = build(h)(
        played.deal_key, played.actor, played.action, played.done
    )
  ok = np.asarray(ok)
  if not ok.all():
    raise RuntimeError(
        f"{int((~ok).sum())} of {rows} games did not replay from their deal "
        "keys (an illegal action, another actor, or a different ending)"
    )
  buf, who, ply, length = (np.asarray(x) for x in (buf, who, ply, length))
  hists = [buf[r, : length[r]].tolist() for r in range(rows)]
  actors = [who[r, : length[r]].tolist() for r in range(rows)]
  plies = [ply[r, : length[r]].tolist() for r in range(rows)]
  return hists, actors, plies, np.asarray(ret, np.float64)


# -- main ----------------------------------------------------------------------


def _estimate(e) -> dict[str, Any]:
  """Returns an estimate's fields as JSON numbers.

  Args:
    e: The estimate.

  Returns:
    {mean, se, lo, hi, n}.
  """
  return dict(
      mean=float(e.mean),
      se=float(e.se),
      lo=float(e.lo),
      hi=float(e.hi),
      n=int(e.n),
  )


def run(
    a_spec: str,
    b_spec: str,
    games: int,
    seed: int,
    out_dir: str,
    rule_flag: str = "default",
    probs: bool = False,
    pyspiel_params: str = "",
    scripted_endgame: bool = False,
    max_rows: int = 0,
    verify: bool = True,
) -> dict[str, Any]:
  """Plays, prints and writes the games (module doc).

  Args:
    a_spec: Side a: an artifact or a built-in side (module doc).
    b_spec: Side b.
    games: The number of games, even and at least 4.
    seed: The match seed.
    out_dir: The directory written.
    rule_flag: "default", "argmax" or "sample".
    probs: Whether to record the acting policy's distribution.
    pyspiel_params: pyspiel game parameters of the artifacts' variant, k=v,...
    scripted_endgame: Whether to route the game's scripted final phase.
    max_rows: The Arena batch rows, or 0 for the memory budget.
    verify: Whether to replay the saved games in pyspiel.

  Returns:
    summary.json's contents.

  Raises:
    app.UsageError: a flag is invalid.
    ValueError: a side or the pyspiel parameters do not match the env.
    RuntimeError: the replay of a saved game failed.
    ModuleNotFoundError: a module other than pyspiel failed to import.
  """
  if games < 4 or games % 2:
    raise app.UsageError(
        f"--games={games}: an even number, at least 4 (two seat-swapped "
        "pairs, for a standard error)"
    )
  rule = None if rule_flag == "default" else types.DecisionRule(rule_flag, 1.0)
  t0 = time.monotonic()

  def builtin(s):
    """Whether a spec is built in (uniform or an arch).

    Args:
      s: The spec.

    Returns:
      True for a built-in.
    """
    return s == UNIFORM or s.startswith(ARCH_PREFIX)

  if builtin(a_spec) and builtin(b_spec):
    raise app.UsageError(
        "at least one side is an artifact (the game comes from its manifest)"
    )
  loaded = {
      k: load_side(s, rule)
      for k, s in (("a", a_spec), ("b", b_spec))
      if not builtin(s)
  }
  env, env_spec = match_env(list(loaded.values()))
  sides = {
      k: loaded[k] if k in loaded else builtin_side(s, env, rule)
      for k, s in (("a", a_spec), ("b", b_spec))
  }
  if scripted_endgame:
    for k in loaded:
      with_endgame(sides[k], env)
  if env.num_players != 2:
    raise app.UsageError(
        f"{env.name} has {env.num_players} players; this script plays two"
    )
  for k, s in sides.items():
    ps = s.entrant.policy.spec
    if (ps.num_actions, ps.num_players) != (env.num_actions, env.num_players):
      raise ValueError(
          f"side {k} ({s.spec}) has {ps.num_actions} actions and "
          f"{ps.num_players} players; {env.name} has {env.num_actions} and "
          f"{env.num_players}"
      )
  a, b = sides["a"].entrant, sides["b"].entrant
  print(
      f"{env.name}: a = {a.name} ({a.rule.kind}), b = {b.name} "
      f"({b.rule.kind}); {games} games on {games // 2} seat-swapped deals",
      flush=True,
  )

  n_deals = games // 2
  played = play(env, a, b, n_deals, jax.random.key(seed), probs, max_rows)
  t_play = time.monotonic() - t0
  pay = payoff.payoffs(played.rewards, played.done, 1)  # f64[rows, P], per seat
  a_seat = np.tile(np.array([0, 1]), n_deals)
  rows = np.arange(2 * n_deals)
  a_pay = pay[rows, a_seat].reshape(n_deals, 2)
  b_pay = pay[rows, 1 - a_seat].reshape(n_deals, 2)
  res = payoff.match_result(
      env.name,
      "play_checkpoints",
      1,
      True,
      a.name,
      b.name,
      a.rule,
      b.rule,
      a_pay,
      b_pay,
  )
  m, s0, s1 = res.mean, res.seat[0], res.seat[1]
  print(
      f"a's mean payoff per game: {m.mean:+.4f} +/- {m.se:.4f} (95% CI "
      f"{m.lo:+.4f} .. {m.hi:+.4f}); b's: {float(b_pay.mean()):+.4f}",
      flush=True,
  )
  print(
      f"  a in seat 0: {s0.mean:+.4f} +/- {s0.se:.4f}; a in seat 1: "
      f"{s1.mean:+.4f} +/- {s1.se:.4f}",
      flush=True,
  )

  hists, actors, plies, returns = raw_histories(env, played)
  if not np.allclose(returns, pay, atol=1e-5, rtol=0):
    raise RuntimeError(
        "the replayed games' returns differ from the Arena's payoffs"
    )
  base_params = pyspiel_params_for(env, env_spec, parse_params(pyspiel_params))
  if base_params is None:
    print(
        f"note: {env.name}'s env histories are not pyspiel's here; histories "
        "are in the env's ids",
        flush=True,
    )

  os.makedirs(out_dir, exist_ok=True)
  path = os.path.join(out_dir, TRAJECTORIES)
  with open(path, "w", encoding="utf-8") as f:
    for r in range(2 * n_deals):
      raw = [int(x) for x in hists[r]]
      gs = None if base_params is None else game_string(env.name, base_params)
      rec = dict(
          index=r,
          deal=r // 2,
          game_string=gs,
          seats=["a", "b"] if a_seat[r] == 0 else ["b", "a"],
          history=raw,
          actors=[int(x) for x in actors[r]],
          returns=[float(x) for x in returns[r]],
          env_history=raw,
      )
      if played.probs is not None:
        pol = []
        for t in plies[r]:
          if t < 0:
            pol.append(None)
            continue
          p = played.probs[r, t]
          pol.append([[int(i), float(p[i])] for i in np.flatnonzero(p > 0)])
        rec["policy"] = pol
      f.write(json.dumps(rec) + "\n")
  summary = dict(
      format=FORMAT,
      format_version=FORMAT_VERSION,
      game=env.name,
      env_spec=env_spec.to_json(),
      rules_sha=next(s.manifest.rules_sha for s in loaded.values()),
      pyspiel_params=base_params,
      entrants={k: s.info for k, s in sides.items()},
      games=games,
      seed=seed,
      num_records=2 * n_deals,
      result=dict(
          a=_estimate(m),
          a_seat0=_estimate(s0),
          a_seat1=_estimate(s1),
          b_mean=float(b_pay.mean()),
      ),
  )
  with open(os.path.join(out_dir, SUMMARY), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, sort_keys=True)
  print(
      f"wrote {2 * n_deals} games to {path} ({FORMAT} format {FORMAT_VERSION}) "
      f"in {time.monotonic() - t0:.0f} s (play {t_play:.0f} s)",
      flush=True,
  )

  if verify and base_params is not None:
    try:
      # Lazy: pyspiel is an optional dependency; only its availability
      # matters here.
      import pyspiel  # pylint: disable=g-import-not-at-top,import-outside-toplevel,unused-import  # noqa: PLC0415

      del pyspiel
    except ModuleNotFoundError as e:
      if e.name != "pyspiel":
        raise
      print(
          f"pyspiel is not installed, so the {2 * n_deals} saved games were "
          "not replayed in"
          " OpenSpiel: the replay check needs pyspiel (OpenSpiel's full "
          "install); the games, the result above and the summary are complete",
          flush=True,
      )
    else:
      _, records = read_trajectories(out_dir)
      n = sum(1 for rec in records if replay_in_pyspiel(rec) is not None)
      print(
          f"replayed {n} saved games in pyspiel: every move legal, same "
          "returns",
          flush=True,
      )
  return summary


def main(argv: Sequence[str], f: dict[str, flags.FlagHolder]) -> None:
  """Runs the script.

  Args:
    argv: The command line; no positional arguments.
    f: The flag holders by name.

  Raises:
    app.UsageError: an argument is unexpected or a required flag is missing.
  """
  # The framework needs 64-bit JAX; on before anything is built.
  x64.enable_x64()
  if len(argv) > 1:
    raise app.UsageError(f"unexpected arguments {argv[1:]}")
  for name, v in (
      ("a", f["a"].value),
      ("b", f["b"].value),
      ("out", f["out"].value),
  ):
    if not v:
      raise app.UsageError(f"--{name} is required")
  val = {k: holder.value for k, holder in f.items()}
  run(
      val["a"],
      val["b"],
      val["games"],
      val["seed"],
      os.path.abspath(val["out"]),
      val["rule"],
      val["probs"],
      val["pyspiel_params"],
      val["scripted_endgame"],
      val["max_rows"],
      val["verify"],
  )


if __name__ == "__main__":
  app.run(functools.partial(main, f=_define_flags()))
