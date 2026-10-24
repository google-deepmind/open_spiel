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

"""The game table: one GameSpec per game, one EnvSpec per constructed env.

Training, serving, evaluation and search all need the same facts about a game:
how to build its Env, which constructor options exist, which modules hold its
game-specific policies and samplers, and how a served pyspiel game maps onto
it. This module is the single place those facts live; nothing else keeps a
second table.

  from open_spiel.python.scale.envs import registry
  env = registry.make("leduc_poker")
  spec = registry.spec("go", board_size=9)        # EnvSpec: name + every option
  env = registry.make(spec)
  assert registry.spec_of(env) == spec
  registry.EnvSpec.from_json(spec.to_json())      # what artifacts record

Concepts:

* `GameSpec` describes a game. Its constructor is a "module:attr" string,
  imported on first use, so naming a game imports nothing.
* `EnvSpec(name, kwargs)` is the identity of a constructed env: a registered
  game and its complete, canonical options (defaults filled in, values
  normalised). Two envs with equal EnvSpecs play the same game.
* `GameSpec.extras` lists modules that register the game's own components
  (scripted policies, belief samplers, reference evaluations). They are
  imported when the game's env is first built and when a registry is asked
  for a component named `<game>.<...>` (`load_extras`), so the generic
  layers never import a game module. A component registered under a name of
  its own (a belief sampler "leduc", a slot factory "gin_mctx_pimc") is
  declared in `GameSpec.components`, which its registry consults
  (`component_module`).
* `rules_sha(env)` is the rules identity of an env: a hash of the module that
  implements the game's rules and of its EnvSpec.
* `register(spec, replace=True)` lets a plugin replace a game's spec (for
  example to change its default options); every replacement is logged.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
from typing import Any, Callable, Mapping, Optional

from absl import logging

from open_spiel.python.scale import deps
from open_spiel.python.scale import registry as path_registry
from open_spiel.python.scale.core import types

ANY = object()  # serving: any value of this pyspiel parameter is accepted
RULES_SHA_VERSION = 1  # the scheme version of rules_sha

_JSON_SCALARS = (bool, int, float, str, type(None))


def _no_params(params: Mapping[str, Any]) -> dict[str, Any]:
  """A game with no pyspiel parameters.

  Args:
    params: The pyspiel params (unused).

  Returns:
    {}.
  """
  del params
  return {}


@dataclasses.dataclass(frozen=True)
class ServingSpec:
  """How a pyspiel game of the same name is served by this env.

  Attributes:
    pyspiel_params: every pyspiel game parameter the env plays, mapped to its
      allowed values (a tuple) or ANY; a request naming another parameter or
      another value is refused.
    served_kwargs: pyspiel game parameters -> the env's constructor options.
    covers: covers(trained_kwargs, served_kwargs) -> whether an artifact
      trained with the first options plays every deal of the second. None
      means the options must be equal.
  """

  pyspiel_params: Mapping[str, Any]
  served_kwargs: Callable[[Mapping[str, Any]], dict[str, Any]] = _no_params
  covers: Optional[Callable[[Mapping[str, Any], Mapping[str, Any]], bool]] = (
      None
  )


@dataclasses.dataclass(frozen=True)
class GameSpec:
  """Everything the framework knows about one game.

  Attributes:
    name: the Env's name; for an OpenSpiel game, its pyspiel short name.
    ctor: "module:attr" of the Env constructor, called with the options.
    default_kwargs: every constructor option and its default (JSON scalars).
    options_of: Env -> its options; None reads the attributes named by
      default_kwargs.
    normalize_options: validates and canonicalises a full option dict (for
      example an int for a float option); None accepts JSON scalars as given.
    canon: "module:attr" of the deal canonical form,
      canon(info [N], labels [N]) -> uint8[N, k] (numpy leaves), the bytes
      sources.deals hashes to keep evaluation deals out of training data.
    extras: modules imported on first build that register the game's
      scripted policies, belief samplers and reference evaluations.
    components: (kind, name, module) of each component the game's modules
      register under a name that does not start with "<game>.": the
      registry of `kind` ("belief", "slot_factory") imports `module` when
      first asked for `name`.
    endgame: policy architecture of a scripted final phase that every seat
      plays (policies.routed.with_scripted_phase).
    reference_evals: names of evaluations defined on this game
      (eval.registry), each a factory (env, mesh) -> Evaluator.
    pyspiel_history_to_raw: "module:attr" of
      fn(history: list[int], params: Mapping) -> list[int], the env's raw
      history for a pyspiel history; None means the two are equal.
    covers: covers(trained_kwargs, other_kwargs) -> whether an artifact
      trained with the first options plays every deal of the second, for a
      game that is not served (a served game's ServingSpec.covers); None
      means the options must be equal.
    serving: how a pyspiel game is served by this env; None: not served.
    exact_eval: the game tree is small enough to build whole
      (eval.exploitability.GameTree), so a training run can track exact
      exploitability in-run.
  """

  name: str
  ctor: str
  default_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
  options_of: Optional[Callable[[Any], Mapping[str, Any]]] = None
  normalize_options: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None
  canon: Optional[str] = None
  extras: tuple[str, ...] = ()
  components: tuple[tuple[str, str, str], ...] = ()
  endgame: Optional[str] = None
  reference_evals: tuple[str, ...] = ()
  pyspiel_history_to_raw: Optional[str] = None
  covers: Optional[Callable[[Mapping[str, Any], Mapping[str, Any]], bool]] = (
      None
  )
  serving: Optional[ServingSpec] = None
  exact_eval: bool = False

  def __post_init__(self):
    if not isinstance(self.name, str) or not self.name:
      raise ValueError(f"a game name is a non-empty string, got {self.name!r}")
    for field in ("ctor", "canon", "pyspiel_history_to_raw"):
      v = getattr(self, field)
      if v is not None and (not isinstance(v, str) or ":" not in v):
        raise ValueError(
            f"{self.name}: {field} must be 'module:attr', got {v!r}"
        )
    for k, v in self.default_kwargs.items():
      if not isinstance(k, str) or not isinstance(v, _JSON_SCALARS):
        raise TypeError(f"{self.name}: option {k}={v!r} is not a JSON scalar")
    for field in ("extras", "reference_evals"):
      v = getattr(self, field)
      if not isinstance(v, tuple) or not all(
          isinstance(x, str) and x for x in v
      ):
        raise TypeError(
            f"{self.name}: {field} must be a tuple of names, got {v!r}"
        )
    if not isinstance(self.components, tuple) or not all(
        isinstance(c, tuple)
        and len(c) == 3
        and all(isinstance(x, str) and x for x in c)
        for c in self.components
    ):
      raise TypeError(
          f"{self.name}: components must be a tuple of (kind, name, module) "
          f"string triples, got {self.components!r}"
      )
    if self.serving is not None and not isinstance(self.serving, ServingSpec):
      raise TypeError(
          f"{self.name}: serving must be a ServingSpec, got {self.serving!r}"
      )

  @property
  def served(self) -> bool:
    """Whether the game has a serving spec.

    Returns:
      True when `serving` is set.
    """
    return self.serving is not None


@dataclasses.dataclass(frozen=True, eq=False)
class EnvSpec:
  """A registered game and its complete canonical options.

  Build one with `spec`.

  Attributes:
    name: The game name.
    kwargs: The game's complete canonical options.
  """

  name: str
  kwargs: Mapping[str, Any]

  def key(self) -> str:
    """Canonical JSON of the spec (equality, hashing, rules identity).

    Returns:
      The spec's canonical JSON string.
    """
    return json.dumps(self.to_json(), sort_keys=True, allow_nan=False)

  def __eq__(self, other):
    return isinstance(other, EnvSpec) and self.key() == other.key()

  def __hash__(self):
    return hash(self.key())

  def to_json(self) -> dict[str, Any]:
    """The JSON form of the spec.

    Returns:
      {"name": the game name, "kwargs": the canonical options}.
    """
    return {"name": self.name, "kwargs": dict(self.kwargs)}

  @staticmethod
  def from_json(d: Mapping[str, Any]) -> "EnvSpec":
    """The EnvSpec of its JSON form.

    Args:
      d: The JSON form, {"name": ..., "kwargs": ...}.

    Returns:
      The canonical EnvSpec.

    Raises:
      ValueError: d is not {name, kwargs}, or is not canonical (a default left
        out, a value not in its normal form).
    """
    if not isinstance(d, Mapping) or set(d) != {"name", "kwargs"}:
      raise ValueError(f"an EnvSpec is {{name, kwargs}}, got {d!r}")
    if not isinstance(d["kwargs"], Mapping):
      raise ValueError(f"EnvSpec kwargs must be a mapping, got {d['kwargs']!r}")
    s = spec(d["name"], **dict(d["kwargs"]))
    if s.to_json() != {"name": d["name"], "kwargs": dict(d["kwargs"])}:
      raise ValueError(
          f"{d!r} is not a canonical EnvSpec (canonical: {s.to_json()})"
      )
    return s


# -- option normalisers and serving maps of the shipped games
# ------------------------


def _gin_options(kw: dict[str, Any]) -> dict[str, Any]:
  """Validates and normalizes the gin_rummy options.

  Args:
    kw: The options.

  Returns:
    The options, with oklahoma_frac a float or None.

  Raises:
    TypeError: An option has the wrong type.
    ValueError: oklahoma_frac is outside [0, 1].
  """
  f = kw["oklahoma_frac"]
  if f is not None:
    if isinstance(f, bool) or not isinstance(f, (int, float)):
      raise TypeError(
          f"gin_rummy oklahoma_frac must be None or a number, got {f!r}"
      )
    f = float(f)
    if not 0.0 <= f <= 1.0:
      raise ValueError(
          f"gin_rummy oklahoma_frac={f!r} must be None or in [0, 1]"
      )
  out = dict(kw, oklahoma_frac=f)
  for k in ("allow_upcard_throwback", "wall_knock_pick", "spade_doubles"):
    if k in out and not isinstance(out[k], bool):
      raise TypeError(f"gin_rummy {k} must be a bool, got {out[k]!r}")
  return out


def _gin_variants(kw: Mapping[str, Any]) -> frozenset[str]:
  """The rule variants a gin oklahoma_frac covers.

  Args:
    kw: The options.

  Returns:
    A subset of {standard, oklahoma}.
  """
  f = kw["oklahoma_frac"]
  if f is None:
    return frozenset({"standard"})
  return frozenset(
      ({"oklahoma"} if f > 0.0 else set())
      | ({"standard"} if f < 1.0 else set())
  )


def gin_covers(trained: Mapping[str, Any], served: Mapping[str, Any]) -> bool:
  """Whether a gin artifact trained on `trained` plays the served env.

  A gin artifact plays a served gin env when it trained on every deal
  variant (standard, Oklahoma) the served env deals, under the same rules.

  Args:
    trained: The training env's options.
    served: The served env's options.

  Returns:
    True when it trained on every deal variant (standard, Oklahoma) the served
    env deals, under the same rules.
  """

  def rules(kw):
    """The options other than oklahoma_frac.

    Args:
      kw: The options.

    Returns:
      The options.
    """
    return {k: v for k, v in kw.items() if k != "oklahoma_frac"}

  return rules(trained) == rules(served) and _gin_variants(
      served
  ) <= _gin_variants(trained)


def gin_served_kwargs(params: Mapping[str, Any]) -> dict[str, Any]:
  """A served pyspiel gin game deals only Oklahoma or only standard hands.

  Args:
    params: The pyspiel game parameters.

  Returns:
    The env options of the served game.
  """
  return {"oklahoma_frac": 1.0 if params.get("oklahoma", False) else None}


GIN_PYSPIEL_PARAMS = {
    "gin_bonus": (25,),
    "undercut_bonus": (25,),
    "hand_size": (10,),
    "knock_card": (10,),
    "num_ranks": (13,),
    "num_suits": (4,),
    "oklahoma": (False, True),
    "starting_player": (0, 1),
}


def _int_options(game_name: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
  """A normaliser for a game whose options are all ints (bools refused).

  Args:
    game_name: The game, for errors.

  Returns:
    The normaliser.
  """

  def normalize(kw: dict[str, Any]) -> dict[str, Any]:
    """Checks that every option is an int.

    Args:
      kw: The options.

    Returns:
      The options as ints.

    Raises:
      TypeError: An option is not an int.
    """
    for k, v in kw.items():
      if isinstance(v, bool) or not isinstance(v, int):
        raise TypeError(f"{game_name} {k} must be an int, got {v!r}")
    return {k: int(v) for k, v in kw.items()}

  return normalize


def _breakthrough_served(params: Mapping[str, Any]) -> dict[str, Any]:
  """Breakthrough's options of pyspiel params.

  Args:
    params: The pyspiel params.

  Returns:
    The options.
  """
  return {
      "rows": int(params.get("rows", 8)),
      "columns": int(params.get("columns", 8)),
  }


def _gomoku_served(params: Mapping[str, Any]) -> dict[str, Any]:
  """Gomoku's options of pyspiel params.

  Args:
    params: The pyspiel params.

  Returns:
    The options.
  """
  return {
      "board_size": int(params.get("size", 15)),
      "connect": int(params.get("connect", 5)),
  }


def _hex_served(params: Mapping[str, Any]) -> dict[str, Any]:
  """Hex's options of pyspiel params.

  Args:
    params: The pyspiel params.

  Returns:
    The options.

  Raises:
    ValueError: The board is not square.
  """
  # pyspiel hex: num_rows / num_cols, when given, override board_size.
  size = int(params.get("board_size", 11))
  rows, cols = int(params.get("num_rows", size)), int(
      params.get("num_cols", size)
  )
  if rows != cols:
    raise ValueError(
        f"hex: num_rows={rows} != num_cols={cols}; the env plays square "
        "boards only"
    )
  return {"board_size": rows}


def _go_options(kw: dict[str, Any]) -> dict[str, Any]:
  """Checks go's options.

  Args:
    kw: The options.

  Returns:
    The checked options.
  """
  # Lazy: the go env needs the optional JAX games package (GAME_PACKAGES).
  from open_spiel.python.scale.envs import go  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  return {
      "board_size": go.check_board_size(kw["board_size"]),
      "komi": go.check_komi(kw["komi"]),
  }


def _go_served(params: Mapping[str, Any]) -> dict[str, Any]:
  """Go's options of pyspiel params.

  Args:
    params: The pyspiel params.

  Returns:
    The options.

  Raises:
    TypeError: board_size is not an int.
    ValueError: max_game_length is not 2 * board_size**2.
  """
  bs = params.get("board_size", 19)  # pyspiel's default size
  if isinstance(bs, bool) or not isinstance(bs, int):
    raise TypeError(f"go: game parameter board_size={bs!r} is not an int")
  mgl = params.get("max_game_length", 2 * bs * bs)
  if mgl != 2 * bs * bs:
    raise ValueError(
        f"go: game parameter max_game_length={mgl!r}; the env plays "
        f"2 * board_size**2 = {2 * bs * bs}"
    )
  return {"board_size": bs, "komi": float(params.get("komi", 7.5))}


def _loa_options(kw: dict[str, Any]) -> dict[str, Any]:
  """Checks lines of action's options.

  Args:
    kw: The options.

  Returns:
    The checked options.
  """
  # Lazy: the env needs the optional JAX games package (GAME_PACKAGES).
  from open_spiel.python.scale.envs import lines_of_action  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  return {
      "max_game_length": lines_of_action.check_max_game_length(
          kw["max_game_length"]
      )
  }


def _loa_served(params: Mapping[str, Any]) -> dict[str, Any]:
  """Lines of action's options of pyspiel params.

  Args:
    params: The pyspiel params.

  Returns:
    The options.
  """
  return {"max_game_length": params.get("max_game_length", 1000)}


# pyspiel dark_hex / dark_hex_ir: 3x3 only; obstype changes only pyspiel's
# observations (the env builds its own InfoState).
DARK_HEX_3X3 = {
    "board_size": (3,),
    "num_rows": (3,),
    "num_cols": (3,),
    "obstype": ANY,
}

# bargaining's options are OpenSpiel's bargaining parameters.
_BARGAINING_OPTIONS = {"max_turns": 10, "discount": 1.0, "prob_end": 0.0}


def _bargaining_options(kw: dict[str, Any]) -> dict[str, Any]:
  """Validates and normalizes the bargaining options.

  Args:
    kw: The options.

  Returns:
    The options: max_turns an int, discount and prob_end floats.

  Raises:
    TypeError: An option has the wrong type.
    ValueError: An option is out of range.
  """
  t = kw["max_turns"]
  if isinstance(t, bool) or not isinstance(t, int):
    raise TypeError(f"bargaining max_turns must be an int, got {t!r}")
  if t < 1:
    raise ValueError(f"bargaining max_turns={t!r} must be >= 1")
  out = {"max_turns": int(t)}
  for k in ("discount", "prob_end"):
    v = kw[k]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
      raise TypeError(f"bargaining {k} must be a number, got {v!r}")
    out[k] = float(v)
  if not 0.0 < out["discount"] <= 1.0:
    raise ValueError(
        f"bargaining discount={out['discount']!r} must be in (0, 1]"
    )
  if not 0.0 <= out["prob_end"] < 1.0:
    raise ValueError(
        f"bargaining prob_end={out['prob_end']!r} must be in [0, 1)"
    )
  return out


def _bargaining_served(params: Mapping[str, Any]) -> dict[str, Any]:
  """Bargaining's options of pyspiel params.

  Args:
    params: The pyspiel params.

  Returns:
    The options.
  """
  return {k: params.get(k, d) for k, d in _BARGAINING_OPTIONS.items()}


# universal_poker: the env plays one configuration, OpenSpiel's heads-up
# no-limit hold'em with 200-chip stacks. pyspiel's universal_poker defaults
# are a different game, so a served request must state the rule parameters.
_UNIVERSAL_POKER_RULES = {
    "betting": "nolimit",
    "bettingAbstraction": "fullgame",
    "blind": "2 1",
    "firstPlayer": "2 1 1 1",
    "numBoardCards": "0 3 1 1",
    "numHoleCards": 2,
    "numPlayers": 2,
    "numRanks": 13,
    "numRounds": 4,
    "numSuits": 4,
    "stack": "200 200",
}
UNIVERSAL_POKER_PARAMS = {
    **{k: (v,) for k, v in _UNIVERSAL_POKER_RULES.items()},
    # The remaining pyspiel parameters at the values that leave the game as is
    # (no preset board or reaches, no raise cap, no starting pot).
    "boardCards": ("",),
    "handReaches": ("",),
    "maxRaises": ("",),
    "potSize": (0,),
    "calcOddsNumSims": (0,),
}


def _universal_poker_served(params: Mapping[str, Any]) -> dict[str, Any]:
  """Checks a served universal_poker request states the rules.

  Args:
    params: The pyspiel params.

  Returns:
    {} (the env has no options).

  Raises:
    ValueError: A rule parameter is missing.
  """
  missing = sorted(set(_UNIVERSAL_POKER_RULES) - set(params))
  if missing:
    raise ValueError(
        "universal_poker: a served request must state the rule parameters "
        f"{missing} (pyspiel's universal_poker defaults are another game; the "
        f"env plays {_UNIVERSAL_POKER_RULES})"
    )
  return {}


def _served(params: Mapping[str, Any], **kw) -> ServingSpec:
  """A ServingSpec of `params`.

  Args:
    params: The pyspiel params.
    **kw: More ServingSpec fields.

  Returns:
    The ServingSpec.
  """
  return ServingSpec(pyspiel_params=params, **kw)


# bridge: the dealer and vulnerability options mirror pyspiel's game
# parameters; random_dealer_vul (the training default) draws them per board
# from the 16-board duplicate cycle (envs/bridge.py); compact_history stores
# the InfoState's action list compactly (same game, same encoding).
_BRIDGE_OPTIONS = {
    "dealer": 0,
    "dealer_vul": False,
    "non_dealer_vul": False,
    "random_dealer_vul": True,
    "compact_history": False,
}


def _bridge_options(kw: dict[str, Any]) -> dict[str, Any]:
  """Checks bridge's options.

  Args:
    kw: The options.

  Returns:
    The checked options.
  """
  # Lazy: the bridge env needs the optional JAX games package.
  from open_spiel.python.scale.envs import bridge  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  return bridge.options_check(kw)


def _bridge_covers(
    trained: Mapping[str, Any], other: Mapping[str, Any]
) -> bool:
  """Whether a bridge model trained on `trained` serves `other`.

  Args:
    trained: The trained options.
    other: The requested options.

  Returns:
    True when it does.
  """
  # Lazy: the bridge env needs the optional JAX games package.
  from open_spiel.python.scale.envs import bridge  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  return bridge.covers(trained, other)


#: The package of the built-in envs (this module's parent), so the table's
#: "module:attr" strings follow the package's import name.
_ENVS = __name__.rpartition(".")[0]

BUILTIN = (
    GameSpec(
        "kuhn_poker",
        f"{_ENVS}.kuhn:KuhnEnv",
        exact_eval=True,
        serving=_served({"players": (2,)}),
    ),
    # Kuhn poker in plain JAX: runs with no game package installed. Not an
    # OpenSpiel game name, so not served.
    GameSpec("kuhn_poker_jax", f"{_ENVS}.kuhn_jax:KuhnJaxEnv"),
    GameSpec(
        "leduc_poker",
        f"{_ENVS}.leduc:LeducEnv",
        exact_eval=True,
        extras=(f"{_ENVS}.leduc.belief",),
        components=(
            ("belief", "leduc", f"{_ENVS}.leduc.belief"),
            ("belief", "leduc_card_removal", f"{_ENVS}.leduc.belief"),
        ),
        serving=_served({
            "players": (2,),
            "starting_player": (0,),
            "action_mapping": (False,),
            "suit_isomorphism": (False,),
        }),
    ),
    GameSpec(
        "tic_tac_toe", f"{_ENVS}.tic_tac_toe:TicTacToeEnv", serving=_served({})
    ),
    GameSpec(
        "connect_four",
        f"{_ENVS}.connect_four:ConnectFourEnv",
        extras=(f"{_ENVS}.connect_four.teacher",),
        serving=_served({
            "rows": (6,),
            "columns": (7,),
            "x_in_row": (4,),
            "egocentric_obs_tensor": ANY,
        }),
    ),
    GameSpec(
        "phantom_ttt",
        f"{_ENVS}.phantom_ttt:PhantomTTTEnv",
        serving=_served({"gameversion": ("classical",), "obstype": ANY}),
    ),
    GameSpec(
        "liars_dice",
        f"{_ENVS}.liars_dice:LiarsDiceEnv",
        serving=_served({
            "players": (2,),
            "numdice": (2,),
            "dice_sides": (5,),
            "bidding_rule": ("reset-face",),
        }),
    ),
    GameSpec(
        "phantom_ttt_ir",
        f"{_ENVS}.phantom_ttt_ir:PhantomTTTIREnv",
        serving=_served({"gameversion": ("abrupt",), "obstype": ANY}),
    ),
    GameSpec(
        "bargaining",
        f"{_ENVS}.bargaining:BargainingEnv",
        default_kwargs=_BARGAINING_OPTIONS,
        normalize_options=_bargaining_options,
        serving=_served(
            {
                "max_turns": ANY,
                "discount": ANY,
                "prob_end": ANY,
                "instances_file": ("",),
                "max_num_instances": (1000,),
            },
            served_kwargs=_bargaining_served,
        ),
    ),
    GameSpec(
        "universal_poker",
        f"{_ENVS}.universal_poker:NoLimitHoldemEnv",
        serving=_served(
            UNIVERSAL_POKER_PARAMS, served_kwargs=_universal_poker_served
        ),
    ),
    GameSpec(
        "gin_rummy",
        f"{_ENVS}.gin:GinEnv",
        default_kwargs={
            "oklahoma_frac": None,
            "allow_upcard_throwback": True,
            "wall_knock_pick": True,
            "spade_doubles": False,
        },
        normalize_options=_gin_options,
        canon=f"{_ENVS}.gin:deal_canon",
        extras=(
            f"{_ENVS}.gin.rule_bot",
            f"{_ENVS}.gin.endgame",
            f"{_ENVS}.gin.belief",
            f"{_ENVS}.gin.reference_eval",
        ),
        components=(
            ("belief", "gin", f"{_ENVS}.gin.belief"),
            ("slot_factory", "gin_mctx_pimc", f"{_ENVS}.gin.search"),
        ),
        endgame="gin_rummy.scripted_endgame",
        reference_evals=("gin_rummy.rule_bot",),
        pyspiel_history_to_raw=f"{_ENVS}.gin:pyspiel_history_to_raw",
        serving=_served(
            GIN_PYSPIEL_PARAMS,
            served_kwargs=gin_served_kwargs,
            covers=gin_covers,
        ),
    ),
    GameSpec(
        "breakthrough",
        f"{_ENVS}.breakthrough:BreakthroughEnv",
        default_kwargs={"rows": 8, "columns": 8},
        normalize_options=_int_options("breakthrough"),
        serving=_served(
            {"rows": ANY, "columns": ANY}, served_kwargs=_breakthrough_served
        ),
    ),
    GameSpec(
        "gomoku",
        f"{_ENVS}.gomoku:GomokuEnv",
        default_kwargs={"board_size": 15, "connect": 5},
        normalize_options=_int_options("gomoku"),
        serving=_served(
            {
                "size": ANY,
                "connect": ANY,
                "dims": (2,),
                "wrap": (False,),
                "anti": (False,),
            },
            served_kwargs=_gomoku_served,
        ),
    ),
    GameSpec(
        "hex",
        f"{_ENVS}.hex:HexEnv",
        default_kwargs={"board_size": 11},
        normalize_options=_int_options("hex"),
        serving=_served(
            {
                "board_size": ANY,
                "num_rows": ANY,
                "num_cols": ANY,
                "string_rep": ANY,
                "swap": (False,),
                "plain_obs_tensor": (False,),
            },
            served_kwargs=_hex_served,
        ),
    ),
    GameSpec("othello", f"{_ENVS}.othello:OthelloEnv", serving=_served({})),
    # Go: 9x9 by default (pyspiel's default is 19x19); any board size and komi.
    GameSpec(
        "go",
        f"{_ENVS}.go:GoEnv",
        default_kwargs={"board_size": 9, "komi": 7.5},
        normalize_options=_go_options,
        serving=_served(
            {
                "board_size": ANY,
                "komi": ANY,
                "handicap": (0,),
                "max_game_length": ANY,
            },
            served_kwargs=_go_served,
        ),
    ),
    GameSpec(
        "lines_of_action",
        f"{_ENVS}.lines_of_action:LinesOfActionEnv",
        default_kwargs={"max_game_length": 1000},
        normalize_options=_loa_options,
        serving=_served({"max_game_length": ANY}, served_kwargs=_loa_served),
    ),
    GameSpec(
        "dark_hex",
        f"{_ENVS}.dark_hex:DarkHexEnv",
        serving=_served({**DARK_HEX_3X3, "gameversion": ("cdh",)}),
    ),
    # Abrupt Dark Hex only: pyspiel's dark_hex_ir defaults to cdh, which is
    # refused.
    GameSpec(
        "dark_hex_ir",
        f"{_ENVS}.dark_hex:DarkHexIREnv",
        serving=_served({**DARK_HEX_3X3, "gameversion": ("adh",)}),
    ),
    # Card play with the auction (pyspiel's use_double_dummy_result=true is
    # another game). Not served. Its rules identity is declared by the env
    # (envs.bridge, "Rules identity").
    GameSpec(
        "bridge",
        f"{_ENVS}.bridge:BridgeEnv",
        default_kwargs=_BRIDGE_OPTIONS,
        normalize_options=_bridge_options,
        canon=f"{_ENVS}.bridge:deal_canon",
        covers=_bridge_covers,
    ),
)

_GAMES: dict[str, GameSpec] = {}
_EXTRAS_LOADED: set[str] = set()


# The keyword is part of the public signature.
def register(game: GameSpec, *, replace: bool = False) -> GameSpec:  # pylint: disable=redefined-outer-name
  """Adds a game to the table.

  Registering an equal spec again is a no-op. A different spec under a taken
  name raises unless replace=True, which swaps it in and logs the change.

  Args:
    game: The spec to register.
    replace: Swap in a different spec under a taken name.

  Returns:
    The registered spec.

  Raises:
    ValueError: the name is taken by a different spec and replace is False.
    TypeError: game is not a GameSpec.
  """
  if not isinstance(game, GameSpec):
    raise TypeError(f"register takes a GameSpec, got {type(game).__name__}")
  old = _GAMES.get(game.name)
  if old is not None and old != game:
    if not replace:
      raise ValueError(f"game {game.name!r} is already registered as {old}")
    logging.info(
        "envs.registry: game %r replaced by a plugin (ctor %s, defaults %s)",
        game.name,
        game.ctor,
        dict(game.default_kwargs),
    )
    _EXTRAS_LOADED.discard(game.name)
  _GAMES[game.name] = game
  return game


def _unique_names(rows) -> None:
  """Refuses a name listed twice in the built-in tables.

  One row per name in the built-in tables: register() accepts an equal
  row twice, so a name listed twice must be refused here.

  Args:
    rows: The GameSpec rows of the built-in tables.

  Raises:
    ValueError: a name is listed twice.
  """
  seen = set()
  for g in rows:
    if g.name in seen:
      raise ValueError(f"game {g.name!r} is listed twice in BUILTIN")
    seen.add(g.name)


_unique_names(BUILTIN)
for _g in BUILTIN:
  register(_g)


def builtin_names() -> tuple[str, ...]:
  """The built-in games (registered at import).

  Returns:
    The sorted built-in game names.
  """
  return tuple(sorted(g.name for g in BUILTIN))


def names() -> tuple[str, ...]:
  """Every registered game.

  Returns:
    The sorted registered game names.
  """
  return tuple(sorted(_GAMES))


def game(name: str) -> GameSpec:
  """The GameSpec of a registered game.

  Args:
    name: The game name.

  Returns:
    The registered GameSpec.

  Raises:
    KeyError: no game of that name; the message lists the registered names.
  """
  if name not in _GAMES:
    raise KeyError(f"no env named {name!r}; known: {names()}")
  return _GAMES[name]


def games() -> tuple[GameSpec, ...]:
  """Every registered game's spec.

  Returns:
    The GameSpecs, sorted by name.
  """
  return tuple(_GAMES[n] for n in names())


def load_extras(name: str) -> None:
  """Imports the GameSpec.extras of game `name` (once per spec).

  Args:
    name: The game name.
  """
  g = game(name)
  if name in _EXTRAS_LOADED:
    return
  for module in g.extras:
    importlib.import_module(path_registry.resolve_path(module))
  _EXTRAS_LOADED.add(name)


def component_module(kind: str, name: str) -> Optional[str]:
  """The module that registers component `name` of `kind`, or None.

  The module that registers component `name` of `kind`, as a registered
  game declares it (GameSpec.components), or None.

  Args:
    kind: The component kind.
    name: The component name.

  Returns:
    The module name, or None when no registered game declares it.

  Raises:
    ValueError: two registered games declare the name with different modules.
  """
  found = set()
  for g in _GAMES.values():
    for k, n, m in g.components:
      if (k, n) == (kind, name):
        found.add(m)
  if len(found) > 1:
    raise ValueError(
        f"{kind} {name!r} is declared by several modules: {sorted(found)}"
    )
  return found.pop() if found else None


def component_names(kind: str) -> tuple[str, ...]:
  """The component names of `kind` that registered games declare.

  Args:
    kind: The component kind.

  Returns:
    The sorted component names.
  """
  found = set()
  for g in _GAMES.values():
    for k, n, _ in g.components:
      if k == kind:
        found.add(n)
  return tuple(sorted(found))


def load_extras_for(component: str) -> bool:
  """Imports the extras of the game a component name "<game>.<...>" names.

  Returns whether the prefix names a registered game.

  Args:
    component: The component name.

  Returns:
    Whether the prefix names a registered game.
  """
  prefix = component.split(".", 1)[0]
  if "." not in component or prefix not in _GAMES:
    return False
  load_extras(prefix)
  return True


#: Game packages an adapter may need that the framework's requirements do not
# install
#: (import root -> how to name it).
GAME_PACKAGES = {
    deps.GAMES_PACKAGE: "the JAX games package (open_spiel/python/jax/games)",
    deps.PYSPIEL: "pyspiel (open_spiel: pip install open-spiel)",
}
BUILTIN_STANDALONE = ("kuhn_poker_jax",)  # games that need no game package


class MissingGamePackageError(ModuleNotFoundError):
  """A game's adapter needs a game package that is not installed.

  The packages are GAME_PACKAGES.
  """


def absent_game_package(module: str | None) -> str | None:
  """The GAME_PACKAGES key that `module`'s import error shows is absent.

  The GAME_PACKAGES key a ModuleNotFoundError for `module` shows is not
  installed, or None (the top-level registry's absent_package).

  Args:
    module: The module a ModuleNotFoundError names.

  Returns:
    The absent package key, or None.
  """
  return path_registry.absent_package(module, tuple(GAME_PACKAGES))


def env_class(name: str) -> Callable[..., Any]:
  """The constructor of game `name` (a class, or a factory function).

  Args:
    name: The game name.

  Returns:
    The constructor.

  Raises:
    MissingGamePackageError: the game's adapter needs a game package that is
      not installed; the message names it and the games that need none.
    ModuleNotFoundError: A missing module that is not a game package.
  """
  module, attr = path_registry.resolve_path(game(name).ctor).split(":", 1)
  try:
    mod = importlib.import_module(module)
  except ModuleNotFoundError as e:
    root = absent_game_package(e.name)
    if root is None:
      raise
    raise MissingGamePackageError(
        f"game {name!r} needs {GAME_PACKAGES[root]}, which is not installed; "
        f"without it only {', '.join(BUILTIN_STANDALONE)} run (e.g. "
        "--config=mmd_kuhn_poker_jax)",
        name=root,
    ) from e
  obj = mod
  for part in attr.split("."):
    obj = getattr(obj, part)
  return obj


def by_field(field: str) -> dict[Any, str]:
  """{value of GameSpec.<field>: game name} over the games that set it.

  A tuple field contributes each of its values.

  Args:
    field: The GameSpec field name.

  Returns:
    The value-to-game mapping.

  Raises:
    AssertionError: two games set the same value.
  """
  out: dict[Any, str] = {}
  for g in games():
    v = getattr(g, field)
    if v is None:
      continue
    for x in v if isinstance(v, tuple) else (v,):
      if x in out:
        raise AssertionError(
            f"{field} {x!r} is set by {out[x]!r} and {g.name!r}"
        )
      out[x] = g.name
  return out


def resolve(path: str) -> Any:
  """The object a GameSpec "module:attr" field names.

  Args:
    path: The "module:attr" path.

  Returns:
    The named object.
  """
  module, attr = path_registry.resolve_path(path).split(":", 1)
  obj = importlib.import_module(module)
  for part in attr.split("."):
    obj = getattr(obj, part)
  return obj


# -- EnvSpec
# ---------------------------------------------------------------------


def _check_json(name: str, kw: Mapping[str, Any]) -> None:
  for k, v in kw.items():
    if not isinstance(v, _JSON_SCALARS):
      raise TypeError(
          f"{name}: option {k}={v!r} is not a JSON scalar (an EnvSpec is "
          "serialisable)"
      )
    if isinstance(v, float) and not math.isfinite(v):
      raise ValueError(f"{name}: option {k}={v!r} is not finite")


def spec(name: str, **kwargs: Any) -> EnvSpec:
  """The canonical EnvSpec of game `name` with options `kwargs`.

  Missing options take their defaults; an int given for a float option becomes
  that float.

  Args:
    name: The game name.
    **kwargs: The env options.

  Returns:
    The canonical EnvSpec.

  Raises:
    TypeError: an unknown option, or a value that is not a JSON scalar.
    ValueError: a value the game's normaliser rejects.
    AssertionError: The normaliser returned a different option set.
  """
  g = game(name)
  unknown = sorted(set(kwargs) - set(g.default_kwargs))
  if unknown:
    raise TypeError(
        f"{name}: unknown env option(s) {unknown}; options: "
        f"{sorted(g.default_kwargs)}"
    )
  kw = {**dict(g.default_kwargs), **kwargs}
  for k, d in g.default_kwargs.items():
    if (
        isinstance(d, float)
        and isinstance(kw[k], int)
        and not isinstance(kw[k], bool)
    ):
      kw[k] = float(kw[k])
  _check_json(name, kw)
  if g.normalize_options is not None:
    kw = g.normalize_options(kw)
    if set(kw) != set(g.default_kwargs):
      raise AssertionError(
          f"{name}: normalize_options returned options {sorted(kw)}"
      )
  return EnvSpec(name, dict(sorted(kw.items())))


def make(name_or_spec: Any, **kwargs: Any) -> types.Env:
  """The Env of an EnvSpec, or of game `name` with options `kwargs`.

  The first build of a game imports its extras.

  Args:
    name_or_spec: An EnvSpec, or a game name.
    **kwargs: The env options (with a game name only).

  Returns:
    The constructed env.

  Raises:
    AssertionError: the constructor built an env whose name or options differ
      from the spec.
    TypeError: Options given with an EnvSpec.
  """
  if isinstance(name_or_spec, EnvSpec):
    if kwargs:
      raise TypeError("make(EnvSpec) takes no options (they are in the spec)")
    s = name_or_spec
  else:
    s = spec(name_or_spec, **kwargs)
  env = env_class(s.name)(**dict(s.kwargs))
  if env.name != s.name:
    raise AssertionError(
        f"{game(s.name).ctor} built an env named {env.name!r}, registered as "
        f"{s.name!r}"
    )
  got = spec_of(env)
  if got != s:
    raise AssertionError(
        f"{s.name}: built from {s.to_json()} but the env reports "
        f"{got.to_json()} (options_of / normalize_options disagree with the "
        "ctor)"
    )
  load_extras(s.name)
  return env


def spec_of(env: types.Env) -> EnvSpec:
  """The EnvSpec of a constructed env (its game must be registered).

  Args:
    env: The env.

  Returns:
    Its canonical EnvSpec.

  Raises:
    TypeError: The env is not of its game's registered constructor.
  """
  g = game(env.name)
  ctor = env_class(env.name)
  if inspect.isclass(ctor) and not isinstance(env, ctor):
    raise TypeError(
        f"{type(env).__name__} is not {g.ctor} (the env registered as "
        f"{env.name!r})"
    )
  if g.options_of is not None:
    kw = dict(g.options_of(env))
  else:
    kw = {k: getattr(env, k) for k in g.default_kwargs}
  return spec(env.name, **kw)


def training_env(manifest: types.Manifest) -> types.Env:
  """The env an artifact was trained on, from its manifest's env_spec.

  A manifest without an env_spec names its game only; that game's env is
  built when the game has no constructor options.

  Args:
    manifest: The artifact's manifest.

  Returns:
    The training env.

  Raises:
    ValueError: no env_spec and the game has options (they cannot be
      inferred), or an env_spec of another game.
  """
  if manifest.env_spec is None:
    if dict(game(manifest.game).default_kwargs):
      raise ValueError(
          f"artifact of {manifest.game!r} has no env_spec and "
          f"{manifest.game!r} has constructor options "
          f"{dict(game(manifest.game).default_kwargs)}: pass the env "
          "explicitly"
      )
    return make(manifest.game)
  env = make(EnvSpec.from_json(manifest.env_spec))
  if env.name != manifest.game:
    raise ValueError(
        f"manifest env_spec {manifest.env_spec} is not its game "
        f"{manifest.game!r}"
    )
  return env


def covers(trained: EnvSpec, served: EnvSpec) -> bool:
  """Whether an artifact trained on `trained` plays every deal of `served`.

  Args:
    trained: The training EnvSpec.
    served: The served EnvSpec.

  Returns:
    True when the artifact plays every deal of `served`.

  Raises:
    TypeError: Either argument is not an EnvSpec.
  """
  if not isinstance(trained, EnvSpec) or not isinstance(served, EnvSpec):
    raise TypeError(f"covers takes two EnvSpecs, got {trained!r}, {served!r}")
  if trained.name != served.name:
    return False
  g = game(trained.name)
  fn = (
      g.serving.covers
      if g.serving is not None and g.serving.covers is not None
      else g.covers
  )
  if fn is None:
    return trained == served
  return bool(fn(trained.kwargs, served.kwargs))


def check_serves(
    trained: Optional[Mapping[str, Any]],
    env: types.Env,
    *,
    allow_cross_variant: bool = False,
) -> None:
  """Checks that an artifact trained on EnvSpec JSON `trained` may play `env`.

  `trained` None (a manifest without an env_spec) checks nothing beyond the
  game name the caller has already matched. allow_cross_variant plays an env
  the training env does not cover (the game must still match).

  Args:
    trained: The training EnvSpec JSON, or None.
    env: The served env.
    allow_cross_variant: Allow a variant the training env does not cover.

  Raises:
    ValueError: another game, or a variant the artifact does not cover.
  """
  if trained is None:
    return
  t = EnvSpec.from_json(trained)
  if t.name != env.name:
    raise ValueError(
        f"the artifact was trained on {t.name!r}, not {env.name!r}"
    )
  s = spec_of(env)
  if not covers(t, s) and not allow_cross_variant:
    raise ValueError(
        f"the artifact was trained on {t.to_json()}, which does not cover the "
        f"served env {s.to_json()} (allow_cross_variant=True to play it anyway)"
    )


_RULES_SHA_TAG = {"tag": f"rules/{RULES_SHA_VERSION}"}


def set_rules_sha_tag(tag: str) -> None:
  """Sets the domain-separation tag hashed first into every rules_sha.

  (default "rules/<RULES_SHA_VERSION>").

  The tag namespaces a deployment's rules hashes: two deployments with
  different tags never share a rules_sha. A plugin sets it once, at import,
  before any rules_sha is computed.

  Args:
    tag: The tag (default "rules/<RULES_SHA_VERSION>").

  Raises:
    ValueError: The tag is not a non-empty string.
  """
  if not isinstance(tag, str) or not tag:
    raise ValueError(f"a rules_sha tag is a non-empty string, got {tag!r}")
  _RULES_SHA_TAG["tag"] = tag


def rules_sha(env: types.Env) -> str:
  """The rules identity of an env.

  Sha256 over the scheme tag, the source of the module that defines its game
  (env.game) and its canonical EnvSpec, so variants of one module (standard
  and Oklahoma gin) differ. An env that declares its rules identity
  (`declared_rules_sha`, a sha256 hex) gets that one, and a stub env
  (envs.stub) the one it was given.

  The hash is over the raw bytes of the module that defines the game class,
  so it is both too sensitive and too narrow: any edit to that file (a
  comment, a reformat, a licence header) changes every artifact's rules
  identity, while a rules change in another module of the engine (a helper
  the game calls) leaves it unchanged. An engine that wants a stable,
  complete identity declares it with `declared_rules_sha` (a versioned
  description of its rules, as envs.bridge does).

  Args:
    env: The env.

  Returns:
    The sha256 hex digest.

  Raises:
    ValueError: A stub env without a rules_sha, or a malformed declared one.
    TypeError: The env has no `game` attribute.
  """
  # Lazy: envs.stub imports JAX, and this registry is part of the JAX-free
  # control plane (league.tests.jax_free_control_test).
  from open_spiel.python.scale.envs import stub  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  if isinstance(env, stub.StubEnv):
    if env.stub_rules_sha is None:
      raise ValueError(
          f"stub env {env.name!r} declares no rules_sha (StubEnv rules_sha)"
      )
    return env.stub_rules_sha
  declared = getattr(env, "declared_rules_sha", None)
  if declared is not None:
    if not (isinstance(declared, str) and len(declared) == 64):
      raise ValueError(
          f"{env.name!r} declares a malformed rules_sha {declared!r}"
      )
    return declared
  g = getattr(env, "game", None)
  if g is None:
    raise TypeError(
        f"{type(env).__name__} has no `game` attribute to read rules from"
    )
  with open(inspect.getsourcefile(type(g)), "rb") as f:
    src = hashlib.sha256(f.read()).hexdigest()
  h = hashlib.sha256()
  for part in (_RULES_SHA_TAG["tag"], src, spec_of(env).key()):
    h.update(part.encode())
    h.update(b"\0")
  return h.hexdigest()
