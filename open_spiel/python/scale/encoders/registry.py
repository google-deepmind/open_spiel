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

"""The encoder registry: obs_format name -> Encoder.

A policy's network reads arrays, a game produces InfoState pytrees; an
Encoder is the declared map between them. Its name is the policy's
obs_format, recorded in every artifact, so a name always means one function,
one shape and one dtype.

  enc = registry.get("leduc_poker_info_state")
  x = enc(info)                 # one InfoState -> f32[30]
  xb = enc.batch(info_b)        # batched InfoState [B, ...] -> f32[B, 30]
  enc = registry.get("go_planes@board_size=13")    # a family member

Concepts:

* An Encoder maps ONE unbatched InfoState to an array of its declared shape
  and dtype; `Encoder.batch` maps a batch (what Policy.encode receives).
  Dtypes are explicit, so jax_enable_x64 never promotes an encoding.
* A family covers every option setting of a game whose encoding shape
  depends on its options (board size, deck size). `register_family(name,
  game, factory, options)` registers it; the obs_format of a member is
  `name` at the game's default options, else `name@k=v,...` over the options
  that differ from their defaults, in sorted order (`family_name`).
* `register_module(module)` adds a module whose import registers encoders
  (a plugin's). A built-in module whose game package (the JAX games, pyspiel) is
  not installed is skipped, and a lookup that misses names the skipped
  package.
"""

import dataclasses
import importlib
import json
from typing import Any, Callable, Optional, Sequence

import jax
import jax.numpy as jnp

from open_spiel.python.scale import deps
from open_spiel.python.scale import registry as path_registry
from open_spiel.python.scale.envs import registry as env_registry

# Modules whose import registers the built-in encoders.
_BUILTIN_MODULES = tuple(
    f"{__name__.rpartition('.')[0]}.{m}"
    for m in (
        "kuhn",
        "leduc",
        "tic_tac_toe",
        "connect_four",
        "phantom_ttt",
        "liars_dice",
        "bargaining",
        "universal_poker",
        "gin",
        "gin_card_tokens",
        "board_games",
        "go",
        "lines_of_action",
        "dark_hex",
        "bridge",
    )
)
OPTIONAL_DEPS = (deps.GAMES_PACKAGE, deps.PYSPIEL)


@dataclasses.dataclass(frozen=True)
class Encoder:
  """A named InfoState -> array map with a declared shape and dtype.

  Attributes:
    name: the obs_format recorded in PolicySpec and artifacts.
    game: the Env.name whose InfoState it reads.
    shape: the per-example output shape.
    dtype: the output dtype.
    fn: one unbatched InfoState -> array[*shape] of dtype.
    reference: free text naming what the encoding reproduces, if anything
      (for example "OpenSpiel kuhn_poker observation_tensor").
  """

  name: str
  game: str
  shape: tuple[int, ...]
  dtype: Any
  fn: Callable[[Any], jax.Array]
  reference: str = ""

  def __post_init__(self):
    if not isinstance(self.name, str) or not self.name:
      raise ValueError(
          f"an encoder name is a non-empty string, got {self.name!r}"
      )
    if not all(isinstance(d, int) and d > 0 for d in self.shape):
      raise ValueError(f"{self.name}: shape {self.shape} must be positive ints")
    object.__setattr__(self, "shape", tuple(self.shape))
    object.__setattr__(self, "dtype", jnp.dtype(self.dtype))

  def __call__(self, info) -> jax.Array:
    """The encoding of one (unbatched) InfoState.

    Args:
      info: one InfoState.

    Returns:
      The declared shape and dtype.

    Raises:
      AssertionError: the output's shape or dtype differs from the declared one.
    """
    out = self.fn(info)
    if out.shape != self.shape or out.dtype != self.dtype:
      raise AssertionError(
          f"{self.name}: got {out.dtype}{out.shape}, declared "
          f"{self.dtype}{self.shape}"
      )
    return out

  def batch(self, info) -> jax.Array:
    """[B, *shape] from a batched InfoState.

    Args:
      info: a batched InfoState, leaves [B, ...].

    Returns:
      [B, *shape] of the declared dtype.

    Raises:
      AssertionError: the output's shape or dtype differs from the declared one.
    """
    out = jax.vmap(self.fn)(info)
    if out.shape[1:] != self.shape or out.dtype != self.dtype:
      raise AssertionError(
          f"{self.name}: got {out.dtype}{out.shape[1:]}, declared "
          f"{self.dtype}{self.shape}"
      )
    return out


@dataclasses.dataclass(frozen=True)
class _Family:
  """A registered encoder family.

  Attributes:
    name: the family name.
    game: the game it encodes.
    factory: EnvSpec -> the family member's Encoder.
    options: the env options the encoding depends on; None for every option.
  """

  name: str
  game: str
  factory: Callable[[Any], Encoder]
  options: Optional[tuple[str, ...]]


_ENCODERS: dict[str, Encoder] = {}
_FAMILIES: dict[str, _Family] = {}
_FAMILY_MEMBERS: dict[str, Encoder] = {}
_EXTRA_MODULES: list[str] = []
_SKIPPED: dict[str, str] = {}  # module -> the missing optional dependency


def register(enc: Encoder) -> Encoder:
  """Registers a plain encoder.

  It may share its name with its game's family: the family member at the
  default options.

  Args:
    enc: the Encoder.

  Returns:
    `enc`.

  Raises:
    TypeError: `enc` is not an Encoder.
    ValueError: the name is taken.
  """
  if not isinstance(enc, Encoder):
    raise TypeError(f"register takes an Encoder, got {type(enc).__name__}")
  if enc.name in _ENCODERS or (
      enc.name in _FAMILIES and _FAMILIES[enc.name].game != enc.game
  ):
    raise ValueError(f"encoder {enc.name!r} is already registered")
  _ENCODERS[enc.name] = enc
  return enc


def register_family(
    name: str,
    game: str,
    factory: Callable[[Any], Encoder],
    options: Optional[Sequence[str]] = None,
) -> None:
  """Registers an encoder family of `game`.

  Args:
    name: the family name, the obs_format at the game's default options.
    game: the game whose EnvSpec options size the encoding.
    factory: EnvSpec -> the Encoder for that spec, named
      family_name(name, spec).
    options: the option names the encoding depends on; None means all.

  Raises:
    ValueError: the name is taken.
  """
  if not isinstance(name, str) or not name or "@" in name:
    raise ValueError(
        f"a family name is a non-empty string without '@', got {name!r}"
    )
  if name in _FAMILIES or (name in _ENCODERS and _ENCODERS[name].game != game):
    raise ValueError(f"encoder family {name!r} is already registered")
  _FAMILIES[name] = _Family(
      name, game, factory, None if options is None else tuple(options)
  )


def family_name(name: str, env_spec: Any) -> str:
  """The obs_format of family `name` at EnvSpec `env_spec` (module doc).

  Args:
    name: the family name.
    env_spec: an EnvSpec of the family's game.

  Returns:
    `name`, or `name@k=v,...` over the options that differ from the defaults.

  Raises:
    ValueError: the family encodes another game.
  """
  fam = _family(name)
  if env_spec.name != fam.game:
    raise ValueError(
        f"family {name!r} encodes {fam.game!r}, not {env_spec.name!r}"
    )
  default = env_registry.spec(fam.game).kwargs
  keys = sorted(env_spec.kwargs) if fam.options is None else sorted(fam.options)
  diff = []
  for k in keys:
    if env_spec.kwargs[k] != default[k]:
      diff.append(f"{k}={json.dumps(env_spec.kwargs[k])}")
  return name if not diff else f"{name}@{','.join(diff)}"


def register_module(module: str) -> None:
  """A module whose import registers encoders (imported with the built-ins).

  Args:
    module: the module name.

  Raises:
    ValueError: `module` is not a non-empty string.
  """
  if not isinstance(module, str) or not module:
    raise ValueError(f"a module name is a non-empty string, got {module!r}")
  if module not in _EXTRA_MODULES:
    _EXTRA_MODULES.append(module)


def _load_builtins() -> None:
  for mod in tuple(_BUILTIN_MODULES) + tuple(_EXTRA_MODULES):
    if mod in _SKIPPED:
      continue
    try:
      importlib.import_module(mod)
    except ModuleNotFoundError as e:
      root = path_registry.absent_package(e.name, OPTIONAL_DEPS)
      if root is None or mod not in _BUILTIN_MODULES:
        raise
      _SKIPPED[mod] = root


def _family(name: str) -> _Family:
  """The encoder family `name`, loading the built-ins on a miss.

  Args:
    name: The family name.

  Returns:
    The family.

  Raises:
    KeyError: There is no such family.
  """
  if name not in _FAMILIES:
    _load_builtins()
  if name not in _FAMILIES:
    raise KeyError(f"no encoder family {name!r}; families: {sorted(_FAMILIES)}")
  return _FAMILIES[name]


def _parse_value(text: str) -> Any:
  """A JSON value, or the text itself when it is not JSON.

  Args:
    text: The value text.

  Returns:
    The parsed value.
  """
  try:
    return json.loads(text)
  except json.JSONDecodeError:
    return text


def _member(name: str) -> Optional[Encoder]:
  """The family member a "family@k=v,..." name (or bare family name) denotes.

  Args:
    name: The encoder name.

  Returns:
    The Encoder, or None when no family matches.

  Raises:
    KeyError: The options are malformed, not the family's or not canonical.
    AssertionError: The family built another encoder.
  """
  if name in _FAMILY_MEMBERS:
    return _FAMILY_MEMBERS[name]
  base, _, rest = name.partition("@")
  if base not in _FAMILIES:
    return None
  fam = _FAMILIES[base]
  kw = {}
  for item in rest.split(",") if rest else ():
    k, sep, v = item.partition("=")
    if not sep:
      raise KeyError(f"encoder {name!r}: {item!r} is not k=v")
    kw[k] = _parse_value(v)
  if fam.options is not None and set(kw) - set(fam.options):
    raise KeyError(
        f"encoder {name!r}: family {base!r} depends only on {fam.options}"
    )
  env_spec = env_registry.spec(fam.game, **kw)
  canonical = family_name(base, env_spec)
  if canonical != name:
    raise KeyError(
        f"encoder {name!r} is not canonical; the obs_format is {canonical!r}"
    )
  enc = fam.factory(env_spec)
  if enc.name != name or enc.game != fam.game:
    raise AssertionError(
        f"family {base!r} built {enc.name!r} of {enc.game!r} for {name!r}"
    )
  _FAMILY_MEMBERS[name] = enc
  return enc


def get(name: str) -> Encoder:
  """The encoder of obs_format `name`: a plain encoder or a family member.

  Args:
    name: the obs_format.

  Returns:
    The Encoder.

  Raises:
    KeyError: an unknown name; the message lists the registered ones and any
      module skipped for a missing game package.
  """
  _load_builtins()
  if name in _ENCODERS:
    return _ENCODERS[name]
  enc = _member(name)
  if enc is not None:
    return enc
  skipped = (
      f"; skipped for a missing optional dependency: {_SKIPPED}"
      if _SKIPPED
      else ""
  )
  err = KeyError(f"unknown encoder {name!r}; registered: {names()}{skipped}")
  if _SKIPPED:
    roots = sorted(set(_SKIPPED.values()))
    raise err from ModuleNotFoundError(
        f"encoder modules were skipped: {', '.join(roots)} not installed",
        name=roots[0],
    )
  raise err


def known(name: str) -> bool:
  """Whether `name` is a plain encoder, a family, or a family member's form.

  Args:
    name: the obs_format.

  Returns:
    bool.
  """
  _load_builtins()
  return name in _ENCODERS or name.partition("@")[0] in _FAMILIES


def names(game: Optional[str] = None) -> tuple[str, ...]:
  """Every plain encoder and family name, optionally of one game.

  Args:
    game: the game name, or None for every game.

  Returns:
    The sorted names.
  """
  _load_builtins()
  out = {n for n, e in _ENCODERS.items() if game is None or e.game == game}
  out |= {n for n, f in _FAMILIES.items() if game is None or f.game == game}
  return tuple(sorted(out))


def families() -> tuple[str, ...]:
  """Every family name.

  Returns:
    The sorted names.
  """
  _load_builtins()
  return tuple(sorted(_FAMILIES))
