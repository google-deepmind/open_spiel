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

"""Gin rummy's step and deal with the meld functions supplied.

A gin rummy backend is the state machine in `game` with two meld functions
chosen: `meld_info` (a hand's optimal melds and the deadwood after each
discard) and `knocker_melds` (the melds a knocker may lay under a knock
limit). The reference backend uses `deadwood.compute_all_meld_info` and
`deadwood.compute_legal_knocker_melds`; the dense-cache backend
(`dense_cache_game`, `GinRummy(backend="dense_cache")`) uses lookup-table
versions of the same functions.

`make_step` and `make_init_dealt_state` return the step and the dealt-state
constructor with the given meld functions bound, not jitted, so a caller can
build a backend, or time one, without patching module globals:

  step = jax.jit(backends.make_step(meld_info=my_meld_info))

Functions that compute the same results as the reference ones give the same
game. Any other choice is not gin rummy (a benchmark may still use one to
bound what the meld work costs); nothing here checks the results.
"""

from collections.abc import Callable
import functools
from typing import Any

from open_spiel.python.jax.games.gin_rummy import deadwood
from open_spiel.python.jax.games.gin_rummy import game

__all__ = [
    "make_init_dealt_state",
    "make_step",
]


def make_step(
    *,
    meld_info: Callable[..., Any] = deadwood.compute_all_meld_info,
    knocker_melds: Callable[..., Any] = deadwood.compute_legal_knocker_melds,
) -> Callable[[Any, Any], Any]:
  """Returns `step(state, action)` with the given meld functions.

  With the defaults the result computes what `game.step` computes; it is
  not jitted.

  Args:
    meld_info: Computes (optimal melds, deadwood per discard) of a hand.
    knocker_melds: Computes the melds a knocker may lay under a knock limit.

  Returns:
    step(state, action) -> the next state.

  Raises:
    TypeError: a meld function is not callable.
  """
  _check_callable(meld_info=meld_info, knocker_melds=knocker_melds)
  # The step body is this package's own extension point (its docstring:
  # "The backend chooses both").
  return functools.partial(
      game._step_body,  # pylint: disable=protected-access
      meld_info=meld_info,
      knocker_melds=knocker_melds,
  )


def make_init_dealt_state(
    *,
    meld_info: Callable[..., Any] = deadwood.compute_all_meld_info,
) -> Callable[..., Any]:
  """Returns `init_dealt_state(key, ...)` with the given meld function.

  The result takes the arguments `game.init_dealt_state` takes (`key`,
  `oklahoma`, `randomized_knock_limit` and the rule variants); with the
  default it deals what `game.init_dealt_state` deals. It is not jitted, and
  it does not run `game.init_dealt_state`'s check of concrete rule values.

  Args:
    meld_info: Computes (optimal melds, deadwood per discard) of a hand.

  Returns:
    init_dealt_state(key, ...) -> (state, new_key), the state at the first
    upcard with its meld caches filled.

  Raises:
    TypeError: meld_info is not callable.
  """
  _check_callable(meld_info=meld_info)
  return functools.partial(
      game._init_dealt_state_body,  # pylint: disable=protected-access
      meld_info=meld_info,
  )


def _check_callable(**fns: Any) -> None:
  """Raises TypeError for an argument that is not callable.

  Args:
    **fns: The arguments, by name.

  Raises:
    TypeError: one of them is not callable.
  """
  for name, fn in fns.items():
    if not callable(fn):
      raise TypeError(f"{name} must be callable, got {fn!r}")
