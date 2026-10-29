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

"""Named evaluations defined on one game.

name -> factory(env, mesh) -> Evaluator.

A game's GameSpec.reference_evals lists the names of the evaluations defined
on it; a module in its GameSpec.extras registers each factory. A league's
in-loop evaluation names one, and the runner builds it for the run's env
without importing any game module.

  ev = registry.build("gin_rummy.rule_bot", env)       # an eval.match.Evaluator
  result = ev.vs_reference(entrant, n_deals=512, key=key)
"""

from __future__ import annotations

from typing import Any, Callable

from open_spiel.python.scale.envs import registry as env_registry

EvalFactory = Callable[
    ..., Any
]  # factory(env, mesh=None) -> eval.match.Evaluator

_FACTORIES: dict[str, EvalFactory] = {}


def register(name: str, factory: EvalFactory) -> None:
  """Registers `factory` under `name` (idempotent for the same factory).

  Args:
    name: the evaluation's name.
    factory: factory(env, mesh) -> Evaluator.

  Raises:
    ValueError: another factory holds the name.
    TypeError: factory is not callable.
  """
  if not isinstance(name, str) or not name or not callable(factory):
    raise TypeError(
        f"register(name: str, factory: callable), got {name!r}, {factory!r}"
    )
  old = _FACTORIES.get(name)
  if old is not None and old is not factory:
    raise ValueError(f"evaluation {name!r} is already registered")
  _FACTORIES[name] = factory


def names() -> tuple[str, ...]:
  """The registered evaluation names, sorted.

  Returns:
    the names.
  """
  return tuple(sorted(_FACTORIES))


def get(name: str) -> EvalFactory:
  """The factory of `name`.

  Loads the extras of the game a "<game>." prefix names.

  Args:
    name: an evaluation name.

  Returns:
    its factory.

  Raises:
    KeyError: no evaluation of that name.
  """
  if name not in _FACTORIES:
    env_registry.load_extras_for(name)
  if name not in _FACTORIES:
    raise KeyError(f"no evaluation named {name!r}; registered: {names()}")
  return _FACTORIES[name]


def build(name: str, env: Any, mesh: Any = None) -> Any:
  """The Evaluator of evaluation `name` on `env`.

  Args:
    name: an evaluation name.
    env: the Env.
    mesh: the device mesh, or None.

  Returns:
    the Evaluator.
  """
  return get(name)(env, mesh=mesh)
