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

"""Array frameworks: their names, and tests of membership that import none.

A policy declares the framework of its params by name (PolicySpec.framework);
checkpoint codecs and pool storage are keyed by that name. "jax" (params and
learner-state codecs) and "numpy" (a params codec, ckpt.codecs) are built in,
and a plugin registers any other.

The population-training control plane (samplers, controllers, meta-solvers,
the artifact index) imports and runs without jax. Code that handles values of
several frameworks asks here instead of importing one: a value can only be a
jax array if jax is already imported.

  frameworks.register_framework("acme")
  frameworks.is_jax_array(x)      # False without importing jax
"""

import os
import sys
from typing import Any, Optional

#: The package's import name (the root this module's own name is under).
PACKAGE = __name__.rsplit(".", 2)[0]

_NAMES: list[str] = ["jax", "numpy"]


def register_framework(name: str) -> None:
  """Adds a framework name (registering a name twice has no effect).

  Args:
    name: an identifier.

  Raises:
    ValueError: name is not an identifier.
  """
  if not isinstance(name, str) or not name.isidentifier():
    raise ValueError(f"framework name {name!r} is not an identifier")
  if name not in _NAMES:
    _NAMES.append(name)


def names() -> tuple[str, ...]:
  """The registered framework names.

  Returns:
    the names, in registration order.
  """
  return tuple(_NAMES)


def loaded(name: str) -> bool:
  """Whether module `name` is imported in this process.

  An entry of None, an import blocked on purpose, counts as not imported.

  Args:
    name: a module name.

  Returns:
    True if imported.
  """
  return sys.modules.get(name) is not None


def is_jax_array(x: Any) -> bool:
  """Whether x is a jax array (False when jax is not imported).

  Args:
    x: any value.

  Returns:
    True for a jax.Array.
  """
  jax = sys.modules.get("jax")
  return jax is not None and isinstance(x, jax.Array)


_FALSE = ("0", "false", "f", "no", "n", "off")


def x64_refused() -> Optional[str]:
  """The value of JAX_ENABLE_X64 when it is set to a false value, else None.

  The one parser of the switch: the package's enable_x64() refuses to run
  when this returns a value.

  Returns:
    the false value JAX_ENABLE_X64 is set to, or None.
  """
  requested = os.environ.get("JAX_ENABLE_X64")
  if requested is not None and requested.strip().lower() in _FALSE:
    return requested
  return None


def enable_x64() -> None:
  """Turns jax_enable_x64 on for this process (idempotent).

  The package's `enable_x64()`. Call it before creating arrays: arrays made
  earlier keep their 32-bit dtypes. It loads no array framework: if jax is
  not imported yet it sets JAX_ENABLE_X64=1 for jax to read at import (and
  for child processes), and if jax is already imported it also updates
  jax's config.

  Raises:
    RuntimeError: JAX_ENABLE_X64 is set to a false value (the environment
      asked for x64 off, which this package cannot run with).
  """
  requested = x64_refused()
  if requested is not None:
    raise RuntimeError(
        f"JAX_ENABLE_X64={requested!r}: {PACKAGE} needs jax_enable_x64 on; "
        "unset it or set it to 1"
    )
  os.environ["JAX_ENABLE_X64"] = "1"
  if sys.modules.get("jax") is not None:  # None: a test blocks the jax import
    sys.modules["jax"].config.update("jax_enable_x64", True)


class X64NotEnabledError(RuntimeError):
  """jax_enable_x64 is off in a process that needs it on.

  This is core.x64.NotEnabledError, defined here so that this module never
  imports core.x64.
  """


def x64_off_message(where: str = "") -> str:
  """The message of X64NotEnabledError.

  Args:
    where: the caller, or "".

  Returns:
    the message.
  """
  at = f" ({where})" if where else ""
  return (
      f"jax_enable_x64 is off{at}: call {PACKAGE}.enable_x64() "
      "once, before creating arrays (the entry points do), and never switch "
      "it off"
  )


def assert_x64_if_loaded(where: str) -> None:
  """core.x64.assert_x64 when jax is imported.

  Without jax nothing in this process computes in 32 bits by accident; jax
  imported later is checked by the components that use it. Reads jax's
  config through sys.modules: this module stays jax-free and never imports
  core.x64 (which imports this module).

  Args:
    where: the caller, for the message.

  Raises:
    X64NotEnabledError: jax is imported and jax_enable_x64 is off.
  """
  if not loaded("jax"):
    return
  if not bool(sys.modules["jax"].config.read("jax_enable_x64")):
    raise X64NotEnabledError(x64_off_message(where))
