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

"""The registry of PolicyOutput.aux keys whose dtype and shape are a contract.

A policy may return any extra heads in PolicyOutput.aux, and the Arena copies
the perspective slot's aux into TrajectoryBatch.aux. Keys that downstream code
consumes (a policy target, search values) are registered here with their
dtype and trailing shape, and the rollout drivers check every registered key
they copy at trace time: a wrong dtype or shape raises, and a non-finite value
on a decided row is a bad ply.

  aux.register_aux_key("search_value", (), "float32", writer="search operator")
  aux.reserve_prefix("search_")   # unregistered "search_*" keys are errors
  aux.spec("search_value").shape  # ()

Shapes are trailing shapes after the batch axis; the symbol "A" stands for
the env's number of actions.
"""

from typing import NamedTuple, Optional

from open_spiel.python.scale.core import types


class AuxSpec(NamedTuple):
  """A registered aux key: its dtype name, trailing shape and writer.

  Attributes:
    name: the key.
    shape: the trailing shape after the batch axis; "A" is num_actions.
    dtype: a numpy dtype name ("float32", "bool", ...).
    writer: which component emits it (documentation only).
  """

  name: str
  shape: tuple[int, ...]
  dtype: str
  writer: str


_SPECS: dict[str, AuxSpec] = {}
_RESERVED: list[str] = []


def register_aux_key(
    name: str, shape: tuple[int, ...], dtype, *, writer: str
) -> None:
  """Registers aux key `name`; re-registering the same spec has no effect.

  Raises:
    ValueError: the name is taken by a different spec, or a dimension is
      neither an int nor "A".

  Args:
    name: the aux key.
    shape: its shape after the row axis: ints or "A" (the action count).
    dtype: its dtype.
    writer: who writes it.
  """
  dims = tuple(shape)
  if any(not (isinstance(d, int) or d == "A") for d in dims):
    raise ValueError(f"aux {name!r}: dimensions are ints or 'A', got {dims}")
  spec_ = AuxSpec(
      str(name), dims, str(getattr(dtype, "name", dtype)), str(writer)
  )
  got = _SPECS.get(spec_.name)
  if got is not None and got[:3] != spec_[:3]:
    raise ValueError(f"aux key {name!r} is already registered as {got}")
  _SPECS[spec_.name] = spec_


def reserve_prefix(prefix: str) -> None:
  """Keys starting with `prefix` must be registered.

  The drivers and the learners refuse an unregistered one.

  Args:
    prefix: a non-empty key prefix.

  Raises:
    ValueError: prefix is empty.
  """
  if not prefix:
    raise ValueError("an empty prefix would reserve every key")
  if prefix not in _RESERVED:
    _RESERVED.append(prefix)


def is_reserved(name: str) -> bool:
  """Whether `name` lies in a reserved namespace.

  Args:
    name: an aux key.

  Returns:
    True if name starts with a reserved prefix.
  """
  return any(str(name).startswith(p) for p in _RESERVED)


def get(name: str) -> Optional[AuxSpec]:
  """The spec of key `name`, or None if it is not registered.

  Args:
    name: an aux key.

  Returns:
    the AuxSpec or None.
  """
  return _SPECS.get(name)


def spec(name: str) -> AuxSpec:
  """The spec of a registered key.

  Args:
    name: an aux key.

  Returns:
    the AuxSpec.

  Raises:
    KeyError: name is not registered (the message lists the registered keys).
  """
  if name not in _SPECS:
    raise KeyError(f"aux key {name!r} is not registered; registered: {names()}")
  return _SPECS[name]


def names() -> tuple[str, ...]:
  """The registered keys, sorted.

  Returns:
    the key names.
  """
  return tuple(sorted(_SPECS))


register_aux_key(types.AUX_TARGET_POLICY, ("A",), "float32", writer="policy")
