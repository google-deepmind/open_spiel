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

"""Host snapshots of serialisable state, compared bit for bit.

Verifying a checkpoint means comparing what is on disk with what the live
process would write. Restoring the file into the live object would overwrite
the very values being compared, and comparing handles (an id that names
weights held elsewhere) says nothing about the weights. This module turns both
sides into plain host values and compares those.

  live = snapshot.to_host(learner.state_dict(state))
  same = snapshot.equal(stored, live)

`to_host` maps nested mappings, lists, tuples and NamedTuples of arrays and
Python scalars to the same nesting with every array a host numpy array (tuples
become lists, a NamedTuple becomes a dict of its fields). Arrays are anything
with `shape` and `dtype` that numpy can convert. A value numpy cannot convert
directly (a tensor of another framework) is converted by a leaf codec that a
plugin registers with `register_leaf_codec`.

`equal(a, b)` holds when both have the same nesting and keys, every array has
the same dtype, shape and bytes (so a NaN equals the same NaN), and scalars
compare equal.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, NamedTuple

import numpy as np

_SCALARS = (bool, int, float, str, bytes, type(None), np.generic)


class LeafCodec(NamedTuple):
  """Converts one kind of foreign leaf to host values and back.

  encode(x) returns (array, meta): the leaf's bits as a host numpy array and a
  small picklable value with whatever else `decode` needs (for example a
  device name). decode(array, meta) rebuilds the leaf.

  Attributes:
    name: the codec's registered name.
    predicate: whether a leaf is of this kind.
    encode: leaf -> (host values, aux).
    decode: (host values, aux) -> leaf.
  """

  name: str
  predicate: Callable[[Any], bool]
  encode: Callable[[Any], tuple[np.ndarray, Any]]
  decode: Callable[[np.ndarray, Any], Any]


_CODECS: list[LeafCodec] = []


def register_leaf_codec(
    predicate: Callable[[Any], bool],
    encode: Callable[[Any], tuple[np.ndarray, Any]],
    decode: Callable[[np.ndarray, Any], Any],
    *,
    name: str,
) -> None:
  """Registers a codec for leaves where `predicate(x)` is True.

  Codecs are tried in registration order. Registering a second codec under
  the same name raises.

  Args:
    predicate: whether the codec claims a leaf.
    encode: leaf -> (array, aux).
    decode: (array, aux) -> leaf.
    name: the codec's unique name.

  Raises:
    ValueError: a codec named `name` is already registered.
  """
  assert isinstance(name, str) and name, name
  if any(c.name == name for c in _CODECS):
    raise ValueError(f"a leaf codec named {name!r} is already registered")
  _CODECS.append(LeafCodec(name, predicate, encode, decode))


def leaf_codec_for(x: Any) -> LeafCodec | None:
  """The first registered codec that claims `x`, or None.

  Args:
    x: a leaf.

  Returns:
    The LeafCodec, or None.
  """
  for c in _CODECS:
    if c.predicate(x):
      return c
  return None


def leaf_codec(name: str) -> LeafCodec:
  """The registered codec named `name`.

  Args:
    name: the codec name.

  Returns:
    The LeafCodec.

  Raises:
    KeyError: no codec is registered under `name`.
  """
  for c in _CODECS:
    if c.name == name:
      return c
  raise KeyError(
      f"no leaf codec named {name!r} (registered: {[c.name for c in _CODECS]})"
  )


def to_host(tree: Any) -> Any:
  """`tree` with every array a host numpy array (module doc).

  Args:
    tree: a tree of arrays, scalars and codec-claimed leaves.

  Returns:
    The host tree.

  Raises:
    TypeError: for a leaf that is neither an array, a scalar nor claimed by a
      registered leaf codec.
  """
  if isinstance(tree, Mapping):
    return {k: to_host(v) for k, v in tree.items()}
  if hasattr(tree, "_fields"):
    return {f: to_host(getattr(tree, f)) for f in tree._fields}
  if isinstance(tree, (list, tuple)):
    return [to_host(v) for v in tree]
  codec = leaf_codec_for(tree)
  if codec is not None:
    return codec.encode(tree)[0]
  if isinstance(tree, _SCALARS):
    return tree
  if hasattr(tree, "shape") and hasattr(tree, "dtype"):
    return np.array(tree)
  raise TypeError(f"to_host: unsupported leaf {type(tree).__name__}")


def equal(a: Any, b: Any) -> bool:
  """Bitwise, structural equality of two `to_host` trees (module doc).

  Args:
    a: a host tree.
    b: another host tree.

  Returns:
    True when the trees are equal.
  """
  if isinstance(a, dict) or isinstance(b, dict):
    return (
        isinstance(a, dict)
        and isinstance(b, dict)
        and set(a) == set(b)
        and all(equal(a[k], b[k]) for k in a)
    )
  if isinstance(a, list) or isinstance(b, list):
    return (
        isinstance(a, list)
        and isinstance(b, list)
        and len(a) == len(b)
        and all(equal(x, y) for x, y in zip(a, b))
    )
  if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
    return (
        isinstance(a, np.ndarray)
        and isinstance(b, np.ndarray)
        and a.dtype == b.dtype
        and a.shape == b.shape
        and a.tobytes() == b.tobytes()
    )
  if type(a) is not type(b):
    return False
  # bitwise, so a NaN equals the same NaN
  if isinstance(a, (float, np.generic)):
    return np.asarray(a).tobytes() == np.asarray(b).tobytes()
  return a == b
