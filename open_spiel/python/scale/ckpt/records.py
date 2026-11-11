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

"""Storage names for the fields of the JSON records a run persists.

A run writes small JSON records beside its checkpoints: the commit record
(train_state.json), the resume record (resume_state.json), the artifact
store's index entries and an artifact's lineage. The library names their
fields itself (for example "run_id"). A store that keeps a field under a
different key, or as a different type, maps it once, with a converter each
way, and every record is then written under the stored name and read back
under the library's:

  records.register_field_name("run_id", "job", to_stored=str.upper,
                              from_stored=str.lower)
  records.encode({"run_id": "a7", "step": 3})   # {"job": "A7", "step": 3}
  records.decode({"job": "A7", "step": 3})      # {"run_id": "a7", "step": 3}

decode inverts encode when from_stored inverts to_stored.

Only top-level fields are mapped. A record that uses the library's name
reads as it is.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, NamedTuple, Optional


class FieldName(NamedTuple):
  """How a field is stored.

  Attributes:
    stored: the stored name.
    to_stored: converts the value on write, or None.
    from_stored: converts the value on read, or None.
  """

  stored: str
  to_stored: Optional[Callable[[Any], Any]]
  from_stored: Optional[Callable[[Any], Any]]


_NAMES: dict[str, FieldName] = {}


def register_field_name(
    field: str,
    stored: str,
    *,
    to_stored: Optional[Callable[[Any], Any]] = None,
    from_stored: Optional[Callable[[Any], Any]] = None,
) -> None:
  """Stores `field` under the name `stored`.

  Args:
    field: the library's field name.
    stored: the name records are written and read under.
    to_stored: converts a non-None value as it is written (for example to
      the type the store uses); None keeps it.
    from_stored: converts a non-None value as it is read back, the inverse
      of to_stored; required with to_stored, so a record round-trips.

  Raises:
    ValueError: `field` already has another stored name, `stored` is already
      another field's stored name, or only one of the two converters is given.
  """
  if not (
      isinstance(field, str) and field and isinstance(stored, str) and stored
  ):
    raise ValueError(
        f"field names are non-empty strings, got {field!r} -> {stored!r}"
    )
  if (to_stored is None) != (from_stored is None):
    raise ValueError(
        f"record field {field!r}: give both to_stored and from_stored, or "
        "neither"
    )
  entry = FieldName(stored, to_stored, from_stored)
  old = _NAMES.get(field)
  if old is not None and old != entry:
    raise ValueError(
        f"record field {field!r} is already stored as {old.stored!r}"
    )
  for other, e in _NAMES.items():
    if other != field and e.stored == stored:
      raise ValueError(
          f"record field {field!r}: stored name {stored!r} is already "
          f"field {other!r}'s"
      )
  _NAMES[field] = entry


def stored_name(field: str) -> str:
  """The name `field` is written under.

  Args:
    field: the library's field name.

  Returns:
    The stored name.
  """
  entry = _NAMES.get(field)
  return field if entry is None else entry.stored


def encode(record: Mapping[str, Any]) -> dict[str, Any]:
  """`record` with its fields under their stored names.

  Args:
    record: a record under the library's names.

  Returns:
    The record to write.
  """
  out: dict[str, Any] = {}
  for k, v in record.items():
    entry = _NAMES.get(k)
    if entry is None:
      out[k] = v
      continue
    out[entry.stored] = (
        v if (v is None or entry.to_stored is None) else entry.to_stored(v)
    )
  return out


def decode(record: Mapping[str, Any]) -> dict[str, Any]:
  """`record` with its fields under the library's names.

  Args:
    record: a record as read.

  Returns:
    The record under the library's names.
  """
  back = {e.stored: (f, e) for f, e in _NAMES.items()}
  out: dict[str, Any] = {}
  for k, v in record.items():
    if k not in back:
      out[k] = v
      continue
    field, entry = back[k]
    out[field] = (
        v if (v is None or entry.from_stored is None) else entry.from_stored(v)
    )
  return out
