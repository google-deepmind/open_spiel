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

"""Artifact formats: the on-disk layouts `load_artifact` can read and write.

The library writes one layout, the native artifact format (`ckpt.artifact`). A
deployment that must keep reading or writing another layout (its own older
checkpoints, a format another tool reads) registers an `ArtifactFormat`
instead of changing the loader. Registered formats are tried before the
native one, in registration order; the first whose `detect` is True reads the
step.

  formats.register_format(MyFormat())
  formats.set_default_writer(MyFormat.name)   # write_artifact now writes it

Every format returns the same `core.types.Manifest` and `Artifact`, so nothing
downstream of the loader knows which layout a step came from.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence

from open_spiel.python.scale.core import types

# The native format's name here; its manifest tag is artifact.FORMAT_TAG.
NATIVE = "artifact"


class ArtifactFormat(Protocol):
  """One on-disk artifact layout.

  Attributes:
    name: the format's id, used by `set_default_writer`.
  """

  name: str

  def detect(self, ckpt_dir: str, step: int) -> bool:
    """Whether `ckpt_dir` holds `step` in this format.

    Args:
      ckpt_dir: the checkpoint dir.
      step: the step.

    Returns:
      True when it does.
    """

  def has_any(self, ckpt_dir: str) -> bool:
    """Whether `ckpt_dir` holds at least one step in this format.

    Args:
      ckpt_dir: the checkpoint dir.

    Returns:
      True when it does.
    """

  def load_manifest(
      self, ckpt_dir: str, step: int, *, verify: bool = True
  ) -> types.Manifest:
    """The step's Manifest without restoring params.

    Args:
      ckpt_dir: the checkpoint dir.
      step: the step.
      verify: give the params files the integrity checks that need no restore.

    Returns:
      The types.Manifest.
    """

  def load_artifact(
      self, ckpt_dir: str, step: int, *, verify: bool = True
  ) -> types.Artifact:
    """The step's Manifest, its rebuilt policy and its params.

    Args:
      ckpt_dir: the checkpoint dir.
      step: the step.
      verify: check the params against the recorded digest.

    Returns:
      The types.Artifact.
    """

  def write(
      self,
      ckpt_dir: str,
      step: int,
      policy: Any,
      params: Any,
      manifest: types.Manifest,
  ) -> types.Manifest:
    """Writes a step (optional: a read-only format omits it).

    Args:
      ckpt_dir: the checkpoint dir.
      step: the step.
      policy: the policy.
      params: the params.
      manifest: every field but the params digest.

    Returns:
      The manifest as written.
    """

  def clear_partial(self, ckpt_dir: str, step: int, tag: str) -> Optional[str]:
    """Moves aside the files of a write of `step` that died before it committed.

    Never deletes them.

    Args:
      ckpt_dir: the checkpoint dir.
      step: the step.
      tag: names the moved-aside files.

    Returns:
      Where they went, or None.
    """


_REGISTERED: list[ArtifactFormat] = []
_NATIVE: list[ArtifactFormat] = []
_WRITER = {"name": NATIVE}


def register_format(fmt: ArtifactFormat) -> None:
  """Adds a format, tried before the native one.

  Registering the same object twice is a no-op.

  Args:
    fmt: the format.

  Raises:
    ValueError: a second format with a registered name.
  """
  if any(f is fmt for f in _REGISTERED):
    return
  if any(f.name == fmt.name for f in _REGISTERED + _NATIVE):
    raise ValueError(
        f"an artifact format named {fmt.name!r} is already registered"
    )
  _REGISTERED.append(fmt)


def set_native(fmt: ArtifactFormat) -> None:
  """Installs the native format (`ckpt.artifact` calls this once).

  Args:
    fmt: the native format.
  """
  assert fmt.name == NATIVE, fmt.name
  _NATIVE[:] = [fmt]


def set_default_writer(name: str) -> None:
  """Makes `write_artifact` write format `name` (default: the native format).

  Args:
    name: a registered format name.

  Raises:
    KeyError: no format is registered under `name`.
  """
  if name != NATIVE and not any(f.name == name for f in _REGISTERED):
    raise KeyError(f"no artifact format named {name!r} (registered: {names()})")
  _WRITER["name"] = name


def default_writer_name() -> str:
  """The name of the format `write_artifact` writes.

  Returns:
    The format name.
  """
  return _WRITER["name"]


def names() -> tuple[str, ...]:
  """Registered format names, in the order they are tried.

  Returns:
    The names.
  """
  return tuple(f.name for f in in_order())


def in_order() -> Sequence[ArtifactFormat]:
  """Registered formats, then the native one.

  Returns:
    The formats.
  """
  return tuple(_REGISTERED) + tuple(_NATIVE)


def get(name: str) -> ArtifactFormat:
  """The format named `name`.

  Args:
    name: the format name.

  Returns:
    The ArtifactFormat.

  Raises:
    KeyError: no format is registered under `name`.
  """
  for f in in_order():
    if f.name == name:
      return f
  raise KeyError(f"no artifact format named {name!r} (registered: {names()})")


def writer() -> ArtifactFormat:
  """The format `write_artifact` writes.

  Returns:
    The ArtifactFormat.

  Raises:
    TypeError: the format is read-only.
  """
  fmt = get(_WRITER["name"])
  if not callable(getattr(fmt, "write", None)):
    raise TypeError(f"artifact format {fmt.name!r} is read-only")
  return fmt


def detect(ckpt_dir: str, step: int) -> Optional[ArtifactFormat]:
  """The first format that holds `ckpt_dir`@`step`, or None.

  Args:
    ckpt_dir: the checkpoint dir.
    step: the step.

  Returns:
    The ArtifactFormat, or None.
  """
  for f in in_order():
    if f.detect(ckpt_dir, step):
      return f
  return None
