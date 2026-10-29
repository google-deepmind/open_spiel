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

"""Params and learner-state codecs: how one array framework's values reach disk.

An artifact format decides where files go and what the manifest says; a codec
decides how the values of one framework are written, read back, hashed and
compared. Keeping the two apart lets any format hold params of any registered
framework, and lets a deployment add a framework without touching a format.

  codecs.register_params_codec("my_framework", MyParamsCodec())
  codec = codecs.params_codec(policy.spec.framework)
  digest = codec.write(path, policy, params)
  params = codec.read(path, policy, digest)

The "jax" codecs and a "numpy" params codec are built in. JAX params are an
orbax checkpoint directory, so
arrays sharded over a mesh are written and read natively. The params digest is
`tree_sha256(params)`, computed in memory from the tree, so writing never reads
the files back.

Dtypes round-trip exactly: an f64 leaf restores as f64 and an int64 counter as
int64. orbax casts a stored leaf to the target dtype silently, so the stored
dtypes are compared with the architecture's before restoring, and any
difference raises.

orbax marks a finished checkpoint directory with an atomic rename on a local
disk and with a `commit_success.txt` object on gs://. A directory written
locally and then copied to gs:// has no marker, and orbax refuses to restore
it. `save_tree` asserts the marker after every write and the JAX codec refuses
a params directory without it, so such a copy fails at its first read. Write
artifacts meant for gs:// to gs:// directly.

NumPy params (a host policy's, framework "numpy") are a flat mapping from
string keys to numpy arrays, written as one .npz file (read without pickle);
their digest is `flat_sha256(params)`. The "numpy" framework has no learner
state codec: a learner that trains NumPy params registers one.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from typing import Any, Mapping, Optional, Protocol

from etils import epath
import numpy as np

from open_spiel.python.scale import plugins
from open_spiel.python.scale.ckpt import snapshot
from open_spiel.python.scale.core import frameworks


class ArtifactError(RuntimeError):
  """An artifact or run checkpoint is incomplete, malformed or unverified."""


class ParamsCodec(Protocol):
  """Writes and reads one framework's policy params.

  Attributes:
    name: the codec id recorded in a manifest (`params_codec`).
    suffix: appended to the params path a format chooses ("" for a directory).
  """

  name: str
  suffix: str

  def write(self, path: epath.Path, policy: Any, params: Any) -> str:
    """Writes `params` at `path` (which must not exist).

    Args:
      path: the params path.
      policy: the policy the params belong to.
      params: the params.

    Returns:
      The params digest.
    """

  def read(self, path: epath.Path, policy: Any, digest: Optional[str]) -> Any:
    """The params at `path`, shaped as `policy` expects.

    Args:
      path: the params path.
      policy: the policy the params belong to.
      digest: the recorded digest, verified when given.

    Returns:
      The params.

    Raises:
      ArtifactError: the params do not match `digest`.
    """

  def read_host(
      self, path: epath.Path, policy: Any, digest: Optional[str]
  ) -> Any:
    """The stored params as host values (`snapshot.to_host` form).

    They are read on their own: nothing is loaded into `policy`.

    Args:
      path: the params path.
      policy: the policy the params belong to.
      digest: the recorded digest, verified when given.

    Returns:
      The host params.
    """

  def check(self, path: epath.Path, digest: str) -> None:
    """A cheap integrity check that needs no policy.

    Args:
      path: the params path.
      digest: the recorded digest.

    Raises:
      ArtifactError: the files at `path` are missing, incomplete or (where the
        digest is a file hash) do not match `digest`.
    """

  def to_host(self, policy: Any, params: Any) -> Any:
    """What `write` would store for (policy, params), as host values.

    Args:
      policy: the policy the params belong to.
      params: the params.

    Returns:
      The host params.
    """


class StateCodec(Protocol):
  """Writes and reads one framework's learner state (its `state_dict`).

  Attributes:
    suffix: the state file's suffix.
  """

  suffix: str

  def save(self, path: epath.Path, learner: Any, state: Any) -> str:
    """Writes learner.state_dict(state) at `path` (which must not exist).

    Args:
      path: the state path.
      learner: the learner.
      state: the learner state.

    Returns:
      The path written.
    """

  def restore(self, path: epath.Path, learner: Any) -> Any:
    """The learner state stored at `path` (through learner.load_state).

    Args:
      path: the state path.
      learner: the learner.

    Returns:
      The learner state.
    """

  def read_host(self, path: epath.Path, learner: Any, state: Any) -> Any:
    """The stored state dict as host values, read without touching the learner.

    Args:
      path: the state path.
      learner: the learner.
      state: supplies a restore template where one is needed.

    Returns:
      The host state dict.
    """


_PARAMS: dict[str, ParamsCodec] = {}
_STATE: dict[str, StateCodec] = {}


def _framework_name(framework: Any) -> str:
  """A framework name, checked to be a non-empty string.

  A framework name, checked to be a non-empty string (a str subclass,
  such as a str enum member, is reduced to its plain string value).

  Args:
    framework: a framework name (str or str subclass).

  Returns:
    The plain string.
  """
  if not isinstance(framework, str) or not framework:
    raise ValueError(f"framework {framework!r} is not a non-empty string")
  return str.__str__(framework)


def register_params_codec(framework: Any, codec: ParamsCodec) -> None:
  """Makes `codec` the params codec of `framework`.

  Args:
    framework: the framework name.
    codec: the ParamsCodec.

  Raises:
    ValueError: a second, different codec for the same framework.
  """
  name = _framework_name(framework)
  if _PARAMS.get(name, codec) is not codec:
    raise ValueError(
        f"a params codec for framework {name!r} is already registered"
    )
  _PARAMS[name] = codec


def register_state_codec(framework: Any, codec: StateCodec) -> None:
  """Makes `codec` the learner-state codec of `framework` (one per framework).

  Args:
    framework: the framework name.
    codec: the StateCodec.

  Raises:
    ValueError: a second, different codec for the same framework.
  """
  name = _framework_name(framework)
  if _STATE.get(name, codec) is not codec:
    raise ValueError(
        f"a state codec for framework {name!r} is already registered"
    )
  _STATE[name] = codec


def _load_plugins() -> None:
  plugins.load_env()


def params_codec(framework: Any) -> ParamsCodec:
  """The params codec of `framework`.

  Args:
    framework: the framework name.

  Returns:
    The ParamsCodec.

  Raises:
    ArtifactError: an unregistered framework.
  """
  name = _framework_name(framework)
  if name not in _PARAMS:
    _load_plugins()
  if name not in _PARAMS:
    raise ArtifactError(
        f"no params codec for framework {name!r} (registered: "
        f"{sorted(_PARAMS)}); a plugin registers one with "
        "ckpt.codecs.register_params_codec"
    )
  return _PARAMS[name]


def params_codec_by_name(name: str) -> ParamsCodec:
  """The registered params codec whose `name` is `name`.

  That is a manifest's `params_codec`.

  Args:
    name: the codec name.

  Returns:
    The ParamsCodec.

  Raises:
    ArtifactError: no registered codec has that name.
  """
  for _ in range(2):
    for codec in _PARAMS.values():
      if codec.name == name:
        return codec
    _load_plugins()
  raise ArtifactError(
      f"no params codec named {name!r} (registered: "
      f"{sorted(c.name for c in _PARAMS.values())})"
  )


def registered_params_codecs() -> tuple[ParamsCodec, ...]:
  """Every registered params codec.

  Returns:
    The codecs.
  """
  return tuple(_PARAMS.values())


def state_codec(framework: Any) -> StateCodec:
  """The learner-state codec of `framework`.

  Args:
    framework: the framework name.

  Returns:
    The StateCodec.

  Raises:
    ArtifactError: an unregistered framework.
  """
  name = _framework_name(framework)
  if name not in _STATE:
    _load_plugins()
  if name not in _STATE:
    raise ArtifactError(
        f"no learner-state codec for framework {name!r} (registered: "
        f"{sorted(_STATE)}); a plugin registers one with "
        "ckpt.codecs.register_state_codec"
    )
  return _STATE[name]


# -- orbax trees ---------------------------------------------------------------


def _jax():
  """The jax module, imported on first use.

  Returns:
    jax.
  """
  # Lazy: ckpt imports without jax (jax_free_control_test).
  import jax  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  frameworks.assert_x64_if_loaded("ckpt (a JAX tree)")
  return jax


def _orbax():
  """The orbax.checkpoint module, imported on first use.

  Returns:
    orbax.checkpoint.
  """
  # Lazy: ckpt imports without orbax (jax_free_control_test).
  import orbax.checkpoint as ocp  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  return ocp


#: The domain-separation tag hashed first into every tree_sha256.
TREE_SHA_TAG = b"params_tree_sha256\0"


def tree_sha256(tree: Any, tag: Optional[bytes] = None) -> str:
  """sha256 of a pytree of arrays, computed in memory.

  The digest covers a tag (TREE_SHA_TAG), then per leaf in flatten order its
  key path, dtype name, shape and the sha256 of its C-order little-endian
  bytes. Equal trees (structure, dtypes, shapes and bits) hash equal on any
  device layout.

  Args:
    tree: a pytree of arrays.
    tag: the domain-separation tag; None is TREE_SHA_TAG (read at call time).
      A reader of digests another deployment recorded passes its tag.

  Returns:
    The hex digest.

  Raises:
    ArtifactError: on a big-endian host, or for a leaf that is not fully
      addressable (a multi-host array must be gathered first).
  """
  if sys.byteorder != "little":
    raise ArtifactError(
        "tree_sha256 hashes little-endian bytes; this host is big-endian"
    )
  h = hashlib.sha256(TREE_SHA_TAG if tag is None else tag)
  jax = _jax()
  for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
    if isinstance(leaf, jax.Array) and not leaf.is_fully_addressable:
      raise ArtifactError(
          f"tree_sha256: leaf {jax.tree_util.keystr(path)} is not fully "
          "addressable (a multi-host array): gather it before hashing"
      )
    a = np.ascontiguousarray(np.asarray(leaf))
    h.update(jax.tree_util.keystr(path).encode())
    h.update(b"\0")
    h.update(np.dtype(a.dtype).name.encode())
    h.update(b"\0")
    h.update(json.dumps([int(d) for d in a.shape]).encode())
    h.update(b"\0")
    h.update(hashlib.sha256(a.tobytes()).digest())
  return h.hexdigest()


def _signature(tree: Any) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
  """{leaf path: (shape, dtype)} of a pytree of arrays (or orbax metadata).

  Args:
    tree: A pytree of arrays or orbax metadata.

  Returns:
    {leaf path: (shape, dtype)}.
  """
  jax = _jax()
  return {
      jax.tree_util.keystr(path): (tuple(leaf.shape), np.dtype(leaf.dtype))
      for path, leaf in jax.tree_util.tree_leaves_with_path(tree)
  }


def compare_signatures(got: Any, want: Any, what: str, where: Any) -> None:
  """Checks that two pytrees have the same leaf paths, shapes and dtypes.

  Args:
    got: the tree checked.
    want: the architecture's tree.
    what: names `got` in errors.
    where: names the artifact in errors.

  Raises:
    ArtifactError: they differ.
  """
  got, want = _signature(got), _signature(want)
  if set(got) != set(want):
    raise ArtifactError(
        f"{where}: {what} leaves {sorted(got)} != architecture leaves "
        f"{sorted(want)}"
    )
  for k in sorted(want):
    if got[k] != want[k]:
      (gs, gd), (ws, wd) = got[k], want[k]
      raise ArtifactError(
          f"{where}: {what} leaf {k} is {gd}{list(gs)}, architecture wants "
          f"{wd}{list(ws)}"
      )


#: The file orbax writes last into a checkpoint directory it cannot rename
#: atomically (gs://); the directory is complete only once it exists.
COMMIT_SUCCESS_FILE = "commit_success.txt"
#: The name suffix of a directory orbax is still writing (renamed on commit).
TMP_DIR_SUFFIX = ".orbax-checkpoint-tmp"


def needs_commit_file(path: epath.Path) -> bool:
  """Whether a checkpoint at `path` is complete only with COMMIT_SUCCESS_FILE.

  orbax commits a gs:// checkpoint by writing that file last (gs:// has no
  atomic rename) and any other path by renaming its temporary directory.
  This package owns the rule, so a new orbax release cannot move it; tests
  patch this function to give a local path gs:// semantics.

  Args:
    path: the checkpoint directory.

  Returns:
    True for a gs:// path.
  """
  return path.as_posix().startswith("gs://")


def params_finalized(path: Any) -> bool:
  """Whether the orbax checkpoint directory `path` is complete.

  The rule orbax applies (its `is_path_finalized`): the directory exists, and
  when `needs_commit_file` it holds COMMIT_SUCCESS_FILE, otherwise its name
  is not an orbax temporary's.

  Args:
    path: the checkpoint directory.

  Returns:
    True when it is complete.
  """
  path = epath.Path(path)
  if not path.is_dir():
    return False
  if needs_commit_file(path):
    return (path / COMMIT_SUCCESS_FILE).exists()
  return TMP_DIR_SUFFIX not in path.name


def save_tree(path: Any, tree: Any) -> None:
  """Write-once orbax save of a pytree of arrays.

  Args:
    path: the checkpoint directory.
    tree: the pytree.

  Raises:
    ArtifactError: if `path` exists, or orbax does not consider the written
      directory complete.
  """
  path = epath.Path(path)
  if path.exists():
    raise ArtifactError(f"{path} already exists (write-once)")
  path.parent.mkdir(parents=True, exist_ok=True)
  ckptr = _orbax().StandardCheckpointer()
  try:
    ckptr.save(path, tree)
    ckptr.wait_until_finished()
  finally:
    ckptr.close()
  if not params_finalized(path):
    raise ArtifactError(
        f"{path}: orbax wrote it but does not consider it complete"
    )


def _stored_tree(ckptr, path: epath.Path) -> Any:
  """The stored tree structure of the checkpoint at `path`.

  Args:
    ckptr: The orbax checkpointer.
    path: The checkpoint path.

  Returns:
    The tree metadata.
  """
  meta = ckptr.metadata(path)
  meta = getattr(meta, "item_metadata", meta)
  return getattr(meta, "tree", meta)


def restore_tree(path: Any, template: Any) -> Any:
  """The orbax tree at `path`, restored at `template`'s leaves.

  Args:
    path: the checkpoint directory.
    template: the tree whose leaves set the restored shapes and dtypes.

  Returns:
    The restored tree.

  Raises:
    ArtifactError: if a stored leaf's shape or dtype differs from the
      template's (orbax would otherwise cast it silently).
  """
  path = epath.Path(path)
  ocp = _orbax()
  ckptr = ocp.StandardCheckpointer()
  try:
    compare_signatures(_stored_tree(ckptr, path), template, "stored", path)
    tree = ckptr.restore(
        path,
        _jax().tree_util.tree_map(ocp.utils.to_shape_dtype_struct, template),
    )
  finally:
    ckptr.close()
  compare_signatures(tree, template, "restored", path)
  return tree


def jax_host(tree: Any) -> dict[str, Any]:
  """A pytree's leaves as host numpy arrays, keyed by leaf path.

  Args:
    tree: a pytree.

  Returns:
    {leaf path: host array}.
  """
  jax = _jax()
  return {
      jax.tree_util.keystr(p): snapshot.to_host(x)
      for p, x in jax.tree_util.tree_leaves_with_path(tree)
  }


EMPTY_PARAMS_MARKER = "EMPTY_PARAMS"


class JaxParamsCodec:
  """Params of a JAX policy: an orbax directory, digest `tree_sha256`.

  A params-free architecture writes a directory holding only an
  EMPTY_PARAMS marker (orbax refuses an empty tree).
  """

  name = "jax.orbax"
  suffix = ""

  def write(self, path: epath.Path, policy: Any, params: Any) -> str:
    """Writes `params` at `path` (which must not exist).

    Args:
      path: the params path.
      policy: the policy the params belong to.
      params: the params.

    Returns:
      The params digest.

    Raises:
      ArtifactError: `path` exists, or the params do not match the policy's.
    """
    path = epath.Path(path)
    if path.exists():
      raise ArtifactError(f"{path} already exists (write-once)")
    compare_signatures(
        params,
        policy.init_params(_jax().random.key(0)),
        "params",
        policy.spec.arch_name,
    )
    digest = tree_sha256(params)
    if not _jax().tree_util.tree_leaves(params):
      path.mkdir(parents=True, exist_ok=False)
      (path / EMPTY_PARAMS_MARKER).write_bytes(b"{}")
      return digest
    save_tree(path, params)
    return digest

  def check(self, path: epath.Path, digest: str) -> None:
    """Checks that the params dir is present and complete.

    Args:
      path: the params path.
      digest: unused; a tree digest needs the restored tree (read checks it).

    Raises:
      FileNotFoundError: the params are missing.
      ArtifactError: the params dir is incomplete.
    """
    del digest  # a tree digest needs the restored tree; read() checks it
    path = epath.Path(path)
    if not path.exists():
      raise FileNotFoundError(f"params {path} are missing")
    if (path / EMPTY_PARAMS_MARKER).exists():
      return
    if not params_finalized(path):
      raise ArtifactError(
          f"{path}: incomplete orbax checkpoint (no commit marker: on gs:// "
          "orbax needs "
          "commit_success.txt, which a local write copied to gs:// lacks; "
          "rewrite it through "
          "orbax at its final path)"
      )

  def _restore(self, path: epath.Path, policy: Any) -> Any:
    """Restores the params at `path` in `policy`'s structure.

    Args:
      path: The params folder.
      policy: The policy whose params structure to restore.

    Returns:
      The params.
    """
    template = policy.init_params(_jax().random.key(0))
    if (path / EMPTY_PARAMS_MARKER).exists():
      compare_signatures({}, template, "stored", path)
      return template
    return restore_tree(path, template)

  def read(self, path: epath.Path, policy: Any, digest: Optional[str]) -> Any:
    """The params at `path`, shaped as `policy` expects.

    Args:
      path: the params path.
      policy: the policy the params belong to.
      digest: the recorded digest, verified when given.

    Returns:
      The params.

    Raises:
      ArtifactError: the params fail to restore or do not match `digest`.
      Exception: a restore error, re-raised as is when there is no digest.
    """
    path = epath.Path(path)
    self.check(path, digest or "")
    try:
      params = self._restore(path, policy)
    except ArtifactError:
      raise
    except Exception as e:
      # Corrupt files: the digest cannot be checked; re-raised, chained.
      if digest is None:
        raise
      raise ArtifactError(
          f"{path}: params do not restore ({type(e).__name__}: {e}); "
          f"params digest {digest} cannot be verified"
      ) from e
    if digest is not None:
      got = tree_sha256(params)
      if got != digest:
        raise ArtifactError(
            f"{path}: restored params tree_sha256 {got} does not match the "
            f"recorded params digest {digest}"
        )
    return params

  def read_host(
      self, path: epath.Path, policy: Any, digest: Optional[str]
  ) -> Any:
    """The stored params as host values (`snapshot.to_host` form).

    They are read on their own: nothing is loaded into `policy`.

    Args:
      path: the params path.
      policy: the policy the params belong to.
      digest: the recorded digest, verified when given.

    Returns:
      The host params.
    """
    return jax_host(self.read(path, policy, digest))

  def to_host(self, policy: Any, params: Any) -> Any:
    """What `write` would store for (policy, params), as host values.

    Args:
      policy: the policy the params belong to.
      params: the params.

    Returns:
      The host params.
    """
    del policy
    return jax_host(params)


class JaxStateCodec:
  """Learner state of a JAX learner: an orbax directory of its state_dict."""

  suffix = ""

  def save(self, path: epath.Path, learner: Any, state: Any) -> str:
    """Saves learner.state_dict(state) with orbax (write-once).

    Args:
      path: the state path.
      learner: the learner.
      state: the learner state.

    Returns:
      The path written.
    """
    save_tree(path, learner.state_dict(state))
    return str(path)

  def restore(self, path: epath.Path, learner: Any) -> Any:
    """The learner state stored at `path`, restored at a fresh state's leaves.

    Args:
      path: the state path.
      learner: the learner; it needs init_from_params.

    Returns:
      The learner state.

    Raises:
      TypeError: the learner has no init_from_params.
    """
    if not callable(getattr(learner, "init_from_params", None)):
      raise TypeError(
          f"{type(learner).__name__} has no init_from_params; a JAX learner "
          "needs it to resume (core.types.Learner)"
      )
    fresh = learner.init_from_params(
        learner.policy().init_params(_jax().random.key(0))
    )
    return learner.load_state(restore_tree(path, learner.state_dict(fresh)))

  def read_host(self, path: epath.Path, learner: Any, state: Any) -> Any:
    """The stored state dict as host values.

    Args:
      path: the state path.
      learner: the learner.
      state: the restore template's state.

    Returns:
      The host state dict.
    """
    return snapshot.to_host(restore_tree(path, learner.state_dict(state)))


def flat_sha256(params: Any) -> str:
  """sha256 of a flat {str: array} mapping. Needs no jax.

  TREE_SHA_TAG, then per key in sorted order the key, the dtype name, the
  shape and the sha256 of the array's C-order little-endian bytes.

  Args:
    params: the flat mapping.

  Returns:
    The hex digest.

  Raises:
    ArtifactError: on a big-endian host.
  """
  if sys.byteorder != "little":
    raise ArtifactError(
        "flat_sha256 hashes little-endian bytes; this host is big-endian"
    )
  h = hashlib.sha256(TREE_SHA_TAG)
  for key in sorted(_flat(params, "flat_sha256")):
    a = np.ascontiguousarray(np.asarray(params[key]))
    h.update(key.encode())
    h.update(b"\0")
    h.update(np.dtype(a.dtype).name.encode())
    h.update(b"\0")
    h.update(json.dumps([int(d) for d in a.shape]).encode())
    h.update(b"\0")
    h.update(hashlib.sha256(a.tobytes()).digest())
  return h.hexdigest()


def _flat(params: Any, where: Any) -> dict[str, Any]:
  """Params as {str: np.ndarray}; anything else raises ArtifactError.

  Args:
    params: The params.
    where: The context for errors.

  Returns:
    {key: array}.

  Raises:
    ArtifactError: The params are not a flat mapping of numeric arrays.
  """
  if not isinstance(params, Mapping) or not all(
      isinstance(k, str) for k in params
  ):
    raise ArtifactError(
        f"{where}: NumPy params are a flat mapping from str keys to arrays, "
        f"got {type(params).__name__}"
    )
  out = {}
  for k, v in params.items():
    a = np.asarray(v)
    if a.dtype == object:
      raise ArtifactError(
          f"{where}: NumPy params leaf {k!r} is not a numeric array"
      )
    out[k] = a
  return out


def _compare_flat(
    got: dict[str, Any], want: dict[str, Any], what: str, where: Any
) -> None:
  """Checks flat params against the architecture's keys, shapes and dtypes.

  Args:
    got: the flat params checked.
    want: the architecture's flat params.
    what: names the checked params in errors.
    where: names the location in errors.

  Raises:
    ArtifactError: a key, shape or dtype differs.
  """

  def sig(d):
    """The (shape, dtype) of each array.

    Args:
      d: {key: array}.

    Returns:
      {key: (shape, dtype)}.
    """
    return {k: (tuple(v.shape), np.dtype(v.dtype)) for k, v in d.items()}

  got, want = sig(got), sig(want)
  if set(got) != set(want):
    raise ArtifactError(
        f"{where}: {what} keys {sorted(got)} != architecture keys "
        f"{sorted(want)}"
    )
  for k in sorted(want):
    if got[k] != want[k]:
      (gs, gd), (ws, wd) = got[k], want[k]
      raise ArtifactError(
          f"{where}: {what} leaf {k} is {gd}{list(gs)}, architecture wants "
          f"{wd}{list(ws)}"
      )


class NumpyParamsCodec:
  """Params of a NumPy policy: one .npz file of a flat mapping.

  Params of a NumPy policy: one .npz file of a flat {str: array} mapping,
  digest `flat_sha256`. The policy's init_params(None) gives the expected
  keys, shapes and dtypes.
  """

  name = "numpy.npz"
  suffix = ".npz"

  def write(self, path: epath.Path, policy: Any, params: Any) -> str:
    """Writes `params` at `path` (which must not exist).

    Args:
      path: the params path.
      policy: the policy the params belong to.
      params: the params.

    Returns:
      The params digest.

    Raises:
      ArtifactError: `path` exists, or the params do not match the policy's.
    """
    path = epath.Path(path)
    if path.exists():
      raise ArtifactError(f"{path} already exists (write-once)")
    flat = _flat(params, path)
    _compare_flat(
        flat,
        _flat(policy.init_params(None), policy.spec.arch_name),
        "params",
        policy.spec.arch_name,
    )
    digest = flat_sha256(flat)
    buf = io.BytesIO()
    np.savez(buf, **flat)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    tmp.write_bytes(buf.getvalue())
    tmp.rename(path)
    return digest

  def check(self, path: epath.Path, digest: str) -> None:
    """Checks that the params file is present.

    Args:
      path: the params path.
      digest: unused; read checks it on the loaded arrays.

    Raises:
      FileNotFoundError: the params are missing.
    """
    del digest  # read() checks it on the loaded arrays
    if not epath.Path(path).exists():
      raise FileNotFoundError(f"params {path} are missing")

  def read(self, path: epath.Path, policy: Any, digest: Optional[str]) -> Any:
    """The params at `path`, shaped as `policy` expects.

    Args:
      path: the params path.
      policy: the policy the params belong to.
      digest: the recorded digest, verified when given.

    Returns:
      The params.

    Raises:
      ArtifactError: the file is corrupt, or the params do not match the
        policy's or `digest`.
    """
    path = epath.Path(path)
    self.check(path, digest or "")
    try:
      with np.load(io.BytesIO(path.read_bytes()), allow_pickle=False) as z:
        params = {k: z[k] for k in z.files}
    except Exception as e:  # corrupt files: re-raised, chained
      raise ArtifactError(
          f"{path}: params do not load ({type(e).__name__}: {e})"
      ) from e
    _compare_flat(
        params,
        _flat(policy.init_params(None), policy.spec.arch_name),
        "stored",
        path,
    )
    if digest is not None:
      got = flat_sha256(params)
      if got != digest:
        raise ArtifactError(
            f"{path}: loaded params flat_sha256 {got} does not match the "
            f"recorded params digest {digest}"
        )
    return params

  def read_host(
      self, path: epath.Path, policy: Any, digest: Optional[str]
  ) -> Any:
    """The stored params as host values (`snapshot.to_host` form).

    They are read on their own: nothing is loaded into `policy`.

    Args:
      path: the params path.
      policy: the policy the params belong to.
      digest: the recorded digest, verified when given.

    Returns:
      The host params.
    """
    return self.read(path, policy, digest)

  def to_host(self, policy: Any, params: Any) -> Any:
    """What `write` would store for (policy, params), as host values.

    Args:
      policy: the policy the params belong to.
      params: the params.

    Returns:
      The host params.
    """
    del policy
    return dict(_flat(params, "to_host"))


register_params_codec("jax", JaxParamsCodec())
register_params_codec("numpy", NumpyParamsCodec())
register_state_codec("jax", JaxStateCodec())
