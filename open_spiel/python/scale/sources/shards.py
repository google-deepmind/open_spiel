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

"""Offline trajectory shards: ShardWriter and ShardReader.

Imitation from recorded play (a teacher's games, an engine's, humans')
needs batches on disk that any later policy can learn from. A shard
directory holds whole TrajectoryBatch rows [N, T, ...] with the InfoState
itself in steps.info, never an encoded observation, so a policy with an
encoder written later can still read the data.

    with shards.ShardWriter(out_dir, env, rules_sha=rs, info_schema_version=1,
                            producer="teacher@4000") as w:
      w.write(batch)
    src = shards.SequentialSource(shards.ShardReader(out_dir, env, rules_sha=rs,
                                                     info_schema_version=1))
    batch = src.next(key)

Layout:
  <dir>/manifest.json      format, format_version, game, rules_sha,
                           info_schema_version, producer (the id of the
                           policy or engine that played), num_steps, leaves
                           [{path, dtype, shape}] (shape = per-row trailing
                           shape), label_keys, aux_keys, init_carry (stored or
                           not), carry_owner ({arch_name, arch_kwargs} of the
                           network whose carry it is, or null), shards [{file,
                           rows, sha256}]
  <dir>/shard-00000.npz    one file per write: leaf_<i> arrays [rows, ...] in
                           manifest leaf order (read with allow_pickle=False)

The writer checks that steps.info has exactly the env's InfoState structure,
dtypes and shapes, and that every write has the same structure; labels and
aux are flat {key: array}. The reader rebuilds the tree from a template made
of the env's InfoState (no pickles, no stored classes), checks game,
rules_sha and info_schema_version against the caller's and every shard's
sha256, and refuses any mismatch (Error). Dtypes round-trip exactly;
bfloat16 leaves are refused (npz cannot hold them without pickle).

Recurrent carries. init_carry is stored only when it has leaves (a stateful
producer), with its owner (the PolicySpec of the network that computed it);
the reader needs `init_carry_like` for its structure. A carry is the hidden
state of the network that computed it, so a stateful student reuses the
stored carry only when carry_owner is its own architecture; otherwise the
student starts every row from its own init_carry, which is exact from a
row's first new_match ply on (every policy resets its state there). Shards
with a trained ply before a row's first new_match ply are refused. A
stateless student gets init_carry () whatever was stored.

Evaluation-deal exclusion (`sources.deals`): a writer given a DealIndex
drops every row with a hand that starts from an indexed deal (recorded in the
manifest's deal_exclusion block with the index's sha256); a reader given one
refuses a shard written without that index (unless index_at_write=False,
for a shard written outside ShardWriter), and any shard where a recomputed
row hits.

The InfoState template comes from env.init + env.info_state, or from
`info_like` for an env without dynamics (envs.stub.StubEnv, a game whose data
is produced outside this library). The reader reads every shard on the host
and holds the concatenated dataset in device memory for its lifetime.

Shards written outside ShardWriter (an external labeller, for example) can
carry extra checks in the manifest; `validator` runs one on the manifest and
the stored rows (for example, one that verifies a receipt the producer wrote).
"""

import datetime
import hashlib
import io
import json
from typing import Any, Callable, Optional

from etils import epath
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.ckpt import artifact
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.sources import deals

#: The manifest's "format" a ShardWriter writes and a ShardReader requires.
MANIFEST_FORMAT = "trajectory_shards"
FORMAT_VERSION = 1
MANIFEST = "manifest.json"
_ROW_TIME = 2  # leading [N, T] dims of every TrajectoryBatch leaf ...
_ROW = 1  # ... except bootstrap_step and init_carry ([N])


class Error(RuntimeError):
  """A shard directory is malformed, mismatched or corrupted."""


def artifact_id(producer: Any) -> str:
  """The producer's id.

  Args:
    producer: a non-empty string ("<run id>@<step>", "path@step", an engine
      name), a types.Manifest with a run id ("<run id>@<step>"), or a
      types.Artifact ("path@step").

  Returns:
    The id string.
  """
  if isinstance(producer, types.Artifact):
    assert producer.path, "artifact without a path"
    return f"{producer.path}@{int(producer.manifest.step)}"
  if isinstance(producer, types.Manifest):
    assert (
        producer.run_id is not None
    ), "a Manifest without a run id has no id; pass the Artifact"
    return f"{producer.run_id}@{int(producer.step)}"
  assert (
      isinstance(producer, str) and producer
  ), f"producer id must be a non-empty str: {producer!r}"
  return producer


def info_template(env: types.Env, info_like: Any = None) -> Any:
  """ShapeDtypeStructs of one seat's InfoState (unbatched).

  info_like's, else the env's (env.init + env.info_state; a stub env provides
  its own info_template()).

  Args:
    env: the env.
    info_like: an InfoState template, optional.

  Returns:
    The template tree.
  """
  if info_like is not None:

    def _sds(x):
      """The ShapeDtypeStruct of a leaf.

      Args:
        x: A leaf.

      Returns:
        The struct.
      """
      if hasattr(x, "shape") and hasattr(x, "dtype"):
        return jax.ShapeDtypeStruct(tuple(x.shape), x.dtype)
      a = np.asarray(x)
      return jax.ShapeDtypeStruct(a.shape, a.dtype)

    return jax.tree_util.tree_map(_sds, info_like)
  own = getattr(env, "info_template", None)
  if callable(own):
    return own()
  state = jax.eval_shape(env.init, jax.random.key(0))
  return jax.eval_shape(env.info_state, state, jnp.int32(0))


def _seatstep_template(info):
  """A SeatStep of placeholders (info = the InfoState template).

  Args:
    info: The InfoState template.

  Returns:
    The SeatStep.
  """
  z = 0
  return types.SeatStep(
      info=info,
      legal_mask=z,
      seat=z,
      to_move=z,
      new_hand=z,
      new_match=z,
      last_action=z,
      last_actor=z,
  )


def _template(info, label_keys, aux_keys):
  """A TrajectoryBatch of placeholders.

  Args:
    info: The InfoState template.
    label_keys: The label keys.
    aux_keys: The aux keys.

  Returns:
    The template.
  """
  z = 0
  steps = _seatstep_template(info)
  return types.TrajectoryBatch(
      steps=steps,
      init_carry=(),
      action=z,
      behavior_logits=z,
      behavior_logp=z,
      value_pred=z,
      target_policy=z,
      target_value=z,
      reward=z,
      done=z,
      valid=z,
      labels={k: z for k in label_keys},
      aux={k: z for k in aux_keys},
      bootstrap_step=_seatstep_template(info),
  )


def _paths(tree):
  """The key paths of the tree's leaves.

  Args:
    tree: A pytree.

  Returns:
    The paths.
  """
  return [
      jax.tree_util.keystr(p)
      for p, _ in jax.tree_util.tree_leaves_with_path(tree)
  ]


def _lead(path: str) -> int:
  """The leading axes of a leaf: rows, or rows and time.

  Args:
    path: The leaf's key path.

  Returns:
    The axis count.
  """
  return _ROW if path.startswith(".bootstrap_step") else _ROW_TIME


def _sha256(data: bytes) -> str:
  """The SHA-256 of `data`.

  Args:
    data: The bytes.

  Returns:
    The hex digest.
  """
  return hashlib.sha256(data).hexdigest()


def carry_owner_id(spec: types.PolicySpec) -> dict[str, Any]:
  """The identity of the network whose carry a shard stores.

  The manifest's carry_owner: its architecture name and kwargs.

  Args:
    spec: the network's PolicySpec.

  Returns:
    {arch_name, arch_kwargs}.
  """
  assert isinstance(spec, types.PolicySpec), type(spec)
  return json.loads(
      json.dumps(
          dict(arch_name=spec.arch_name, arch_kwargs=dict(spec.arch_kwargs)),
          sort_keys=True,
      )
  )


def unaligned_valid_plies(steps: types.SeatStep, valid) -> np.ndarray:
  """i64[N]: per row, the valid plies before its first new_match ply.

  All of them when the row has none: the plies a student starting from its own
  init_carry cannot reproduce.

  Args:
    steps: the SeatSteps [N, T].
    valid: bool[N, T].

  Returns:
    The counts.
  """
  nm = np.asarray(steps.new_match, np.bool_)
  v = np.asarray(valid, np.bool_)
  assert nm.shape == v.shape and nm.ndim == 2, (nm.shape, v.shape)
  before = np.cumsum(nm, axis=1) == 0  # plies before the first reset
  return np.sum(v & before, axis=1, dtype=np.int64)


def _info_signature(info, lead: int):
  """(treedef, [(trailing shape, dtype)]) of a batched InfoState.

  Args:
    info: The batched InfoState.
    lead: The leading axes to drop.

  Returns:
    (treedef, [(trailing shape, dtype)]).
  """
  leaves, treedef = jax.tree_util.tree_flatten(info)
  return treedef, [
      (tuple(np.shape(x))[lead:], np.dtype(x.dtype)) for x in leaves
  ]


class ShardWriter:
  """Writes TrajectoryBatches as shards.

  `close()` (or the context manager) writes the manifest. Write-once: refuses a
  directory that has a manifest.

  Attributes:
    root: the shard directory.
    env: the env the batches were played in.
    rules_sha: the rules' sha.
    info_schema_version: the InfoState schema version.
    producer: the producer's artifact_id.
    carry_owner: carry_owner_id of the network whose carry the batches store,
      or None.
    deal_exclusion: the sources.deals.DealIndex, or None.
  """

  def __init__(
      self,
      path: str,
      env: types.Env,
      *,
      rules_sha: str,
      info_schema_version: int,
      producer: Any,
      carry_owner: Optional[types.PolicySpec] = None,
      deal_exclusion: Any = None,
      info_like: Any = None,
  ):
    """Opens a shard directory for writing.

    Args:
      path: the shard directory.
      env: the env the batches were played in.
      rules_sha: the rules' sha.
      info_schema_version: the InfoState schema version.
      producer: the producer (artifact_id).
      carry_owner: the PolicySpec of the network whose carry the batches'
        init_carry is (required when it has leaves).
      deal_exclusion: a sources.deals.DealIndex (module doc).
      info_like: the InfoState template (module doc).

    Raises:
      Error: the directory already has a manifest, or an argument is
        refused.
    """
    x64.assert_x64("ShardWriter")
    assert (
        isinstance(rules_sha, str) and rules_sha
    ), f"rules_sha must be non-empty: {rules_sha!r}"
    assert (
        isinstance(info_schema_version, int) and info_schema_version >= 1
    ), info_schema_version
    self.root = epath.Path(str(path).rstrip("/"))
    if (self.root / MANIFEST).exists():
      raise Error(
          f"{self.root} already has a {MANIFEST} (shard directories are "
          "write-once)"
      )
    self.env = env
    self.rules_sha = rules_sha
    self.info_schema_version = info_schema_version
    self.producer = artifact_id(producer)
    self.carry_owner = (
        None if carry_owner is None else carry_owner_id(carry_owner)
    )
    info = info_template(env, info_like)
    self._info_sig = _info_signature(info, 0)
    self.deal_exclusion = deal_exclusion
    if deal_exclusion is not None and (
        deal_exclusion.game != env.name or deal_exclusion.rules_sha != rules_sha
    ):
      raise Error(
          f"deal index for {deal_exclusion.game} / {deal_exclusion.rules_sha}, "
          f"shards of {env.name} / {rules_sha}"
      )
    self._refused_rows = 0
    self._checked_hands = 0
    self._layout: Optional[dict[str, Any]] = None
    self._shards: list[dict[str, Any]] = []
    self._closed = False

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc, tb):
    if exc_type is None:
      self.close()

  def _check_and_layout(self, batch: types.TrajectoryBatch) -> dict[str, Any]:
    """Checks batch against the shard format and returns its layout.

    Args:
      batch: the TrajectoryBatch to write.

    Returns:
      The layout dict: num_steps, leaves, label_keys, aux_keys, init_carry.

    Raises:
      Error: batch does not fit the shard format.
    """
    if not isinstance(batch, types.TrajectoryBatch):
      raise Error(f"want a TrajectoryBatch, got {type(batch).__name__}")
    for name, part in (
        ("steps", batch.steps),
        ("bootstrap_step", batch.bootstrap_step),
    ):
      lead = _ROW if name == "bootstrap_step" else _ROW_TIME
      if _info_signature(part.info, lead) != self._info_sig:
        raise Error(
            f"{name}.info is not {self.env.name}'s InfoState "
            "(structure/shape/dtype): shards "
            "store the InfoState PyTree, not an encoded obs; got "
            f"{_info_signature(part.info, lead)}"
            f", want {self._info_sig}"
        )
    for group in ("labels", "aux"):
      d = getattr(batch, group)
      for k, v in d.items():
        if not hasattr(v, "shape"):
          raise Error(
              f"{group}[{k!r}] must be an array (flat {{key: array}}), got "
              f"{type(v)}"
          )
    main = batch._replace(init_carry=())
    leaves = jax.tree_util.tree_leaves_with_path(main)
    b, t = batch.valid.shape
    out = []
    for p, x in leaves:
      path = jax.tree_util.keystr(p)
      lead = _lead(path)
      shape = tuple(np.shape(x))
      if shape[:lead] != (b, t)[:lead]:
        raise Error(
            f"leaf {path} has shape {shape}, want leading {(b, t)[:lead]}"
        )
      out.append(
          dict(
              path=path, dtype=np.dtype(x.dtype).name, shape=list(shape[lead:])
          )
      )
    carry = [
        dict(
            path=jax.tree_util.keystr(p),
            dtype=np.dtype(x.dtype).name,
            shape=list(np.shape(x)[1:]),
        )
        for p, x in jax.tree_util.tree_leaves_with_path(batch.init_carry)
    ]
    if carry and self.carry_owner is None:
      raise Error(
          "the batches store an init_carry: name the network that computed it "
          "(ShardWriter carry_owner); a carry without its owner is unusable"
      )
    for leaf in out + carry:
      if leaf["dtype"] == "bfloat16":
        raise Error(
            f"leaf {leaf['path']} is bfloat16; npz cannot store it without "
            "pickle"
        )
    return dict(
        num_steps=int(t),
        leaves=out,
        label_keys=sorted(batch.labels),
        aux_keys=sorted(batch.aux),
        init_carry=carry,
    )

  def _exclude(
      self, batch: types.TrajectoryBatch
  ) -> Optional[types.TrajectoryBatch]:
    """Batch without its rows that start a hand from an indexed eval deal.

    None if no row is left; refuses rows whose deal is not in the row.

    Args:
      batch: the TrajectoryBatch.

    Returns:
      The kept rows, or None.

    Raises:
      Error: a row has decisions before its first hand start.
    """
    rows, hashes, unaligned = deals.batch_hashes(
        self.env.name, self.rules_sha, batch
    )
    if unaligned.any():
      raise Error(
          f"{int(unaligned.sum())} rows have decisions before their first hand "
          "start: their deal cannot be checked against the eval deal index "
          "(record "
          "hand-aligned rows: first_hand_only / restarted segments)"
      )
    self._checked_hands += len(hashes)
    bad = np.zeros((int(batch.valid.shape[0]),), bool)
    hit = self.deal_exclusion.hits(hashes)
    bad[rows[hit]] = True
    if bad.any():
      self._refused_rows += int(bad.sum())
      if bad.all():
        return None
      keep = np.flatnonzero(~bad)
      batch = jax.tree.map(lambda x: x[keep], batch)
    return batch

  def write(self, batch: types.TrajectoryBatch) -> None:
    """Writes one batch as a shard.

    Args:
      batch: the TrajectoryBatch.

    Raises:
      Error: the batch is refused or the shard file exists.
    """
    assert not self._closed, "writer is closed"
    if self.deal_exclusion is not None:
      batch = self._exclude(batch)
      if batch is None:
        return
    layout = self._check_and_layout(batch)
    if self._layout is None:
      self._layout = layout
    elif layout != self._layout:
      raise Error(
          "batch structure differs from the earlier writes of this directory: "
          f"{layout} vs {self._layout}"
      )
    host = jax.device_get((
        jax.tree.leaves(batch._replace(init_carry=())),
        jax.tree.leaves(batch.init_carry),
    ))
    arrays = {f"leaf_{i:05d}": np.asarray(x) for i, x in enumerate(host[0])}
    arrays.update(
        {f"carry_{i:05d}": np.asarray(x) for i, x in enumerate(host[1])}
    )
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    data = buf.getvalue()
    name = f"shard-{len(self._shards):05d}.npz"
    path = self.root / name
    if path.exists():
      raise Error(f"{path} exists (shard directories are write-once)")
    artifact.atomic_write(path, data)
    self._shards.append(
        dict(file=name, rows=int(batch.valid.shape[0]), sha256=_sha256(data))
    )

  def close(self) -> None:
    """Writes the manifest (once).

    Raises:
      Error: nothing was written.
    """
    if self._closed:
      return
    if not self._shards:
      raise Error("no batch was written; refusing an empty shard directory")
    lay = self._layout
    manifest = dict(
        format=MANIFEST_FORMAT,
        format_version=FORMAT_VERSION,
        game=self.env.name,
        rules_sha=self.rules_sha,
        info_schema_version=self.info_schema_version,
        producer=self.producer,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        num_steps=lay["num_steps"],
        leaves=lay["leaves"],
        label_keys=lay["label_keys"],
        aux_keys=lay["aux_keys"],
        init_carry=lay["init_carry"],
        carry_owner=self.carry_owner if lay["init_carry"] else None,
        shards=self._shards,
    )
    if self.deal_exclusion is not None:
      manifest["deal_exclusion"] = dict(
          index_sha256=self.deal_exclusion.sha256,
          refused_rows=int(self._refused_rows),
          checked_hands=int(self._checked_hands),
      )
    artifact.atomic_write(
        self.root / MANIFEST,
        json.dumps(manifest, indent=2, sort_keys=True).encode(),
    )
    self._closed = True


class ShardReader:
  """TrajectorySource over a shard directory.

  next(key): batch_size rows drawn uniformly without replacement (the row
  indices of the last draw are in `last_rows`). read_all() / rows(start,
  size): the stored rows in order (shard order, then row order).
  SequentialSource(reader): the stored batches in write order, one per next
  (with skip(n) for a resume), the BC replay of recorded teacher games.

  student: the PolicySpec of the learner the batches feed (module doc): a
  stateful student reuses a stored carry only when it owns it, else gets
  init_carry () (rows reset-aligned, checked here); a stateless student
  gets (). None: the stored rows as written (inspection, tests).
  `carry` records which: "stored", "student_init" or "none".

  Attributes:
    root: the shard directory.
    manifest: the parsed manifest.
    producer: the producer's artifact_id (from the manifest).
    num_steps: plies per row (T).
    batch_size: rows per `next`, or None.
    carry: "stored", "student_init" or "none".
    validation: the validator's result, or None without a validator.
    num_rows: the stored rows.
    last_rows: the row indices of the last `next` draw (np.int64), or None.
  """

  def __init__(
      self,
      path: str,
      env: types.Env,
      *,
      rules_sha: str,
      info_schema_version: int,
      batch_size: Optional[int] = None,
      init_carry_like: Any = None,
      student: Optional[types.PolicySpec] = None,
      deal_exclusion: Any = None,
      info_like: Any = None,
      validator: Optional[
          Callable[[epath.Path, dict[str, Any], list[dict[str, Any]]], Any]
      ] = None,
      index_at_write: bool = True,
  ):
    """Reads and checks a shard directory.

    Args:
      path: the shard directory.
      env: the env.
      rules_sha: the rules' sha the shards must carry.
      info_schema_version: the InfoState schema version the shards must carry.
      batch_size: rows per `next`.
      init_carry_like: the stored init_carry's template.
      student: the PolicySpec of the network that reads the shards.
      deal_exclusion: a sources.deals.DealIndex (module doc).
      info_like: the InfoState template (module doc).
      validator: validator(root, manifest, arrays) -> result, run on the
        manifest and the stored rows (one host array per manifest leaf) before
        they reach the device; it raises Error to refuse the shard, and its
        result is kept as `validation`. None: no extra check.
      index_at_write: with deal_exclusion, require the manifest to name the same
        index (ShardWriter applied it at write time). False: the shard was
        written outside ShardWriter and the read-time recompute alone enforces
        the index, so the manifest may name none (one naming another index is
        still refused).

    Raises:
      FileNotFoundError: there is no manifest.
      Error: the shard is refused.
    """
    x64.assert_x64("ShardReader")
    if student is not None:
      types.check_policy_admitted(env, student)
    self.root = epath.Path(str(path).rstrip("/"))
    mpath = self.root / MANIFEST
    if not mpath.exists():
      raise FileNotFoundError(f"no shard manifest: {mpath}")
    try:
      m = json.loads(mpath.read_text())
    except ValueError as e:
      raise Error(f"{mpath}: {e}") from e
    if (
        m.get("format") != MANIFEST_FORMAT
        or m.get("format_version") != FORMAT_VERSION
    ):
      raise Error(
          f"{mpath}: format {m.get('format')!r} v{m.get('format_version')!r}, "
          "want "
          f"{MANIFEST_FORMAT!r} v{FORMAT_VERSION}"
      )
    for field, want in (
        ("game", env.name),
        ("rules_sha", rules_sha),
        ("info_schema_version", info_schema_version),
    ):
      if m.get(field) != want:
        raise Error(f"{mpath}: {field} {m.get(field)!r} != expected {want!r}")
    self.manifest = m
    self.producer = m["producer"]
    self.num_steps = int(m["num_steps"])
    self.batch_size = batch_size
    if deal_exclusion is not None:
      de = m.get("deal_exclusion") or {}
      if (index_at_write or de) and de.get(
          "index_sha256"
      ) != deal_exclusion.sha256:
        raise Error(
            f"{mpath}: written without the eval deal index "
            f"{deal_exclusion.sha256} "
            f"(manifest deal_exclusion {de or None}); a reader given an index "
            "refuses it"
        )
    template = _template(
        info_template(env, info_like), m["label_keys"], m["aux_keys"]
    )
    want_paths = _paths(template)
    got_paths = [leaf["path"] for leaf in m["leaves"]]
    if want_paths != got_paths:
      raise Error(
          f"{mpath}: stored leaves {got_paths} do not match the {env.name} "
          f"TrajectoryBatch layout {want_paths}"
      )
    treedef = jax.tree_util.tree_structure(template)
    stored = bool(m["init_carry"])
    if student is None:
      self.carry = "stored" if stored else "none"
    elif not student.stateful:
      self.carry = "none"
    elif stored and m.get("carry_owner") == carry_owner_id(student):
      self.carry = "stored"
    else:
      self.carry = "student_init"
    if self.carry != "stored":
      m = dict(m, init_carry=[])  # the stored carry (if any) is not read
    if m["init_carry"]:
      if init_carry_like is None:
        raise Error(
            f"{mpath}: shards store an init_carry; pass init_carry_like"
        )
      carry_def = jax.tree_util.tree_structure(init_carry_like)
      cpaths = _paths(init_carry_like)
      if cpaths != [c["path"] for c in m["init_carry"]]:
        raise Error(
            f"{mpath}: stored init_carry leaves {m['init_carry']} do not match "
            f"init_carry_like {cpaths}"
        )
    else:
      carry_def = jax.tree_util.tree_structure(())
    leaves = [[] for _ in m["leaves"]]
    carries = [[] for _ in m["init_carry"]]
    for sh in m["shards"]:
      data = (self.root / sh["file"]).read_bytes()
      if _sha256(data) != sh["sha256"]:
        raise Error(
            f"{self.root / sh['file']}: sha256 does not match the manifest"
        )
      with np.load(io.BytesIO(data), allow_pickle=False) as z:
        for i, spec in enumerate(m["leaves"]):
          x = z[f"leaf_{i:05d}"]
          self._check_leaf(x, spec, sh, _lead(spec["path"]))
          leaves[i].append(x)
        for i, spec in enumerate(m["init_carry"]):
          x = z[f"carry_{i:05d}"]
          self._check_leaf(x, spec, sh, 1)
          carries[i].append(x)
    host = [np.concatenate(x, axis=0) for x in leaves]
    self.validation = (
        None if validator is None else validator(self.root, m, host)
    )

    def cat(xs):
      """Concatenates host arrays onto the device.

      Args:
        xs: The host arrays.

      Returns:
        The array.
      """
      return jnp.asarray(np.concatenate(xs, axis=0))

    batch = jax.tree_util.tree_unflatten(
        treedef, [jnp.asarray(x) for x in host]
    )
    del host
    carry = jax.tree_util.tree_unflatten(carry_def, [cat(x) for x in carries])
    self._data = batch._replace(init_carry=carry)
    self.num_rows = int(self._data.valid.shape[0])
    if self.carry == "student_init":
      bad = unaligned_valid_plies(batch.steps, batch.valid)
      if bad.any():
        rows = np.flatnonzero(bad)
        if stored:
          why = "its stored carry belongs to " + json.dumps(
              self.manifest.get("carry_owner")
          )
        else:
          why = "it stores no carry"
        raise Error(
            f"{mpath}: the stateful student {student.arch_name} must start "
            "from its own "
            f"init_carry ({why}), but {rows.size} of {self.num_rows} rows "
            f"(first: {rows[:5]}) "
            f"have {int(bad.sum())} valid plies before their first new_match "
            "ply: no "
            "reset-aligned prefix exists to rebuild its state there (record "
            "whole matches, "
            "or the student's own carry)"
        )
    assert self.num_rows == sum(int(s["rows"]) for s in m["shards"])
    if deal_exclusion is not None:
      rows, hashes, _ = deals.batch_hashes(env.name, rules_sha, self._data)
      hit = deal_exclusion.hits(hashes)
      if hit.any():
        raise Error(
            f"{mpath}: {int(hit.sum())} hands in rows "
            f"{sorted(set(rows[hit]))[:5]} "
            "start from an eval deal (sources.deals index "
            f"{deal_exclusion.sha256[:12]}): the shard is refused"
        )
    self.last_rows: Optional[np.ndarray] = None

  @staticmethod
  def _check_leaf(x, spec, sh, lead):
    want_shape = (int(sh["rows"]),) + tuple(spec["shape"])
    if lead == _ROW_TIME:
      want_shape = (int(sh["rows"]), None) + tuple(spec["shape"])
    ok_shape = len(x.shape) == len(want_shape) and all(
        w is None or int(s) == w for s, w in zip(x.shape, want_shape)
    )
    if not ok_shape or x.dtype.name != spec["dtype"]:
      raise Error(
          f"{sh['file']} {spec['path']}: {x.dtype}{x.shape} != manifest "
          f"{spec['dtype']}{want_shape}"
      )

  def read_all(self) -> types.TrajectoryBatch:
    """Every stored row.

    Returns:
      The TrajectoryBatch.
    """
    return self._data

  def rows(self, start: int, size: int) -> types.TrajectoryBatch:
    """Stored rows [start, start + size).

    Args:
      start: the first row.
      size: the row count.

    Returns:
      The TrajectoryBatch.
    """
    assert 0 <= start and size >= 1 and start + size <= self.num_rows, (
        start,
        size,
        self.num_rows,
    )
    return jax.tree.map(lambda x: x[start : start + size], self._data)

  def next(self, key) -> types.TrajectoryBatch:
    """batch_size rows drawn with `key` (TrajectorySource protocol).

    Args:
      key: the PRNG key.

    Returns:
      The TrajectoryBatch.
    """
    assert self.batch_size is not None, "ShardReader.next needs batch_size"
    assert self.batch_size <= self.num_rows, (self.batch_size, self.num_rows)
    idx = np.asarray(
        jax.random.choice(key, self.num_rows, (self.batch_size,), replace=False)
    )
    self.last_rows = idx.astype(np.int64)
    return jax.tree.map(lambda x: x[idx], self._data)


class SequentialSource:
  """TrajectorySource over a ShardReader: the stored batches in write order.

  One per `next` (the key is ignored: the data is fixed). Raises when the shards
  are exhausted.

  Attributes:
    reader: the ShardReader.
    frames: plies emitted (np.int64).
    decisions: valid plies emitted (np.int64).
  """

  def __init__(self, reader: ShardReader):
    """Builds the source at the first stored batch.

    Args:
      reader: the ShardReader.
    """
    self.reader = reader
    self._sizes = [int(s["rows"]) for s in reader.manifest["shards"]]
    self._i = 0
    self._start = 0
    self.frames = np.int64(0)
    self.decisions = np.int64(0)

  def skip(self, n: int) -> None:
    """Advances past the first n batches (a resumed run's trained batches).

    They are not counted (the counters come from the checkpoint).

    Args:
      n: the batch count.
    """
    assert self._i == 0 and 0 <= n <= len(self._sizes), (
        self._i,
        n,
        len(self._sizes),
    )
    self._start = sum(self._sizes[:n])
    self._i = n

  def next(self, key) -> types.TrajectoryBatch:
    """The next stored batch.

    Args:
      key: ignored.

    Returns:
      The TrajectoryBatch.

    Raises:
      IndexError: the shards are exhausted.
    """
    del key
    if self._i >= len(self._sizes):
      raise IndexError(
          f"shards exhausted after {self._i} batches ({self.reader.root})"
      )
    n = self._sizes[self._i]
    b = self.reader.rows(self._start, n)
    self._i += 1
    self._start += n
    self.frames = np.int64(self.frames + np.int64(b.valid.size))
    self.decisions = np.int64(
        self.decisions + np.int64(jnp.sum(b.valid, dtype=jnp.int64))
    )
    return b
