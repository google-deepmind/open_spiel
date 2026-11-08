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

"""League checkpoint commits and ArenaSource snapshots.

League checkpoint commits: attempt-scoped outputs behind one atomic commit
reference, and the ArenaSource snapshots that let a resume continue the
checkpointed hands.

A run may be preempted at any time and resumed by a new process. Each process
attempt (one League object) draws a fresh attempt id and writes everything
before a commit under paths that name it:

  <ckpt>/league/<step>-<attempt>/state.pkl       the League's host state
  <ckpt>/league/<step>-<attempt>/learners/<name>/ each learner's state at <step>
  <run_dir>/members/<base>/<attempt>/             member artifacts it admitted

`<ckpt>/train_state.json` is the commit: `league_ref` names the committed
step's ref ("league/<step>-<attempt>") and `league_commits` maps every
committed step to its ref. Nothing an attempt wrote is read unless a commit
names it, so a resumed attempt recomputes uncommitted work without comparing
it to (or overwriting) what an earlier attempt left behind.

    r = commit.ref(step, attempt)                   # write under r, then
    record = commit.with_commit(record, previous, step, r)
    loc = commit.locate(ckpt_dir, step)  # loc.state_path, loc.learners_root

Other checkpoint layouts are read through `register_layout_reader`.

ArenaSource snapshots hold a source's rollout state (env states, policy
carries, keys, counters), its slot table and its frame and decision counters
as host data: jax arrays as numpy, typed PRNG keys as their key data and
implementation, and leaves of other array types through the codecs
registered with `ckpt.snapshot.register_leaf_codec`. A restored league deals
the same hands with the same keys. Params are not snapshotted: the driver
sets them before every rollout.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import secrets
from typing import Any, Callable, Mapping, NamedTuple, Optional

from etils import epath
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.ckpt import layout
from open_spiel.python.scale.ckpt import snapshot
from open_spiel.python.scale.core import arena as arena_lib

COMMITS_DIR = "league"
STATE_FILE = "state.pkl"
LEARNERS_SUBDIR = "learners"
MEMBERS_DIR = "members"
TRAIN_STATE = layout.TRAIN_STATE


def new_attempt_id() -> str:
  """A fresh attempt id: UTC time and random hex, so ids sort by start time.

  Returns:
    The id.
  """
  t = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
  return f"{t}-{secrets.token_hex(4)}"


def ref(step: int, attempt: str) -> str:
  """The commit ref of an attempt's checkpoint at `step`.

  Args:
    step: the step.
    attempt: the attempt id.

  Returns:
    The ref, relative to the ckpt dir.
  """
  assert attempt and "/" not in attempt, attempt
  return f"{COMMITS_DIR}/{int(step)}-{attempt}"


def ref_step(r: str) -> int:
  """The step of a commit ref.

  Args:
    r: the ref.

  Returns:
    The step.

  Raises:
    ValueError: r is not a league commit ref.
  """
  head, _, tail = r.partition("/")
  if head != COMMITS_DIR or "-" not in tail:
    raise ValueError(f"not a league commit ref: {r!r}")
  return int(tail.split("-", 1)[0])


def _train_state(ckpt_dir: str) -> Optional[dict[str, Any]]:
  """The train state saved in `ckpt_dir`, if any.

  Args:
    ckpt_dir: The checkpoint folder.

  Returns:
    The state, or None.
  """
  p = epath.Path(ckpt_dir) / TRAIN_STATE
  return json.loads(p.read_text()) if p.exists() else None


# -- registered layouts --------------------------------------------------------
class Location(NamedTuple):
  """Where a committed league checkpoint lives.

  Attributes:
    state_path: the epath.Path of the state pickle.
    learners_root: the directory holding <name>/ per learner.
    label: the ref, or the layout reader's description.
  """

  state_path: Any  # epath.Path of the state pickle
  learners_root: str  # holds <name>/ per learner
  label: str  # the ref, or the layout reader's description


_LAYOUT_READERS: list[Any] = []  # (detect, restore)


def register_layout_reader(
    detect: Callable[[str], bool], restore: Callable[[str], Mapping[str, Any]]
) -> None:
  """Lets the league read checkpoints stored in another layout.

  Lets the league read league checkpoints stored in a directory layout
  other than the `league_commits` refs this module writes (for example one
  state file per step). `locate`, and so League.restore, consults the readers
  when no ref is given; the state found goes through
  loop.register_state_translation like any other, so a reader only says where
  the files are. A population reader for such a layout is registered apart
  (population.register_reader).

  Args:
    detect: ckpt_dir -> whether it holds league checkpoints in this layout
      (its train_state.json has no `league_commits`).
    restore: ckpt_dir -> {"label": a description for logs, "checkpoints":
      {step: {"state_path": the state pickle, "learners_root": the dir
      holding <name>/ per learner}}}.
  """
  entry = (detect, restore)
  if entry not in _LAYOUT_READERS:
    _LAYOUT_READERS.append(entry)


def _layout_reader(ckpt_dir: str):
  """The restore of the first layout reader that detects `ckpt_dir`.

  Args:
    ckpt_dir: The checkpoint folder.

  Returns:
    The restore function, or None.
  """
  for detect, restore in _LAYOUT_READERS:
    if detect(ckpt_dir):
      return restore
  return None


def committed_ref(ckpt_dir: str, step: int) -> str:
  """The ref train_state.json commits for `step`.

  Args:
    ckpt_dir: the run's checkpoint dir.
    step: the step.

  Returns:
    The ref.

  Raises:
    FileNotFoundError: nothing is committed at `step`.
    ValueError: the commits name another step, or use an unregistered layout.
  """
  ts = _train_state(ckpt_dir)
  if ts is None:
    raise FileNotFoundError(f"{ckpt_dir}: no {TRAIN_STATE} (nothing committed)")
  commits = ts.get("league_commits")
  if commits is None:
    raise ValueError(
        f"{ckpt_dir}: {TRAIN_STATE} has no league_commits and no registered "
        "layout reader recognises it"
    )
  r = commits.get(str(int(step)))
  if r is None:
    raise FileNotFoundError(
        f"{ckpt_dir}: no commit at step {step} (committed: "
        f"{sorted(commits, key=int)})"
    )
  if ref_step(r) != int(step):
    raise ValueError(
        f"{ckpt_dir}: league_commits[{step}] = {r!r} names another step"
    )
  return r


def locate(ckpt_dir: str, step: int, r: Optional[str] = None) -> Location:
  """The committed checkpoint of `step`.

  Ref r if given (checked against step), else the one train_state.json
  commits, or a registered layout's.

  Args:
    ckpt_dir: the run's checkpoint dir.
    step: the step.
    r: the ref, or None.

  Returns:
    The Location.

  Raises:
    FileNotFoundError: nothing is committed at `step` (by train_state.json or
      a registered layout).
    ValueError: r is not a checkpoint of `step`.
  """
  if r is not None:
    if ref_step(r) != int(step):
      raise ValueError(f"ref {r!r} is not a checkpoint of step {step}")
  else:
    restore = _layout_reader(ckpt_dir)
    if restore is not None:
      got = dict(restore(ckpt_dir))
      ck = {int(k): v for k, v in dict(got["checkpoints"]).items()}
      if int(step) not in ck:
        raise FileNotFoundError(
            f"{ckpt_dir}: no {got['label']} checkpoint at step {step} "
            f"(steps {sorted(ck)})"
        )
      c = ck[int(step)]
      return Location(
          epath.Path(c["state_path"]),
          str(c["learners_root"]),
          str(got["label"]),
      )
    r = committed_ref(ckpt_dir, step)
  return Location(
      epath.Path(ckpt_dir) / r / STATE_FILE,
      os.path.join(ckpt_dir, r, LEARNERS_SUBDIR),
      r,
  )


def state_path(ckpt_dir: str, step: int, r: str) -> epath.Path:
  """The state pickle of ref r, a checkpoint of `step`.

  Args:
    ckpt_dir: the run's checkpoint dir.
    step: the step.
    r: the ref.

  Returns:
    The path.

  Raises:
    ValueError: r is not a checkpoint of `step`.
  """
  if ref_step(r) != int(step):
    raise ValueError(f"ref {r!r} is not a checkpoint of step {step}")
  return epath.Path(ckpt_dir) / r / STATE_FILE


def learner_dir(
    ckpt_dir: str,
    name: str,
    step: Optional[int] = None,
    r: Optional[str] = None,
) -> str:
  """The learner-state dir of learner `name`.

  In ref r, or in the commit of `step` (default: the latest commit).

  Args:
    ckpt_dir: the run's checkpoint dir.
    name: the learner name.
    step: the step, or None for the latest commit.
    r: the ref, or None.

  Returns:
    The dir.

  Raises:
    FileNotFoundError: nothing is committed.
  """
  if r is None:
    if step is None:
      ts = _train_state(ckpt_dir)
      if ts is None:
        raise FileNotFoundError(
            f"{ckpt_dir}: no {TRAIN_STATE} (nothing committed)"
        )
      step = int(ts["step"])
    return os.path.join(locate(ckpt_dir, step).learners_root, name)
  return os.path.join(ckpt_dir, r, LEARNERS_SUBDIR, name)


def with_commit(
    record: Mapping[str, Any],
    previous: Optional[Mapping[str, Any]],
    step: int,
    r: str,
) -> dict[str, Any]:
  """Record + league_ref r + league_commits (previous commits + step -> r).

  Args:
    record: the commit record.
    previous: the previous commit record, or None.
    step: the step.
    r: the ref.

  Returns:
    The record.
  """
  commits = dict((previous or {}).get("league_commits") or {})
  commits[str(int(step))] = r
  return dict(record, league_ref=r, league_commits=commits)


# -- ArenaSource snapshots -----------------------------------------------------
@dataclasses.dataclass(frozen=True)
class _JaxLeaf:
  """A jax array leaf of a snapshot, held as a host array.

  Attributes:
    data: the array's host values.
  """

  data: np.ndarray


@dataclasses.dataclass(frozen=True)
class _KeyLeaf:
  """A typed jax PRNG key leaf of a snapshot, held as its key data.

  Attributes:
    impl: the key's PRNG implementation name.
    data: the key data as a host array.
  """

  impl: str
  data: np.ndarray


@dataclasses.dataclass(frozen=True)
class _HostLeaf:
  """A leaf of a type with a registered snapshot leaf codec.

  Its bits as host numpy and the codec's metadata, decoded by the same codec.

  Attributes:
    codec: the codec name.
    data: the leaf's bits as host numpy.
    meta: the codec's metadata.
  """

  codec: str
  data: np.ndarray
  meta: Any


def _is_leaf(x) -> bool:
  """Whether `x` is a commit leaf wrapper.

  Args:
    x: A value.

  Returns:
    True for one.
  """
  return isinstance(x, (_JaxLeaf, _KeyLeaf, _HostLeaf))


def _to_host(x):
  """One state leaf in its host (picklable) form.

  Args:
    x: the state leaf.

  Returns:
    A _KeyLeaf, _JaxLeaf or _HostLeaf for a PRNG key, jax array or codec leaf;
    a host numpy or Python scalar leaf unchanged.

  Raises:
    TypeError: a leaf of no supported kind.
  """
  if isinstance(x, jax.Array):
    if jnp.issubdtype(x.dtype, jax.dtypes.prng_key):
      return _KeyLeaf(
          str(jax.random.key_impl(x)), np.asarray(jax.random.key_data(x))
      )
    return _JaxLeaf(np.asarray(x))
  if isinstance(x, (np.ndarray, np.generic, bool, int, float)):
    return x
  codec = snapshot.leaf_codec_for(x)
  if codec is not None:
    data, meta = codec.encode(x)
    return _HostLeaf(codec.name, data, meta)
  raise TypeError(
      f"ArenaSource snapshot: unsupported state leaf {type(x).__name__} (host "
      "numpy, jax arrays and leaves of a registered snapshot leaf codec)"
  )


def _from_host(x):
  """One state leaf back from its host form (_to_host's inverse).

  Args:
    x: the host leaf.

  Returns:
    The restored leaf; any other leaf unchanged.
  """
  if isinstance(x, _KeyLeaf):
    return jax.random.wrap_key_data(jnp.asarray(x.data), impl=x.impl)
  if isinstance(x, _JaxLeaf):
    out = jnp.asarray(x.data)
    # x64 is on: no narrowing
    assert out.dtype == x.data.dtype, (
        out.dtype,
        x.data.dtype,
    )
    return out
  if isinstance(x, _HostLeaf):
    return snapshot.leaf_codec(x.codec).decode(x.data, x.meta)
  return x


def host_tree(tree: Any) -> Any:
  """Tree with every jax array as host numpy (typed keys wrapped).

  Args:
    tree: a tree.

  Returns:
    The host tree.
  """
  return jax.tree_util.tree_map(_to_host, tree)


def device_tree(tree: Any) -> Any:
  """The inverse of host_tree.

  Args:
    tree: a host tree.

  Returns:
    The tree with jax arrays.
  """
  return jax.tree_util.tree_map(_from_host, tree, is_leaf=_is_leaf)


def snapshot_source(src: Any) -> dict[str, Any]:
  """A picklable snapshot of an ArenaSource (module doc).

  Args:
    src: the ArenaSource.

  Returns:
    The snapshot.
  """
  assert isinstance(src.frames, np.int64) and isinstance(
      src.decisions, np.int64
  )
  return dict(
      state=None if src.state is None else host_tree(src.state),
      table=host_tree(tuple(src.table)),
      table_type=type(src.table),
      num_steps=int(src.num_steps),
      frames=np.int64(src.frames),
      decisions=np.int64(src.decisions),
  )


def restore_source(arena: Any, snap: Mapping[str, Any]) -> Any:
  """The ArenaSource of snapshot_source(src) over `arena`, params unset.

  The driver sets the params before the next rollout.

  Args:
    arena: the Arena.
    snap: the snapshot.

  Returns:
    The ArenaSource.
  """
  table = snap["table_type"](*device_tree(snap["table"]))
  src = arena_lib.ArenaSource(arena, None, table, int(snap["num_steps"]))
  src.state = None if snap["state"] is None else device_tree(snap["state"])
  src.frames, src.decisions = np.int64(snap["frames"]), np.int64(
      snap["decisions"]
  )
  return src


def same_table(a: Any, b: Any) -> bool:
  """Whether two slot tables are equal leaf for leaf.

  Args:
    a: a slot table.
    b: another slot table.

  Returns:
    True when the shapes and values match.
  """
  la, lb = jax.tree_util.tree_leaves(tuple(a)), jax.tree_util.tree_leaves(
      tuple(b)
  )
  return len(la) == len(lb) and all(
      np.shape(x) == np.shape(y)
      and np.array_equal(np.asarray(x), np.asarray(y))
      for x, y in zip(la, lb)
  )
