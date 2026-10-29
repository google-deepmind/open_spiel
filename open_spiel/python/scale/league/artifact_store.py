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

"""ArtifactStore: a league's member artifacts by id ("<run id>@<step>").

A league admits snapshots of its learners as members and must find them
again by name (to seat them, evaluate them, warm-start from them, resume).
The store is an index, not a second checkpoint format:
<root>/artifact_index.json maps an artifact id to (path, step) plus the
identity read at registration (game, architecture, framework, capability,
params digest, parents). Artifacts stay where their run wrote them (local or
gs://) and are loaded by `ckpt.artifact.load_artifact`; a load whose params
digest differs from the registered one raises, so an id names exactly those
params.

    store = ArtifactStore(run_dir + "/store")
    aid = store.register(path, step)              # "<run id>@<step>"
    art = store.load(aid)
    slot, params = store.slot([aid, other_id])    # K members in one Arena slot

Ids are "<run id>@<step>" from the artifact's manifest; an artifact without a
run id (tests, local runs) needs an explicit "<name>@<step>" whose step is
the artifact's. Registering the same (path, step) again is a no-op; another
artifact claiming a registered id raises. `unregister` drops an index entry
under a given run's dir (files are never deleted).

One process writes the index; every call re-reads it, so readers in other
processes see registrations. The index write is atomic, not merged:
concurrent writers are not supported.

`slot(ids)` puts K same-architecture artifacts into one Arena slot through
`policies.members.stack`. The index operations read manifests only and
import no jax; jax comes in when an artifact is loaded or stacked.
"""

import datetime
import json
import re
from typing import Any, NamedTuple, Optional, Sequence

from open_spiel.python.scale.ckpt import artifact
from open_spiel.python.scale.ckpt import records
from open_spiel.python.scale.core import slots as slots_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.policies import members as members_lib

INDEX_FILENAME = "artifact_index.json"
INDEX_SCHEMA = 1
_ID_RE = re.compile(r"^([A-Za-z0-9_.\-]+)@(\d+)$")


class Error(RuntimeError):
  """The store index is missing pieces, malformed or of the wrong schema."""


def parse_artifact_id(artifact_id: str) -> tuple[str, int]:
  """Parses an artifact id: '123@3000' -> ('123', 3000).

  Args:
    artifact_id: the id.

  Returns:
    (run id or name, step).

  Raises:
    ValueError: the id is malformed.
  """
  m = _ID_RE.match(artifact_id) if isinstance(artifact_id, str) else None
  if not m:
    raise ValueError(
        f"artifact id must be '<run id or name>@<step>', got {artifact_id!r}"
    )
  return m.group(1), int(m.group(2))


def _id_key(artifact_id: str):
  """The sort key of an artifact id: numeric names first.

  Args:
    artifact_id: The artifact id.

  Returns:
    The key.
  """
  name, step = parse_artifact_id(artifact_id)
  return (0, int(name), "", step) if name.isdigit() else (1, 0, name, step)


class StoreEntry(NamedTuple):
  """One index entry: where an artifact is and what it was at registration.

  Attributes:
    artifact_id: the artifact's id, "<run>@<step>".
    path: where the artifact is.
    step: the artifact's step.
    run_id: the run that wrote it, or None.
    game: the game it plays.
    arch_name: its policy architecture.
    framework: its policy's array framework.
    capability: its policy's capability.
    schema_version: its manifest schema version.
    params_sha256: the digest of its params.
    parents: the artifact ids it descends from.
    registered_at: when it was registered (ISO 8601).
  """

  artifact_id: str
  path: str
  step: int
  run_id: Optional[str]
  game: str
  arch_name: str
  framework: str
  capability: str
  schema_version: int
  params_sha256: str
  parents: tuple[str, ...]
  registered_at: str


class ArtifactStore:
  """The index at <root>/artifact_index.json (module doc).

  The index at <root>/artifact_index.json (module doc). cache=True keeps
  loaded artifacts in process memory.

  Attributes:
    root: the store's root directory, normalised.
  """

  def __init__(self, root: str, cache: bool = True):
    self.root = artifact.norm_path(root)
    self._index_path = artifact.ckpt_root(self.root) / INDEX_FILENAME
    self._cache: Optional[dict[str, types.Artifact]] = {} if cache else None

  # -- index -------------------------------------------------------------------
  def _read(self) -> dict[str, Any]:
    """The index, or an empty one when there is no index file yet.

    Returns:
      the index: {"schema": INDEX_SCHEMA, "artifacts": {id: entry}}.

    Raises:
      Error: the index is not JSON, has another schema or no artifacts mapping.
    """
    if not self._index_path.exists():
      return {"schema": INDEX_SCHEMA, "artifacts": {}}
    try:
      idx = json.loads(self._index_path.read_text())
    except ValueError as e:
      raise Error(f"{self._index_path}: {e}") from e
    if not isinstance(idx, dict) or idx.get("schema") != INDEX_SCHEMA:
      got = idx.get("schema") if isinstance(idx, dict) else idx
      raise Error(f"{self._index_path}: index schema {got!r} != {INDEX_SCHEMA}")
    if not isinstance(idx.get("artifacts"), dict):
      raise Error(f"{self._index_path}: 'artifacts' must be a mapping")
    return idx

  def _write(self, idx: dict[str, Any]) -> None:
    artifact.atomic_write(
        self._index_path, json.dumps(idx, indent=2, sort_keys=True).encode()
    )

  @staticmethod
  def _entry(artifact_id: str, d: dict[str, Any]) -> StoreEntry:
    """The StoreEntry of an index entry.

    Args:
      artifact_id: the id.
      d: its index entry.

    Returns:
      the StoreEntry.

    Raises:
      Error: the entry is missing a field or has a malformed one.
    """
    d = records.decode(d)
    try:
      return StoreEntry(
          artifact_id=artifact_id,
          path=d["path"],
          step=int(d["step"]),
          run_id=None if d["run_id"] is None else str(d["run_id"]),
          game=d["game"],
          arch_name=d["arch_name"],
          framework=d["framework"],
          capability=d["capability"],
          schema_version=int(d["schema_version"]),
          params_sha256=d["params_sha256"],
          parents=tuple(d["parents"]),
          registered_at=d["registered_at"],
      )
    except (KeyError, TypeError, ValueError) as e:
      raise Error(f"index entry {artifact_id}: {e!r}") from e

  # -- API ---------------------------------------------------------------------
  def register(
      self, path: str, step: int, artifact_id: Optional[str] = None
  ) -> str:
    """Registers path@step and returns its id.

    Its manifest is validated by load_manifest; the params digest is checked
    when `load` restores them.

    Args:
      path: the artifact's checkpoint dir.
      step: the artifact step.
      artifact_id: the id; "<run id>@<step>" when None.

    Returns:
      The id.

    Raises:
      ValueError: there is no id to register under, or the id is taken by
        another artifact.
    """
    path = artifact.norm_path(path)
    man = artifact.load_manifest(path, step)
    run = man.run_id
    if artifact_id is None:
      if run is None:
        raise ValueError(
            f"{path}@{step} has no run id: pass artifact_id='<name>@{man.step}'"
        )
      artifact_id = f"{run}@{man.step}"
    name, id_step = parse_artifact_id(artifact_id)
    if id_step != man.step:
      raise ValueError(
          f"artifact id {artifact_id} names step {id_step}, artifact is step "
          f"{man.step}"
      )
    if run is not None and name != str(run):
      raise ValueError(
          f"artifact id {artifact_id} does not match the artifact's run id "
          f"{run}"
      )
    idx = self._read()
    old = idx["artifacts"].get(artifact_id)
    if old is not None:
      if (old["path"], int(old["step"]), old["params_sha256"]) == (
          path,
          man.step,
          man.params_sha256,
      ):
        return artifact_id
      raise ValueError(
          f"{artifact_id} is already registered to {old['path']}@{old['step']} "
          f"(sha256 {old['params_sha256']}); refusing {path}@{step}"
      )
    idx["artifacts"][artifact_id] = records.encode(
        dict(
            path=path,
            step=man.step,
            run_id=run,
            game=man.game,
            arch_name=man.arch_name,
            framework=man.framework,
            capability=man.capability.value,
            schema_version=man.schema_version,
            params_sha256=man.params_sha256,
            parents=list(man.parents),
            registered_at=datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
        )
    )
    self._write(idx)
    return artifact_id

  def ids(self) -> tuple[str, ...]:
    """The registered ids, run ids numerically first.

    Returns:
      The ids.
    """
    return tuple(sorted(self._read()["artifacts"], key=_id_key))

  def entries(
      self, game: Optional[str] = None, arch_name: Optional[str] = None
  ) -> list[StoreEntry]:
    """The registered entries, optionally filtered.

    Args:
      game: keep only entries of this game.
      arch_name: keep only entries of this architecture.

    Returns:
      The entries, in id order.
    """
    arts = self._read()["artifacts"]
    out = [self._entry(k, arts[k]) for k in sorted(arts, key=_id_key)]
    return [
        e
        for e in out
        if (game is None or e.game == game)
        and (arch_name is None or e.arch_name == arch_name)
    ]

  def get(self, artifact_id: str) -> StoreEntry:
    """The entry registered under `artifact_id`.

    Args:
      artifact_id: the id.

    Returns:
      The StoreEntry.

    Raises:
      KeyError: the id is not registered.
    """
    arts = self._read()["artifacts"]
    if artifact_id not in arts:
      raise KeyError(
          f"artifact {artifact_id!r} is not registered in {self._index_path}"
      )
    return self._entry(artifact_id, arts[artifact_id])

  def load(self, artifact_id: str, cache: bool = True) -> types.Artifact:
    """The Artifact for an id, through load_artifact; refuses changed params.

    Args:
      artifact_id: the id.
      cache: False neither reads nor fills the in-process cache (the League
        loads members uncached: it decides where their weights reside).

    Returns:
      The types.Artifact.

    Raises:
      ArtifactError: the artifact changed since registration.
    """
    e = self.get(artifact_id)
    cache = cache and self._cache is not None
    if cache and artifact_id in self._cache:
      return self._cache[artifact_id]
    art = artifact.load_artifact(e.path, e.step)
    man = art.manifest
    if man.params_sha256 != e.params_sha256:
      raise artifact.ArtifactError(
          f"{artifact_id}: params_sha256 {man.params_sha256} at {e.path}@"
          f"{e.step} differs "
          f"from the registered {e.params_sha256}"
      )
    if (man.step, man.run_id) != (e.step, e.run_id):
      raise artifact.ArtifactError(
          f"{artifact_id}: artifact step / run id {(man.step, man.run_id)} != "
          f"registered {(e.step, e.run_id)}"
      )
    if cache:
      self._cache[artifact_id] = art
    return art

  def unregister(self, artifact_id: str, under: str) -> StoreEntry:
    """Drops the index entry of artifact_id; the files stay where they are.

    The entry must point under `under` (a run's own members dir). The League
    drops the registrations of an uncommitted attempt this way.

    Args:
      artifact_id: the id.
      under: the dir the entry must point under.

    Returns:
      The dropped StoreEntry.

    Raises:
      KeyError: the id is not registered.
      ValueError: the entry points elsewhere.
    """
    under = artifact.norm_path(under).rstrip("/") + "/"
    idx = self._read()
    old = idx["artifacts"].get(artifact_id)
    if old is None:
      raise KeyError(
          f"artifact {artifact_id!r} is not registered in {self._index_path}"
      )
    if not (old["path"].rstrip("/") + "/").startswith(under):
      raise ValueError(
          f"{artifact_id} is registered to {old['path']}, not under {under}: "
          "refusing to unregister it"
      )
    entry = self._entry(artifact_id, old)
    del idx["artifacts"][artifact_id]
    self._write(idx)
    if self._cache is not None:
      self._cache.pop(artifact_id, None)
    return entry

  def slot(
      self,
      artifact_ids: Sequence[str],
      rule: Optional[types.DecisionRule] = None,
  ) -> tuple[slots_lib.Slot, Any]:
    """(Slot, params) for K same-architecture artifacts, member k = ids[k].

    Args:
      artifact_ids: the artifact ids.
      rule: the slot's decision rule; defaults to the artifacts' default_rule
        (which must agree).

    Returns:
      (Slot, stacked params).

    Raises:
      ValueError: no ids, or the artifacts differ in architecture or rule.
    """
    if not artifact_ids:
      raise ValueError("slot needs at least one artifact id")
    arts = [self.load(a) for a in artifact_ids]
    first = arts[0].manifest
    for a, aid in zip(arts, artifact_ids):
      m = a.manifest
      if (m.arch_name, dict(m.arch_kwargs)) != (
          first.arch_name,
          dict(first.arch_kwargs),
      ):
        raise ValueError(
            f"slot needs one architecture: {aid} is {m.arch_name}, "
            f"{artifact_ids[0]} is {first.arch_name} (or their arch_kwargs "
            "differ)"
        )
    if rule is None:
      rules = {a.manifest.default_rule for a in arts}
      if len(rules) != 1:
        raise ValueError(
            f"default rules differ across {list(artifact_ids)}: {rules}; pass "
            "rule"
        )
      rule = rules.pop()
    policy = arts[0].policy
    params = members_lib.stack(policy, [(a.policy, a.params) for a in arts])
    return slots_lib.Slot(policy, len(arts), rule), params
