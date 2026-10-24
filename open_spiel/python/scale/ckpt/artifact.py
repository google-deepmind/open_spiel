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

"""Artifacts: a trained policy's params and what rebuilding and serving needs.

An artifact step is self-describing. Its manifest names the game, the rules
hash, the architecture and its kwargs, the framework, the decision rule and
the params digest, so `load_artifact` rebuilds the exact policy through
`policies.registry` and restores its params without any other context. Steps
are written once and never modified.

  manifest = artifact.write_artifact(ckpt_dir, step, policy, params,
                                     rules_sha=rules_sha, env_spec=env_spec)
  art = artifact.load_artifact(ckpt_dir, step)
  out = art.policy.evaluate(art.params, carry, seat_step, key)

The native layout is the artifact format, version 1:

  <ckpt_dir>/<step>/manifest.json   the manifest (flat JSON, below)
  <ckpt_dir>/<step>/params          params, written by the framework's codec

  {"format": "artifact", "schema_version": 1,
   "game", "rules_sha", "env_spec", "obs_format", "arch_name", "arch_kwargs",
   "framework", "capability", "decider", "default_rule", "params_codec",
   "params_sha256", "step", "parents", "run_id", "provenance", "extensions",
   "created_at"}

"format" is the format's tag ("artifact"), "params_codec" names the codec
that wrote the params (`ckpt.codecs`) and "created_at" is the UTC time of the
write; these three are format bookkeeping. "decider" is the policy's decider
(`PolicySpec.decider`); a load checks it against the rebuilt policy. Each other
key carries the Manifest field of the same name (core.types.Manifest);
`provenance` holds the training run's git_commit, config_hash, algo,
env_name, code_repo and lineage.

The manifest is written after the params, so a step whose manifest exists is
complete. Other layouts plug in through `ckpt.formats`; a framework's params
encoding through `ckpt.codecs`.

Invariants:
  * x64 is asserted on write and load whenever jax is loaded; leaf dtypes
    round-trip exactly.
  * The JAX params digest is `tree_sha256(params)`, computed in memory on
    write and checked against the restored tree on load.
  * A warm start records both `parents` (artifact ids "<run>@<step>") and a
    `lineage` record ({source, step, ...}); a fresh init records neither.
  * `extensions` holds records keyed by names registered with
    `register_meta_extension`. Writing an unregistered key raises; a reader
    validates the keys it has registered and keeps the others as they are.
  * A policy whose spec says `publishable=False` (a training-only behaviour
    wrapper) is refused.
  * Paths are local or gs://; `norm_path` keeps the scheme.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import posixpath
from typing import Any, Callable, Mapping, Optional, Sequence

from etils import epath

from open_spiel.python.scale import plugins
from open_spiel.python.scale.ckpt import codecs
from open_spiel.python.scale.ckpt import formats
from open_spiel.python.scale.ckpt import provenance as provenance_lib
from open_spiel.python.scale.ckpt import records
from open_spiel.python.scale.core import frameworks
from open_spiel.python.scale.core import types
from open_spiel.python.scale.policies import registry

FORMAT = formats.NATIVE  # the native format's name in ckpt.formats
FORMAT_VERSION = 1
FORMAT_TAG = "artifact"  # the "format" value of a native manifest.json
MANIFEST_FILENAME = "manifest.json"
PARAMS_NAME = "params"
GCS_PREFIX = "gs://"

ArtifactError = codecs.ArtifactError
tree_sha256 = codecs.tree_sha256
save_tree = codecs.save_tree
restore_tree = codecs.restore_tree
params_finalized = codecs.params_finalized

_MANIFEST_KEYS = (
    "format",
    "schema_version",
    "game",
    "rules_sha",
    "env_spec",
    "obs_format",
    "arch_name",
    "arch_kwargs",
    "framework",
    "capability",
    "decider",
    "default_rule",
    "params_codec",
    "params_sha256",
    "step",
    "parents",
    "run_id",
    "provenance",
    "extensions",
    "created_at",
)
_LINEAGE_KEYS = ("source", "step")


def _assert_x64(where: str) -> None:
  frameworks.assert_x64_if_loaded(where)


# -- paths ---------------------------------------------------------------------


def norm_path(path: str) -> str:
  """A checkpoint path, normalised.

  gs://bucket/a/b (scheme kept, duplicate and trailing slashes dropped) or an
  absolute local path.

  Args:
    path: a local or gs:// path.

  Returns:
    The normalised path.

  Raises:
    ValueError: for an empty path, a gs:// path without a bucket, or another
      URL scheme.
  """
  path = str(path)
  if not path:
    raise ValueError("empty checkpoint path")
  if path.startswith(GCS_PREFIX):
    rest = posixpath.normpath(path[len(GCS_PREFIX) :]).lstrip("/")
    if not rest or rest == ".":
      raise ValueError(f"no bucket in {path!r}")
    return GCS_PREFIX + rest
  if "://" in path:
    raise ValueError(
        f"unsupported checkpoint path scheme: {path!r} (local or gs://)"
    )
  return os.path.abspath(path)


def ckpt_root(path: str) -> epath.Path:
  """`norm_path(path)` as an etils path.

  Args:
    path: a local or gs:// path.

  Returns:
    The etils path.
  """
  return epath.Path(norm_path(path))


def atomic_write(path: epath.Path, data: bytes) -> None:
  """Writes bytes atomically: tmp file + rename locally, one PUT on gs://.

  Args:
    path: the destination.
    data: the bytes.
  """
  path = epath.Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  if str(path).startswith(GCS_PREFIX):
    path.write_bytes(data)
    return
  tmp = path.parent / f".{path.name}.tmp"
  tmp.write_bytes(data)
  os.replace(str(tmp), str(path))


def read_json(path: epath.Path) -> Any:
  """JSON at `path`.

  Args:
    path: the file.

  Returns:
    The parsed JSON.

  Raises:
    ArtifactError: malformed JSON (the error names the file).
  """
  try:
    return json.loads(epath.Path(path).read_text())
  except ValueError as e:
    raise ArtifactError(f"{path}: {e}") from e


# -- extensions ----------------------------------------------------------------

_EXTENSIONS: dict[str, Callable[[Any], None]] = {}


def register_meta_extension(key: str, validate: Callable[[Any], None]) -> None:
  """Allows `key` in a manifest's `extensions`.

  `validate(value)` raises on a bad value and runs on write and on read.

  Args:
    key: the extension key.
    validate: raises on a bad value.

  Raises:
    ValueError: a second validator for the same key.
  """
  assert isinstance(key, str) and key, key
  assert callable(validate), validate
  if _EXTENSIONS.get(key, validate) is not validate:
    raise ValueError(f"meta extension {key!r} is already registered")
  _EXTENSIONS[key] = validate


def check_extensions(
    extensions: Mapping[str, Any], where: str, *, allow_unknown: bool = False
) -> dict[str, Any]:
  """A JSON copy of `extensions` with every registered key's value validated.

  Args:
    extensions: the extension records.
    where: names the artifact in errors.
    allow_unknown: keep unregistered keys instead of raising (a reader keeps
      records of extensions this process has not loaded).

  Returns:
    The validated copy.

  Raises:
    ArtifactError: extensions is not a mapping, or holds an unregistered key
      (unless allow_unknown) or an invalid value.
  """
  if not isinstance(extensions, Mapping):
    raise ArtifactError(
        f"{where}: extensions must be a mapping, got {extensions!r}"
    )
  unknown = sorted(set(extensions) - set(_EXTENSIONS))
  if unknown and not allow_unknown:
    raise ArtifactError(
        f"{where}: unregistered meta extension(s) {unknown} (registered: "
        f"{sorted(_EXTENSIONS)}; ckpt.artifact.register_meta_extension)"
    )
  out = json.loads(json.dumps(dict(extensions), sort_keys=True))
  for k, v in out.items():
    if k not in _EXTENSIONS:
      continue
    try:
      _EXTENSIONS[k](v)
    except (TypeError, ValueError) as e:
      raise ArtifactError(f"{where}: extension {k}: {e}") from e
  return out


def check_value_scale(v: Any) -> None:
  """A value scale is a finite positive number (not a bool).

  Args:
    v: the value scale.

  Raises:
    ValueError: v is not one.
  """
  if (
      isinstance(v, bool)
      or not isinstance(v, (int, float))
      or not math.isfinite(v)
      or v <= 0
  ):
    raise ValueError(f"value_scale must be a finite positive number, got {v!r}")


#: The units of a policy's value head: value = return / value_scale. A
#: consumer that mixes values with returns reads it with `value_scale`.
register_meta_extension("value_scale", check_value_scale)


# -- validation ----------------------------------------------------------------


def lineage(source: str, step: Optional[int], **extra: Any) -> dict[str, Any]:
  """The lineage record of a warm start from `source` at `step`.

  Args:
    source: the warm-start source.
    step: the source's step, or None.
    **extra: further fields.

  Returns:
    The record.
  """
  return dict(source=source, step=None if step is None else int(step), **extra)


def check_lineage(parents: Any, lineage_record: Any) -> None:
  """Checks that parents and lineage are both present or both absent.

  Args:
    parents: the parent artifact ids.
    lineage_record: the lineage record, or None.

  Raises:
    ArtifactError: they are not, or either is malformed.
  """
  if isinstance(parents, str) or not isinstance(parents, Sequence):
    raise ArtifactError(
        f"parents must be a list of artifact ids, got {parents!r}"
    )
  if not all(isinstance(p, str) and "@" in p for p in parents):
    raise ArtifactError(f"parents must be '<run>@<step>' ids, got {parents!r}")
  if lineage_record is None:
    if parents:
      raise ArtifactError(
          f"parents {list(parents)} without a lineage record: a warm start "
          "records its lineage {source, step}"
      )
    return
  if not isinstance(lineage_record, Mapping):
    raise ArtifactError(f"lineage must be a mapping, got {lineage_record!r}")
  missing = [k for k in _LINEAGE_KEYS if k not in lineage_record]
  if missing:
    raise ArtifactError(f"lineage missing {missing}")
  if not parents:
    raise ArtifactError(
        "a lineage record without parents: a warm start names its parent "
        "artifact ids"
    )


def check_env_spec(env_spec: Any, game: str) -> None:
  """Checks that env_spec is None or {name, kwargs} naming `game`.

  Args:
    env_spec: the recorded EnvSpec JSON, or None.
    game: the artifact's game.

  Raises:
    ArtifactError: it is not.
  """
  if env_spec is None:
    return
  if not isinstance(env_spec, Mapping) or set(env_spec) != {"name", "kwargs"}:
    raise ArtifactError(f"env_spec must be {{name, kwargs}}, got {env_spec!r}")
  if env_spec["name"] != game or not isinstance(env_spec["kwargs"], Mapping):
    raise ArtifactError(
        f"env_spec {env_spec!r} does not describe game {game!r}"
    )


def _publishable(policy: types.Policy) -> bool:
  """Whether `policy` may be published in an artifact.

  Args:
    policy: The policy.

  Returns:
    The spec's publishable flag.
  """
  return policy.spec.publishable


def _load_plugins() -> None:
  """Loads the plugins before a step is read.

  A step names registered things (architecture, encoder, codec, format):
  the plugins named in the plugins environment variable (plugins.ENV) are
  loaded before one is read.
  """
  plugins.load_env()


def build_policy(manifest: types.Manifest) -> types.Policy:
  """The policy (arch_name, arch_kwargs) names, rebuilt through the registry.

  It is checked against the manifest's game, obs_format, framework and
  capability.

  Args:
    manifest: the Manifest.

  Returns:
    The policy.

  Raises:
    ArtifactError: the rebuilt policy disagrees with the manifest.
  """
  policy = registry.build(manifest.arch_name, manifest.arch_kwargs)
  for field in ("game", "obs_format", "framework", "capability"):
    if getattr(policy.spec, field) != getattr(manifest, field):
      raise ArtifactError(
          f"{field}: the registry built {getattr(policy.spec, field)!r}, the "
          f"manifest says {getattr(manifest, field)!r}"
      )
  return policy


# -- the native format ---------------------------------------------------------


class NativeFormat:
  """The native artifact format, version 1 (module doc).

  Attributes:
    name: the format's id in `ckpt.formats`.
    tag: the "format" value its manifests carry and its reader requires.
  """

  name = FORMAT
  tag = FORMAT_TAG

  def _step_dir(self, ckpt_dir: str, step: int) -> epath.Path:
    """The folder of `step`.

    Args:
      ckpt_dir: The artifact folder.
      step: The step.

    Returns:
      The step folder.
    """
    return ckpt_root(ckpt_dir) / str(int(step))

  def detect(self, ckpt_dir: str, step: int) -> bool:
    """Whether `ckpt_dir` holds `step` in the native format.

    Args:
      ckpt_dir: the checkpoint dir.
      step: the step.

    Returns:
      True when the step's manifest exists.
    """
    return (self._step_dir(ckpt_dir, step) / MANIFEST_FILENAME).exists()

  def has_any(self, ckpt_dir: str) -> bool:
    """Whether `ckpt_dir` holds at least one native step.

    Args:
      ckpt_dir: the checkpoint dir.

    Returns:
      True when it does.
    """
    root = ckpt_root(ckpt_dir)
    if not root.exists():
      return False
    return any(
        c.name.isdigit() and (c / MANIFEST_FILENAME).exists()
        for c in root.iterdir()
    )

  def _read(self, ckpt_dir: str, step: int) -> dict[str, Any]:
    """The manifest.json of `step`.

    Args:
      ckpt_dir: The artifact folder.
      step: The step.

    Returns:
      The parsed manifest.

    Raises:
      FileNotFoundError: There is no manifest.
    """
    path = self._step_dir(ckpt_dir, step) / MANIFEST_FILENAME
    if not path.exists():
      raise FileNotFoundError(
          f"no artifact at {ckpt_dir}@{step}: {path} is missing"
      )
    return read_json(path)

  def manifest_from_json(
      self, m: Mapping[str, Any], where: str
  ) -> types.Manifest:
    """The Manifest of a parsed manifest.json, validated.

    Args:
      m: the parsed manifest.json.
      where: names the artifact in errors.

    Returns:
      The Manifest.

    Raises:
      ArtifactError: the manifest fails validation.
    """
    if not isinstance(m, Mapping):
      raise ArtifactError(f"{where}: the manifest is not a JSON object")
    missing = [k for k in _MANIFEST_KEYS if k not in m]
    if missing:
      raise ArtifactError(f"{where}: manifest missing {missing}")
    if m["format"] != self.tag or m["schema_version"] != FORMAT_VERSION:
      raise ArtifactError(
          f"{where}: format {m['format']!r} schema_version "
          f"{m['schema_version']!r}; this reader reads {self.tag!r} "
          f"schema_version {FORMAT_VERSION}"
      )
    if m["decider"] not in types.DECIDERS:
      raise ArtifactError(
          f"{where}: decider {m['decider']!r} is not one of {types.DECIDERS}"
      )
    if not isinstance(m["arch_name"], str) or not m["arch_name"]:
      raise ArtifactError(f"{where}: arch_name is empty")
    rule = m["default_rule"]
    prov = m["provenance"]
    if not isinstance(prov, Mapping):
      raise ArtifactError(f"{where}: provenance must be a mapping")
    check_lineage(m["parents"], prov.get("lineage"))
    check_env_spec(m["env_spec"], m["game"])
    extensions = check_extensions(m["extensions"], where, allow_unknown=True)
    run_id = m["run_id"]
    if run_id is not None and not isinstance(run_id, str):
      raise ArtifactError(
          f"{where}: run_id must be a string or null, got {run_id!r}"
      )
    return types.Manifest(
        schema_version=int(m["schema_version"]),
        game=m["game"],
        rules_sha=m["rules_sha"],
        obs_format=m["obs_format"],
        arch_name=m["arch_name"],
        arch_kwargs=dict(m["arch_kwargs"]),
        framework=str(m["framework"]),
        capability=types.Capability(m["capability"]),
        default_rule=types.DecisionRule(
            rule["kind"], float(rule["temperature"])
        ),
        params_sha256=m["params_sha256"],
        step=int(m["step"]),
        parents=tuple(m["parents"]),
        env_spec=m["env_spec"],
        run_id=run_id,
        provenance=dict(prov),
        extensions=extensions,
    )

  def _params_path(
      self, ckpt_dir: str, step: int, codec: codecs.ParamsCodec
  ) -> epath.Path:
    """The params file of `step` for `codec`.

    Args:
      ckpt_dir: The artifact folder.
      step: The step.
      codec: The params codec.

    Returns:
      The params path.
    """
    return self._step_dir(ckpt_dir, step) / f"{PARAMS_NAME}{codec.suffix}"

  def load_manifest(
      self, ckpt_dir: str, step: int, *, verify: bool = True
  ) -> types.Manifest:
    """The step's Manifest without restoring params.

    Args:
      ckpt_dir: the checkpoint dir.
      step: the step.
      verify: give the params files the integrity checks that need no restore.

    Returns:
      The Manifest.
    """
    return self._manifest_of(
        self._read(ckpt_dir, step), ckpt_dir, step, verify=verify
    )

  def _manifest_of(
      self, m: Mapping[str, Any], ckpt_dir: str, step: int, *, verify: bool
  ) -> types.Manifest:
    """The validated Manifest of step's parsed manifest.json `m`.

    Args:
      m: The parsed manifest.json.
      ckpt_dir: The artifact folder.
      step: The step.
      verify: Whether to check the params digest.

    Returns:
      The Manifest.

    Raises:
      ArtifactError: The manifest records another step.
    """
    manifest = self.manifest_from_json(m, f"{ckpt_dir}@{step}")
    if manifest.step != int(step):
      raise ArtifactError(
          f"{ckpt_dir}: the manifest of step {step} records step "
          f"{manifest.step}"
      )
    if verify:
      codec = codecs.params_codec_by_name(m["params_codec"])
      codec.check(
          self._params_path(ckpt_dir, step, codec), manifest.params_sha256
      )
    return manifest

  def load_artifact(
      self, ckpt_dir: str, step: int, *, verify: bool = True
  ) -> types.Artifact:
    """The step's Manifest, its rebuilt policy and its params.

    Args:
      ckpt_dir: the checkpoint dir.
      step: the step.
      verify: check the params against the recorded digest.

    Returns:
      The Artifact.

    Raises:
      ArtifactError: the rebuilt policy disagrees with the manifest, or the
        params fail their checks.
    """
    m = self._read(ckpt_dir, step)
    manifest = self._manifest_of(m, ckpt_dir, step, verify=False)
    policy = build_policy(manifest)
    if types.decider_of(policy) != m["decider"]:
      raise ArtifactError(
          f"{ckpt_dir}@{step}: the rebuilt policy's decider is "
          f"{types.decider_of(policy)!r}, the manifest says {m['decider']!r}"
      )
    codec = codecs.params_codec_by_name(m["params_codec"])
    if codec is not codecs.params_codec(policy.spec.framework):
      raise ArtifactError(
          f"{ckpt_dir}@{step}: params codec {m['params_codec']!r} is not the "
          f"codec of framework {policy.spec.framework}"
      )
    params = codec.read(
        self._params_path(ckpt_dir, step, codec),
        policy,
        manifest.params_sha256 if verify else None,
    )
    return types.Artifact(
        manifest=manifest,
        params=params,
        policy=policy,
        path=norm_path(ckpt_dir),
    )

  def write(
      self,
      ckpt_dir: str,
      step: int,
      policy: Any,
      params: Any,
      manifest: types.Manifest,
  ) -> types.Manifest:
    """Writes a native step (write-once).

    Args:
      ckpt_dir: the checkpoint dir.
      step: the step.
      policy: the policy.
      params: the params.
      manifest: every field but the params digest.

    Returns:
      The manifest as written.

    Raises:
      ArtifactError: the step exists, or the params do not match the
        architecture.
    """
    sdir = self._step_dir(ckpt_dir, step)
    if (sdir / MANIFEST_FILENAME).exists():
      raise ArtifactError(
          f"step {step} already written under {ckpt_dir} (write-once)"
      )
    codec = codecs.params_codec(policy.spec.framework)
    ppath = self._params_path(ckpt_dir, step, codec)
    if ppath.exists():
      raise ArtifactError(
          f"{ppath} exists without a manifest (a write that died); "
          "clear_partial moves it aside"
      )
    digest = codec.write(ppath, policy, params)
    manifest = manifest._replace(params_sha256=digest)
    rule = manifest.default_rule
    m = dict(
        format=self.tag,
        schema_version=FORMAT_VERSION,
        game=manifest.game,
        rules_sha=manifest.rules_sha,
        env_spec=None if manifest.env_spec is None else dict(manifest.env_spec),
        obs_format=manifest.obs_format,
        arch_name=manifest.arch_name,
        arch_kwargs=dict(manifest.arch_kwargs),
        framework=manifest.framework,
        capability=manifest.capability.value,
        decider=types.decider_of(policy),
        default_rule=dict(kind=rule.kind, temperature=float(rule.temperature)),
        params_codec=codec.name,
        params_sha256=digest,
        step=int(step),
        parents=list(manifest.parents),
        run_id=manifest.run_id,
        provenance=dict(manifest.provenance),
        extensions=dict(manifest.extensions),
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    data = json.dumps(m, indent=2, sort_keys=True).encode()
    written = self.manifest_from_json(json.loads(data), f"{ckpt_dir}@{step}")
    atomic_write(sdir / MANIFEST_FILENAME, data)
    return written

  def clear_partial(self, ckpt_dir: str, step: int, tag: str) -> Optional[str]:
    """Moves aside a step dir that has no manifest (a write that died).

    Args:
      ckpt_dir: the checkpoint dir.
      step: the step.
      tag: names the moved-aside dir.

    Returns:
      Where it went, or None when there was nothing.
    """
    sdir = self._step_dir(ckpt_dir, step)
    if not sdir.exists() or (sdir / MANIFEST_FILENAME).exists():
      return None
    aside = ckpt_root(ckpt_dir) / "unpublished" / f"{int(step)}.{tag}"
    aside.parent.mkdir(parents=True, exist_ok=True)
    sdir.rename(aside)
    return str(aside)


NATIVE_FORMAT = NativeFormat()
formats.set_native(NATIVE_FORMAT)


# -- the API -------------------------------------------------------------------


def write_artifact(
    ckpt_dir: str,
    step: int,
    policy: types.Policy,
    params: Any,
    *,
    rules_sha: str,
    env_spec: Optional[Mapping[str, Any]] = None,
    default_rule: types.DecisionRule = types.ARGMAX,
    config_hash: Optional[str] = None,
    algo: Optional[str] = None,
    env_name: Optional[str] = None,
    git_commit: Optional[str] = None,
    run_id: Optional[str] = None,
    parents: Sequence[str] = (),
    # The keyword is part of the public signature.
    lineage: Optional[Mapping[str, Any]] = None,  # pylint: disable=redefined-outer-name
    extensions: Optional[Mapping[str, Any]] = None,
) -> types.Manifest:
  """Writes artifact step `step` of (policy, params) under `ckpt_dir`.

  The step goes to the default writer format (`ckpt.formats`; the native
  format unless a plugin sets another).

  Args:
    ckpt_dir: local or gs:// directory of the run's artifacts.
    step: the step id; each step is written once.
    policy: the policy; its spec supplies the architecture and framework.
    params: params matching `policy.init_params` in tree, shapes and dtypes.
    rules_sha: hash of the game rules the policy was trained under.
    env_spec: EnvSpec JSON of the training env (envs.registry.spec_of(env)
      .to_json()), recorded so serving can check it covers the served env.
    default_rule: the decision rule a consumer applies by default.
    config_hash: provenance: the training run's config hash.
    algo: provenance: the training algorithm.
    env_name: provenance: the training env's name.
    git_commit: provenance: the training code's commit.
    run_id: the id of the run, or None.
    parents: a warm start's parent ids (with lineage, or neither).
    lineage: a warm start's lineage record (with parents, or neither;
      `lineage()` builds the record).
    extensions: registered extension records (`register_meta_extension`).

  Returns:
    The Manifest as written (equal to `load_manifest` of the step).

  Raises:
    ArtifactError: the step exists, params do not match the architecture, the
      policy is not publishable, or a field fails validation.
  """
  _assert_x64("write_artifact")
  _load_plugins()
  spec = policy.spec
  if not _publishable(policy):
    raise ArtifactError(
        f"{type(policy).__name__} ({spec.arch_name}) is a training-only "
        "behaviour wrapper and is never published"
    )
  if run_id is not None and (not isinstance(run_id, str) or not run_id):
    raise ArtifactError(
        f"run_id must be a non-empty string or None, got {run_id!r}"
    )
  parents = list(parents)
  lineage = (
      None
      if lineage is None
      else json.loads(json.dumps(records.encode(lineage)))
  )
  check_lineage(parents, lineage)
  env_spec = (
      None if env_spec is None else json.loads(json.dumps(dict(env_spec)))
  )
  check_env_spec(env_spec, spec.game)
  where = f"{ckpt_dir}@{step}"
  prov = dict(
      git_commit=git_commit,
      config_hash=config_hash,
      algo=algo,
      env_name=env_name,
      code_repo=provenance_lib.code_repo(),
      lineage=lineage,
  )
  manifest = types.Manifest(
      schema_version=FORMAT_VERSION,
      game=spec.game,
      rules_sha=rules_sha,
      obs_format=spec.obs_format,
      arch_name=spec.arch_name,
      arch_kwargs=dict(spec.arch_kwargs),
      framework=spec.framework,
      capability=spec.capability,
      default_rule=default_rule,
      params_sha256="",
      step=int(step),
      parents=tuple(parents),
      env_spec=env_spec,
      run_id=run_id,
      provenance=prov,
      extensions=check_extensions(extensions or {}, where),
  )
  return formats.writer().write(ckpt_dir, int(step), policy, params, manifest)


def _format_of(path: str, step: int) -> formats.ArtifactFormat:
  """The format of the artifact at `path`@`step`.

  Args:
    path: The artifact folder.
    step: The step.

  Returns:
    The detected format.

  Raises:
    FileNotFoundError: No format recognises the artifact.
  """
  _load_plugins()
  fmt = formats.detect(path, step)
  if fmt is None:
    raise FileNotFoundError(
        f"no artifact at {path}@{step} (formats tried: {formats.names()})"
    )
  return fmt


def load_manifest(
    path: str, step: int, *, verify: bool = True
) -> types.Manifest:
  """The Manifest of `path`@`step`, without restoring params.

  Args:
    path: the checkpoint dir.
    step: the step.
    verify: give the params the checks their codec can make without a
      restore (presence, completeness, a file digest).

  Returns:
    The Manifest.

  Raises:
    FileNotFoundError: no format holds the step.
    ArtifactError: the manifest or the params fail validation.
  """
  _assert_x64("load_manifest")
  return _format_of(path, step).load_manifest(path, step, verify=verify)


def load_artifact(
    path: str, step: int, *, verify: bool = True
) -> types.Artifact:
  """The artifact at `path`@`step`: its Manifest, rebuilt policy and params.

  Args:
    path: the checkpoint dir.
    step: the step.
    verify: check the params against their digest.

  Returns:
    The Artifact.

  Raises:
    FileNotFoundError: no format holds the step.
    ArtifactError: the manifest fails validation, the rebuilt policy disagrees
      with it, or (with `verify`) the params do not match their digest.
  """
  _assert_x64("load_artifact")
  return _format_of(path, step).load_artifact(path, step, verify=verify)


def exists(path: str, step: Optional[int] = None) -> bool:
  """Whether `path` holds artifact step `step` (any step when None).

  In any format.

  Args:
    path: the checkpoint dir.
    step: the step, or None.

  Returns:
    True when it does.
  """
  _load_plugins()
  if step is None:
    return any(f.has_any(path) for f in formats.in_order())
  return formats.detect(path, int(step)) is not None


def clear_partial(path: str, step: int, tag: str) -> Optional[str]:
  """Moves aside what a write of `step` by the default writer left.

  Args:
    path: the checkpoint dir.
    step: the step.
    tag: names the moved-aside files.

  Returns:
    Where it went, or None when there was nothing.
  """
  return formats.writer().clear_partial(path, int(step), tag)


def value_scale(path: str, step: int) -> Optional[float]:
  """The recorded value units of `path`@`step`, or None when not recorded.

  Args:
    path: the checkpoint dir.
    step: the step.

  Returns:
    The value scale, or None.
  """
  v = load_manifest(path, step, verify=False).extensions.get("value_scale")
  return None if v is None else float(v)
