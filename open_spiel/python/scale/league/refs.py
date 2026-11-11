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

"""Policy references in run configs: PolicyRef and FixedOpponent.

A run config names the policies it uses (teachers, fixed opponents, warm
starts) without holding them: a PolicyRef is built into (policy, params) when
the run starts. Both classes are JSON-ready frozen dataclasses, hashed with
the config (`league.config_common.config_hash`).

    teacher  = refs.PolicyRef(artifact="/runs/teacher@4000")
    uniform  = refs.PolicyRef(arch_name="uniform", arch_kwargs={...})
    opponent = refs.FixedOpponent("uniform", uniform, types.SAMPLE)
    policy, params = teacher.build()

A PolicyRef is exactly one of:
  * a registry architecture with fresh params (`arch_name`, `arch_kwargs`):
    scripted policies, whose params are {};
  * an artifact "<ckpt_dir>@<step>" (`artifact`), read through
    `ckpt.artifact.load_artifact`;
  * a checkpoint of another format (`format`, `uri`), read by the loader
    registered for that format (`register_policy_format`).
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Callable, Mapping, NamedTuple, Optional

import jax

from open_spiel.python.scale.ckpt import artifact
from open_spiel.python.scale.core import types
from open_spiel.python.scale.policies import registry


class _Format(NamedTuple):
  """A registered policy format's hooks.

  Attributes:
    load: reads a ref into its (policy, params) tuple.
    ref_id: None, or the artifact id of a ref.
    lineage: None, or the lineage record of a ref.
  """

  load: Callable[[str], tuple[Any, ...]]
  ref_id: Optional[Callable[[str], str]]
  lineage: Optional[Callable[[str], Mapping[str, Any]]]


_FORMATS: dict[str, _Format] = {}


def register_policy_format(
    name: str,
    load: Callable[[str], tuple[Any, ...]],
    *,
    ref_id: Optional[Callable[[str], str]] = None,
    lineage: Optional[Callable[[str], Mapping[str, Any]]] = None,
) -> None:
  """Registers a checkpoint format PolicyRef(format=name, uri=...) can name.

  Args:
    name: the format name.
    load: uri -> (policy, params).
    ref_id: uri -> the id recorded as a run's parent or a shard's producer
      (default "<name>:<uri>").
    lineage: uri -> the {"parents", "lineage"} artifact fields of a run
      warm-started from it (default: `kinds.lineage_extra` of its id).

  Raises:
    ValueError: the name is registered with another loader.
  """
  if not isinstance(name, str) or not name:
    raise ValueError(f"policy format name {name!r}")
  fmt = _Format(load, ref_id, lineage)
  old = _FORMATS.get(name)
  if old is not None and old != fmt:
    raise ValueError(f"policy format {name!r} is already registered")
  _FORMATS[name] = fmt


def _format(name: str) -> _Format:
  """The registered policy format `name`.

  Args:
    name: The format name.

  Returns:
    The format.

  Raises:
    KeyError: No such format is registered.
  """
  if name not in _FORMATS:
    raise KeyError(
        f"no policy format {name!r} is registered (known {sorted(_FORMATS)}); "
        "a "
        "plugin registers it with league.refs.register_policy_format"
    )
  return _FORMATS[name]


@dataclasses.dataclass(frozen=True)
class PolicyRef:
  """A policy: an architecture, an artifact or a (format, uri) checkpoint.

  Exactly one of the three (module doc).

  Attributes:
    arch_name: a registered architecture name.
    arch_kwargs: the architecture's kwargs.
    artifact: an artifact id "<path>@<step>".
    format: a registered policy format.
    uri: the checkpoint's uri under `format`.
  """

  arch_name: str = ""
  arch_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
  artifact: str = ""
  format: str = ""
  uri: str = ""

  def __post_init__(self):
    kinds = bool(self.arch_name) + bool(self.artifact) + bool(self.format)
    if kinds != 1:
      raise ValueError(
          "a PolicyRef is an architecture, an artifact or a (format, uri) "
          f"checkpoint, exactly one: {self}"
      )
    if bool(self.format) != bool(self.uri):
      raise ValueError(f"format and uri go together: {self}")
    if self.artifact and "@" not in self.artifact:
      raise ValueError(f"artifact {self.artifact!r} is not <ckpt_dir>@<step>")
    json.dumps(dict(self.arch_kwargs), sort_keys=True)

  @property
  def id(self) -> str:
    """The id recorded as a run's parent or a shard's producer."""
    if self.format:
      f = _format(self.format)
      return (
          f.ref_id(self.uri)
          if f.ref_id is not None
          else f"{self.format}:{self.uri}"
      )
    return self.artifact or self.arch_name

  def build(self, key=None) -> tuple[types.Policy, Any]:
    """Builds (policy, params).

    An architecture's params are init_params(key or key(0)).

    Args:
      key: the init key of an architecture's params.

    Returns:
      (policy, params).
    """
    if self.format:
      return _format(self.format).load(self.uri)
    if self.artifact:
      path, step = self.artifact.rsplit("@", 1)
      art = artifact.load_artifact(path, int(step))
      return art.policy, art.params

    pol = registry.build(self.arch_name, dict(self.arch_kwargs))
    return pol, pol.init_params(jax.random.key(0) if key is None else key)

  def lineage(self) -> dict[str, Any]:
    """The {"parents", "lineage"} artifact fields of a run warm-started from it.

    Returns:
      The fields.
    """
    if self.format:
      f = _format(self.format)
      if f.lineage is not None:
        return dict(f.lineage(self.uri))
    # Lazy: cycle: kinds.base imports this module.
    from open_spiel.python.scale.league.kinds import base  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

    return base.lineage_extra(self.id, ())


@dataclasses.dataclass(frozen=True)
class FixedOpponent:
  """An opponent outside the population.

  Attributes:
    name: the name the sampler mixes over; never "self" (the self-play
      category) and without "@" (member ids carry one).
    ref: the policy.
    rule: the decision rule it plays by.

  The League seats opponents whose built policies have the same spec and
  rule in one Arena slot (members stacked [K] in config order).
  """

  name: str
  ref: PolicyRef
  rule: types.DecisionRule = types.SAMPLE

  def __post_init__(self):
    if not (isinstance(self.name, str) and self.name and "@" not in self.name):
      raise ValueError(
          "FixedOpponent.name must be a non-empty string without '@', got "
          f"{self.name!r}"
      )
    if self.name == "self":
      raise ValueError(
          "FixedOpponent.name 'self' is the sampler's self-play category"
      )
    if not isinstance(self.ref, PolicyRef):
      raise TypeError(
          "FixedOpponent.ref must be a PolicyRef, got "
          f"{type(self.ref).__name__}"
      )
    if not isinstance(self.rule, types.DecisionRule):
      raise TypeError(
          f"FixedOpponent.rule must be a DecisionRule, got {self.rule!r}"
      )
