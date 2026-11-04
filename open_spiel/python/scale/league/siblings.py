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

"""Expert-iteration artifacts: the sibling layout and the manifest records.

An expert-iteration run publishes two artifacts per generation: the plain
network and the network-plus-search operator that labels its training data.
Evaluating both against the same opponents separates what the network learned
from what search adds. Each is an ordinary artifact directory under the run's
checkpoint dir, the plain one written first:

  <ckpt>/plain/<step>    the network (cell "<run>@<step>:plain")
  <ckpt>/search/<step>   the search operator (cell "<run>@<step>:search")

The run's root holds no step of its own, so a cell without a mode is refused
rather than guessed.

Two manifest extensions (ckpt.artifact.register_meta_extension) describe a
generation's artifacts:
  value_link  what the value head predicts, in value_scale units:
              "inherited"   the parent network's value semantics, unchanged by
                            training (a distilled network trained with value
                            coefficient 0);
              "mc_outcome"  Monte Carlo outcomes of its own games;
              "search_mix"  (1 - lambda) outcome + lambda search value.
  expert_iteration
              the generation record (`sibling_record`): its mode, decider,
              operator id and the environment provenance of the run that
              wrote it, so targets can be reproduced from the artifact.

    d = siblings.mode_dir(ckpt_dir, "plain")
    siblings.has_mode_siblings(ckpt_dir)
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from open_spiel.python.scale.ckpt import artifact
from open_spiel.python.scale.ckpt import provenance
from open_spiel.python.scale.core import types

MODES = ("plain", "search")  # write order: plain first
VALUE_LINKS = ("inherited", "mc_outcome", "search_mix")
RECORD_KEY = "expert_iteration"  # the manifest extension holding sibling_record


def mode_dir(ckpt_dir: str, mode: str) -> str:
  """The checkpoint dir of a run's `mode` sibling.

  Args:
    ckpt_dir: the run's checkpoint dir.
    mode: one of MODES.

  Returns:
    The sibling's dir.

  Raises:
    ValueError: mode is not one of MODES.
  """
  if mode not in MODES:
    raise ValueError(f"mode {mode!r} not in {MODES}")
  return artifact.norm_path(ckpt_dir).rstrip("/") + "/" + mode


def has_mode_siblings(ckpt_dir: str) -> bool:
  """Whether `ckpt_dir` holds an artifact under any mode sibling.

  Args:
    ckpt_dir: the run's checkpoint dir.

  Returns:
    True when some mode sibling holds an artifact.
  """
  return any(artifact.exists(mode_dir(ckpt_dir, m)) for m in MODES)


def check_value_link(v: Any) -> str:
  """Checks that `v` is one of VALUE_LINKS.

  Args:
    v: the value link.

  Returns:
    v.

  Raises:
    ValueError: v is not one of VALUE_LINKS.
  """
  if not isinstance(v, str) or v not in VALUE_LINKS:
    raise ValueError(f"value_link {v!r} not in {VALUE_LINKS}")
  return v


def check_generation_record(v: Any) -> None:
  """Checks that `v` is a mapping (a sibling_record).

  Args:
    v: the generation record.

  Raises:
    ValueError: v is not a mapping.
  """
  if not isinstance(v, Mapping):
    raise ValueError(
        f"the expert_iteration record must be a mapping, got {type(v).__name__}"
    )


artifact.register_meta_extension("value_link", check_value_link)
artifact.register_meta_extension(RECORD_KEY, check_generation_record)


def value_link(ckpt_dir: str, step: int) -> Optional[str]:
  """The recorded value link of `ckpt_dir`@`step`, or None.

  Args:
    ckpt_dir: the artifact's checkpoint dir.
    step: the artifact step.

  Returns:
    The value link, or None.
  """
  v = artifact.load_manifest(ckpt_dir, step, verify=False).extensions.get(
      "value_link"
  )
  return None if v is None else check_value_link(v)


def generation_record(ckpt_dir: str, step: int) -> Optional[dict[str, Any]]:
  """The recorded generation record of `ckpt_dir`@`step`, or None.

  Args:
    ckpt_dir: the artifact's checkpoint dir.
    step: the artifact step.

  Returns:
    The record, or None.
  """
  v = artifact.load_manifest(ckpt_dir, step, verify=False).extensions.get(
      RECORD_KEY
  )
  return None if v is None else dict(v)


def sibling_record(
    mode: str, policy: types.Policy, params: Any, git_commit: Optional[str]
) -> dict[str, Any]:
  """The `expert_iteration` record written with a sibling artifact.

  It holds the mode, decider, operator id and provenance (the params digest
  of a JAX policy, package versions, devices).

  Args:
    mode: one of MODES.
    policy: the sibling's policy.
    params: the sibling's params.
    git_commit: the commit of the running code, or None.

  Returns:
    The record.

  Raises:
    ValueError: mode is not one of MODES.
  """
  # Lazy: this module imports without jax.
  from open_spiel.python.scale.policies import operators  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  if mode not in MODES:
    raise ValueError(f"mode {mode!r} not in {MODES}")
  # The digest of the tree write_artifact writes; another framework's
  # artifact records its own digest in its manifest.
  sha = artifact.tree_sha256(params) if policy.spec.framework == "jax" else None
  return dict(
      mode=mode,
      decider=operators.decider_of(policy),
      operator_id=operators.operator_id(policy.spec),
      provenance=provenance.environment_provenance(
          params_sha256=sha, git_commit_sha=git_commit
      ),
  )
