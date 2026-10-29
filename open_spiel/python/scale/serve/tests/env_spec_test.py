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

"""Serving and the EnvSpec.

Serving and the EnvSpec: artifacts record the env
they were trained on (the manifest's env_spec),
HostAgent serves on that env by default, refuses an env the training env
does not cover (Oklahoma-only gin served standard) unless
allow_cross_variant.
"""

import json
import os
import sys

import jax
import pytest

from open_spiel.python.scale.ckpt import artifact as ckpt
from open_spiel.python.scale.envs import registry
from open_spiel.python.scale.policies import uniform
from open_spiel.python.scale.serve import host_agent
from open_spiel.python.scale.serve import server as server_lib
from open_spiel.python.scale.serve.tests import helpers


def _write_uniform(path, env, env_spec, step=1):
  """Writes a uniform artifact recording `env_spec`.

  Args:
    path: The folder.
    env: The env.
    env_spec: The env spec.
    step: The step.

  Returns:
    (policy, manifest).
  """
  pol = uniform.UniformPolicy(env.name, env.num_actions, env.num_players)
  return pol, ckpt.write_artifact(
      str(path),
      step,
      pol,
      pol.init_params(jax.random.key(0)),
      rules_sha=registry.rules_sha(env),
      config_hash="c",
      algo="test",
      env_name=env.name,
      env_spec=env_spec,
  )


def test_write_artifact_records_the_env_spec(tmp_path):
  """Write artifact records the env spec."""
  env = registry.make("gin_rummy", oklahoma_frac=1.0)
  spec = registry.spec_of(env).to_json()
  _, man = _write_uniform(tmp_path, env, spec)
  assert man.env_spec == registry.spec("gin_rummy", oklahoma_frac=1.0).to_json()
  meta = json.loads(
      open(
          os.path.join(tmp_path, "1", ckpt.MANIFEST_FILENAME), encoding="utf-8"
      ).read()
  )
  assert meta["env_spec"] == spec
  assert ckpt.load_manifest(str(tmp_path), 1).env_spec == spec
  with pytest.raises(ckpt.ArtifactError, match="env_spec"):
    _write_uniform(tmp_path / "bad", env, {"name": "kuhn_poker", "kwargs": {}})


def test_artifact_without_env_spec_still_loads(tmp_path):
  """Artifact without env spec still loads."""
  env = registry.make("kuhn_poker")
  _write_uniform(tmp_path, env, None)
  art = ckpt.load_artifact(str(tmp_path), 1)
  assert art.manifest.env_spec is None
  agent = host_agent.HostAgent(art, 0)
  assert registry.spec_of(agent.env) == registry.spec("kuhn_poker")


def test_host_agent_serves_the_training_env_and_refuses_other_variants(
    tmp_path,
):
  """Host agent serves the training env and refuses other variants."""
  okl = registry.make("gin_rummy", oklahoma_frac=1.0)
  _write_uniform(tmp_path, okl, registry.spec_of(okl).to_json())
  art = ckpt.load_artifact(str(tmp_path), 1)
  agent = host_agent.HostAgent(art, 0)  # the default env: the training env
  assert registry.spec_of(agent.env) == registry.spec(
      "gin_rummy", oklahoma_frac=1.0
  )
  assert agent.identity["env_spec"] == art.manifest.env_spec
  std = registry.make("gin_rummy")
  with pytest.raises(ValueError, match="does not cover"):
    host_agent.HostAgent(art, 0, env=std)
  cross = host_agent.HostAgent(art, 0, env=std, allow_cross_variant=True)
  assert cross.env is std
  assert server_lib.PolicyServer(allow_cross_variant=True).allow_cross_variant


def test_a_mixed_variant_artifact_serves_both_variants():
  """A mixed variant artifact serves both variants."""
  mixed = registry.make("gin_rummy", oklahoma_frac=0.5)
  pol = uniform.UniformPolicy(mixed.name, mixed.num_actions, mixed.num_players)
  art = helpers.artifact(pol, {})
  art = art._replace(
      manifest=art.manifest._replace(env_spec=registry.spec_of(mixed).to_json())
  )
  for frac in (None, 1.0):
    host_agent.HostAgent(
        art, 1, env=registry.make("gin_rummy", oklahoma_frac=frac)
    )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
