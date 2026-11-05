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

"""A freshly initialised network written to disk as a servable artifact.

What a training run writes, without the training: a NetworkPolicy on one of
the game's registered encoders, its initial params, and a manifest naming
the env spec and rules identity of the env it plays (the fields serving
checks), so the end-to-end serving tests load it from disk through the
public loader as they would a trained artifact.

  path = fresh_artifact.write(tmp_dir, env, "hearts_info_state")
  bot = checkpoint_bot.checkpoint_bot(path, step=1)
"""

import jax

from open_spiel.python.scale.ckpt import artifact as artifact_lib
from open_spiel.python.scale.encoders import registry as enc_registry
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.networks import policy as policy_lib


def network_config(widths=(32,)):
  """A small separate-torso MLP config (policy and value heads).

  Args:
    widths: the torso's hidden widths.

  Returns:
    The network config.
  """
  return {
      "compute_dtype": "float32",
      "torso": {"kind": "mlp", "widths": list(widths), "layout": "separate"},
      "policy_head": {"kind": "dense"},
      "value_head": {"kind": "scalar"},
  }


def write(
    path,
    env,
    encoder: str,
    *,
    step: int = 1,
    seed: int = 0,
    widths=(32,),
    network=None
) -> str:
  """Writes a freshly initialised network for `env` as artifact path@step.

  Args:
    path: the artifact folder.
    env: the env it plays (its EnvSpec and rules identity are recorded).
    encoder: a registered encoder of the env's game.
    step: the artifact step.
    seed: the init seed.
    widths: the torso's hidden widths (of the default MLP network).
    network: a network config (actor_critic.build_network's input) instead
      of the default MLP.

  Returns:
    The folder, as a string.
  """
  enc = enc_registry.get(encoder)
  assert enc.game == env.name, (enc.game, env.name)
  pol = policy_lib.NetworkPolicy(
      env.name,
      enc.name,
      env.num_actions,
      env.num_players,
      enc.name,
      list(enc.shape),
      network_config(widths) if network is None else network,
  )
  params = pol.init_params(jax.random.key(seed))
  artifact_lib.write_artifact(
      str(path),
      step,
      pol,
      params,
      rules_sha=env_registry.rules_sha(env),
      config_hash="fresh-init",
      algo="none",
      env_name=env.name,
      env_spec=env_registry.spec_of(env).to_json(),
  )
  return str(path)
