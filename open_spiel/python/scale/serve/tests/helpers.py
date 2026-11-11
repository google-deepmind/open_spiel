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

"""Policies and in-memory artifacts for the serving tests."""

import jax
import jax.numpy as jnp

from open_spiel.python.scale.ckpt import artifact as artifact_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.encoders import registry as enc_registry
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.networks import policy as policy_lib
from open_spiel.python.scale.networks.tests import helpers as net_helpers
from open_spiel.python.scale.policies import tabular
from open_spiel.python.scale.testing import fixtures

KUHN_ONEHOT = net_helpers.KUHN_ONEHOT
KUHN_INDEX = f"{kuhn.__name__}:info_index"


def artifact(
    policy, params, rule=types.ARGMAX, run_id="424", step=7, path="mem://test"
):
  """An Artifact without a disk round trip.

  load_artifact is tested separately.

  Args:
    policy: the policy.
    params: its params.
    rule: the DecisionRule.
    run_id: the manifest's run id.
    step: the step.
    path: the artifact path.

  Returns:
    The Artifact.
  """
  spec = policy.spec
  manifest = types.Manifest(
      schema_version=artifact_lib.FORMAT_VERSION,
      game=spec.game,
      rules_sha="rules-test",
      obs_format=spec.obs_format,
      arch_name=spec.arch_name,
      arch_kwargs=dict(spec.arch_kwargs),
      framework=spec.framework,
      capability=spec.capability,
      default_rule=rule,
      params_sha256="sha-test",
      step=step,
      parents=(),
      provenance={"git_commit": "commit-test"},
      run_id=run_id,
  )
  return types.Artifact(
      manifest=manifest, params=params, policy=policy, path=path
  )


def _mlp():
  """A small MLP network config.

  Returns:
    The config.
  """
  return {
      "compute_dtype": "float32",
      "torso": {"kind": "mlp", "widths": [16, 16], "layout": "shared"},
      "policy_head": {"kind": "dense"},
      "value_head": {"kind": "scalar"},
  }


def _rnn():
  """A small recurrent network config.

  Returns:
    The config.
  """
  return {
      "compute_dtype": "float32",
      "torso": {
          "kind": "recurrent",
          "hidden_size": 8,
          "use_last_action": True,
          "inner": {"kind": "mlp", "widths": [16], "layout": "shared"},
      },
      "policy_head": {"kind": "dense"},
      "value_head": {"kind": "scalar"},
  }


def _encoder_for(env):
  """The encoder and its shape for a test env.

  Args:
    env: The env.

  Returns:
    (encoder, shape).
  """
  if env.name == "kuhn_poker":
    return KUHN_ONEHOT, (kuhn.NUM_INFO_STATES,)
  name = {
      "leduc_poker": "leduc_poker_info_state",
      "connect_four": "connect_four_flat_planes",
  }[env.name]
  return name, tuple(enc_registry.get(name).shape)


def network(env, kind, seed=1):
  """(policy, params): an MLP or RNN NetworkPolicy with random params.

  Args:
    env: the env.
    kind: "mlp" or "rnn".
    seed: the init seed.

  Returns:
    (policy, params).
  """
  enc, shape = _encoder_for(env)
  cfg = {"mlp": _mlp, "rnn": _rnn}[kind]()
  pol = policy_lib.NetworkPolicy(
      env.name, enc, env.num_actions, env.num_players, enc, list(shape), cfg
  )
  return pol, pol.init_params(jax.random.key(seed))


def history(env, seed=1, cross_hand=True):
  """(policy, params): HistoryPolicy with random weights.

  Logits depend on the whole history (count, every last action, hands begun).

  Args:
    env: the env.
    seed: the init seed.
    cross_hand: carry the history across hands.

  Returns:
    (policy, params).
  """
  pol = fixtures.HistoryPolicy(
      env.name, env.num_actions, env.num_players, cross_hand=cross_hand
  )
  a = env.num_actions
  k = jax.random.split(jax.random.key(seed), 4)
  params = pol.make_params(
      c=jax.random.normal(k[0], (a,), jnp.float32),
      w=jax.random.normal(k[1], (a, a), jnp.float32),
      b=jax.random.normal(k[2], (a,), jnp.float32),
      d=jax.random.normal(k[3], (a,), jnp.float32),
  )
  return pol, params


def kuhn_tabular(env, seed=1):
  """A kuhn TabularPolicy with random logits.

  Args:
    env: the kuhn env.
    seed: the logits seed.

  Returns:
    (policy, params).
  """
  pol = tabular.TabularPolicy(
      env.name,
      kuhn.NUM_INFO_STATES,
      env.num_actions,
      env.num_players,
      KUHN_INDEX,
  )
  logits = jax.random.normal(
      jax.random.key(seed), (kuhn.NUM_INFO_STATES, env.num_actions), jnp.float32
  )
  return pol, pol.params_from_logits(logits)


def make(env, name, seed=1):
  """(policy, params) of a named policy kind.

  Args:
    env: the env.
    name: "mlp", "rnn", "history" or "tabular".
    seed: the init seed.

  Returns:
    (policy, params).
  """
  if name == "history":
    return history(env, seed)
  if name == "tabular":
    return kuhn_tabular(env, seed)
  return network(env, name, seed)
