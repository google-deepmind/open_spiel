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

"""The registered bridge env (envs.bridge).

One env under the name, trainable, not served, with the pinned rules hash and
InfoState template.
"""

import hashlib
import json
import sys

import pytest

from open_spiel.python.scale.envs import bridge
from open_spiel.python.scale.envs import registry
from open_spiel.python.scale.league import config_common
from open_spiel.python.scale.serve import games
from open_spiel.python.scale.sources import shards

# The pinned rules hash of envs.bridge (bridge.RULES_SHA).
RULES_SHA = "8f0c3f24a6542b978a082cbe7221ef7f30e85b2d4884acf24867b69139158f7b"

# The pinned InfoState template of envs.bridge: keys, dtypes and shapes of one
# seat's view.
INFO_TEMPLATE = {
    "actions": ("int16", (371,)),
    "dealer": ("int8", ()),
    "dummy": ("bool", (52,)),
    "dummy_shown": ("bool", ()),
    "hand": ("bool", (52,)),
    "num_actions": ("int16", ()),
    "seat": ("int8", ()),
    "vul": ("bool", (2,)),
}


def test_bridge_resolves_by_name_to_the_one_env():
  """Bridge resolves by name to the one env."""
  env = registry.make("bridge")
  assert isinstance(env, bridge.BridgeEnv)
  assert (env.name, env.num_players, env.num_actions, env.max_len) == (
      "bridge",
      4,
      90,
      371,
  )
  assert registry.spec_of(env) == registry.spec("bridge")
  assert registry.game("bridge").ctor == f"{bridge.__name__}:BridgeEnv"


def test_bridge_is_trainable_but_not_served():
  """Bridge is trainable but not served."""
  assert "bridge" in registry.builtin_names()
  assert "bridge" in config_common.env_names()
  assert "bridge" not in games.served_games()
  assert not registry.game("bridge").served
  with pytest.raises(KeyError, match="no env for game"):
    games.env_for("bridge", {"use_double_dummy_result": False})


def test_rules_sha_is_pyspiel_bridges_declared_rule_set():
  """Rules sha is pyspiel bridges declared rule set."""
  env = registry.make("bridge")
  want = hashlib.sha256(
      json.dumps(bridge.RULES, sort_keys=True, separators=(",", ":")).encode()
  ).hexdigest()
  assert bridge.RULES["game"] == "bridge(use_double_dummy_result=false)"
  assert (
      registry.rules_sha(env)
      == config_common.rules_sha(env)
      == bridge.RULES_SHA
      == want
  )
  assert want == RULES_SHA
  # compact_history is storage, not rules.
  assert (
      registry.rules_sha(registry.make("bridge", compact_history=True))
      == RULES_SHA
  )


def test_a_fixed_board_is_another_rule_set():
  """A fixed board is another rule set."""
  fixed = registry.make(
      "bridge", dealer=2, dealer_vul=True, random_dealer_vul=False
  )
  other = registry.make(
      "bridge", dealer=3, dealer_vul=True, random_dealer_vul=False
  )
  shas = {registry.rules_sha(fixed), registry.rules_sha(other), RULES_SHA}
  assert len(shas) == 3
  assert registry.rules_sha(fixed) == registry.rules_sha(
      registry.make(
          "bridge",
          dealer=2,
          dealer_vul=True,
          random_dealer_vul=False,
          compact_history=True,
      )
  )


def test_info_template_is_the_shard_template():
  """Info template is the shard template."""
  for t in (
      shards.info_template(registry.make("bridge")),
      bridge.info_template(),
  ):
    got = {k: (v.dtype.name, tuple(v.shape)) for k, v in t.items()}
    assert got == INFO_TEMPLATE


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
