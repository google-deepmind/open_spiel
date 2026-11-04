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

"""The names the library records into what it writes.

the deal hash tag and the deal index format (sources.deals), the shard manifest
format (sources.shards), and decide's named scope with its setter (core.decide).
The params digest tag and the artifact manifest tag are pinned beside their
modules (ckpt/tests). Every environment variable the library reads starts with
plugins.ENV_PREFIX.
"""

import hashlib
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale import plugins
from open_spiel.python.scale.ckpt import provenance
from open_spiel.python.scale.core import decide
from open_spiel.python.scale.core import types
from open_spiel.python.scale.launch import hardware
from open_spiel.python.scale.launch import subprocess_env
from open_spiel.python.scale.sources import deals
from open_spiel.python.scale.sources import shards
from open_spiel.python.scale.testing import infoset_invariance

_GAME = "recorded_names_toy"


@pytest.fixture(name="defaults")
def _defaults(monkeypatch):
  # A deployment's plugin may rebind these in this process; the known
  # answers below are the library's own.
  # Reaches private _CANON of the code under test.
  # pylint: disable=protected-access
  monkeypatch.setattr(deals, "HASH_TAG", b"deal_hash\0")
  monkeypatch.setattr(deals, "INDEX_FORMAT", "deal_index")
  monkeypatch.setattr(shards, "MANIFEST_FORMAT", "trajectory_shards")
  monkeypatch.setattr(decide, "_SCOPE", {"name": decide.DEFAULT_SCOPE})
  # Reaches private _CANON of the code under test.
  monkeypatch.setitem(
      deals._CANON,
      _GAME,  # pylint: disable=protected-access
      lambda info, labels: np.asarray(labels["deal"], np.uint8),
  )


def test_the_defaults():
  """The recorded names hold their defaults."""
  assert decide.DEFAULT_SCOPE == "policy_decide"


# The fixture is requested for its side effect.
def test_deal_hash_known_answer_and_tag(defaults):  # pylint: disable=unused-argument
  """deal_hashes on a known deal, under its tag."""
  labels = {"deal": np.array([[1, 2, 3]], np.uint8)}
  want = hashlib.sha256(
      b"deal_hash\0" + _GAME.encode() + b"\0r\0" + bytes([1, 2, 3])
  )
  assert deals.deal_hashes(_GAME, "r", None, labels) == [want.hexdigest()]


# The fixture is requested for its side effect.
def test_deal_index_format(defaults, tmp_path):  # pylint: disable=unused-argument
  """The deal index format name."""
  index = deals.DealIndex(_GAME, "r", ["ab", "cd"])
  assert index.to_json()["format"] == "deal_index"
  index.save(str(tmp_path / "a.json"))
  assert deals.DealIndex.load(str(tmp_path / "a.json")).hashes == index.hashes
  d = json.loads((tmp_path / "a.json").read_text())
  (tmp_path / "b.json").write_text(
      json.dumps(dict(d, format="other.deal_index"))
  )
  with pytest.raises(ValueError, match="not a deal_index"):
    deals.DealIndex.load(str(tmp_path / "b.json"))  # saved under another format


# The fixture is requested for its side effect.
def test_shard_manifest_format(defaults):  # pylint: disable=unused-argument
  """The shard manifest format name."""
  assert shards.MANIFEST_FORMAT == "trajectory_shards"


# The fixture is requested for its side effect.
def test_decide_traces_under_its_scope(defaults):  # pylint: disable=unused-argument
  """Decide traces under its named scope."""
  logits = jnp.zeros((2, 3), jnp.float32)
  mask = jnp.ones((2, 3), jnp.bool_)

  def stacks(closed):
    """The name stacks of a jaxpr's equations.

    Args:
      closed: a closed jaxpr.

    Returns:
      the set of name stacks.
    """
    return {str(e.source_info.name_stack) for e in closed.jaxpr.eqns}

  first = jax.make_jaxpr(
      lambda lg: decide.decide(lg, mask, types.ARGMAX, None)[1]
  )(logits)
  assert any("policy_decide" in s for s in stacks(first)), stacks(first)
  decide.set_scope("acme_decide")
  second = jax.make_jaxpr(
      lambda lg: decide.decide(lg, mask, types.ARGMAX, None)[1]
  )(logits)
  assert any("acme_decide" in s for s in stacks(second)), stacks(second)
  with pytest.raises(ValueError, match="identifier"):
    decide.set_scope("not an identifier")


def test_every_environment_variable_has_the_prefix():
  """Every environment variable the package reads has its prefix."""
  assert plugins.ENV_PREFIX == "OPEN_SPIEL_SCALE_"
  for name in (
      plugins.ENV,
      provenance.ENV_VAR,
      provenance.CODE_REPO_ENV,
      hardware.EXPECT_PLATFORM_ENV,
      infoset_invariance.FULL_ENV,
      subprocess_env.PYTHON_ENV,
  ):
    assert name.startswith(plugins.ENV_PREFIX), name


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
