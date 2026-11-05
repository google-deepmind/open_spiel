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

"""PolicyServer's per-artifact HostAgent options: pad_rows and compiler_options.

Each reaches the HostAgents a server builds, given as a value or as a
function of the loaded artifact, and a bad value is refused.
"""

import sys

import pytest

from open_spiel.python.scale.ckpt import artifact as ckpt
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.serve import host_agent
from open_spiel.python.scale.serve import server as server_lib
from open_spiel.python.scale.serve.tests import helpers
import pyspiel


@pytest.fixture(name="kuhn_ckpt", scope="module")
def _kuhn_ckpt_fixture(tmp_path_factory):
  """A Kuhn tabular checkpoint.

  Args:
    tmp_path_factory: The pytest fixture.

  Returns:
    The checkpoint folder.
  """
  d = tmp_path_factory.mktemp("kuhn_ckpt")
  pol, params = helpers.kuhn_tabular(kuhn.KuhnEnv())
  ckpt.write_artifact(
      str(d),
      3,
      pol,
      params,
      rules_sha="r",
      config_hash="c",
      algo="test",
      env_name="kuhn_poker",
      default_rule=types.SAMPLE,
      run_id="42",
  )
  return str(d)


def _kuhn_state(actions):
  """The pyspiel Kuhn state after `actions`.

  Args:
    actions: The actions.

  Returns:
    The state.
  """
  s = pyspiel.load_game("kuhn_poker").new_initial_state()
  for a in actions:
    s.apply_action(a)
  return s


def test_pad_rows_per_artifact(kuhn_ckpt):
  """pad_rows reaches the HostAgents: an int, or a function of the artifact."""
  state = _kuhn_state([0, 1])
  default = server_lib.PolicyServer()
  reply = default.act(state, checkpoint_dir=kuhn_ckpt, step=3, seed=11)
  agent = default.agent(kuhn_ckpt, 3, state.current_player(), "kuhn_poker", {})
  assert agent.pad_rows == host_agent.PAD_ROWS
  seen = []

  def rows(art):
    """pad_rows as a function: records the artifact's step.

    Args:
      art: the artifact.

    Returns:
      1.
    """
    seen.append(art.manifest.step)
    return 1

  one = server_lib.PolicyServer(pad_rows=rows)
  assert (
      one.act(state, checkpoint_dir=kuhn_ckpt, step=3, seed=11).action
      == reply.action
  )
  assert (
      one.agent(kuhn_ckpt, 3, state.current_player(), "kuhn_poker", {}).pad_rows
      == 1
  )
  assert seen and set(seen) == {3}
  with pytest.raises(ValueError, match="pad_rows"):
    server_lib.PolicyServer(pad_rows=0).act(
        state, checkpoint_dir=kuhn_ckpt, step=3
    )


def test_compiler_options_per_artifact(kuhn_ckpt):
  """compiler_options reach the HostAgents and compile the policy path.

  They may be a mapping or a function of the artifact.

  Args:
    kuhn_ckpt: a kuhn_poker checkpoint folder (the fixture).
  """
  state = _kuhn_state([0, 1])
  want = server_lib.PolicyServer().act(
      state, checkpoint_dir=kuhn_ckpt, step=3, seed=11
  )
  opts = {"xla_allow_excess_precision": False}
  for arg in (opts, lambda art: opts):
    srv = server_lib.PolicyServer(compiler_options=arg)
    got = srv.act(state, checkpoint_dir=kuhn_ckpt, step=3, seed=11)
    assert got.action == want.action and got.probs == want.probs
    agent = srv.agent(kuhn_ckpt, 3, state.current_player(), "kuhn_poker", {})
    assert agent.compiler_options == opts
  default = (
      server_lib.PolicyServer()
      .agent(kuhn_ckpt, 3, state.current_player(), "kuhn_poker", {})
      .compiler_options
  )
  assert isinstance(default, dict) and not default, default
  with pytest.raises(ValueError, match="compiler_options"):
    server_lib.PolicyServer(compiler_options=lambda art: 3).act(
        state, checkpoint_dir=kuhn_ckpt, step=3
    )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
