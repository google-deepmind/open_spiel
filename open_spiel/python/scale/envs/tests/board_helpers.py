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

"""JitEnv: an Env whose methods run jitted.

(one compile per method and input shape), so host-side known-answer tests do not
run the board games' engines op by op. Attributes pass through; check_env
accepts it as an Env.
"""

import jax

_JITTED = (
    "root",
    "apply",
    "apply_chance",
    "init",
    "step",
    "is_chance",
    "chance_probs",
    "terminal",
    "actor",
    "legal_mask",
    "returns",
    "info_state",
    "labels",
    "visible_action",
    "world_from_info",
)


class JitEnv:
  """An Env whose methods run jitted; attributes pass through.

  Attributes:
    env: The wrapped Env.
  """

  def __init__(self, env):
    self.env = env
    self._fns = {m: jax.jit(getattr(env, m)) for m in _JITTED}

  def __getattr__(self, name):
    fns = self.__dict__.get("_fns", {})
    if name in fns:
      return fns[name]
    return getattr(self.__dict__["env"], name)


def play(env, actions):
  """The state after `actions` from the env's start; each must be legal.

  Args:
    env: The env.
    actions: The actions to apply.

  Returns:
    The state.
  """
  s = env.init(jax.random.key(0))
  for a in actions:
    assert not bool(env.terminal(s)), actions
    assert bool(env.legal_mask(s)[a]), a
    s = env.step(s, jax.numpy.int32(a), jax.random.key(0))
  return s
