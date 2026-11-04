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

"""The gin rummy reference evaluation: an entrant against the rule bot.

The rule bot (envs.gin.rule_bot) is a fixed, deterministic
opponent, so points per hand against it measure a gin agent on one scale
across runs. Both sides play KNOCK / LAYOFF / WALL with the game's scripted
endgame (GameSpec.endgame) and the other phases with their own policy, play
greedily, and meet on seat-swapped duplicate deals, one hand per match.

  from open_spiel.python.scale.eval import registry
  ev = registry.build("gin_rummy.rule_bot", env)       # env: a GinEnv
  result = ev.vs_reference(entrant, n_deals=512, key=jax.random.key(0))

The payoff is points per hand, from the entrant's seat.
"""

import dataclasses

from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import gin
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.envs.gin import rule_bot as gin_rule_bot
from open_spiel.python.scale.eval import match
from open_spiel.python.scale.eval import registry
from open_spiel.python.scale.policies import routed

NAME = "gin_rummy.rule_bot"
RULE_BOT_NAME = "gin_rule_bot"


def route_endgame(e: match.Entrant, game_spec) -> match.Entrant:
  """The entrant `e` with its endgame played by the scripted endgame.

  E with KNOCK / LAYOFF / WALL played by the game's scripted endgame; an
  entrant that already routes a scripted phase is returned as is.

  Args:
    e: The entrant.
    game_spec: The gin GameSpec (its endgame).

  Returns:
    The routed entrant.
  """
  if routed.is_scripted_phase(e.policy):
    return e
  pol, params = routed.routed_pair(e.policy, e.params, game_spec.endgame)
  return dataclasses.replace(e, policy=pol, params=params)


def rule_bot_entrant() -> match.Entrant:
  """The rule bot as an entrant.

  Returns:
    The Entrant.
  """
  return match.Entrant(
      RULE_BOT_NAME, gin_rule_bot.GinRuleBotPolicy(), {}, types.ARGMAX
  )


def config(game_spec, name: str = NAME) -> match.EvaluatorConfig:
  """The EvaluatorConfig of the evaluation (module doc).

  Args:
    game_spec: The gin GameSpec.
    name: The evaluation name.

  Returns:
    The EvaluatorConfig.
  """
  return match.EvaluatorConfig(
      name=name,
      hands_per_match=1,
      duplicate=True,
      rule=types.ARGMAX,
      wrap=lambda e: route_endgame(e, game_spec),
      reference=rule_bot_entrant,
      max_rows=1024,
  )


def evaluator(
    env: gin.GinEnv, mesh=None, *, name: str = NAME
) -> match.Evaluator:
  """The Evaluator on `env`, a GinEnv.

  Args:
    env: The GinEnv.
    mesh: The device mesh, or None.
    name: The evaluation name.

  Returns:
    The Evaluator.

  Raises:
    TypeError: env is not a GinEnv.
  """
  if not isinstance(env, gin.GinEnv):
    raise TypeError(
        f"the gin rule-bot evaluation plays a GinEnv, got {type(env).__name__}"
    )
  return match.Evaluator(
      env, config(env_registry.game(env.name), name), mesh=mesh
  )


registry.register(NAME, evaluator)
