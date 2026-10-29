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

"""The random-rollout MCTS-N opponent as an Evaluator reference.

A fixed-strength search opponent measures progress in perfect-information
games (tic_tac_toe, connect_four) independently of the training population.

  cfg = mcts_baseline.vs_mcts_config("connect_four", num_simulations=1000)
  match.Evaluator(env, cfg).vs_reference(subject, n_deals=512, key=key)

  mcts_entrant(game, n)    the random-rollout MCTS-N bot (policies.rollout) of
                           the game named `game` as an Entrant named
                           "mcts<n>"; it plays the search's own action under
                           any rule. env_kwargs: the bot's env options when
                           not the defaults (for example a 9x9 go board).
  vs_mcts_config(game, n)  an EvaluatorConfig whose reference is that bot.
"""

from typing import Any, Mapping, Optional

from open_spiel.python.scale.core import types
from open_spiel.python.scale.eval import match
from open_spiel.python.scale.policies import rollout

DEFAULT_SIMULATIONS = 1000


def mcts_entrant(
    game: str,
    num_simulations: int = DEFAULT_SIMULATIONS,
    env_kwargs: Optional[Mapping[str, Any]] = None,
) -> match.Entrant:
  """The MCTS-N bot of `game` as an Entrant named "mcts<n>" (ARGMAX).

  Args:
    game: the game name.
    num_simulations: MCTS simulations per move.
    env_kwargs: the bot's env options when not the defaults.

  Returns:
    the Entrant.
  """
  pol, params = rollout.mcts_bot(game, num_simulations, env_kwargs)
  return match.Entrant(f"mcts{int(num_simulations)}", pol, params, types.ARGMAX)


def vs_mcts_config(
    game: str,
    num_simulations: int = DEFAULT_SIMULATIONS,
    env_kwargs: Optional[Mapping[str, Any]] = None,
    **kwargs,
) -> match.EvaluatorConfig:
  """EvaluatorConfig(name="vs_mcts<n>", reference=the MCTS-N bot, **kwargs).

  env_kwargs: the bot's env options when not the defaults.

  Args:
    game: the game name.
    num_simulations: MCTS simulations per move.
    env_kwargs: the bot's env options when not the defaults.
    **kwargs: further EvaluatorConfig fields.

  Returns:
    the EvaluatorConfig.
  """
  bot = mcts_entrant(game, num_simulations, env_kwargs)
  return match.EvaluatorConfig(
      name=f"vs_mcts{int(num_simulations)}", reference=lambda: bot, **kwargs
  )
