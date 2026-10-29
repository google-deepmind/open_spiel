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

"""Policies: composable implementations of core.types.Policy.

Also the architecture registry that builds them by name.

  registry      arch_name -> constructor; build, register, temporary.
  uniform       uniform random play.
  tabular       a lookup table over an InfoState index, for small games.
  routed        one policy per row, chosen by an InfoState predicate.
  opening       random opening plies, then a base policy.
  search        Monte Carlo tree search (mctx) over a base policy.
  rollout       a random-rollout value policy and the MCTS baseline bot.
  operators     facts of search operators and the BehaviorMixture wrapper.
  members       stacking K loaded policies into one slot.

A game's own policies (scripted policies, game-specific search) live with
the game in envs/<game>/ and register their architectures on import.
"""
