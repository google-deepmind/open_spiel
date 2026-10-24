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

"""Evaluation of policies: head-to-head matches and exact metrics.

  match           Evaluator: head-to-head on seat-swapped duplicate deals,
                  payoff matrices and population mixtures, every mean with
                  its SE and 95% CI
  payoff          the numpy payoff accounting and the MatchEngine protocol
  stats           mean, SE and normal 95% interval
  exploitability  exact NashConv of a policy in a small game
  heldout         agreement of a student with a teacher on a fixed state set
  mcts_baseline   the random-rollout MCTS-N opponent as an Evaluator reference
  registry        named per-game evaluations (GameSpec.reference_evals)

The default MatchEngine (core.arena_engine), the in-loop eval deadline
(core.deadline) and a small Env's game tree (core.game_tree) live in core, so
beliefs and data sources use them without importing eval.

A game's reference evaluations live with the game (envs/<game>/) and
register with `registry` on import.
"""
