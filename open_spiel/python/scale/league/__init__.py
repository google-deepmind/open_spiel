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

"""Population training: one loop with pluggable parts.

The parts are samplers, controllers, evaluators and learner kinds.

Every training run is a league. A plain self-play run is one learner slot under
a fixed opponent mix; PSRO, snapshot pools, exploiters and population-based
training are the same loop with another controller.

    cfg = runner.one_slot("ppo_kuhn", "kuhn_poker", kinds.PG(learner),
                          num_envs=256, num_steps=64, num_updates=1000,
                          policy_kwargs=network_kwargs)
    runner.run(cfg, checkpoint_dir="/tmp/ppo_kuhn")

Modules:
  runner          LeagueRunConfig, run(), the recipes, and the controller,
                  on-commit and in-loop-eval registries
  loop            League: the loop itself
  kinds           learner kinds (PG, BC, AZ, ExIt) and their drivers
  sampler         opponent mixes (uniform, PFSP, pmix, lagged audit, snapshot
                  pool, meta-strategy, pipeline) and the slot-table draw
  controller      controllers (PSRO, pipeline PSRO, gated pool, continual
                  exploiter, best-response oracle, PBT, lagged audit,
                  scheduled promotion, fixed mix)
  meta_solvers    Nash (LP), fictitious play, regret matching, alpha-rank,
                  replicator dynamics, maximum-entropy Nash, uniform
  mixture         population mixtures as behaviour policies; exact payoffs
  population      a run's final population and its exact exploitability
  refs            PolicyRef, FixedOpponent: policies named in configs
  artifact_store  member artifacts by id
  pool_params     how pool positions are stored per policy type
  commit          checkpoint commits and ArenaSource snapshots
  resume          the overwritten resume checkpoint
  siblings        expert-iteration sibling artifacts and their manifest records
  slot_factories  operators an expert-iteration run builds over its network

The sampler, controller, meta-solvers, the artifact store's index and
pool_params import no array framework.
"""
