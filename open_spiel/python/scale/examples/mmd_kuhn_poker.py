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

"""Magnetic mirror descent (MMD) on Kuhn poker, with exact exploitability.

What it shows:
  * a ready-made recipe, `configs.recipes.mmd_self_play` (Sokota et al.
    2023), adjusted with the same dotted overrides the command-line trainer
    takes (`league.config_common.apply_overrides`, `with_overrides`);
  * the metrics stream: a custom sink (`metrics.MetricSink`) handed to the
    trainer receives every row the run writes, including the exact
    exploitability the trainer tracks in-run for small games;
  * `eval.exploitability.exploitability`, which enumerates the whole game
    tree, so the number has no sampling error. Exploitability is zero exactly
    at a Nash equilibrium; the uniform policy's is 11/24 = 0.4583;
  * the reference the curve should approach, computed exactly on the same
    game tree (`core.game_tree.GameTree`).

MMD with a fixed entropy temperature does not converge to a Nash equilibrium:
its last iterate converges to the entropy-regularised equilibrium of the game
at that temperature (the agent logit quantal response equilibrium: at every
information state the policy is the softmax of its action values divided by
the temperature). That point has a positive exploitability which grows with
the temperature, so a curve that levels off above zero is the expected
behaviour, not a failure to learn. The script computes the regularised
equilibrium exactly (tabular MMD on the 12 information states, no network)
and prints its exploitability as the reference line. Driving exploitability
to zero means annealing the temperature towards zero, and the step size with
it: at small temperatures the same update with a large step cycles instead
of converging.

The advantages are left unnormalised here (the recipe normalises them), so
the temperature is in payoff units and the reference is a fixed number:
0.0436 at the default temperature 0.2.

The game is Kuhn poker from the JAX games package (OpenSpiel's
`kuhn_poker`). The default 100 updates take two to three minutes on a CPU,
most of it the exact evaluation after every update. Exploitability falls from
about 0.65 (the untrained network) past the uniform policy's 0.46 and levels
off a little above the regularised equilibrium's 0.0436: over the last 50
updates it averaged 0.052, 0.056 and 0.069 (seeds 0, 1, 2), moving between
about 0.03 and 0.09 with the sampling noise of each update. The gap is
consistent with that noise: a policy that jitters around the equilibrium is
on average more exploitable than the equilibrium itself.

  python -m open_spiel.python.scale.examples.mmd_kuhn_poker
  python -m open_spiel.python.scale.examples.mmd_kuhn_poker --num_updates=20
  python -m open_spiel.python.scale.examples.mmd_kuhn_poker --temperature=0.5
"""

import time

from absl import app
from absl import flags
import numpy as np

from open_spiel.python.scale import metrics
from open_spiel.python.scale.configs import recipes
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.eval import exploitability
from open_spiel.python.scale.league import config_common
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.policies import uniform

_NUM_UPDATES = flags.DEFINE_integer("num_updates", 100, "Training updates.")
_NUM_ENVS = flags.DEFINE_integer(
    "num_envs", 256, "Games played in parallel per update."
)
_NUM_STEPS = flags.DEFINE_integer("num_steps", 16, "Plies per row per update.")
_LR = flags.DEFINE_float("lr", 1e-3, "Adam step size.")
_ANNEAL = flags.DEFINE_bool(
    "anneal", True, "Anneal the step size linearly to zero."
)
_TEMPERATURE = flags.DEFINE_float(
    "temperature",
    0.2,
    "Entropy temperature (the entropy coefficient on unnormalised "
    "advantages, in payoff units). The exact reference needs 0.1 or more.",
)
_SEED = flags.DEFINE_integer("seed", 0, "Seed of the run.")

GAME = "kuhn_poker"
TAIL = 50  # updates averaged for the plateau


class ExploitabilityPrinter(metrics.MetricSink):
  """A metric sink that prints the in-run exact exploitability as it arrives.

  Attributes:
    curve: (update, exploitability) pairs, one per eval.
  """

  def __init__(self):
    self.curve = []  # (update, exploitability)

  def write(self, step, values):
    """Prints and records the row's exploitability, if it has one.

    Args:
      step: The update.
      values: {prefixed name: number}.
    """
    if "eval/exploitability" in values:
      e = float(values["eval/exploitability"])
      self.curve.append((int(step), e))
      print(f"update {int(step):5d}  exploitability {e:.4f}", flush=True)

  def params(self, values):
    """Ignores the run's parameters.

    Args:
      values: Unused.
    """
    del values

  def flush(self):
    """Does nothing: every row is printed as it arrives."""

  def close(self):
    """Does nothing."""


def _edge_weights(tree, table):
  """Per node: the probability of the edge from its parent under `table`.

  Args:
    tree: The game tree.
    table: The policy table.

  Returns:
    f64[num_nodes], the edge weights.
  """
  w = np.ones(tree.num_nodes)
  kids = np.arange(1, tree.num_nodes)
  chance = tree.in_chance[kids]
  w[kids[chance]] = tree.edge_p[kids[chance]]
  dec = kids[~chance]
  w[dec] = table[tree.infoset[tree.parent[dec]], tree.in_action[dec]]
  return w


def action_values(tree, table):
  """Returns each information state's action values for the seat acting there.

  Every seat plays `table`; the histories are weighted by how likely the other
  seats and chance make them.

  Args:
    tree: The game tree.
    table: The policy table.

  Returns:
    f64[I, A].
  """
  w = _edge_weights(tree, table)
  levels = [
      np.flatnonzero(tree.level == lvl)
      for lvl in range(int(tree.level.max()) + 1)
  ]
  value = tree.node_returns.copy()  # f64[N, P]
  for nodes in reversed(levels[1:]):
    np.add.at(value, tree.parent[nodes], w[nodes][:, None] * value[nodes])
  q = np.zeros((tree.num_infosets, tree.num_actions))
  for p in range(tree.num_players):
    reach = np.ones(
        tree.num_nodes
    )  # the others' and chance's share of each node's reach
    for nodes in levels[1:]:
      parent = tree.parent[nodes]
      own = ~tree.in_chance[nodes] & (tree.actor[parent] == p)
      reach[nodes] = reach[parent] * np.where(own, 1.0, w[nodes])
    for row in np.flatnonzero(tree.player == p):
      m = tree.members[row]
      for a in np.flatnonzero(tree.legal[row]):
        q[row, a] = reach[m] @ value[tree.child_index[m, a], p] / reach[m].sum()
  return q


def _softmax(tree, logits):
  """The softmax of `logits` over the legal actions.

  Args:
    tree: The game tree.
    logits: The logits per infoset.

  Returns:
    The probabilities.
  """
  z = np.where(tree.legal, logits, -np.inf)
  e = np.where(tree.legal, np.exp(z - np.max(z, axis=1, keepdims=True)), 0.0)
  return e / e.sum(1, keepdims=True)


def regularized_equilibrium(
    tree, temperature, step=0.05, max_iters=20_000, tol=1e-12
):
  """Returns the entropy-regularised equilibrium at `temperature`.

  It is the table with table = softmax(action_values(table) / temperature) at
  every information state, found by exact tabular MMD from the uniform policy
  (logits <- (logits + eta * Q) / (1 + eta * temperature), eta * temperature =
  step).

  Args:
    tree: The game tree.
    temperature: The regularisation temperature.
    step: eta * temperature.
    max_iters: The iteration limit.
    tol: The fixed-point tolerance.

  Returns:
    The table.

  Raises:
    RuntimeError: the iteration did not reach the fixed point (it cycles at
      temperatures below about 0.1 with this step).
  """
  logits = np.zeros((tree.num_infosets, tree.num_actions))
  table = _softmax(tree, logits)
  for _ in range(max_iters):
    logits = (logits + step / temperature * action_values(tree, table)) / (
        1.0 + step
    )
    new = _softmax(tree, logits)
    done = np.abs(new - table).max() < tol
    table = new
    if done:
      break
  gap = np.abs(
      _softmax(tree, action_values(tree, table) / temperature) - table
  ).max()
  if gap > 1e-9:
    raise RuntimeError(
        f"no regularised equilibrium found at temperature {temperature} "
        f"(fixed-point gap {gap:.1e}); use a temperature of 0.1 or more"
    )
  return table


def main(argv):
  """Trains MMD on Kuhn poker and compares it with the exact equilibrium.

  Args:
    argv: The command line; no positional arguments.

  Raises:
    app.UsageError: an argument is invalid.
    RuntimeError: the run wrote no exploitability rows.
  """
  # The framework needs 64-bit JAX; on before anything is built.
  x64.enable_x64()
  if len(argv) > 1:
    raise app.UsageError(f"unexpected arguments {argv[1:]}")
  temperature = _TEMPERATURE.value
  if not temperature > 0:
    raise app.UsageError(f"--temperature must be positive, got {temperature}")
  cfg = recipes.mmd_self_play(
      "example_mmd_kuhn_poker",
      GAME,
      "kuhn_poker_observation_tensor",
      [7],
      2,
      num_updates=_NUM_UPDATES.value,
      num_steps=_NUM_STEPS.value,
      num_envs=_NUM_ENVS.value,
  )
  cfg = config_common.apply_overrides(
      cfg,
      [
          f"learner.learner.lr={_LR.value}",
          f"learner.learner.lr_end={0.0 if _ANNEAL.value else _LR.value}",
          f"learner.learner.anneal_lr={_ANNEAL.value}",
          f"learner.learner.ent_coef.start={temperature}",
          "learner.learner.pg.normalize_advantages=False",
      ],
  )
  cfg = config_common.with_overrides(cfg, seed=_SEED.value)

  env = env_registry.make(GAME)
  tree = exploitability.GameTree(env)
  uni = uniform.UniformPolicy(GAME, env.num_actions, env.num_players)
  uni_expl = exploitability.exploitability(env, uni, {}, tree=tree)
  print(
      f"uniform policy: exploitability {uni_expl.exploitability:.4f}",
      flush=True,
  )
  reference = regularized_equilibrium(tree, temperature)
  ref_expl = exploitability.table_result(tree, reference).exploitability
  print(
      f"regularised equilibrium at temperature {temperature:g}: exploitability "
      f"{ref_expl:.4f} (where MMD at this temperature converges to)",
      flush=True,
  )

  printer = ExploitabilityPrinter()
  t0 = time.time()
  res = runner.run(cfg, sinks=(printer,))
  train_s = time.time() - t0
  table = tree.policy_table(
      res.learner.policy(), res.state.params, types.SAMPLE
  )
  final = exploitability.table_result(tree, table)
  print(f"trained {len(res.metrics)} updates in {train_s:.0f} s", flush=True)
  print(
      f"final policy: exploitability {final.exploitability:.4f} "
      f"(NashConv {final.nash_conv:.4f}, {final.num_infosets} information "
      "states)",
      flush=True,
  )
  if not printer.curve:
    raise RuntimeError("the run wrote no eval/exploitability rows")
  print(
      f"exploitability {printer.curve[0][1]:.4f} -> {final.exploitability:.4f}",
      flush=True,
  )
  tail = [e for _, e in printer.curve[-TAIL:]]
  print(
      f"last {len(tail)} updates: exploitability mean {np.mean(tail):.4f} "
      f"(min {min(tail):.4f}, max {max(tail):.4f}); regularised equilibrium "
      f"{ref_expl:.4f}",
      flush=True,
  )
  print(
      "final policy vs regularised equilibrium: largest probability "
      f"difference {np.abs(table - reference).max():.3f}",
      flush=True,
  )


if __name__ == "__main__":
  app.run(main)
