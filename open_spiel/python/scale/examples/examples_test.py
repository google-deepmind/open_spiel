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

"""Runs every example end to end with tiny flags and checks what it prints.

Each example runs in its own process, exactly as a user runs it
(`python -m open_spiel.python.scale.examples.<example>`). The package's import
name and root come from launch/subprocess_env.py (derived from the
package's own `__name__`), so the test works wherever the package is
installed, nested in another package or not.

A long test, not in the CI tier: each example is a fresh process that imports
JAX, jit-compiles and trains, 2-27 s each on 12 CPU cores (72 s for the file
under ctest) and longer on a loaded machine, over the tier's 60 s per-file
budget.
"""

import os
import re
import tempfile

from absl.testing import absltest

from open_spiel.python.scale.examples import play_checkpoints
from open_spiel.python.scale.launch import subprocess_env


def run_example(name: str, *flags: str) -> str:
  """Runs an example as a module in a child interpreter.

  Args:
    name: The example module, under examples.
    *flags: Its flags.

  Returns:
    Its stdout.

  Raises:
    AssertionError: it exited non-zero.
  """
  proc = subprocess_env.run_python(
      ["-m", subprocess_env.module(f"examples.{name}"), *flags], timeout=900
  )
  if proc.returncode != 0:
    raise AssertionError(
        f"{name} exited {proc.returncode}\n--- stdout\n{proc.stdout}\n"
        f"--- stderr (tail)\n{proc.stderr[-6000:]}"
    )
  return proc.stdout


class ExamplesTest(absltest.TestCase):
  """Every example runs end to end at a small size."""

  def test_quickstart(self):
    """The quickstart example trains and evaluates."""
    with tempfile.TemporaryDirectory() as out_dir:
      out = run_example(
          "quickstart",
          "--updates=2",
          "--rows=16",
          "--plies=8",
          "--eval_deals=4",
          f"--out_dir={out_dir}",
      )
      self.assertTrue(
          os.path.exists(os.path.join(out_dir, "train_state.json")), out
      )
      self.assertIn(
          f"checkpoint and metrics in {os.path.abspath(out_dir)}", out
      )
    m = re.search(
        r"vs uniform random, \d+ seat-swapped deal pairs: mean payoff"
        r" ([+-]\d\.\d{3})",
        out,
    )
    self.assertIsNotNone(m, out)
    self.assertBetween(float(m.group(1)), -1.0, 1.0)

  def test_jax_games_vs_openspiel(self):
    """jax_games_vs_openspiel agrees with OpenSpiel."""
    out = run_example(
        "jax_games_vs_openspiel",
        "--playouts=3",
        "--batch=64",
        "--games=tic_tac_toe,kuhn_poker",
    )
    self.assertRegex(
        out, r"tic_tac_toe: 3 random playouts agree with OpenSpiel"
    )
    self.assertRegex(out, r"kuhn_poker: 3 random playouts agree with OpenSpiel")
    self.assertRegex(
        out, r"kuhn_poker: 64 random games in one jitted vmap loop"
    )

  def test_ppo_tic_tac_toe(self):
    """ppo_tic_tac_toe trains."""
    out = run_example(
        "ppo_tic_tac_toe",
        "--num_updates=3",
        "--num_envs=64",
        "--num_steps=16",
        "--eval_games=64",
    )
    self.assertIn("trained 3 updates", out)
    self.assertRegex(
        out, r"trained network vs uniform: win \d\.\d{3}  draw \d\.\d{3}"
    )

  def test_mmd_kuhn_poker(self):
    """mmd_kuhn_poker prints the exact equilibrium."""
    out = run_example("mmd_kuhn_poker", "--num_updates=6", "--num_envs=128")
    # The exact entropy-regularised equilibrium at the default temperature 0.2.
    self.assertIn(
        "regularised equilibrium at temperature 0.2: exploitability 0.0436", out
    )
    curve = [
        float(x)
        for x in re.findall(r"update +\d+  exploitability (\d\.\d+)", out)
    ]
    self.assertGreaterEqual(len(curve), 2, out)
    m = re.search(r"final policy: exploitability (\d\.\d+)", out)
    self.assertIsNotNone(m, out)
    self.assertLess(float(m.group(1)), curve[0], out)

  def test_alphazero_tic_tac_toe(self):
    """alphazero_tic_tac_toe trains."""
    out = run_example(
        "alphazero_tic_tac_toe",
        "--num_iterations=2",
        "--batch_size=8",
        "--num_simulations=4",
        "--eval_games=32",
    )
    self.assertIn("trained 2 iterations", out)
    self.assertRegex(out, r"search \+ trained vs uniform:  win \d\.\d{3}")

  def test_play_checkpoints(self):
    """Trains 2 updates with a checkpoint and plays the artifact.

    It plays against uniform and against itself, reads the games back and
    replays them in pyspiel.
    """
    with tempfile.TemporaryDirectory() as tmp:
      ckpt = os.path.join(tmp, "ttt")
      out = run_example(
          "ppo_tic_tac_toe",
          "--num_updates=2",
          "--num_envs=16",
          "--num_steps=8",
          "--eval_games=4",
          f"--checkpoint_dir={ckpt}",
      )
      self.assertIn(f"artifact: {ckpt}@2", out)
      vs_uniform = os.path.join(tmp, "vs_uniform")
      out = run_example(
          "play_checkpoints",
          f"--a={ckpt}@2",
          "--b=uniform",
          "--games=8",
          f"--out={vs_uniform}",
      )
      self.assertRegex(
          out, r"a's mean payoff per game: [+-]\d\.\d{4} \+/- \d\.\d{4}"
      )
      self.assertRegex(
          out, r"a in seat 0: [+-]\d\.\d{4} \+/- \d\.\d{4}; a in seat 1"
      )
      self.assertIn(f"wrote 8 games to {vs_uniform}", out)
      self_play = os.path.join(tmp, "self_play")
      out = run_example(
          "play_checkpoints",
          f"--a={ckpt}",
          f"--b={ckpt}@2",
          "--games=8",
          "--probs",
          f"--out={self_play}",
      )
      self.assertIn("wrote 8 games", out)
      for d in (vs_uniform, self_play):
        summary, records = play_checkpoints.read_trajectories(d)
        records = list(records)
        self.assertEqual(summary["game"], "tic_tac_toe")
        self.assertEqual(summary["entrants"]["a"]["step"], 2)
        self.assertLen(records, 8)
        for i, rec in enumerate(records):
          self.assertEqual((rec["index"], rec["deal"]), (i, i // 2))
          self.assertEqual(
              rec["seats"], ["a", "b"] if i % 2 == 0 else ["b", "a"]
          )
          self.assertEqual(rec["game_string"], "tic_tac_toe")
          self.assertEqual(
              rec["history"], rec["env_history"]
          )  # no chance in tic-tac-toe
          self.assertEqual(
              rec["actors"], [j % 2 for j in range(len(rec["history"]))]
          )
          self.assertEqual(sum(rec["returns"]), 0.0)
        # a's payoff per seat, from the records, is what the summary reports.
        a_pay = [r["returns"][r["seats"].index("a")] for r in records]
        self.assertAlmostEqual(
            sum(a_pay) / 8, summary["result"]["a"]["mean"], places=9
        )
      for rec in records:  # self-play, with --probs
        self.assertLen(rec["policy"], len(rec["history"]))
        for a, dist in zip(rec["history"], rec["policy"]):
          self.assertAlmostEqual(sum(p for _, p in dist), 1.0, places=6)
          self.assertIn(a, [x for x, _ in dist])
      try:
        # Lazy: pyspiel is an optional dependency; only its availability
        # matters here.
        import pyspiel  # pylint: disable=g-import-not-at-top,import-outside-toplevel,unused-import  # noqa: PLC0415

        del pyspiel
        has_pyspiel = True
      except ImportError:
        has_pyspiel = False
      if not has_pyspiel:
        self.assertIn("pyspiel is not installed", out)
        return
      self.assertIn("replayed 8 saved games in pyspiel", out)
      for rec in records:
        state = play_checkpoints.replay_in_pyspiel(rec)
        self.assertEqual(state.history(), rec["history"])


class KnownReturnDifferenceTest(absltest.TestCase):

  def test_gin_equal_deadwood_undercut_only(self):
    try:
      # Lazy: pyspiel is an optional dependency.
      import pyspiel  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415
    except ModuleNotFoundError:
      self.skipTest("needs pyspiel")
    gin = pyspiel.load_game("gin_rummy(oklahoma=true)")
    known = play_checkpoints.known_return_difference
    self.assertIsNotNone(known(gin, [0.0, 0.0], [25.0, -25.0]))
    self.assertIsNotNone(known(gin, [-0.0, 0.0], [-25.0, 25.0]))
    self.assertIsNone(known(gin, [0.0, 0.0], [27.0, -27.0]))
    self.assertIsNone(known(gin, [3.0, -3.0], [25.0, -25.0]))
    self.assertIsNone(
        known(pyspiel.load_game("kuhn_poker"), [0.0, 0.0], [25.0, -25.0])
    )


if __name__ == "__main__":
  absltest.main()
