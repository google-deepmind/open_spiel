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

"""The contract checkers catch planted bugs (positive controls).

Each checker must fail a component with a known defect; a checker that
passes everything is vacuous.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import decide as decide_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.policies import uniform
from open_spiel.python.scale.testing import check_decide
from open_spiel.python.scale.testing import check_env
from open_spiel.python.scale.testing import check_policy
from open_spiel.python.scale.testing import fixtures
from open_spiel.python.scale.testing import steps as steps_lib
from open_spiel.python.scale.testing.errors import ContractError


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """Returns the Kuhn poker env the checks run on.

  Returns:
    The env.
  """
  return kuhn.KuhnEnv()


@pytest.fixture(name="steps", scope="module")
def _steps_fixture(env):
  """Returns two plies of SeatSteps from `env` for seat 1.

  Args:
    env: The Kuhn poker env fixture.

  Returns:
    The steps that check_policy takes after its params.
  """
  return steps_lib.two_ply_steps(env, batch=16, seat=1)


# -- check_env -----------------------------------------------------------------
def test_planted_leak_is_caught():
  """check_env catches an info state that leaks the hidden cards."""

  class LeakyKuhn(kuhn.KuhnEnv):
    """Kuhn poker whose info state encodes both private cards."""

    def info_state(self, s, seat):
      """Returns the info state with both cards folded into num_bets.

      Args:
        s: The world state.
        seat: The observing seat.

      Returns:
        The leaky info state.
      """
      info = super().info_state(s, seat)
      return info._replace(
          num_bets=info.num_bets + 10 * s.p1_card + 100 * s.p0_card
      )

  with pytest.raises(ContractError, match="leaks"):
    check_env.check_env(
        LeakyKuhn(), hide_fn=kuhn.hide_opponent_card, num_playouts=32
    )


def test_vacuous_privacy_hook_is_caught(env):
  """check_env rejects a hide_fn that changes nothing.

  Args:
    env: The Kuhn poker env fixture.
  """
  with pytest.raises(ContractError, match="vacuous"):
    check_env.check_env(env, hide_fn=lambda e, s, q, k: s, num_playouts=16)


def test_planted_bad_chance_probs_are_caught():
  """check_env catches chance probabilities that do not sum to 1."""

  class HalfProbs(kuhn.KuhnEnv):
    """Kuhn poker whose chance probabilities sum to 0.5."""

    def chance_probs(self, s):
      """Returns half the true chance probabilities.

      Args:
        s: The world state.

      Returns:
        The halved probabilities.
      """
      return 0.5 * super().chance_probs(s)

  with pytest.raises(ContractError, match="sum to 1"):
    check_env.check_env(HalfProbs(), num_playouts=16)


def test_planted_f16_chance_probs_are_caught():
  """check_env catches chance probabilities that are not f32."""

  class F16Probs(kuhn.KuhnEnv):
    """Kuhn poker whose chance probabilities are f16."""

    def chance_probs(self, s):
      """Returns the true chance probabilities cast to f16.

      Args:
        s: The world state.

      Returns:
        The f16 probabilities.
      """
      return super().chance_probs(s).astype(jnp.float16)

  with pytest.raises(ContractError, match="f32"):
    check_env.check_env(F16Probs(), num_playouts=16)


def test_planted_init_bypassing_the_raw_layer_is_caught():
  """check_env catches an init that bypasses root, apply_chance and resolve."""

  class OwnDeal(kuhn.KuhnEnv):
    """Kuhn poker whose init deals a fixed hand itself."""

    def init(self, deal_key):
      """Returns a fixed deal, ignoring the raw chance layer.

      Args:
        deal_key: Unused; the deal is fixed.

      Returns:
        The world state of the fixed deal.
      """
      del deal_key  # Unused.
      return self.deal(0, 1)  # ignores root/apply_chance/resolve

  with pytest.raises(ContractError):
    check_env.check_env(OwnDeal(), num_playouts=16)


def test_planted_bad_world_from_info_is_caught():
  """check_env catches a world_from_info that does not round-trip."""

  class WrongWorld(kuhn.KuhnEnv):
    """Kuhn poker whose world_from_info returns a fixed deal."""

    def world_from_info(self, info):
      """Returns a fixed deal, not the world `info` came from.

      Args:
        info: Unused.

      Returns:
        The world state of the fixed deal.
      """
      del info  # Unused.
      return self.deal(0, 1)  # not the world the info came from

  with pytest.raises(ContractError, match="round-trip"):
    check_env.check_env(WrongWorld(), num_playouts=16)


def test_planted_non_mapping_labels_are_caught():
  """check_env catches labels that are not a Mapping."""

  class ListLabels(kuhn.KuhnEnv):
    """Kuhn poker whose labels are a list."""

    def labels(self, s, seat):
      """Returns the labels as a list instead of a Mapping.

      Args:
        s: The world state.
        seat: The labelled seat.

      Returns:
        A one-element list.
      """
      del seat  # Unused.
      return [s.p0_card]

  with pytest.raises(ContractError, match="Mapping"):
    check_env.check_env(ListLabels(), num_playouts=16)


# -- check_policy --------------------------------------------------------------
def test_planted_stateful_without_state_is_caught(env, steps):
  """check_policy catches a policy that declares state but carries nothing.

  Args:
    env: The Kuhn poker env fixture.
    steps: The SeatSteps fixture.
  """

  class Liar(uniform.UniformPolicy):
    """A uniform policy whose spec claims it is stateful.

    Attributes:
      spec: The PolicySpec, with stateful set.
    """

    def __init__(self, *a):
      super().__init__(*a)
      self.spec = type(self.spec)(**{**self.spec.__dict__, "stateful": True})

  pol = Liar("kuhn_poker", env.num_actions, env.num_players)
  with pytest.raises(ContractError, match="carries nothing"):
    check_policy.check_policy(pol, {}, {}, *steps)


def test_planted_carry_mutation_is_caught(env, steps):
  """check_policy catches an evaluate that mutates the carry in place.

  Args:
    env: The Kuhn poker env fixture.
    steps: The SeatSteps fixture.
  """

  class Mutator(uniform.UniformPolicy):
    """A uniform policy whose evaluate increments its numpy carry."""

    def init_carry(self, params, batch, seat, key):
      """Returns a zero counter carry.

      Args:
        params: The policy's parameters.
        batch: The number of rows.
        seat: i32[batch], each row's seat.
        key: The PRNG key.

      Returns:
        {"n": f32[batch] zeros}.
      """
      del params, seat, key  # Unused.
      return {"n": np.zeros((batch,), np.float32)}

    def evaluate(self, params, carry, step, key):
      """Evaluates acting rows and, wrongly, mutates the carry.

      Args:
        params: The policy's parameters.
        carry: The seat's carry; incremented in place (the planted bug).
        step: The SeatStep of the acting rows.
        key: The PRNG key.

      Returns:
        The uniform policy's output.
      """
      if isinstance(carry["n"], np.ndarray):
        carry["n"] += 1.0  # the bug: evaluate must never touch the carry
      return super().evaluate(params, (), step, key)

  pol = Mutator("kuhn_poker", env.num_actions, env.num_players)
  with pytest.raises(ContractError, match="mutated"):
    check_policy.check_policy(pol, {}, {}, *steps)


def test_planted_ignores_new_hand_is_caught(env, steps):
  """check_policy catches a policy that does not reset at new_hand.

  Args:
    env: The Kuhn poker env fixture.
    steps: The SeatSteps fixture.
  """

  class NoHandReset(fixtures.HistoryPolicy):
    """A history policy that resets only at new_match."""

    def observe(self, params, carry, step):
      """Observes `step` as if new_hand were new_match.

      Args:
        params: The policy's parameters.
        carry: The seat's carry.
        step: The SeatStep observed.

      Returns:
        The carry after the step.
      """
      return super().observe(
          params, carry, step._replace(new_hand=step.new_match)
      )

  pol = NoHandReset("kuhn_poker", env.num_actions, env.num_players)
  with pytest.raises(ContractError, match="new_hand"):
    check_policy.check_policy(pol, pol.init_params(), pol.init_params(), *steps)


def test_planted_new_hand_treated_as_no_reset_by_cross_hand_policy_is_caught(
    steps,
):
  """A cross-hand policy must still reset everything at new_match.

  Args:
    steps: The SeatSteps fixture.
  """

  class NoMatchReset(fixtures.HistoryPolicy):
    """A history policy that counts hands across matches."""

    def observe(self, params, carry, step):
      """Observes `step` and adds a hand count never reset at new_match.

      Args:
        params: The policy's parameters.
        carry: The seat's carry.
        step: The SeatStep observed.

      Returns:
        The carry after the step.
      """
      out = super().observe(params, carry, step)
      out["hands"] = carry["hands"] + step.new_hand.astype(jnp.float32)
      return out

  pol = NoMatchReset("kuhn_poker", 2, 2, cross_hand=True)
  with pytest.raises(ContractError, match="new_match"):
    check_policy.check_policy(
        pol, pol.init_params(), pol.init_params(), *steps, cross_hand_state=True
    )


def test_planted_bad_observe_seq_override_is_caught(steps):
  """check_policy catches an observe_seq that differs from observe.

  Args:
    steps: The SeatSteps fixture.
  """

  class BadSeq(fixtures.HistoryPolicy):
    """A history policy whose observe_seq skips every step."""

    def observe_seq(self, params, carry, steps):
      """Returns `carry` unchanged instead of observing `steps`.

      Args:
        params: The policy's parameters.
        carry: The seat's carry.
        steps: Steps [B, T, ...]; ignored (the planted bug).

      Returns:
        `carry`, unchanged.
      """
      del params, steps
      return carry  # skips every step

  pol = BadSeq("kuhn_poker", 2, 2)
  with pytest.raises(ContractError, match="observe_seq"):
    check_policy.check_policy(pol, pol.init_params(), pol.init_params(), *steps)


def test_planted_bad_evaluate_seq_override_is_caught(steps):
  """check_policy catches an evaluate_seq that differs from evaluate.

  Args:
    steps: The SeatSteps fixture.
  """

  class BadEvalSeq(fixtures.HistoryPolicy):
    """A history policy whose evaluate_seq shifts the logits."""

    def evaluate_seq(self, params, carry, steps, keys):
      """Returns the default scan's outputs with every logit plus 1.

      Args:
        params: The policy's parameters.
        carry: The seat's carry.
        steps: Steps [B, T, ...].
        keys: The per-ply keys stacked on a time axis, or None.

      Returns:
        (final carry, PolicyOutput [B, T, ...]) with shifted logits.
      """
      c, out = super().evaluate_seq(params, carry, steps, keys)
      return c, out._replace(logits=out.logits + 1.0)

  pol = BadEvalSeq("kuhn_poker", 2, 2)
  with pytest.raises(ContractError, match="evaluate_seq"):
    check_policy.check_policy(pol, pol.init_params(), pol.init_params(), *steps)


def test_host_policy_inheriting_the_jax_scan_is_caught(steps):
  """check_policy catches a host policy that keeps the JAX-only scans.

  Args:
    steps: The SeatSteps fixture.
  """

  class NoSeq(fixtures.NumpyHistoryPolicy):
    """A numpy history policy that inherits the JAX scans."""

    observe_seq = types.Policy.observe_seq
    evaluate_seq = types.Policy.evaluate_seq

  pol = NoSeq("kuhn_poker", 2, 2)
  with pytest.raises(ContractError, match="must implement observe_seq"):
    check_policy.check_policy(pol, pol.init_params(), pol.init_params(), *steps)


def test_numpy_host_policy_passes_check_policy(steps):
  """The numpy history policy passes check_policy (negative control).

  Args:
    steps: The SeatSteps fixture.
  """
  pol = fixtures.NumpyHistoryPolicy("kuhn_poker", 2, 2)
  w = np.array([[0.0, 1.0], [0.5, 0.0]], np.float32)
  check_policy.check_policy(
      pol, pol.make_params(w=w), pol.make_params(), *steps
  )


def test_planted_init_params_mismatch_is_caught(steps):
  """check_policy catches init_params that differ from the params' tree.

  Args:
    steps: The SeatSteps fixture.
  """

  class WrongInit(fixtures.HistoryPolicy):
    """A history policy whose init_params add an extra leaf."""

    def init_params(self, key=None):
      """Returns the base params plus an extra leaf "c".

      Args:
        key: The PRNG key, or None.

      Returns:
        The mismatched params.
      """
      p = super().init_params(key)
      p["c"] = jnp.zeros((3,), jnp.float32)
      return p

  pol = WrongInit("kuhn_poker", 2, 2)
  ok = fixtures.HistoryPolicy("kuhn_poker", 2, 2).init_params()
  with pytest.raises(ContractError, match="init_params"):
    check_policy.check_policy(pol, ok, ok, *steps)


# -- check_decide --------------------------------------------------------------
def test_planted_bad_decide_is_caught():
  """check_decide catches a decide that ignores the legal mask."""

  def bad(out, mask, rule, key):
    """Decides with every action marked legal.

    Args:
      out: The policy logits.
      mask: The legal mask; ignored (the planted bug).
      rule: The decide rule.
      key: The PRNG key.

    Returns:
      decide's (actions, probs) over all actions.
    """
    del mask
    return decide_lib.decide(out, jnp.ones(jnp.shape(out), bool), rule, key)

  with pytest.raises(AssertionError):
    check_decide.check_decide(bad)


def _decide_variant(dtype, clip):
  """Decide's maths at `dtype`, optionally without the upper clip.

  Args:
    dtype: The dtype.
    clip: Whether to clip above.

  Returns:
    The variant decide function.
  """

  def fn(out, mask, rule, key):
    """Decides at `dtype`, clipping above only if `clip`.

    Args:
      out: The policy logits.
      mask: bool[B, A], the legal mask.
      rule: The decide rule.
      key: The PRNG key.

    Returns:
      (i32[B] actions, -1 where no action is legal; f64[B, A] probs).
    """
    logits = jnp.asarray(out).astype(dtype) / jnp.asarray(
        rule.temperature, dtype
    )
    hi = jnp.finfo(dtype).max if clip else jnp.inf
    masked = jnp.where(
        mask, jnp.clip(logits, jnp.finfo(dtype).min, hi), -jnp.inf
    )
    any_legal = jnp.any(mask, -1)
    safe = jnp.where(any_legal[:, None], masked, jnp.zeros((), dtype))
    probs = jnp.where(
        any_legal[:, None], jax.nn.softmax(safe, -1), jnp.zeros((), dtype)
    )
    if rule.kind == "argmax":
      a = jnp.argmax(masked, -1).astype(jnp.int32)
    else:
      a = jax.random.categorical(key, safe, -1).astype(jnp.int32)
    return jnp.where(any_legal, a, jnp.int32(-1)), probs.astype(jnp.float64)

  return fn


def test_planted_decide_without_upper_clip_is_caught():
  """check_decide catches a decide without the upper logit clip."""
  with pytest.raises(ContractError, match="overflow"):
    check_decide.check_decide(_decide_variant(jnp.float64, clip=False))


def test_planted_f32_decide_is_caught():
  """An f32 decide that returns f64-typed probs still fails.

  Its arithmetic misses the f64 known answers (rtol 1e-12).
  """
  with pytest.raises(ContractError, match="softmax row 0|f64 known answer"):
    check_decide.check_decide(_decide_variant(jnp.float32, clip=True))
  check_decide.check_decide(
      _decide_variant(jnp.float64, clip=True)
  )  # the control passes


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
