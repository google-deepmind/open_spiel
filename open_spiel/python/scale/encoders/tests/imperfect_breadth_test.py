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

"""Tests for the liars_dice, phantom_ttt_ir and bargaining research encoders.

The encoder contract (declared shape and dtype under x64, batch == per example,
jit == eager, deterministic), hand-computed known vectors (liars_dice's against
a pyspiel information_state_tensor read off pyspiel), and agreement with the JAX
engine's (pyspiel-equivalent) tensors on random playouts.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.encoders import registry
from open_spiel.python.scale.envs import bargaining
from open_spiel.python.scale.envs import liars_dice
from open_spiel.python.scale.envs import phantom_ttt_ir
from open_spiel.python.scale.testing import check_env

NAMES = {
    "liars_dice_info_state": liars_dice.LiarsDiceEnv,
    "phantom_ttt_ir_observation": phantom_ttt_ir.PhantomTTTIREnv,
    "bargaining_info_state": bargaining.BargainingEnv,
}


def _raw(env, events):
  """Applies `events`, chance outcomes at chance nodes, from the root.

  Args:
    env: The env.
    events: The chance outcomes and actions.

  Returns:
    The state.
  """
  s = env.root()
  for e in events:
    s = env.apply_chance(s, e) if bool(env.is_chance(s)) else env.apply(s, e)
  return s


def _check(name, info, expected):
  enc = registry.get(name)
  out = enc(info)
  assert out.dtype == enc.dtype == jnp.float32
  np.testing.assert_array_equal(
      np.asarray(out), np.asarray(expected, np.float32)
  )


@pytest.fixture(name="states", scope="module")
def _states_fixture():
  """Every state of random hands of each game.

  Returns:
    {name: (env, states, terminal flags)}.
  """
  # Reaches private _playout of the code under test.
  # pylint: disable=protected-access
  out = {}
  for name, ctor in NAMES.items():
    env = ctor()
    kd, kp = jax.random.split(jax.random.key(7))
    traj = jax.jit(
        jax.vmap(lambda d, q, env=env: check_env._playout(env, d, q))
    )(jax.random.split(kd, 64), jax.random.split(kp, 64))
    flat = jax.tree_util.tree_map(
        lambda x: x.reshape((-1,) + x.shape[2:]), traj
    )
    out[name] = (env, flat, np.asarray(jax.vmap(env.terminal)(flat)))
  return out


def test_registered_for_their_games():
  """Registered for their games."""
  for name, ctor in NAMES.items():
    assert registry.get(name).game == ctor.name
    assert name in registry.names(game=ctor.name)


@pytest.mark.parametrize("name", sorted(NAMES))
def test_encoder_contract(states, name):
  """Encoder contract."""
  assert jax.config.jax_enable_x64
  enc = registry.get(name)
  env, flat, _ = states[name]
  n = jax.tree_util.tree_leaves(flat)[0].shape[0]
  for seat in (0, 1):
    infos = jax.vmap(env.info_state)(flat, jnp.full((n,), seat, jnp.int32))
    out = enc.batch(infos)
    assert out.shape == (n,) + enc.shape and out.dtype == enc.dtype
    assert np.isfinite(np.asarray(out)).all()
    np.testing.assert_array_equal(
        np.asarray(jax.jit(enc.batch)(infos)), np.asarray(out)
    )
    np.testing.assert_array_equal(np.asarray(enc.batch(infos)), np.asarray(out))
    for i in (0, n // 2, n - 1):
      one = jax.tree_util.tree_map(lambda x, i=i: x[i], infos)
      np.testing.assert_array_equal(np.asarray(enc(one)), np.asarray(out[i]))


def test_liars_dice_known_vector_is_pyspiels():
  """Liars dice known vector is pyspiels.

  Seat 0 rolled 5, 1 (sorted 1, 5); seat 1 rolled 3, 4; no bid yet. The
  expected vector is pyspiel's information_state_tensor(0) at this node
  (liars_dice(numdice=2,dice_sides=5), history [4, 0, 2, 3]): bits 0, 2, 11.
  """
  env = liars_dice.LiarsDiceEnv()
  s = _raw(env, (4, 0, 2, 3))
  want = np.zeros(33)
  want[[0, 2, 11]] = 1
  _check("liars_dice_info_state", env.info_state(s, 0), want)
  s = _raw(env, (4, 0, 2, 3, 7, 12))  # bids "2 3s", "3 3s"
  want = np.zeros(33)
  # seat 1, faces 3 and 4, two bids
  want[[1, 2 + 2, 2 + 5 + 3, 12 + 7, 12 + 12]] = 1
  _check("liars_dice_info_state", env.info_state(s, 1), want)
  pyspiel = pytest.importorskip("pyspiel")
  st = pyspiel.load_game(
      "liars_dice(numdice=2,dice_sides=5)"
  ).new_initial_state()
  for a in (4, 0, 2, 3, 7, 12):
    st.apply_action(a)
  np.testing.assert_array_equal(
      np.asarray(st.information_state_tensor(1), np.float32), want
  )


def test_phantom_ttt_ir_known_vector():
  """Phantom ttt ir known vector."""
  env = phantom_ttt_ir.PhantomTTTIREnv()
  s = env.init(jax.random.key(0))
  for a in (4, 4, 0):  # p1 failed at 4 (sees X); p0 took 0
    s = env.step(s, jnp.int32(a), jax.random.key(0))
  view1 = np.array([0, 0, 0, 0, 2, 0, 0, 0, 0])  # X = 2 in the phantom marks
  planes1 = np.concatenate([view1 == 0, view1 == 1, view1 == 2])
  _check(
      "phantom_ttt_ir_observation",
      env.info_state(s, 1),
      np.concatenate([planes1, [0, 1]]),
  )
  view0 = np.array([2, 0, 0, 0, 2, 0, 0, 0, 0])
  planes0 = np.concatenate([view0 == 0, view0 == 1, view0 == 2])
  _check(
      "phantom_ttt_ir_observation",
      env.info_state(s, 0),
      np.concatenate([planes0, [1, 0]]),
  )


def test_bargaining_known_vector():
  """Bargaining known vector."""
  env = bargaining.BargainingEnv()
  s = _raw(env, (0, bargaining.offer_id(1, 1, 0)))  # instance 0, seat 1 to act

  def th(x, top):
    """A thermometer: 1 at indices 0..x of top + 1.

    Args:
      x: the value.
      top: the largest value.

    Returns:
      f32[top + 1].
    """
    return (np.arange(top + 1) <= x).astype(np.float32)

  num = np.zeros(12)
  num[1] = 1
  pool = np.concatenate([th(1, 7), th(2, 7), th(3, 7)])
  vals1 = np.concatenate([th(4, 10), th(0, 10), th(2, 10)])
  offers = np.zeros((10, 24))
  offers[0] = np.concatenate([th(1, 7), th(1, 7), th(0, 7)])
  _check(
      "bargaining_info_state",
      env.info_state(s, 1),
      np.concatenate([num, pool, vals1, offers.reshape(-1), [0, 1]]),
  )


def test_bargaining_encoder_refuses_other_max_turns():
  """Bargaining encoder refuses other max turns."""
  env = bargaining.BargainingEnv(max_turns=4)
  with pytest.raises(ValueError, match="max_turns"):
    registry.get("bargaining_info_state")(
        env.info_state(env.init(jax.random.key(0)), 0)
    )


@pytest.mark.parametrize(
    "name,tensor,width",
    [
        ("liars_dice_info_state", "information_state_tensor", 33),
        ("phantom_ttt_ir_observation", "observation_tensor", 27),
        ("bargaining_info_state", "information_state_tensor", 309),
    ],
)
def test_encoder_equals_engine_tensor(states, name, tensor, width):
  """Encoder equals engine tensor.

  The info-state encoder's first `width` features equal the engine's tensor
  at every non-terminal state.

  Args:
    states: the states fixture.
    name: the encoder name.
    tensor: the engine's tensor method.
    width: the number of leading features compared.
  """
  enc = registry.get(name)
  env, flat, term = states[name]
  n = term.shape[0]
  for seat in (0, 1):
    seats = jnp.full((n,), seat, jnp.int32)
    ours = jax.jit(lambda s, q: enc.batch(jax.vmap(env.info_state)(s, q)))(
        flat, seats
    )
    ref = jax.jit(jax.vmap(getattr(env.game, tensor)))(flat, seats)
    assert ref.shape[1] == width
    if name == "bargaining_info_state":
      # OpenSpiel's header is an agreement bit (0 before the end), then the
      # offer count one-hot at 1 + num_offers; the encoder one-hots the count
      # at num_offers over the same max_turns + 2 slots.
      k = env.max_turns + 2
      head = ref[:, :k]
      assert not np.asarray(head[~term, 0]).any()
      ref = jnp.concatenate([head[:, 1:], head[:, :1], ref[:, k:]], axis=1)
    np.testing.assert_array_equal(
        np.asarray(ours[~term, :width]), np.asarray(ref[~term], np.float32)
    )
  assert (~term).sum() > 100


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
