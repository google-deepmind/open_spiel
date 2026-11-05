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

"""networks.card_move_transformer: shapes, size, pointer logits."""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.league import loop
from open_spiel.python.scale.networks import actor_critic
from open_spiel.python.scale.networks import card_move_transformer as cmt
from open_spiel.python.scale.networks import catalog


def _n_params(params):
  """The parameter count.

  Args:
    params: a params tree.

  Returns:
    Its number of scalars.
  """
  return sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(params))


def test_hearts_network_shapes_and_size():
  net = actor_critic.build_network(
      catalog.get("hearts_card_move_transformer_bf16"), 52
  )
  x = jax.random.normal(jax.random.key(0), (3, 56, 52), jnp.float32)
  params = net.init(jax.random.key(1), x)
  out = net.apply(params, x)
  assert out["logits"].shape == (3, 52) and out["logits"].dtype == jnp.float32
  assert out["value"].shape == (3,)
  assert np.isfinite(np.asarray(out["logits"])).all()
  # depth 8, D 384: about 14.5M params (embedder 56 -> 384, 55 tokens)
  n = _n_params(params)
  assert 14_400_000 < n < 14_700_000, n
  pos = params["params"]["torso"]["positional_encoding"]
  assert pos.shape == (55, 384)
  # a normal of std 0.1 truncated at +-2 (absolute)
  assert float(jnp.abs(pos).max()) <= 2.0
  assert 0.09 < float(jnp.std(pos)) < 0.11


def test_euchre_network_shapes_and_size():
  """24 card tokens + 7 special-action tokens: 31 moves.

  The size is the same-shaped torch move transformer's (value / decision /
  no-op + 7 special + 24 card tokens, 64 planes per card).
  """
  net = actor_critic.build_network(
      catalog.get("euchre_card_move_transformer_bf16"), 31
  )
  x = jax.random.normal(jax.random.key(0), (3, 64, 24), jnp.float32)
  params = net.init(jax.random.key(1), x)
  out = net.apply(params, x)
  assert out["logits"].shape == (3, 31) and out["value"].shape == (3,)
  assert np.isfinite(np.asarray(out["logits"])).all()
  assert _n_params(params) == 14_530_561
  assert params["params"]["torso"]["positional_encoding"].shape == (34, 384)


def test_hearts_network_exact_size():
  net = actor_critic.build_network(
      catalog.get("hearts_card_move_transformer_bf16"), 52
  )
  params = net.init(jax.random.key(1), jnp.zeros((1, 56, 52), jnp.float32))
  assert _n_params(params) == 14_535_553


def test_special_action_tokens_follow_the_cards():
  cfg = cmt.card_move_transformer(
      depth=1,
      embed_dim_per_head_over8=1,
      n_head=2,
      ff_factor=2,
      in_channels=5,
      n_card=24,
      n_special_action=7,
      compute_dtype="float32",
  )
  net = actor_critic.build_network(cfg, 31)
  x = jax.random.normal(jax.random.key(0), (2, 5, 24), jnp.float32)
  params = net.init(jax.random.key(1), x)
  assert params["params"]["torso"]["positional_encoding"].shape == (
      3 + 7 + 24,
      16,
  )
  assert net.apply(params, x)["logits"].shape == (2, 31)
  with pytest.raises(ValueError, match="num_actions"):
    actor_critic.build_network(cfg, 30).init(jax.random.key(1), x)


def test_card_logits_are_permutation_equivariant_up_to_position():
  """A card's logit reads its own token.

  Zero positional encodings make the card logits permute with the card
  columns.
  """
  cfg = cmt.card_move_transformer(
      depth=2,
      embed_dim_per_head_over8=1,
      n_head=2,
      ff_factor=2,
      in_channels=4,
      n_card=6,
      n_special_action=0,
      compute_dtype="float32",
  )
  net = actor_critic.build_network(cfg, 6)
  x = jax.random.normal(jax.random.key(0), (1, 4, 6), jnp.float32)
  params = net.init(jax.random.key(1), x)
  params = jax.tree_util.tree_map(lambda a: a, params)
  p = dict(params["params"])
  torso = dict(p["torso"])
  torso["positional_encoding"] = jnp.zeros_like(torso["positional_encoding"])
  p["torso"] = torso
  params = {"params": p}
  perm = np.array([3, 0, 5, 1, 4, 2])
  a = np.asarray(net.apply(params, x)["logits"])[0]
  b = np.asarray(net.apply(params, x[:, :, perm])["logits"])[0]
  np.testing.assert_allclose(b, a[perm], rtol=1e-5, atol=1e-5)


def _small(**kw):
  """A small card move transformer config.

  Args:
    **kw: further card_move_transformer arguments.

  Returns:
    The network config.
  """
  return cmt.card_move_transformer(
      depth=2,
      embed_dim_per_head_over8=1,
      n_head=2,
      ff_factor=2,
      in_channels=5,
      n_card=12,
      n_special_action=0,
      **kw,
  )


def test_default_config_carries_no_opt_in_keys():
  # the opt-in knobs leave the default config dict, so its hash, unchanged
  cfg = _small(compute_dtype="bfloat16")
  assert cfg == _small(
      compute_dtype="bfloat16", value_init="lecun", center_keys=False
  )
  assert cfg["value_head"] == {"kind": "scalar", "source": "value"}
  assert "center_keys" not in cfg["torso"]
  with pytest.raises(ValueError, match="value_init"):
    _small(compute_dtype="float32", value_init="orthogonal")


def test_value_head_torch_uniform_init():
  x = jax.random.normal(jax.random.key(0), (2, 5, 12), jnp.float32)
  d = 16
  for value_init, kernel_bound, bias_zero in (
      ("lecun", None, True),
      ("torch_uniform", 1 / np.sqrt(d), False),
  ):
    net = actor_critic.build_network(
        _small(compute_dtype="float32", value_init=value_init), 12
    )
    vh = net.init(jax.random.key(1), x)["params"]["value_head"]["out"]
    kernel, bias = np.asarray(vh["kernel"]), np.asarray(vh["bias"])
    assert kernel.shape == (d, 1) and bias.shape == (1,)
    assert (bias == 0).all() == bias_zero
    if kernel_bound is not None:
      assert (
          np.abs(kernel).max() <= kernel_bound
          and np.abs(bias).max() <= kernel_bound
      )


@pytest.mark.parametrize("dtype,tol", [("float32", 1e-5), ("bfloat16", 5e-2)])
def test_center_keys_is_the_same_function(dtype, tol):
  x = jax.random.normal(jax.random.key(0), (3, 5, 12), jnp.float32)
  plain = actor_critic.build_network(_small(compute_dtype=dtype), 12)
  centred = actor_critic.build_network(
      _small(compute_dtype=dtype, center_keys=True), 12
  )
  params = plain.init(jax.random.key(1), x)
  # same parameter tree, so a fork restores its parent's params
  assert jax.tree_util.tree_structure(params) == jax.tree_util.tree_structure(
      centred.init(jax.random.key(1), x)
  )
  a, b = plain.apply(params, x), centred.apply(params, x)
  np.testing.assert_allclose(
      np.asarray(a["logits"]), np.asarray(b["logits"]), atol=tol
  )
  np.testing.assert_allclose(
      np.asarray(a["value"]), np.asarray(b["value"]), atol=tol
  )
  g = jax.grad(lambda p: centred.apply(p, x)["value"].sum())(params)
  k = g["params"]["torso"]["layer_1"]["k_proj"]["kernel"]
  assert k.dtype == jnp.float32 and np.isfinite(np.asarray(k)).all()


def test_league_hash_ignores_center_keys():
  # a fork that turns key centring on restores its parent's league
  plain = {"encoder": "e", "network": _small(compute_dtype="bfloat16")}
  centred = {
      "encoder": "e",
      "network": _small(compute_dtype="bfloat16", center_keys=True),
  }
  assert (
      loop.hashed_arch_kwargs(centred)
      == loop.hashed_arch_kwargs(plain)
      == plain
  )
  assert loop.hashed_arch_kwargs({"network": {"torso": "named"}}) == {
      "network": {"torso": "named"}
  }


def test_with_options_is_the_constructor_rule():
  base = _small(compute_dtype="bfloat16")
  for vi in ("lecun", "torch_uniform"):
    for ck in (False, True):
      assert cmt.with_options(base, value_init=vi, center_keys=ck) == _small(
          compute_dtype="bfloat16", value_init=vi, center_keys=ck
      )
  assert (
      cmt.with_options(_small(compute_dtype="bfloat16", center_keys=True))
      == base
  )
  with pytest.raises(ValueError, match="not a card move transformer"):
    cmt.with_options({"torso": {"kind": "mlp"}, "value_head": {}})
  with pytest.raises(ValueError, match="value_init"):
    cmt.with_options(base, value_init="orthogonal")


if __name__ == "__main__":
  sys.exit(pytest.main([__file__, "-p", "no:cacheprovider"]))
