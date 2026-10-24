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

"""NetworkPolicy: a Policy over a network (arch_name "network").

An encoder turns the InfoState into features and an ActorCritic network maps
them to logits and a value. Everything needed to rebuild the policy is JSON
in its arch_kwargs, so an artifact restores it without code.

  pol = NetworkPolicy(game="leduc_poker", obs_format="leduc_poker_info_state",
                      num_actions=3, num_players=2,
                      encoder="leduc_poker_info_state", obs_shape=[30],
                      network=catalog.mlp([64, 64], "relu", "shared"))
  params = pol.init_params(jax.random.key(0))
  out = pol.evaluate(params, (), step, key)       # PolicyOutput

arch_kwargs (JSON, recorded in the artifact):
  game, obs_format, num_actions, num_players
  encoder:   "module:attr" of a function InfoState[B] -> features
             [B, *obs_shape],
             or a name in the encoders registry (resolved lazily)
  obs_shape: [..] per-row feature shape (init_params builds from it)
  network:   the actor_critic.build_network config
  value_layout: "seat" (value[b, seat] = v, other seats 0) or "zero_sum"
             (2 players: value[b, seat] = v, value[b, other] = -v)

obs_format must identify the encoder: for a registry encoder it equals the
encoder name and obs_shape its declared shape; a 'module:attr' or injected
encoder may not use a registry name as its obs_format (checked at
construction, so an artifact never names an encoder it does not use).

An encoder function may instead be injected (`encoder_fn=`); such a policy
cannot be rebuilt from its arch_kwargs, so its encoder name is recorded as
"injected:<qualname>" and the registry refuses to build it.

Stateless networks: carry (), observe is the identity, evaluate runs the
forward on encode(step.info), chunked above the network's chunk_size with
the encoder inside each chunk. Recurrent networks (torso kind "recurrent"):
carry {"h": f32[B, H]}; observe resets h on new_hand / new_match, then
h = GRU step on encode(step.info) (plus the last-action one-hot when the
torso config sets "use_last_action"); evaluate reads h only (observe has
already consumed this ply's InfoState).

PolicyOutput: logits f32[B, A]; value f32[B, P]; aux {"value_logits" (HL-Gauss
bin logits) if any, each aux head by key}, all f32.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Callable, Mapping, Optional

import jax
import jax.numpy as jnp

from open_spiel.python.scale.core import types
from open_spiel.python.scale.networks import actor_critic
from open_spiel.python.scale.policies import registry

ARCH = "network"
VALUE_LAYOUTS = ("seat", "zero_sum")
INJECTED = types.INJECTED_PREFIX


def _encoder_registry():
  """The encoder registry module, imported lazily (it imports game encoders).

  Returns:
    The encoders.registry module.
  """
  # Optional dependency: the encoder registry (and the game encoders it
  # loads) is imported only for a registry encoder name; a 'module:attr'
  # encoder needs none, and resolve_encoder reports a registry that is not
  # importable.
  # pylint: disable-next=g-import-not-at-top,import-outside-toplevel
  from open_spiel.python.scale.encoders import registry as enc  # noqa: PLC0415

  return enc


def resolve_encoder(name: str) -> Callable[[Any], jax.Array]:
  """The batched encoder function an encoder name denotes.

  'module:attr' -> that function (InfoState[B] -> [B, *obs_shape]); a bare
  name -> encoders.registry.get(name).batch (registry encoders map ONE
  InfoState; Policy.encode receives a batch).

  Args:
    name: 'module:attr' or an encoder registry name.

  Returns:
    The function InfoState[B] -> [B, *obs_shape].

  Raises:
    ValueError: the name records an injected encoder.
    ModuleNotFoundError: a registry name but the registry is not importable.
  """
  assert isinstance(name, str) and name, name
  if name.startswith(INJECTED):
    raise ValueError(
        f"encoder {name!r} was injected; it cannot be rebuilt from arch_kwargs"
    )
  if ":" in name:
    return registry.resolve_callable(name)
  try:
    enc_registry = _encoder_registry()
  except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        f"encoder {name!r} is a registry name but the encoder registry is not "
        "importable; "
        "use 'module:attr'"
    ) from e
  return enc_registry.get(name).batch


def value_rows(
    v: jax.Array, seat: jax.Array, num_players: int, layout: str
) -> jax.Array:
  """Spreads a value from `seat`'s view over the players.

  Args:
    v: f32[B], the value from `seat`'s view.
    seat: the seat of each row.
    num_players: P.
    layout: "seat" (other seats 0) or "zero_sum" (the other seat gets -v).

  Returns:
    f32[B, P].

  Raises:
    ValueError: an unknown layout, or zero_sum without 2 players.
  """
  oh = jax.nn.one_hot(seat, num_players, dtype=jnp.float32)
  v = v.astype(jnp.float32)[:, None]
  if layout == "seat":
    return oh * v
  if layout == "zero_sum":
    if num_players != 2:
      raise ValueError("value_layout 'zero_sum' needs num_players == 2")
    return (2.0 * oh - 1.0) * v
  raise ValueError(
      f"value_layout must be one of {VALUE_LAYOUTS}, got {layout!r}"
  )


def _check_encoder_identity(
    obs_format: str, encoder: str, obs_shape: tuple[int, ...]
) -> None:
  """Checks that obs_format identifies the encoder actually used.

  obs_format is recorded in the artifact manifest (manifest.json). A registry
  encoder: obs_format == its name and obs_shape == its declared shape. A
  'module:attr' or injected encoder (tests, custom experiments): obs_format
  must NOT be a registry name, so it can never claim a registered encoding it
  does not compute.

  Args:
    obs_format: the recorded observation format name.
    encoder: the encoder name ('module:attr', a registry name, or injected).
    obs_shape: the declared observation shape.

  Raises:
    ValueError: obs_format or obs_shape does not identify the encoder.
  """
  if encoder.startswith(INJECTED) or ":" in encoder:
    if _encoder_registry().known(obs_format):
      raise ValueError(
          f"obs_format {obs_format!r} names a registry encoder but the encoder "
          "is "
          f"{encoder!r}; use encoder={obs_format!r} (the registry one)"
      )
    return
  enc = _encoder_registry().get(encoder)
  if obs_format != encoder:
    raise ValueError(
        f"obs_format {obs_format!r} != registry encoder {encoder!r}; a "
        "registry "
        "encoder's name IS the obs_format"
    )
  if tuple(obs_shape) != tuple(enc.shape):
    raise ValueError(
        f"obs_shape {list(obs_shape)} != encoder {encoder!r}'s declared shape "
        f"{list(enc.shape)}"
    )


class NetworkPolicy(types.Policy):
  """A Policy over an encoder and an ActorCritic network (module doc).

  Attributes:
    obs_shape: the observation shape.
    network_config: the network config.
    value_layout: "seat" or "zero_sum" (value_rows).
    recurrent: the torso is recurrent.
    hidden_size: the recurrent hidden size; 0 when not recurrent.
    use_last_action: the recurrent torso also takes the last action.
    module: the ActorCritic module.
    spec: the PolicySpec.
  """

  def __init__(
      self,
      game: str,
      obs_format: str,
      num_actions: int,
      num_players: int,
      encoder: str,
      obs_shape,
      network: Mapping[str, Any],
      value_layout: str = "seat",
      encoder_fn: Optional[Callable[[Any], jax.Array]] = None,
  ):
    """Builds the policy.

    Args:
      game: the game's name.
      obs_format: the recorded observation format name.
      num_actions: the game's number of actions.
      num_players: the game's number of players.
      encoder: 'module:attr' or an encoder registry name.
      obs_shape: the observation shape.
      network: the network config (actor_critic.build_network).
      value_layout: "seat" or "zero_sum" (value_rows).
      encoder_fn: an encoder function injected as an object; overrides
        `encoder`.

    Raises:
      ValueError: an unknown value_layout, or an obs_format / obs_shape that
        does not identify the encoder.
    """
    if value_layout not in VALUE_LAYOUTS:
      raise ValueError(
          f"value_layout must be one of {VALUE_LAYOUTS}, got {value_layout!r}"
      )
    if encoder_fn is not None:
      encoder = INJECTED + getattr(encoder_fn, "__qualname__", repr(encoder_fn))
      self._encoder = encoder_fn
    else:
      self._encoder = resolve_encoder(encoder)
    self.obs_shape = tuple(int(d) for d in obs_shape)
    _check_encoder_identity(obs_format, encoder, self.obs_shape)
    self.network_config = copy.deepcopy(dict(network))
    self.value_layout = value_layout
    tcfg = self.network_config["torso"]
    self.recurrent = tcfg["kind"] == "recurrent"
    self.hidden_size = int(tcfg["hidden_size"]) if self.recurrent else 0
    self.use_last_action = (
        bool(tcfg.get("use_last_action", False)) if self.recurrent else False
    )
    # use_last_action is a policy-level input option, not a module field.
    module_cfg = copy.deepcopy(self.network_config)
    if self.recurrent:
      module_cfg["torso"].pop("use_last_action", None)
    self.module = actor_critic.build_network(module_cfg, num_actions)
    self.spec = types.PolicySpec(
        game=game,
        obs_format=obs_format,
        num_actions=num_actions,
        num_players=num_players,
        stateful=self.recurrent,
        capability=types.Capability.NATIVE_JAX,
        framework="jax",
        arch_name=ARCH,
        arch_kwargs=dict(
            game=game,
            obs_format=obs_format,
            num_actions=num_actions,
            num_players=num_players,
            encoder=encoder,
            obs_shape=list(self.obs_shape),
            network=copy.deepcopy(dict(network)),
            value_layout=value_layout,
        ),
    )

  def stack_key(self) -> Optional[tuple[str, str]]:
    """A key equal for policies that can share one params stack.

    They share it when they have the same network, encoder and layout.

    Returns:
      The key, or None when the encoder was injected as an object, which cannot
      be compared by value.
    """
    if self.spec.arch_kwargs["encoder"].startswith(INJECTED):
      return None
    return (ARCH, json.dumps(dict(self.spec.arch_kwargs), sort_keys=True))

  def value_support(self):
    """The HL-Gauss support the value head decodes with.

    Learners train on this support (asserted by
    learners.losses.check_value_support), never on a second copy.

    Returns:
      The hl_gauss.Support, or None for a scalar head.
    """
    vh = self.module.value_head
    return vh.support() if vh.kind == "hl_gauss" else None

  # --- params -------------------------------------------------------------
  def _extra(self, step_or_batch, batch=None):
    """The last-action one-hot input, if the policy uses it.

    Args:
      step_or_batch: A SeatStep, or None for zeros.
      batch: The rows when it is None.

    Returns:
      f32[B, A], or None when unused.
    """
    if not self.use_last_action:
      return None
    if step_or_batch is None:
      return jnp.zeros((batch, self.spec.num_actions), jnp.float32)
    st = step_or_batch
    seen = (jnp.asarray(st.last_actor) >= 0) & (
        jnp.asarray(st.last_action) >= 0
    )
    oh = jax.nn.one_hot(
        jnp.maximum(jnp.asarray(st.last_action), 0),
        self.spec.num_actions,
        dtype=jnp.float32,
    )
    return jnp.where(seen[:, None], oh, jnp.float32(0.0))

  def init_params(self, key):
    """The network's initial params, from a zero input of obs_shape.

    Args:
      key: a PRNG key.

    Returns:
      The flax params.
    """
    x = jnp.zeros((1,) + self.obs_shape, jnp.float32)
    if self.recurrent:
      h = jnp.zeros((1, self.hidden_size), jnp.float32)
      return self.module.init(
          key,
          x,
          h,
          self._extra(None, 1),
          method=actor_critic.ActorCritic.init_recurrent,
      )
    return self.module.init(key, x)

  # --- Policy -------------------------------------------------------------
  def encode(self, info):
    """The encoder's features of a batched InfoState.

    Args:
      info: the InfoState, leaves [B, ...].

    Returns:
      [B, *obs_shape] features.

    Raises:
      ValueError: the rows do not have shape obs_shape.
    """
    x = self._encoder(info)
    if tuple(x.shape[1:]) != self.obs_shape:
      raise ValueError(
          f"encoder gave rows of shape {tuple(x.shape[1:])}, obs_shape is "
          f"{self.obs_shape}"
      )
    return x

  def init_carry(self, params, batch, seat, key):
    """The initial carry: zeros for a recurrent network, () otherwise.

    Args:
      params: unused.
      batch: the number of rows B.
      seat: unused.
      key: unused.

    Returns:
      {"h": f32[B, H]} or ().
    """
    del params, seat, key
    if not self.recurrent:
      return ()
    return {"h": jnp.zeros((batch, self.hidden_size), jnp.float32)}

  def observe(self, params, carry, step):
    """A recurrent network's GRU step on this ply; the identity otherwise.

    Args:
      params: the params.
      carry: the carry.
      step: the SeatStep, leaves [B, ...].

    Returns:
      The carry after observing `step`.
    """
    if not self.recurrent:
      return carry
    reset = jnp.asarray(step.new_hand) | jnp.asarray(step.new_match)
    h = jnp.where(reset[:, None], jnp.float32(0.0), carry["h"])
    h = self.module.apply(
        params,
        self.encode(step.info),
        h,
        self._extra(step),
        method=actor_critic.ActorCritic.observe,
    )
    return {"h": h.astype(jnp.float32)}

  def forward(self, params, carry, step) -> Mapping[str, Any]:
    """The composer's raw output dict for this ply (f32).

    Args:
      params: the params.
      carry: the carry.
      step: the SeatStep, leaves [B, ...].

    Returns:
      {"logits", "value", "value_logits", "aux"}.
    """
    if self.recurrent:
      return self.module.apply(
          params, carry["h"], method=actor_critic.ActorCritic.evaluate_carry
      )
    return self._stateless_forward(params, step.info)

  def _stateless_forward(self, params, info) -> Mapping[str, Any]:
    """module(encode(info)) over N rows.

    Above the network's chunk_size the encoder runs inside the chunk loop too,
    so a memory-hungry encoding is never materialised for every row at once: the
    rows are split by actor_critic.chunk_plan, pad rows repeat row 0 and are
    dropped, and each chunk is encoded and run under a scan, rematerialised
    unless the network says remat=false. Inside, a chunk is <= chunk_size, so
    the module runs it whole.

    Args:
      params: the network's params.
      info: the InfoStates, leaves [N, ...].

    Returns:
      The network's outputs over the N rows.
    """
    chunk = self.module.chunk_size
    n = int(jax.tree.leaves(info)[0].shape[0])
    if chunk is None or n <= int(chunk):
      return self.module.apply(params, self.encode(info))
    num, c = actor_critic.chunk_plan(n, int(chunk))
    pad = num * c - n

    def split(x):
      """Pads a leaf to num * c rows and splits it into chunks.

      Args:
        x: a leaf with n rows.

      Returns:
        [num, c, ...].
      """
      x = jnp.asarray(x)
      assert x.shape[0] == n, (x.shape, n)
      if pad:
        x = jnp.concatenate([x, jnp.broadcast_to(x[:1], (pad,) + x.shape[1:])])
      return x.reshape((num, c) + x.shape[1:])

    def body(_, info_c):
      """The outputs of one chunk (scan step).

      Args:
        _: the unused scan carry.
        info_c: one chunk of InfoState rows.

      Returns:
        (None, the chunk's outputs).
      """
      return None, self.module.apply(params, self.encode(info_c))

    # The network's "remat" (default true) applies here too: off stores every
    # chunk's activations instead of recomputing them (same math).
    step = (
        jax.checkpoint(body, prevent_cse=False) if self.module.remat else body
    )
    _, out = jax.lax.scan(step, None, jax.tree.map(split, info))
    return jax.tree.map(lambda y: y.reshape((num * c,) + y.shape[2:])[:n], out)

  def evaluate(self, params, carry, step, key):
    """The PolicyOutput of this ply.

    Args:
      params: the params.
      carry: the carry.
      step: the SeatStep, leaves [B, ...].
      key: unused.

    Returns:
      The PolicyOutput: f32 logits, values [B, P] and aux.
    """
    del key
    out = self.forward(params, carry, step)
    aux = dict(out["aux"])
    if out["value_logits"] is not None:
      aux["value_logits"] = out["value_logits"]
    value = value_rows(
        out["value"],
        jnp.asarray(step.seat),
        self.spec.num_players,
        self.value_layout,
    )
    return types.PolicyOutput(
        logits=out["logits"].astype(jnp.float32), value=value, aux=aux
    )


registry.register(ARCH, NetworkPolicy)
