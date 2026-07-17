# Copyright 2019 DeepMind Technologies Limited
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

"""Export a trained AlphaZero model to ONNX or compiled JAX."""

import json
import os
from typing import Callable, Any

import jax
import jax.numpy as jnp
import pyspiel
from absl import app, flags

from open_spiel.python.algorithms.alpha_zero import utils

FLAGS = flags.FLAGS

flags.DEFINE_string("game", None, "Name of the game")
flags.DEFINE_string("path", None, "Checkpoint directory")
flags.DEFINE_integer("checkpoint_step", None, "Checkpoint step to load")
flags.DEFINE_enum("format", "onnx", ["onnx", "jit"], "Export format")
flags.DEFINE_string("output", None, "Output file path")
flags.DEFINE_integer("batch_size", 1, "Batch size for export")

flags.mark_flag_as_required("game")
flags.mark_flag_as_required("path")
flags.mark_flag_as_required("output")


def _load_config():
  """Load config from file or infer from flags."""
  config = {}
  config_path = os.path.join(FLAGS.path, "config.json")
  if FLAGS.path and os.path.exists(config_path):
    with open(config_path) as f:
      config = json.load(f)
  return config


def _build_model(config, game):
  """Build and optionally load model."""
  print("CONFIG", config)
  model = utils.api_selector(
    config.get("nn_api_version", "nnx")
  ).Model.build_model(
    config.get("nn_model", "resnet"),
    game.observation_tensor_shape(),
    game.num_distinct_actions(),
    config.get("nn_width", 128),
    config.get("nn_depth", 10),
    config.get("weight_decay", 0.0001),
    config.get("learning_rate", 0.001),
    os.path.abspath(FLAGS.path),
  )
  if FLAGS.checkpoint_step is not None:
    model.load_checkpoint(str(FLAGS.checkpoint_step))
  return model


def _make_inference_fn(model: Any) -> Callable:
  """Create a pure inference function suitable for export."""

  def inference_fn(obs, mask):
    # obs: [B, ...obs_shape], mask: [B, num_actions]
    value, policy = model.inference(obs, mask)
    return value, policy

  return inference_fn


def _export_onnx(model: Any, game: pyspiel.Game, output_path: str) -> None:
  """Export to ONNX via jax2onnx.
      Creates a batched inference function with explicit batch dimension.
  """
  try:
    import jax2onnx
    import logging

    # Mute the jax2onnx plugin registration logger
    logging.getLogger("jax2onnx").setLevel(logging.ERROR)
  except ImportError as e:
    raise ImportError("Install jax2onnx: pip install 'jax2onnx==0.14.1'") from e

  obs_shape = game.observation_tensor_shape()
  num_actions = game.num_distinct_actions()

  inference_fn = _make_inference_fn(model)

  # Wrap with vmap for batched export
  @jax.jit
  def inference_batched(obs, mask):
    # obs: [batch, *obs_shape], mask: [batch, num_actions]
    return jax.vmap(inference_fn)(obs, mask)

  # Build dummy inputs with explicit batch dimension
  dummy_obs = jnp.zeros((FLAGS.batch_size, *obs_shape), dtype=jnp.float32)
  dummy_mask = jnp.ones((FLAGS.batch_size, num_actions), dtype=jnp.bool)

  # Verify the function works before export
  print(
    f"Testing inference with batch_size={FLAGS.batch_size}, "
    f"obs_shape={dummy_obs.shape}, mask_shape={dummy_mask.shape}"
  )
  test_value, test_policy = inference_batched(dummy_obs, dummy_mask)
  print(
    f"  Output: value_shape={test_value.shape}, policy_shape={test_policy.shape}"
  )

  # Export to ONNX
  jax2onnx.to_onnx(
    inference_batched,
    inputs=[dummy_obs, dummy_mask],
    return_mode="file",
    output_path=output_path,
  )
  print(f"ONNX exported: {output_path}")
  print(f"  Inputs: 'obs' {dummy_obs.shape}, 'mask' {dummy_mask.shape}")
  print(f"  Outputs: 'value' {test_value.shape}, 'policy' {test_policy.shape}")


def _export_jit(model: Any, game: pyspiel.Game, output_path: str) -> None:
  """Export compiled JAX artifact (StableHLO/MLIR)."""
  obs_shape = game.observation_tensor_shape()
  num_actions = game.num_distinct_actions()

  inference_fn = _make_inference_fn(model)

  @jax.jit
  def inference_jit(obs, mask):
    return inference_fn(obs, mask)

  # Lower with explicit batch dimension
  # Use polymorphic shapes if you want variable batch size at runtime
  batch_dim = FLAGS.batch_size if FLAGS.batch_size > 0 else None

  if batch_dim:
    dummy_obs = jax.ShapeDtypeStruct((batch_dim, *obs_shape), jnp.float32)
    dummy_mask = jax.ShapeDtypeStruct((batch_dim, num_actions), jnp.bool)
  else:
    dummy_obs = jax.ShapeDtypeStruct(obs_shape, jnp.float32)
    dummy_mask = jax.ShapeDtypeStruct((num_actions,), jnp.bool)

  print(
    f"Lowering JIT with batch={batch_dim}, obs={dummy_obs.shape}, mask={dummy_mask.shape}"
  )
  lowered = inference_jit.lower(dummy_obs, dummy_mask)

  # Serialize compiled artifact
  # JAX API evolves; try newer API first, fall back to MLIR text
  serialized = None

  try:
    # JAX 0.4.30+ export API
    from jax.experimental import export as jax_export

    compiled = lowered.compile()
    serialized = jax_export.serialize(compiled)
    print("Used jax.experimental.export.serialize")
  except (ImportError, AttributeError):
    pass

  if serialized is None:
    try:
      # Try AOT compilation path
      serialized = lowered.compile().as_text()
      print("Used compiled AOT text serialization")
    except (AttributeError, NotImplementedError):
      pass

  if serialized is None:
    # Fallback: save MLIR IR (can be loaded with jax.mlir)
    serialized = lowered.as_text()
    print("Fallback: saved MLIR text (load with jax.mlir)")

  # Write output
  mode = "wb" if isinstance(serialized, bytes) else "w"
  with open(output_path, mode) as f:
    f.write(serialized)

  print(f"JIT compiled artifact: {output_path}")
  print(f"  Size: {os.path.getsize(output_path) / 1024:.1f} KB")


def main(_):
  config = _load_config()
  game = pyspiel.load_game(FLAGS.game)
  model = _build_model(config, game)

  os.makedirs(
    os.path.dirname(os.path.abspath(FLAGS.output)) or ".", exist_ok=True
  )

  if FLAGS.format == "onnx":
    _export_onnx(model, game, FLAGS.output)
  elif FLAGS.format == "jit":
    _export_jit(model, game, FLAGS.output)
  else:
    raise ValueError(f"Unknown format: {FLAGS.format}")


if __name__ == "__main__":
  app.run(main)
