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

"""Export the model's graph."""

import json
import os

from absl import app
from absl import flags
from flax import linen
from flax import nnx

from open_spiel.python.algorithms.alpha_zero import utils
import pyspiel


FLAGS = flags.FLAGS
flags.DEFINE_string("game", None, "Name of the game")

flags.DEFINE_string("save_path", None, "Directory to save graph")
flags.DEFINE_string(
    "path", None, "A path (directory) for a pretrained model chekpoint"
)
flags.DEFINE_integer(
    "checkpoint_step", None, "A step for a pretrained model chekpoint"
)
flags.DEFINE_string(
    "config_path", None, "Filename for the a training config.json"
)


flags.DEFINE_enum(
    "nn_model",
    "resnet",
    utils.api_selector(utils.AVIALABLE_APIS[0]).Model.valid_model_types,
    "What type of model should be used?.",
)
flags.DEFINE_integer("nn_width", 2**7, "How wide should the network be.")
flags.DEFINE_integer("nn_depth", 10, "How deep should the network be.")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate used for training")
flags.DEFINE_float("weight_decay", 0.0001, "L2 regularization strength.")
flags.DEFINE_enum(
    "nn_api_version",
    "linen",
    ["linen", "nnx"],
    "What type of flax api should be used for training?.                 "
    " Currently, linen and nnx are supported",
)

flags.DEFINE_bool("verbose", False, "Print information about the model.")
flags.DEFINE_bool(
    "visualise", False, "Use it to draw a treescope layout of the model"
)


flags.mark_flag_as_required("game")
flags.mark_flag_as_required("path")


def build_model() -> nnx.Module | linen.Module:
  """Builds a model."""
  game = pyspiel.load_game(FLAGS.game)

  config = {}
  if FLAGS.config_path is not None and os.path.exists(FLAGS.config_path):
    with open(FLAGS.config_path, "r") as f:
      config = json.load(f)

  model = utils.api_selector(
      config.get("nn_api_version", FLAGS.nn_api_version)
  ).Model.build_model(
      config.get("nn_model", FLAGS.nn_model),
      game.observation_tensor_shape(),
      game.num_distinct_actions(),
      config.get("nn_width", FLAGS.nn_width),
      config.get("nn_depth", FLAGS.nn_depth),
      config.get("weight_decay", FLAGS.weight_decay),
      config.get("learning_rate", FLAGS.learning_rate),
      config.get("path", FLAGS.path),
  )

  if FLAGS.checkpoint_step is not None:
    model.load_checkpoint(FLAGS.checkpoint_step)

  return model


def main(unused_argv):
  model = build_model()

  if FLAGS.visualise:
    nnx_model = (
        utils.linen_to_nnx(model._model)  # pylint: disable=protected-access
        if not isinstance(model, nnx.Module)
        else model
    )
    nnx.display(nnx_model)
  else:
    # In essence, just prepared the model graph for training
    model.save_checkpoint(0)


if __name__ == "__main__":
  app.run(main)
