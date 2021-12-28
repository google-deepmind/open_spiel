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

# Lint as: python3
"""Export the model's Tensorflow graph as a protobuf."""

from absl import app
from absl import flags

from open_spiel.python.algorithms.alpha_zero import model as model_lib
import pyspiel

FLAGS = flags.FLAGS
flags.DEFINE_string("game", None, "Name of the game")
flags.DEFINE_string("path", None, "Directory to save graph")
flags.DEFINE_string("graph_def", None, "Filename for the graph")
flags.DEFINE_enum("nn_model", "resnet", model_lib.Model.valid_model_types,
                  "What type of model should be used?.")
flags.DEFINE_integer("nn_width", 2 ** 7, "How wide should the network be.")
flags.DEFINE_integer("nn_depth", 10, "How deep should the network be.")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate used for training")
flags.DEFINE_float("weight_decay", 0.0001, "L2 regularization strength.")
flags.DEFINE_bool("verbose", False, "Print information about the model.")
flags.mark_flag_as_required("game")
flags.mark_flag_as_required("path")
flags.mark_flag_as_required("graph_def")


def main(_):
  game = pyspiel.load_game(FLAGS.game)
  model = model_lib.Model.build_model(
      FLAGS.nn_model, game.observation_tensor_shape(),
      game.num_distinct_actions(), FLAGS.nn_width, FLAGS.nn_depth,
      FLAGS.weight_decay, FLAGS.learning_rate, FLAGS.path)
  model.write_graph(FLAGS.graph_def)

  if FLAGS.verbose:
    print("Game:", FLAGS.game)
    print("Model type: %s(%s, %s)" % (FLAGS.nn_model, FLAGS.nn_width,
                                      FLAGS.nn_depth))
    print("Model size:", model.num_trainable_variables, "variables")
    print("Variables:")
    model.print_trainable_variables()


if __name__ == "__main__":
  app.run(main)
