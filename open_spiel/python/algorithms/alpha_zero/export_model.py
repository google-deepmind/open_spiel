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

"""Export the model's Tensorflow graph as a protobuf."""

from absl import app
from absl import flags
from typing import Optional

from open_spiel.python.algorithms.alpha_zero.utils import api_selector, AVIALABLE_APIS
import pyspiel

from flax.nnx import bridge, Module, display, Rngs
import flax.linen as nn
import chex


FLAGS = flags.FLAGS
flags.DEFINE_string("game", None, "Name of the game")
flags.DEFINE_string("save_path", None, "Directory to save graph")
flags.DEFINE_string("checkpoint_path", None, "Filename for the graph")
flags.DEFINE_enum("nn_model", "resnet", api_selector(AVIALABLE_APIS[0]).Model.valid_model_types,
                  "What type of model should be used?.")
flags.DEFINE_integer("nn_width", 2 ** 7, "How wide should the network be.")
flags.DEFINE_integer("nn_depth", 10, "How deep should the network be.")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate used for training")
flags.DEFINE_float("weight_decay", 0.0001, "L2 regularization strength.")
flags.DEFINE_bool("verbose", False, "Print information about the model.")

flags.DEFINE_bool("linen_to_nnx", False)
flags.DEFINE_bool("nnx_to_linen", False)
flags.DEFINE_bool("visualise", False)


flags.mark_flag_as_required("game")
flags.mark_flag_as_required("save_path")



def build_model(api_version: str, checkpoint_path: Optional[str]=None) -> Module | nn.Module:
  game = pyspiel.load_game(FLAGS.game)
  model = api_selector(api_version).Model.build_model(
    FLAGS.nn_model, 
    game.observation_tensor_shape(),
    game.num_distinct_actions(), 
    FLAGS.nn_width, 
    FLAGS.nn_depth,
    FLAGS.weight_decay, 
    FLAGS.learning_rate, 
    FLAGS.path
  )
  
  if checkpoint_path is not None:
    #TODO: try except
    model.load_checkpoint(checkpoint_path)
  
  return model

def linen_to_nnx(model, sample_input: chex.Array, save_path: str, seed: int) -> Module:
  model = bridge.ToNNX(model, rngs=Rngs(seed))    
  bridge.lazy_init(model, sample_input)
  model.save_checkpoint(save_path)
  return model              

def nnx_to_linen(model_class: Module, save_path: str, *args, **kwargs) -> nn.Module:
  model = bridge.to_linen(model_class, *args, **kwargs)    
  model.save_checkpoint(save_path)
  return model

def main(unused_argv):
  model = build_model()

  if FLAGS.linen_to_nnx:
    linen_to_nnx(model)
  elif FLAGS.nnx_to_linen:
    nnx_to_linen(model)
  elif FLAGS.visualise:
    nnx_model = linen_to_nnx() if not isinstance(model, Module) else model
    display(nnx_model)


if __name__ == "__main__":
  app.run(main)
