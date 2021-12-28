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

"""An example of building and exporting a Tensorflow graph.

Adapted from the Travis Ebesu's blog post:
https://tebesu.github.io/posts/Training-a-TensorFlow-graph-in-C++-API
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf
import pyspiel

FLAGS = flags.FLAGS
flags.DEFINE_string("game", "breakthrough", "Name of the game")
flags.DEFINE_string("dir", "/tmp", "Directory to save graph")
flags.DEFINE_string("filename", "graph.pb", "Filename for the graph")


def main(_):
  game = pyspiel.load_game(FLAGS.game)

  # Information state length
  info_state_shape = game.observation_tensor_shape()
  flat_info_state_length = np.prod(info_state_shape)

  # Output
  num_actions = game.num_distinct_actions()

  with tf.Session() as sess:
    net_input = tf.placeholder(
        tf.float32, [None, flat_info_state_length], name="input")

    # pylint: disable=unused-variable
    output = tf.placeholder(tf.float32, [None, num_actions], name="output")
    legals_mask = tf.placeholder(
        tf.float32, [None, num_actions], name="legals_mask")

    policy_net = tf.layers.dense(net_input, 128, activation=tf.nn.relu)
    policy_net = tf.layers.dense(policy_net, 128, activation=tf.nn.relu)
    policy_net = tf.layers.dense(policy_net, num_actions)

    # Note: subtracting the max here is to help with numerical stability.
    # However, there can still be numerical problems. If you are doing a softmax
    # here, it can return NaN when the max for the policy net is high on one of
    # the illegal actions, because policy_net - max will be small for legal
    # actions, giving all exp(small) == 0 in the denominator, returning NaN at
    # the end. One fix is to set the logits to -inf and define a custom cross
    # entropy op that ignores over the illegal actions.
    policy_net = policy_net - tf.reduce_max(policy_net, axis=-1, keepdims=True)

    masked_exp_logit = tf.multiply(tf.exp(policy_net), legals_mask)
    renormalizing_factor = tf.reduce_sum(
        masked_exp_logit, axis=-1, keepdims=True)
    # pylint: disable=unused-variable
    policy_softmax = tf.where(
        tf.equal(legals_mask, 0.),
        tf.zeros_like(masked_exp_logit),
        tf.divide(masked_exp_logit, renormalizing_factor),
        name="policy_softmax")

    policy_targets = tf.placeholder(shape=[None, num_actions], dtype=tf.float32)

    policy_cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=policy_net, labels=policy_targets),
        axis=0)

    # We make one sample.
    sampled_actions = tf.random.categorical(
        tf.log(policy_softmax), 1, name="sampled_actions")

    # pylint: disable=unused-variable
    optimizer = tf.train.AdamOptimizer(0.0001).minimize(
        policy_cost, name="train")

    # pylint: disable=unused-variable
    init = tf.variables_initializer(tf.global_variables(),
                                    name="init_all_vars_op")

    print("Writing file: {}/{}".format(FLAGS.dir, FLAGS.filename))
    tf.train.write_graph(
        sess.graph_def, FLAGS.dir, FLAGS.filename, as_text=False)


if __name__ == "__main__":
  app.run(main)
