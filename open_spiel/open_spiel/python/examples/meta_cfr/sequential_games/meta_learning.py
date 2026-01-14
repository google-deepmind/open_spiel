# Copyright 2022 DeepMind Technologies Limited
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

"""Meta learning algorithm."""

import os
from typing import Dict, List, Any

from absl import flags
from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from open_spiel.python.examples.meta_cfr.sequential_games import cfr
from open_spiel.python.examples.meta_cfr.sequential_games import dataset_generator
from open_spiel.python.examples.meta_cfr.sequential_games import game_tree_utils
from open_spiel.python.examples.meta_cfr.sequential_games import models
from open_spiel.python.examples.meta_cfr.sequential_games import openspiel_api
from open_spiel.python.examples.meta_cfr.sequential_games import typing
from open_spiel.python.examples.meta_cfr.sequential_games import utils


FLAGS = flags.FLAGS

flags.DEFINE_integer("batch_size", 250, "Batch size.")
flags.DEFINE_integer("num_batches", 1, "Number of batches.")
flags.DEFINE_integer("meta_learner_training_epochs", 1,
                     "Number of meta_learner_training_epochs")
flags.DEFINE_integer("num_tasks", 1, "Number tasks to train meta learner.")
flags.DEFINE_integer("random_seed", 2, "Random seed.")
flags.DEFINE_integer("checkpoint_interval", 50,
                     "Checkpoint every checkpoint_interval.")
flags.DEFINE_string("game", "leduc_poker", "Name of the game")
flags.DEFINE_integer("players", 2, "Number of players")
flags.DEFINE_bool("perturbation", True, "Random perturbation of the game.")
flags.DEFINE_bool(
    "use_infostate_representation", True,
    "Use infostate representation as extra input to meta network.")
flags.DEFINE_float("init_lr", 0.2, "Initial learning rate")
flags.DEFINE_string("lstm_sizes", "64", "Size of lstm layers.")
flags.DEFINE_string("mlp_sizes", "20, 20", "Size of mlp layers.")
flags.DEFINE_string("model_type", "MLP", "Model type.")


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.5"


def append_counterfactual_values(
    infostates: List[typing.InfostateNode],
    counterfactual_values: Dict[str, List[List[float]]]):
  for infostate in infostates:
    counterfactual_values[infostate.infostate_string].append([
        infostate.counterfactual_action_values[a]
        for a in infostate.get_actions()
    ])


def compute_next_policy_invariants(
    infostates: typing.InfostateMapping, all_actions: List[int],
    infostate_map: typing.InfostateMapping
) -> tuple[Dict[str, jnp.ndarray], Dict[str, List[int]]]:
  """Computes information needed to calculate next policy.

  This function computes one hot encodings of infostates and returns mappings
  from infostate strings to one hot representations of infostates as well as
  illegal actions.

  Args:
    infostates: List of infostate mappings.
    all_actions: List of actions.
    infostate_map: Mapping from infostate string to infostate.

  Returns:
    Returns mappings of infostate strings to one hot representation for
    infostates and illegal actions
  """
  one_hot_representations = {}
  illegal_actions = {}

  for (infostate_str, infostate) in infostates.items():
    if infostate.is_terminal():
      continue

    legal_actions = infostate.get_actions()

    if len(legal_actions) == 1:
      infostate.policy[infostate.get_actions()[0]] = 1
      continue
    infostate_str_one_hot = jax.nn.one_hot(infostate_map[infostate_str],
                                           len(infostates))
    one_hot_representations[infostate_str] = infostate_str_one_hot
    illegal_actions[infostate_str] = [
        i for i, a in enumerate(all_actions) if a not in legal_actions
    ]
  return one_hot_representations, illegal_actions


def compute_next_policy(infostates: typing.InfostateMapping,
                        net_apply: typing.ApplyFn, net_params: typing.Params,
                        epoch: int, all_actions: List[int],
                        one_hot_representations: Dict[str, jnp.ndarray],
                        illegal_actions: Dict[str,
                                              List[int]], key: hk.PRNGSequence):
  """Computes next step policy from output of the model.

  Args:
    infostates: List of infostate mappings.
    net_apply: Apply function.
    net_params: Model params.
    epoch: epoch.
    all_actions: List of actions.
    one_hot_representations: Dictionary from infostate string to infostate.
    illegal_actions: Dictionary from infostate string to the list of illegal
      actions.
    key: Haiku Pseudo random number generator.
  """

  infostate_lst = []
  input_lst = []
  illegal_action_lst = []

  batched_net_output = []
  for (infostate_str, infostate) in infostates.items():
    if infostate.is_terminal():
      continue

    legal_actions = infostate.get_actions()
    if len(legal_actions) == 1:
      infostate.policy[infostate.get_actions()[0]] = 1
      continue
    regret_vec = np.array([
        infostate.regret[a] /
        (epoch + 1) if a in infostate.get_actions() else 0
        for a in all_actions
    ])
    if FLAGS.use_infostate_representation:
      one_hot_representation = one_hot_representations[infostate_str]
      net_input = jnp.concatenate([regret_vec, one_hot_representation])
    else:
      net_input = regret_vec
    input_lst.append(net_input)
    infostate_lst.append(infostate)
    illegal_action_lst.append(illegal_actions[infostate_str])
  batched_inputs, output_mappings, relevant_illegal_actions = (
      utils.get_batched_input(
          input_lst, infostate_lst, illegal_action_lst, FLAGS.batch_size
      )
  )
  idx = 0

  for _ in range(int(len(batched_inputs) / FLAGS.batch_size)):
    batched_input, output_mapping, relevant_illegal_action = batched_inputs[
        idx:idx + FLAGS.batch_size], output_mappings[
            idx:idx +
            FLAGS.batch_size], relevant_illegal_actions[idx:idx +
                                                        FLAGS.batch_size]
    idx += FLAGS.batch_size

    batched_input_jnp = jnp.array(
        np.expand_dims(np.array(batched_input), axis=1))
    batched_net_output = utils.get_network_output_batched(  # pytype: disable=wrong-arg-types  # jnp-type
        net_apply, net_params,
        batched_input_jnp,
        relevant_illegal_action, key)
    for i, infostate in enumerate(output_mapping):
      net_output = jnp.squeeze(batched_net_output[i])
      for ai, action in enumerate(infostate.get_actions()):
        infostate.policy[action] = float(net_output[ai])


def cfr_br_meta_data(
    history_tree_node: typing.HistoryNode,
    infostate_nodes: List[typing.InfostateNode],
    all_infostates_map: List[typing.InfostateMapping], epochs: int,
    net_apply: typing.ApplyFn, net_params: typing.Params,
    all_actions: List[int], infostate_map: typing.InfostateMapping,
    key: hk.PRNGSequence
) -> tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray], List[float]]:
  """Collects counterfactual values for both players and best response for player_2.

  Args:
    history_tree_node: Game tree HistoryTreeNode which is the root of the game
      tree.
    infostate_nodes: Infostates.
    all_infostates_map: List of mappings from infostate strings to infostates.
    epochs: Number of epochs.
    net_apply: Apply function.
    net_params: Network parameters.
    all_actions: List of all actions.
    infostate_map: A mapping from infostate strings to infostates.
    key: Haiku pseudo random number generator.

  Returns:
    Returns counterfactual values for player_1, counterfactual values for
    player_2 and best response values for player_2.
  """
  counterfactual_values_player1 = {
      infostate.infostate_string: []
      for infostate in list(all_infostates_map[1].values())
  }
  counterfactual_values_player2 = {
      infostate.infostate_string: []
      for infostate in list(all_infostates_map[2].values())
  }

  non_terminal_infostates_map_player1 = utils.filter_terminal_infostates(
      all_infostates_map[1]
  )
  one_hot_representations_player1, illegal_actions_player1 = (
      compute_next_policy_invariants(
          non_terminal_infostates_map_player1, all_actions, infostate_map
      )
  )
  player_2_last_best_response_values = []
  for epoch in range(epochs):
    compute_next_policy(non_terminal_infostates_map_player1, net_apply,
                        net_params, epoch, all_actions,
                        one_hot_representations_player1,
                        illegal_actions_player1, key)

    cfr.compute_reach_probabilities(history_tree_node, all_infostates_map)
    cfr.cumulate_average_policy(list(all_infostates_map[1].values()))
    cfr.compute_best_response_policy(infostate_nodes[2])
    cfr.compute_reach_probabilities(history_tree_node, all_infostates_map)
    cfr.compute_counterfactual_values(infostate_nodes[1])
    cfr.update_regrets(list(all_infostates_map[1].values()))
    append_counterfactual_values(
        list(all_infostates_map[1].values()), counterfactual_values_player1)
    cfr.normalize_average_policy(all_infostates_map[1].values())
    cfr.compute_reach_probabilities(history_tree_node, all_infostates_map)
    player_2_last_best_response_values.append(
        float(cfr.compute_best_response_values(infostate_nodes[2]))
    )

    logging.info(
        "Epoch %d: player_2 best response value is %f",
        epoch,
        player_2_last_best_response_values[-1],
    )

  return (  # pytype: disable=bad-return-type  # jax-ndarray
      counterfactual_values_player1,
      counterfactual_values_player2,
      player_2_last_best_response_values,
  )


class MetaCFRRegretAgent:
  """Meta regret minimizer agent.

  Attributes:
    training_epochs: Number of training epochs.
    meta_learner_training_epochs: Number of epochs for meta learner.
    game_name: Name of the game.
    game_config: Game configuration.
    perturbation: Binary variable to specify perturbation.
    seed: Random seed.
    model_type: Type of NN model for meta learner.
    best_response: Binary variable to specify if using best response.
    optimizer: Optimizer model.
  """

  def __init__(self,
               training_epochs,
               meta_learner_training_epochs,
               game_name,
               game_config,
               perturbation,
               seed,
               model_type="MLP",
               best_response=True):
    self._training_epochs = training_epochs
    self._meta_learner_training_epochs = meta_learner_training_epochs
    self._game_name = game_name
    self._model_type = model_type
    self._perturbation = perturbation
    self._game_config = game_config
    self._best_response = best_response
    self._seed = seed
    self._rng = hk.PRNGSequence(100)
    self._world_state = openspiel_api.WorldState(self._game_name,
                                                 self._game_config,
                                                 self._perturbation,
                                                 self._seed)
    self._all_actions = self._world_state.get_distinct_actions()
    self._num_infostates, self._infostate_map = self.get_num_infostates()
    self._step = 0

  def get_num_infostates(self):
    """Returns number of infostates and infostate mapping.

    Returns:
      Returns sum of number of infostates for both players and a mapping from
      infostate string to infostates.
    """
    all_infostates_map = [{}, {}, {}]
    _, _ = game_tree_utils.build_tree_dfs(
        self._world_state, all_infostates_map)
    non_terminal_infostates_map_player1 = utils.filter_terminal_infostates(
        all_infostates_map[1])
    non_terminal_infostates_map_player2 = utils.filter_terminal_infostates(
        all_infostates_map[2])
    if self._best_response:
      infostate_map = {
          infostate_str: infostate_node
          for (infostate_node, infostate_str
              ) in enumerate(list(non_terminal_infostates_map_player1.keys()))
      }
      return len(non_terminal_infostates_map_player1), infostate_map
    nont_terminal_infostates_map_both_players = list(
        non_terminal_infostates_map_player1.keys()) + list(
            non_terminal_infostates_map_player2.keys())
    infostate_map = {
        infostate_str: infostate_node
        for (infostate_node, infostate_str
            ) in enumerate(nont_terminal_infostates_map_both_players)
    }
    return len(non_terminal_infostates_map_player1) + len(
        non_terminal_infostates_map_player2), infostate_map

  def train(self):
    self.training_optimizer()

  def next_policy(self, world_state: openspiel_api.WorldState):
    """Computes best reponses for the next step of cfr.

    Args:
      world_state: Current state of the world.

    Returns:
      Returns best response values for player_2.

    """
    all_infostates_map = [{}, {}, {}]
    first_history_node, infostate_nodes = game_tree_utils.build_tree_dfs(
        world_state, all_infostates_map)

    _, _, player_2_best_response_values = cfr_br_meta_data(  # pytype: disable=wrong-arg-types
        history_tree_node=first_history_node,
        infostate_nodes=infostate_nodes,
        all_infostates_map=all_infostates_map,
        epochs=self._meta_learner_training_epochs,
        net_apply=self.optimizer.net_apply,  # pytype: disable=attribute-error
        net_params=self.optimizer.net_params,  # pytype: disable=attribute-error
        all_actions=self._all_actions,
        infostate_map=self._infostate_map,
        key=self._rng)
    return player_2_best_response_values

  def optimize_infoset(self, cfvalues: Any, infoset: List[typing.InfostateNode],
                       infostate_map: typing.InfostateMapping,
                       rng: hk.PRNGSequence):
    """Apply updates to optimizer state.

    Args:
      cfvalues: Counterfactual values.
      infoset: Infostates.
      infostate_map: Mapping from infostate string to infostate.
      rng: Next random seed.
    """
    grads = jax.grad(
        utils.meta_loss, has_aux=False)(self.optimizer.net_params, cfvalues,  # pytype: disable=attribute-error
                                        self.optimizer.net_apply,  # pytype: disable=attribute-error
                                        self._meta_learner_training_epochs,
                                        len(self._all_actions), infoset,
                                        infostate_map, FLAGS.batch_size,
                                        next(rng),
                                        FLAGS.use_infostate_representation)
    updates, self.optimizer.opt_state = self.optimizer.opt_update(  # pytype: disable=attribute-error
        grads, self.optimizer.opt_state)  # pytype: disable=attribute-error

    self.optimizer.net_params = optax.apply_updates(self.optimizer.net_params,  # pytype: disable=attribute-error
                                                    updates)

  def training_optimizer(self):
    """Train an optimizer for meta learner."""

    self.optimizer = models.OptimizerModel(
        mlp_sizes=FLAGS.mlp_sizes,
        lstm_sizes=FLAGS.lstm_sizes,
        initial_learning_rate=FLAGS.init_lr,
        batch_size=FLAGS.batch_size,
        num_actions=len(self._all_actions),
        num_infostates=self._num_infostates,
        model_type=self._model_type,
        use_infostate_representation=FLAGS.use_infostate_representation)
    self.optimizer.initialize_optimizer_model()

    while self._step < FLAGS.num_tasks:
      if self._perturbation:
        self._seed = np.random.choice(np.array(list(range(100))))
      self._world_state = openspiel_api.WorldState(
          self._game_name,
          self._game_config,
          perturbation=self._perturbation,
          random_seed=self._seed)

      for epoch in range(self._training_epochs):
        logging.info("Training epoch %d", epoch)
        all_infostates_map = [{}, {}, {}]
        first_history_node, infostate_nodes = game_tree_utils.build_tree_dfs(
            self._world_state, all_infostates_map)
        cfr_values_player1, cfr_values_player2, _ = cfr_br_meta_data(  # pytype: disable=wrong-arg-types
            history_tree_node=first_history_node,
            infostate_nodes=infostate_nodes,
            all_infostates_map=all_infostates_map,
            epochs=self._meta_learner_training_epochs,
            net_apply=self.optimizer.net_apply,
            net_params=self.optimizer.net_params,
            all_actions=self._all_actions,
            infostate_map=self._infostate_map,
            key=self._rng)

        train_dataset = []
        cfvalues_per_player = [
            cfr_values_player1, cfr_values_player2
        ]
        # for CFRBR we consider player 0.
        player_ix = 0
        infosets = [
            infoset for infoset in all_infostates_map[player_ix + 1].values()
            if len(infoset.get_actions()) >= 2
        ]
        for infoset in infosets:
          cfvalues = cfvalues_per_player[player_ix][infoset.infostate_string]
          train_dataset.append((cfvalues, infoset))

        dataset = dataset_generator.Dataset(train_dataset, FLAGS.batch_size)  # pytype: disable=wrong-arg-types  # jax-ndarray
        data_loader = dataset.get_batch()
        for _ in range(FLAGS.num_batches):
          batch = next(data_loader)
          cfvalues, infoset = zip(*batch)
          cfvalues = np.array(list(cfvalues), dtype=object)
          cfvalues = utils.mask(cfvalues, infoset, len(self._all_actions),
                                FLAGS.batch_size)
          self.optimize_infoset(cfvalues, infoset, self._infostate_map,  # pytype: disable=wrong-arg-types
                                self._rng)
      logging.info("Game: %d", self._step)
      self._step += 1
