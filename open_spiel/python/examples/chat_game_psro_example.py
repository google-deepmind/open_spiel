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

"""Prompt-Space Response-Oracle (PSRO) experiment.

Runs PSRO exploring the space of `tones` with which to construct messages. Only
works with `tones` for now.
"""

import enum
import itertools
import math

from absl import app
from absl import flags
from absl import logging

import ml_collections

import nashpy
import numpy as np

from open_spiel.python.games import chat_game  # pylint: disable=unused-import
from open_spiel.python.games.chat_games.configs import config_schedule_meeting_w_tone
from open_spiel.python.games.chat_games.configs import config_trade_fruit_w_tone
from open_spiel.python.games.chat_games.envs.utils import text
from open_spiel.python.games.chat_games.utils import test_utils as chat_test_utils

import pyspiel


_SAVE_PATH = flags.DEFINE_string("save_path",
                                 default="",
                                 help="path for writing results")

LLM_TYPE = chat_test_utils.TestLLM.MOCK


class Domain(enum.StrEnum):
  TRADE_FRUIT_W_TONE = enum.auto()
  SCHEDULE_MEETING_W_TONE = enum.auto()


def get_config():
  """Get configuration for imitation dataset construction."""
  config = ml_collections.config_dict.ConfigDict()

  config.game_string = "chat_game"
  config.seed = 34239871
  config.num_iters = 4
  config.num_trials = 10
  config.num_candidates = 2
  config.domain = Domain.SCHEDULE_MEETING_W_TONE

  if config.domain == Domain.TRADE_FRUIT_W_TONE:
    config.env_config = config_trade_fruit_w_tone.get_config()
  elif config.domain == Domain.SCHEDULE_MEETING_W_TONE:
    config.env_config = config_schedule_meeting_w_tone.get_config()
  else:
    raise ValueError("Unknown domain: %s" % config.domain)

  return config


def sym(pt):
  """Symmetrize stack of payoff tensors (stacked along first dimension).

  A payoff tensor can be `symmetrized' by averaging over all possible
  permutations of the players. This means permuting the axes corresponding to
  the player strategies as well as the payoffs assigned to the players. E.g.,
  player A playing strategy 1 and player B playing strategy 3 is no different
  from player A playing strategy 3 and player B playing strategy 1 in a
  symmetric game. Note we permuted the strategies, but we must also permute the
  payoffs.

  Args:
    pt: tensor of shape: (num_players,) + (num_strategies,) * num_players
  Returns:
    pt_sym: symmetrized payoff tensor of same shape
  """
  num_players = len(pt.shape[1:])
  num_perms = math.factorial(num_players)
  pt_sym = np.zeros_like(pt)
  for _, perm_players in enumerate(itertools.permutations(range(num_players))):
    perm_axes = tuple([pi + 1 for pi in perm_players])
    permuted_tensor = np.transpose(pt, (0,) + perm_axes)[list(perm_players)]
    pt_sym += permuted_tensor / float(num_perms)
  return pt_sym


def random_policy(rnd, state):
  # all actions are legal for now
  rnd_action = tuple([rnd.choice(a) for a in state.num_actions])
  return np.ravel_multi_index(rnd_action, state.num_actions)


def fixed_prompt_policy(rnd, state, prompt_action_dict):
  # all actions are legal for now
  action = [rnd.choice(a) for a in state.num_actions]
  for prompt_key, prompt_action in prompt_action_dict.items():
    prompt_key_idx = 1 + state.header.action_keys.index(prompt_key)
    prompt_val_idx = state.prompt_actions[prompt_key].index(prompt_action)
    action[prompt_key_idx] = prompt_val_idx
  action = tuple(action)
  return np.ravel_multi_index(action, state.num_actions)


def mixed_prompt_policy(rnd, state, prompt_keys, mixture):
  # all actions are legal for now
  action = [rnd.choice(a) for a in state.num_actions]
  for prompt_key in prompt_keys:
    prompt_key_idx = 1 + state.header.action_keys.index(prompt_key)
    actions = state.prompt_actions[prompt_key]
    num_actions = len(actions)
    prompt_val_idx = rnd.choice(num_actions, p=mixture)
    action[prompt_key_idx] = prompt_val_idx
  action = tuple(action)
  return np.ravel_multi_index(action, state.num_actions)


def build_player_policy(policies):
  def player_policy(player_id, state):
    return policies[player_id](state)
  return player_policy


def simulate_dialogue(game, policy):
  """Simulate a dialogue and returns payoffs for each player."""

  state = game.new_initial_state()

  while not state.is_terminal():
    if state.is_chance_node():
      # Chance node: sample an outcome
      outcomes = state.chance_outcomes()
      action_list, prob_list = zip(*outcomes)
      action = np.random.choice(action_list, p=prob_list)
      state.apply_action(action)
    else:
      # Decision node: sample action for the single current player
      action = policy(state.current_player(), state)
      state.apply_action(action)

  # Game is now done. Print utilities for each player
  returns = state.returns()

  return returns


def estimate_payoff_tensor(game, rnd, num_trials=5):
  """Simulate a batch of dialogues and returns payoffs for each player."""

  num_players = game.num_players()
  num_actions = len(game.given_prompt_actions["tone"])
  payoff_tensor = np.zeros(
      (num_trials, num_players) + (num_actions,) * num_players
  )

  joint_actions = list(itertools.product(range(num_actions),
                                         repeat=num_players))

  for trial in range(num_trials):
    for joint_action_idx in joint_actions:
      policies = []
      for _, tone_idx in zip(range(num_players), joint_action_idx):
        fixed_tone = {"tone": game.given_prompt_actions["tone"][tone_idx]}
        policy = lambda state: fixed_prompt_policy(rnd, state, fixed_tone)  # pylint:disable=cell-var-from-loop
        policies.append(policy)
      player_policy = build_player_policy(policies)

      returns = simulate_dialogue(game, player_policy)

      pt_index = (trial, slice(None)) + joint_action_idx

      payoff_tensor[pt_index] = returns

  return payoff_tensor


def score_candidate_responses(game_str, config, load_dict, rnd,
                              background_policies, candidates,
                              player_ids=(0,), num_trials=5):
  """Simulate a batch of dialogues and returns payoffs for each player."""

  num_players = config.params["num_players"]

  num_candidates = len(candidates)

  config.game.given_prompt_actions["tone"] += candidates
  num_actions = len(config.game.given_prompt_actions["tone"])
  config.params["num_distinct_actions"] = num_players * num_actions

  game = pyspiel.load_game(game_str, config.params.to_dict())

  game.load_chat_game(**load_dict, **config.game)

  payoffs = np.zeros((num_trials, len(player_ids), num_candidates))

  for player_id in player_ids:
    for trial in range(num_trials):
      for candidate_idx in range(num_candidates):
        policies = []
        for i in range(num_players):
          if player_id == i:
            fixed_tone = {"tone": candidates[candidate_idx]}
            policy = lambda state: fixed_prompt_policy(rnd, state, fixed_tone)  # pylint:disable=cell-var-from-loop
            policies.append(policy)
          else:
            policies.append(background_policies[i])
        player_policy = build_player_policy(policies)

        returns = simulate_dialogue(game, player_policy)

        payoffs[trial, player_id, candidate_idx] = returns[player_id]

  # undo changes to config (is this inplace?)
  config.game.given_prompt_actions["tone"] = config.game.given_prompt_actions[
      "tone"
  ][:-num_candidates]
  num_tones = len(config.game.given_prompt_actions["tone"])
  config.params["num_distinct_actions"] = num_players * num_tones

  return payoffs, candidates


def compute_sym_eq(pt):
  game = nashpy.Game(pt[0], pt[1])
  p1_traj, p2_traj = game.asymmetric_replicator_dynamics()
  p1_strat = np.mean(p1_traj, axis=0)
  p2_strat = np.mean(p2_traj, axis=0)
  return 0.5 * p1_strat + 0.5 * p2_strat


class PSRO():
  """Run prompt-space response oracle algorithm on chat game."""

  def __init__(self, save_path, config):
    self.save_path = save_path
    self.game_string = config.game_string
    self.seed = config.seed
    self.num_iters = config.num_iters
    self.num_trials = config.num_trials
    self.num_candidates = config.num_candidates
    self.domain = config.domain.value
    self.config = config.env_config

    self.rnd = np.random.RandomState(self.seed)

    self.num_players = self.config.params["num_players"]

    self.game = pyspiel.load_game(self.game_string,
                                  self.config.params.to_dict())

    vectorizer = chat_test_utils.MockVectorizer()
    vectorize = vectorizer.vectorize

    self.load_dict = {"llm_type": LLM_TYPE,
                      "vectorize": vectorize,
                      "seed": self.seed}

    self.game.load_chat_game(**self.load_dict, **self.config.game)

    self.reporting = PSROReporting(
        save_path=self.save_path,
        experiment_name="psro",
        game_string=self.game_string,
        seed=self.seed,
        num_iters=self.num_iters,
        num_trials=self.num_trials,
        num_candidates=self.num_candidates,
        domain=self.domain,
        base_candidates=list(self.config.game.given_prompt_actions["tone"]))

  def run(self):
    """Evaluate an imitation-learned policy."""

    for psro_iter in range(self.num_iters):

      pt = estimate_payoff_tensor(self.game,
                                  self.rnd,
                                  num_trials=self.num_trials)
      pt = pt.mean(axis=0)  # mean over trials
      pt = sym(pt)  # symmetrize the pt

      # compute eq
      sub_eq = compute_sym_eq(pt)  # assume symmetric ne

      # generate num_candidate tones
      actions = self.config.game.given_prompt_actions["tone"]
      candidates = self.game.generate_prompts("tone",
                                              actions,
                                              self.num_candidates,
                                              text.retrieve_alpha_block)
      new_actions = actions + candidates
      new_num_actions = len(new_actions)

      eq = np.zeros(new_num_actions) / float(new_num_actions)
      eq[:pt.shape[1]] = sub_eq

      background_policies = []
      for _ in range(self.num_players):
        bg_policy = lambda state: mixed_prompt_policy(self.rnd,
                                                      state,
                                                      ["tone"],
                                                      eq)  # pylint:disable=cell-var-from-loop
        background_policies.append(bg_policy)

      scores, candidates = score_candidate_responses(
          self.game_string,
          self.config,
          self.load_dict,
          self.rnd,
          background_policies,
          candidates,
          player_ids=(0,),
          num_trials=self.num_trials)

      mean_scores = np.mean(scores, axis=0)[0]  # only need player 0's scores
      br_idx = np.argmax(mean_scores)
      br = candidates[br_idx]

      self.config.game.given_prompt_actions["tone"] += [br]
      new_num_tones = len(self.config.game.given_prompt_actions["tone"])
      self.num_players = self.config.params["num_players"]
      new_num_distinct_actions = self.num_players * new_num_tones
      self.config.params["num_distinct_actions"] = new_num_distinct_actions

      self.game = pyspiel.load_game(self.game_string,
                                    self.config.params.to_dict())

      self.game.load_chat_game(**self.load_dict, **self.config.game)

      self.reporting.report(psro_iter,
                            pt,
                            br,
                            mean_scores,
                            candidates,
                            sub_eq)


class PSROReporting(object):
  """Utilities for logging an experiment run."""

  def __init__(self,
               save_path: str,
               experiment_name: str,
               game_string: str,
               seed: int,
               num_iters: int,
               num_trials: int,
               num_candidates: int,
               domain: str,
               base_candidates: list[str]):
    self.save_path = save_path
    self.experiment_name = experiment_name
    self.game_string = game_string
    self.seed = seed
    self.num_iters = num_iters
    self.num_trials = num_trials
    self.num_candidates = num_candidates
    self.domain = domain
    self.base_candidates = base_candidates

    config_dict_params = {}
    config_dict_params["game_string"] = self.game_string
    config_dict_params["seed"] = self.seed
    config_dict_params["num_iters"] = self.num_iters
    config_dict_params["num_trials"] = self.num_trials
    config_dict_params["num_candidates"] = self.num_candidates
    config_dict_params["domain"] = self.domain
    config_dict_params["base_candidates"] = self.base_candidates

    print("Config parameters:\n{:}".format(config_dict_params))

  def report(self,
             psro_iter: int,
             payoff_tensor: np.ndarray,
             br: str,
             mean_scores: np.ndarray,
             candidates: np.ndarray,
             eq: np.ndarray):
    """Report the psro statistics."""
    psro_stats_dict = {}
    psro_stats_dict["psro_iter"] = psro_iter
    psro_stats_dict["payoff_tensor"] = payoff_tensor
    psro_stats_dict["br"] = br
    psro_stats_dict["mean_scores"] = mean_scores
    psro_stats_dict["candidates"] = candidates
    psro_stats_dict["eq"] = eq

    print("PSRO statistics ({:d}):\n{:}".format(psro_iter, psro_stats_dict))


def main(_):
  logging.set_verbosity(logging.ERROR)  # silence internal game logging
  save_path = _SAVE_PATH.value
  config = get_config()
  psro = PSRO(save_path, config)
  psro.run()


if __name__ == "__main__":
  app.run(main)
