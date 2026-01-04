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

"""Counterfactual regret minimization (CFR) experiment.

Runs OpenSpiel CFR on a chat game.
"""

import dataclasses
import enum

from typing import Callable, Union

from absl import app
from absl import flags
from absl import logging

import ml_collections

import numpy as np

from open_spiel.python import policy as pyspiel_policy
from open_spiel.python.algorithms import expected_game_score

from open_spiel.python.games import chat_game  # pylint: disable=unused-import
from open_spiel.python.games.chat_games import chat_game_base

from open_spiel.python.games.chat_games.configs import config_debate
from open_spiel.python.games.chat_games.configs import config_schedule_meeting_w_dow
from open_spiel.python.games.chat_games.configs import config_schedule_meeting_w_tone
from open_spiel.python.games.chat_games.configs import config_trade_fruit_w_tone

from open_spiel.python.games.chat_games.envs.comm_substrates import schedules

from open_spiel.python.games.chat_games.utils import test_utils as chat_test_utils

import pyspiel


_SAVE_PATH = flags.DEFINE_string("save_path",
                                 default="",
                                 help="path for writing results")

LLM_TYPE = chat_test_utils.TestLLM.MOCK


class Domain(enum.StrEnum):
  TRADE_FRUIT_W_TONE = enum.auto()
  DEBATE_W_STYLE = enum.auto()
  SCHEDULE_MEETING_W_DOW = enum.auto()
  SCHEDULE_MEETING_W_TONE = enum.auto()


def new_debate_scenario_config(
    config: ml_collections.config_dict.ConfigDict,
    game_id: int,
) -> ml_collections.config_dict.ConfigDict:
  """Creates a new debate scenario config with a new topic.

  Arguments:
    config: the original debate scenario config dict (this should contain
      examples for generating new scenarios)
    game_id: int, will index into set of 20 debate topics found in
      https://www.englishclub.com/speaking/agreeing-disagreeing-topics.php
  Returns:
    new_config: debate config with redefined debate topic
  """
  # https://www.englishclub.com/speaking/agreeing-disagreeing-topics.php
  topics = ["Breakfast is the most important meal of the day.",
            "Swimming in the ocean is better than swimming in a public pool.",
            "Alcohol should be illegal.",
            "Children should provide room and board for their aging parents.",
            "Studying grammar is more important than practising conversation " +
            "skills.",
            "Television is the leading cause of violence in todays society.",
            "Dogs make better companions than cats.",
            "Smoking should be permitted in public places.",
            "Females are better students than males.",
            "A parent shouldn't pierce a babys ears.",
            "Women should be allowed to go topless in public.",
            "Lawyers should make a higher salary than nurses.",
            "Everyone should plan their own funeral.",
            "Reading English is more difficult than writing English.",
            "Summer is the best season of the year.",
            "Children under 13 should not be allowed to babysit.",
            "High school students should wear uniforms.",
            "21 should be the legal driving age around the world.",
            "Rock and Roll is the best kind of music.",
            "The government should pay for post secondary education."]

  topic = topics[game_id]
  config.game.given_private_info["topic"] = [topic, topic]

  return config


def same_scenario_config(
    config: ml_collections.config_dict.ConfigDict,
    game_id: int,
) -> ml_collections.config_dict.ConfigDict:
  """Dummy function for games that don't need any config modification.

  Arguments:
    config: the original game scenario config dict (this should contain
      examples for generating new scenarios)
    game_id: int, unused
  Returns:
    new_config: original game config
  """
  del game_id

  return config


def get_config_debate(config: ml_collections.config_dict.ConfigDict):
  """Get config for imitation dataset construction of debates."""

  config.config_rnd = config_debate.get_config()
  config.new_config = new_debate_scenario_config

  return config


def get_config_trade_fruit_w_tone(
    config: ml_collections.config_dict.ConfigDict,
):
  """Get config for imitation dataset construction of trading fruit."""

  config.config_rnd = config_trade_fruit_w_tone.get_config()
  config.new_config = same_scenario_config

  return config


def get_config_schedule_meeting_w_dow(
    config: ml_collections.config_dict.ConfigDict,
):
  """Get config for imitation dataset construction of meeting scheduling dow."""

  config.config_rnd = config_schedule_meeting_w_dow.get_config()
  config.new_config = same_scenario_config

  return config


def get_config_schedule_meeting_w_tone(
    config: ml_collections.config_dict.ConfigDict,
):
  """Get config for imitation dataset construction of meeting scheduling dow."""

  config.config_rnd = config_schedule_meeting_w_tone.get_config()
  config.new_config = same_scenario_config

  return config


def get_config():
  """Get configuration for imitation dataset construction."""
  config = ml_collections.config_dict.ConfigDict()

  config.game_string = "chat_game"
  config.game_id = 0
  config.seed = 34239871
  config.num_demos = 10
  config.num_iters = 4
  config.domain = Domain.SCHEDULE_MEETING_W_TONE

  if config.domain == Domain.DEBATE_W_STYLE:
    config = get_config_debate(config)
  elif config.domain == Domain.TRADE_FRUIT_W_TONE:
    config = get_config_trade_fruit_w_tone(config)
  elif config.domain == Domain.SCHEDULE_MEETING_W_DOW:
    config = get_config_schedule_meeting_w_dow(config)
    config.substrate = schedules
  elif config.domain == Domain.SCHEDULE_MEETING_W_TONE:
    config = get_config_schedule_meeting_w_tone(config)
  else:
    raise ValueError("Unknown domain: %s" % config.domain)

  return config


@dataclasses.dataclass(frozen=True)
class InfoStateRecord:
  observation: str | np.ndarray
  observation_str: str
  probabilities: list[float]
  actions: list[int]
  prev_message: str
  prev_speaker: int
  prev_action_strs: list[str]


@dataclasses.dataclass(frozen=False)
class GameStats:
  num_states: int = 0
  num_chance_nodes: int = 0
  num_decision_nodes: int = 0
  num_simultaneous_nodes: int = 0
  num_terminals: int = 0
  info_state_dict: dict[str, InfoStateRecord] = dataclasses.field(
      default_factory=dict)


@dataclasses.dataclass(frozen=True)
class EqRecord:
  nash_conv: float
  payoffs_eq_vs_bg_any: list[float]
  payoffs_any: list[float]
  payoffs_eq: list[float]


def record_info_state_data(
    state: pyspiel.State,
    policy: pyspiel.Policy,
    observer: Union[None, chat_game_base.ChatGameObserverBase] = None,
    vectorize: Union[None, Callable[[str, int], np.ndarray]] = None,
) -> InfoStateRecord:
  """Return observation and equilibrium strategy for a given state+policy."""
  pi = policy.action_probabilities(state)
  action_list = list(pi.keys())
  prob_list = list(pi.values())
  if observer is not None:
    info_str = observer.string_from(state, player=state.current_player())
    if vectorize is not None:
      info = vectorize(info_str, 768)
    else:
      info = info_str
  else:
    info = info_str = str(state)
  prev_msg = ""
  prev_speaker = -1
  prev_action_strs = []
  if state.played_actions:
    prev_action = state.played_actions[-1]
    prev_msg = state.dialogue[-1]
    prev_speaker = state.speakers[-1]
    prev_speaker = int(prev_speaker)
    prev_action_dict = state.unravel_flat_action_to_dict(prev_speaker,
                                                         prev_action)
    action_keys = state.prompt_actions.keys()
    prev_action_strs = [prev_action_dict["action"][key] for key in action_keys]
  sample = InfoStateRecord(info, info_str, prob_list, action_list,
                           prev_msg, prev_speaker, prev_action_strs)
  return sample


# traverses game tree and records game stats like info states.
def traverse_game_tree(
    game: pyspiel.Game,
    state: pyspiel.State,
    game_stats: GameStats,
    policy: pyspiel.Policy,
    observer: Union[None, chat_game_base.ChatGameObserverBase] = None,
    vectorize: Union[None, Callable[[str, int], np.ndarray]] = None,
):
  """Traverse the game tree and record GameStats in place.

  Args:
    game: pyspiel.Game
    state: initial pyspiel.State
    game_stats: empty GameStats object
    policy: pyspiel Policy
    observer: pyspiel Observer
    vectorize: method to vectorize a string
  """
  if state.is_terminal():
    game_stats.num_terminals += 1
  elif state.is_chance_node():
    game_stats.num_chance_nodes += 1
    for outcome in state.legal_actions():
      child = state.child(outcome)
      traverse_game_tree(game, child, game_stats, policy, observer, vectorize)
  elif state.is_simultaneous_node():
    game_stats.num_simultaneous_nodes += 1
    # TODO(imgemp): need to implement recording data for simultaneous
    # Using joint actions for convenience. Can use legal_actions(player) to
    # and state.apply_actions when walking over individual players
    for joint_action in state.legal_actions():
      child = state.child(joint_action)
      traverse_game_tree(game, child, game_stats, policy, observer, vectorize)
  else:
    game_stats.num_decision_nodes += 1
    if game.get_type().provides_information_state_string:
      sample = record_info_state_data(state, policy, observer, vectorize)
      game_stats.info_state_dict[
          state.information_state_string()] = sample
    for outcome in state.legal_actions():
      child = state.child(outcome)
      traverse_game_tree(game, child, game_stats, policy, observer, vectorize)


class ImitationDatasetConstructor():
  """Construct a dataset of (observation, CFR strategy) for imitation."""

  def __init__(self, save_path, config):
    self.save_path = save_path
    self.game_string = config.game_string
    self.game_id = config.game_id
    self.seed = config.seed
    self.num_demos = config.num_demos
    self.num_iters = config.num_iters
    self.domain = config.domain.value
    self.config_rnd = config.config_rnd
    self.new_config = config.new_config

    self._rnd = np.random.RandomState(self.seed)

    self.reporting = ImitationDatasetConstructorReporting(
        save_path=self.save_path,
        experiment_name="imitation_dataset_construction",
        game_string=self.game_string,
        game_id=self.game_id,
        seed=self.seed,
        num_demos=self.num_demos,
        num_iters=self.num_iters,
        domain=self.domain)

  def sample_to_dict(
      self,
      info_state_string: str,
      sample: InfoStateRecord,
      eq_record: EqRecord):
    """Constructs a dict mapping named keys to values in arguments."""

    sample_dict = {}
    sample_dict["info_state_string"] = info_state_string
    sample_dict["observation"] = sample.observation
    sample_dict["observation_str"] = sample.observation_str
    sample_dict["probabilities"] = sample.probabilities
    sample_dict["actions"] = sample.actions
    sample_dict["prev_message"] = sample.prev_message
    sample_dict["prev_speaker"] = sample.prev_speaker
    sample_dict["prev_action_strs"] = sample.prev_action_strs
    sample_dict["nash_conv"] = eq_record.nash_conv
    sample_dict["payoffs_eq_vs_bg_any"] = eq_record.payoffs_eq_vs_bg_any
    sample_dict["payoffs_any"] = eq_record.payoffs_any
    sample_dict["payoffs_eq"] = eq_record.payoffs_eq
    return sample_dict

  def eval_vs_any(self, game: pyspiel.Game, eq: pyspiel.Policy
                  ) -> EqRecord:
    """Evaluates the equilibrium against a background 'any' policy.

    Arguments:
      game: pyspiel.Game
      eq: pyspiel.Policy equilibrium policy (e.g., result of CFR)
    Returns:
      EqRecord containing
        ne_conv: float, sum of gains from each player best responding to eq
        payoffs_eq_vs_bg_any: list of floats, payoffs for each player when
          playing their side of equilibrium against background agents that all
          play 'any'
        payoff_any: list of floats, payoffs for each player when everyone plays
          'any' policy
        payoff_eq: list of floats, payoffs for each player when everyone plays
          equilibrium policy
    """
    ne_conv = pyspiel.nash_conv(game, eq)

    # construct pyspiel.Policy to play "any" tone (null strategy)
    # the action set is assumed to be (msg_receiver, prompt_action)
    # and "any" is assumed to be the last action in the prompt_action_list
    num_players = game.num_players()
    num_prompt_actions = game.num_distinct_actions() // num_players
    payoffs_eq_vs_bg_any = []
    one_hot_any = [0.0 for _ in range(game.num_distinct_actions())]
    for i in range(num_players):
      idx = i * num_prompt_actions + (num_prompt_actions - 1)
      one_hot_any[idx] = 1 / float(num_players)
    policy_any = dict(zip(range(len(one_hot_any)), one_hot_any))

    def callable_policy(state):
      del state
      return policy_any  # pylint:disable=cell-var-from-loop

    # compute expected payoffs for each player playing eq against "any" bg strat
    for i in range(num_players):
      policies = []
      for j in range(num_players):
        if i == j:
          # grab player i's side of avg_policy (eq_i)
          eq_i = pyspiel_policy.pyspiel_policy_to_python_policy(game,
                                                                eq,
                                                                players=[i])
          policies.append(eq_i)
        else:
          # setting player j policy to "any"
          p_j = pyspiel_policy.tabular_policy_from_callable(game,
                                                            callable_policy,
                                                            players=[j])
          policies.append(p_j)
      state = game.new_initial_state()
      payoff_array = expected_game_score.policy_value(state, policies)
      payoffs_eq_vs_bg_any.append(payoff_array[i])

    # compute expected payoffs when everyone plays "any" strategy
    policies = []
    for j in range(num_players):
      p_j = pyspiel_policy.tabular_policy_from_callable(game,
                                                        callable_policy,
                                                        players=[j])
      policies.append(p_j)
    state = game.new_initial_state()
    payoffs_any = expected_game_score.policy_value(state, policies)

    # compute expected payoffs when everyone plays eq strategy
    policies = []
    for j in range(num_players):
      # grab player j's side of avg_policy (eq_j)
      p_j = pyspiel_policy.pyspiel_policy_to_python_policy(game,
                                                           eq,
                                                           players=[j])
      policies.append(p_j)
    state = game.new_initial_state()
    payoffs_eq = expected_game_score.policy_value(state, policies)

    eq_record = EqRecord(ne_conv,
                         payoffs_eq_vs_bg_any,
                         payoffs_any,
                         payoffs_eq)

    return eq_record

  def construct_dataset(self):
    """Construct a dataset of (observation, optimal strategy) for imitation."""

    logging.info("Building vectorizer")
    vectorizer = chat_test_utils.MockVectorizer()
    vectorize = vectorizer.vectorize

    for demo in range(self.num_demos):
      logging.info("Creating new config for demo %d", demo)

      config = self.new_config(self.config_rnd, self.game_id)

      game = pyspiel.load_game(self.game_string, config.params.to_dict())

      seed = self._rnd.randint(42, 12345 + 1)
      game.load_chat_game(llm_type=LLM_TYPE,
                          vectorize=vectorize,
                          seed=seed,
                          **config.game)

      game_cached = pyspiel.convert_to_cached_tree(game)

      logging.info("Constructing CFR solver")
      cfr_solver = pyspiel.CFRSolver(game_cached)

      logging.info("Evaluating and Updating CFR policy")
      for i in range(self.num_iters):
        logging.info("CFR iteration %d", i)
        cfr_solver.evaluate_and_update_policy()

      logging.info("Averaging CFR policy")
      average_policy = cfr_solver.tabular_average_policy()

      eq_record = self.eval_vs_any(game_cached, average_policy)
      logging.info("NashConv: %f", eq_record.nash_conv)
      logging.info("Payoffs vs background any policy: %s",
                   eq_record.payoffs_eq_vs_bg_any)
      logging.info("Payoffs using any policy: %s", eq_record.payoffs_any)
      logging.info("Payoffs using eq policy: %s", eq_record.payoffs_eq)

      logging.info("Building info_state -> observation vectorizer")
      observer = game.make_py_observer()
      vectorizer = chat_test_utils.MockVectorizer()
      vectorize = vectorizer.vectorize

      logging.info("Traversing game tree and storing imitation policy")
      game_stats = GameStats()
      state = game.new_initial_state()
      traverse_game_tree(game, state, game_stats, average_policy,
                         observer=observer, vectorize=vectorize)
      h = f = "*" * 50
      for info_state_string in game_stats.info_state_dict:
        logging.info("%s\nInfo state string:\n%s\n%s", h, info_state_string, f)
        sample = game_stats.info_state_dict[info_state_string]
        results = self.sample_to_dict(info_state_string, sample, eq_record)
        self.reporting.report(demo, results)

      logging.info("Number of info states (length of policy): %d",
                   len(average_policy))


class ImitationDatasetConstructorReporting(object):
  """Utilities for logging an experiment run."""

  def __init__(
      self,
      save_path: str,
      experiment_name: str,
      game_string: str,
      game_id: int,
      seed: int,
      num_demos: int,
      num_iters: int,
      domain: str,
  ):
    self.save_path = save_path
    self.experiment_name = experiment_name
    self.game_string = game_string
    self.game_id = game_id
    self.seed = seed
    self.num_demos = num_demos
    self.num_iters = num_iters
    self.domain = domain

    config_dict_params = {}
    config_dict_params["experiment_name"] = self.experiment_name
    config_dict_params["game_string"] = self.game_string
    config_dict_params["seed"] = self.seed
    config_dict_params["num_demos"] = self.num_demos
    config_dict_params["num_iters"] = self.num_iters
    config_dict_params["domain"] = self.domain

    print("Config parameters:\n{:}".format(config_dict_params))

  def report(self, demo: int, results):
    """Report the exploitability."""
    print("CFR statistics ({:d}):\n{:}".format(demo, results))


def main(_):
  logging.set_verbosity(logging.ERROR)  # silence internal game logging
  save_path = _SAVE_PATH.value
  config = get_config()
  im = ImitationDatasetConstructor(save_path, config)
  im.construct_dataset()


if __name__ == "__main__":
  app.run(main)
