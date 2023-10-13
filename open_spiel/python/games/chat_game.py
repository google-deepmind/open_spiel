# Copyright 2023 DeepMind Technologies Limited
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

"""Creates a chat game as an OpenSpiel Environment."""

from typing import Any, Callable, Dict, OrderedDict, List, Tuple, Union
from absl import logging
import numpy as np

from open_spiel.python.games.chat_games import chat_game_base
from open_spiel.python.games.chat_games.configs import config_fixed_mock
from open_spiel.python.games.chat_games.configs import config_rnd_mock
from open_spiel.python.games.chat_games.envs.observations import utils as observation_utils
from open_spiel.python.games.chat_games.envs.payoffs import utils as payoff_utils
from open_spiel.python.games.chat_games.envs.termination import utils as term_utils
from open_spiel.python.games.chat_games.envs.utils import header as header_utils
from open_spiel.python.games.chat_games.utils import test_utils as chat_test_utils

import pyspiel


GAME_TYPE = pyspiel.GameType(
        short_name='chat_game',
        long_name='Chat Game',
        utility=pyspiel.GameType.Utility.GENERAL_SUM,
        provides_information_state_string=False,
        provides_information_state_tensor=False,
        **chat_game_base.GAME_TYPE_KWARGS)


class ChatGameObserver(chat_game_base.ChatGameObserverBase):
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  def _build_str_to_info_state(self) -> bool:
    """Initializes map from str to infostate. Returns True if successful."""
    # Build a string tokenizer here
    # --------------------------- #
    # Build a string tokenizer here
    return True

  def _info_state(self, input_text: str, obs_size: int) -> np.ndarray:
    """Returns a len-obs_size np.ndarray given an input string and obs_size."""
    if not self._str_to_info_state_built:
      raise ValueError('String to info state mapping not built!')
    del input_text
    # Vectorize a str (ideally lossless for info state) using a tokenizer here
    # ---------------------------------------------------------------------- #
    # Vectorize a str (ideally lossless for info state) using a tokenizer here
    return np.zeros(obs_size, dtype=np.int32)


class ChatGame(chat_game_base.BaseChatGame):
  """Chat game."""

  # pylint:disable=dangerous-default-value
  def __init__(
      self,
      params: Dict[str, Any] = chat_game_base.DEFAULT_PARAMS,
  ):
    """Constructor.

    Args:
      params: dict, parameter dict with the following keys

        num_distinct_actions- int, # of actions at each info set
        num_llm_seeds- int, # of seeds to use for generating LLM response
        num_players- int, # of speakers (action: recipient) on the message chain
        min_utility- float, minimum utility any player can attain
        max_utility- float, maximum utility any player can attain
        num_max_replies- int, total # of messages each player can send in an
          episode
    """
    self._game_loaded = False

    super().__init__(params)  # initializes self.game_info via base init
    super(chat_game_base.BaseChatGame, self).__init__(
        GAME_TYPE, self.game_info, params or dict())

  def load_chat_game(self,
                     llm_type: chat_test_utils.TestLLM,
                     observations: List[observation_utils.Observation],
                     vectorize: ...,
                     header: header_utils.Header,
                     payoffs: List[payoff_utils.Payoff],
                     aggregate_payoffs: Callable[[List[int]], float] = np.mean,
                     given_names: Union[List[str], None] = None,
                     given_llm_seeds: Union[List[int], None] = None,
                     given_prompt_actions: Union[OrderedDict[str, List[str]],
                                                 None] = None,
                     given_private_info: Union[OrderedDict[str, List[str]],
                                               None] = None,
                     initial_scenario: Union[Any, None] = None,
                     num_names: int = 2,
                     num_prompt_actions: Tuple[int, ...] = (4,),
                     num_private_info: Tuple[int, ...] = (4,),
                     examples_names: Union[List[str], None] = None,
                     examples_prompt_actions: Union[OrderedDict[str, List[str]],
                                                    None] = None,
                     examples_private_info: Union[OrderedDict[str, List[str]],
                                                  None] = None,
                     examples_scenarios: Union[List[Any], None] = None,
                     llm_list_suffix: str = 'Continue the list from here.',
                     llm_termination_prompt: Union[term_utils.Termination,
                                                   None] = None,
                     seed: Union[int, None] = None
                     ):
    """Constructor.

    Args:
      llm_type: item of enum type chat_test_utils.TestLLM
      observations: List of Observation items used for prompting llms to extract
        observations (string features) from dialogues
      vectorize: converts any length string into a length obs_size vector

      header: List of Header items used for prompting llms to take actions
        (construct messages) based on latent action variables and private
        information

      payoffs: list of Payoff items used for constructing queries and scoring
        dialogue for each agent
      aggregate_payoffs: function that maps from vector to nonnegative scalar
      
      given_names: list of strings representing names of players
      given_llm_seeds: list of ints to seed llm with to generate each message
      given_prompt_actions: ordered dict mapping action_keys
        (see envs/utils/header) to list of strings representing the set of
        available prompt actions (e.g., personalities or msg tones). Overrides
        examples_prompt_actions.
      given_private_info: ordered dict mapping info_keys
        (see envs/utils/header) to length-[num_players] list of strings
        representing the private information available to each player (e.g.,
        inventory / valuations of fruits). Overrides examples_private_info.
      initial_scenario: Scenario items representing an initial message

      num_names: int, # of names to generate (can be greater than # of players)
      num_prompt_actions: tuple of int, # of prompts to consider for each
        action_key (i.e., size of action space for each prompt action)
      num_private_info: tuple of int, # of private info states to consider for
        each info_key
      
      examples_names: list of strings representing examples of names of players
      examples_prompt_actions: ordered dict mapping action_keys
        (see envs/utils/header) to list of strings representing examples of
        prompt actions (e.g., personalities or msg tones).
      examples_private_info: ordered dict mapping info_keys
        (see envs/utils/header) to list of strings representing examples of
        private information available to players (e.g., inventory / valuations
        of fruits). Overrides examples_private_info.
      examples_scenarios: list of Scenario items used for meta-generating new
        scenarios
      
      llm_list_suffix: str, gets appended to a prompt to induce an llm to
        generate a list of items (different llms like different prompts).
        chinchilla likes ``, llmit likes `Continue the list from here.`
      llm_termination_prompt: Termination item w/ [attrs query,
        obs_trans_postfix, postfix]. llm will be asked to score a binary
        response `yes`/`no` given query.format(msg=last_msg) to determine
        whether the episode has reached a terminal state (e.g., deal has been
        agreed upon). default is empty string in which case llm terminal
        condition is left unused and episode terminates after
        num_players * num_max_replies

      seed: int, master seed for experiment (used to generate all subsequent
        seeds for any random generation)
    """

    # Define LLM model here
    self._llm_type = llm_type
    if self._llm_type == chat_test_utils.TestLLM.MOCK:
      self._lm = chat_test_utils.MockLLM()
    else:
      raise NotImplementedError(f'llm_type {self._llm_type} not available.')
    # Define LLM model here

    super()._load_chat_game(observations,
                            vectorize,
                            header,
                            payoffs,
                            aggregate_payoffs,
                            given_names,
                            given_llm_seeds,
                            given_prompt_actions,
                            given_private_info,
                            initial_scenario,
                            num_names,
                            num_prompt_actions,
                            num_private_info,
                            examples_names,
                            examples_prompt_actions,
                            examples_private_info,
                            examples_scenarios,
                            llm_list_suffix,
                            llm_termination_prompt,
                            seed)

    self._game_loaded = True

  def generate_response(self, prompt: str, seed: int,
                        num_output_tokens: Union[int, None] = None) -> str:
    """Returns LLM generated string given prompt and seed."""
    # Define generate response here
    if self._llm_type == chat_test_utils.TestLLM.MOCK:
      return self._lm.generate_response(prompt, seed, num_output_tokens)
    else:
      raise NotImplementedError(f'llm_type {self._llm_type} not available.')
    # Define generate response here

  def generate_bool(self, prompt: str, seed: int) -> bool:
    """Returns LLM generated boolean given prompt and seed."""
    # Define generate bool here (e.g., for terminating an episode)
    if self._llm_type == chat_test_utils.TestLLM.MOCK:
      return self._lm.generate_bool(prompt, seed)
    else:
      raise NotImplementedError(f'llm_type {self._llm_type} not available.')
    # Define generate bool here

  def make_py_observer(self,
                       iig_obs_type: Union[pyspiel.IIGObservationType,
                                           None] = None,
                       params: Union[Dict[str, Any], None] = None
                       ) -> ChatGameObserver:
    """Returns an object used for observing game state."""
    return ChatGameObserver(
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
        params)

  def new_initial_state(self) -> chat_game_base.ChatGameState:
    """Generates a new dialogue game.

    Returns:
      chat_game_base.ChatGameState (see chat_games/chat_game_base.py)
    """
    # KEEP THIS IF-BLOCK FOR OPEN_SPIEL TESTS
    if not self._game_loaded:
      # load mock game for testing
      if self._num_players == 2:
        config = config_fixed_mock.get_config()
        tones = config.game.given_prompt_actions.values()[0]
        num_prompt_actions = (len(tones),)
      else:
        config = config_rnd_mock.get_config()
        num_prompt_actions = config.game.num_prompt_actions
      # open_spiel attempts to run several simulation tests of games. this
      # chat_game, however, requires calling `load_chat_game` explicitly after
      # __init__ which is unique. we do this because the most obvious place to
      # pass game configs would be via `params`, but everything in params must
      # be `pickleable` which rules out passing things like `vectorizers` and
      # messsy llm string generators. therefore, we need to check to see if
      # `load_chat_game` has been called here and call it if not.
      # also, open_spiel tests run with variable numbers of players which are
      # different from those in chat_game_base.DEFAULT_PARAMS. More importantly,
      # this affects the number of distinct actions since the number of players
      # affects who we can choose to speak to. hence, we explicitly recalculate
      # the number of distinct actions here (overwriting what was specified in
      # the original chat_game_base.DEFAULT_PARAMS)
      self._num_distinct_actions = np.prod(num_prompt_actions +
                                           (self._num_players,))
      vectorizer = chat_test_utils.MockVectorizer()
      self.load_chat_game(llm_type=chat_test_utils.TestLLM.MOCK,
                          vectorize=vectorizer.vectorize,
                          seed=1234,
                          **config.game)
      logging.warning('Loading chat_game with default config. Only meant for ' +
                      'open_spiel testing.')

    return chat_game_base.ChatGameState(self,
                                        *super().new_initial_state_specs())

# Register the game with the OpenSpiel library

pyspiel.register_game(GAME_TYPE, ChatGame)
