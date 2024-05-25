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

"""Creates a chat game base class as an OpenSpiel Environment."""

import collections
import dataclasses
import string

from typing import Any, Callable, Dict, OrderedDict, List, Tuple, Union
from absl import logging
import numpy as np

from open_spiel.python.games.chat_games.envs.observations import utils as observation_utils
from open_spiel.python.games.chat_games.envs.payoffs import utils as payoff_utils
from open_spiel.python.games.chat_games.envs.termination import utils as term_utils
from open_spiel.python.games.chat_games.envs.utils import header as header_utils
from open_spiel.python.games.chat_games.envs.utils import text

from open_spiel.python.games.chat_games.utils import logging_utils

import pyspiel


ct = logging_utils.ColorText()

REWARD_MODEL = pyspiel.GameType.RewardModel.TERMINAL

ALL_PLAYERS = 'Everyone'

MIN_RND_SEED = 42
MAX_RND_SEED = 9999
DEFAULT_LLM_SEED = 42

LLM_LENGTH_MESSAGE_TOKENS = 300
LLM_LENGTH_MESSAGE_CHARS = 300
LLM_LENGTH_OBS_TOKENS = 300
LLM_LENGTH_OBS_CHARS = 300
LLM_LENGTH_PAYOFF_OBS_TOKENS = 300
LLM_LENGTH_PAYOFF_OBS_CHARS = 300

LLM_LENGTH_LIST_OF_WORDS_TOKENS = 30
LLM_LIST_GEN_ATTEMPTS = 30

LLM_LENGTH_SCORE_TOKENS = 10

ITEM_PREFIX = '* '

MIN_PLAYERS = 2  # any less and it's not a game, is it ;)
MAX_PLAYERS = 10  # this is set arbitrarily for now, should be upper bound
MAX_NUM_REPLIES = 5

VEC_SIZE = 100  # int, length of vector returned by `vectorize` on string input

DEFAULT_PARAMS = {'num_distinct_actions': 2,
                  'num_llm_seeds': 1,
                  'num_players': MIN_PLAYERS,
                  'players': 0,  # open_spiel tests use this for `num_players`
                  'min_utility': -10.0,
                  'max_utility': 10.0,
                  'num_max_replies': 1,
                  'silence_logging': True}

GAME_TYPE_KWARGS = {
    'dynamics': pyspiel.GameType.Dynamics.SEQUENTIAL,
    'chance_mode': pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    'information': pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    'reward_model': REWARD_MODEL,
    'max_num_players': MAX_PLAYERS,
    'min_num_players': MIN_PLAYERS,
    'provides_observation_string': True,
    'provides_observation_tensor': True,
    'provides_factored_observation_string': True,
    'parameter_specification': DEFAULT_PARAMS,
    'default_loadable': True
    }

GAME_TYPE = pyspiel.GameType(
        short_name='chat_game',
        long_name='Chat Game',
        utility=pyspiel.GameType.Utility.GENERAL_SUM,
        provides_information_state_string=False,
        provides_information_state_tensor=False,
        **GAME_TYPE_KWARGS)


class ChatGameState(pyspiel.State):
  """Chat game state."""

  def __init__(self,
               game: ...,
               actions: OrderedDict[str, List[str]],
               seeds: List[int],
               scenario_prompt: str,
               private_info: OrderedDict[str, List[str]]):
    """Constructor.

    Args:
      game: see ChatGame class (should inherit from BaseChatGame)
      actions: dict, {'player_names': list of str,
                      <prompt_action_i>: list of str,
                      ...}
      seeds: list of ints, llm seeds (chance nodes)
      scenario_prompt: str, initial message with header (no tone)
      private_info: dict mapping info-type to list of str, one for each player
        i.e., private (prior) info available to each player
    """
    super().__init__(game)  # access game with self.get_game()

    self._num_actions = tuple([len(a) for a in actions.values()])
    prompt_action_vals = [
        actions[key] for key in self.get_game().header.action_keys
    ]
    self._prompt_actions = OrderedDict(zip(self.get_game().header.action_keys,
                                           prompt_action_vals))
    self._names = actions['player_names']

    self._llm_seeds = seeds
    assert self.get_game().num_llm_seeds == len(self._llm_seeds)

    self._scenario_prompt = scenario_prompt

    self._private_info = private_info

    self._llm_termination = False

    self._rnd = self.get_game().rnd

    self._played_actions = []
    self._dialogue = [scenario_prompt]
    self._current_speaker = 1
    self._current_player = 1
    self._speakers = []
    self._num_actions_played = 0
    self._returns = None
    self._player_action = None

  def __str__(self):
    """String for debug purposes. No particular semantics are required."""
    return self._dialogue[-1]

  def _unravel_flat_action(self, action: int) -> Tuple[int, ...]:
    """Returns an action tuple with action types separated.
    
    Args:
      action: int
    Returns:
      action_tuple: tuple of ints, each int represents a separate component of
        the combinatorial action-space  
    """
    idxs = np.unravel_index([action], self._num_actions)
    return tuple([idx[0] for idx in idxs])

  def _build_payoff_query(self,
                          payoff_query: str,
                          msg: str,
                          player_str: str) -> str:
    """Construct prompt for LLM to perform sentiment analysis.
    
    Args:
      payoff_query: str, query to be formatted for llm
      msg: str, message to be analyzed
      player_str: str, player message is analyzed (scored) for
    Returns:
      str: str, payoff prompt to feed to LLM
    """
    payoff_dict = {'m': msg, 'p': player_str}
    return payoff_query.format(**payoff_dict)

  def _llm_is_terminal(self) -> bool:
    ct.set_color(logging_utils.RED)
    prefix = self.get_game().llm_termination_prompt.obs_trans_prefix
    postfix = self.get_game().llm_termination_prompt.obs_trans_postfix
    if prefix or postfix:
      prompt = prefix + self.dialogue_str + postfix
      term_obs = self.get_game().generate_response(prompt,
                                                   seed=DEFAULT_LLM_SEED)
      logging.info(ct.color('LLM summary:\n%s'), term_obs)
    else:
      term_obs = self.dialogue_str
    llm_termination = self.get_game().generate_bool(
        self.get_game().llm_termination_prompt.query.format(msg=term_obs),
        seed=DEFAULT_LLM_SEED)
    logging.info(ct.color('LLM termination condition met? %s'),
                 str(llm_termination))
    return llm_termination

  def _names_from_validated_receiver(self, receiver: int, speaker: int
                                     ) -> Tuple[Tuple[str, str, str], int]:
    """Modify receiver if sending to self. Then return names of all roles.
    
    Args:
      receiver: integer action indicating receiver to send message to
      speaker: integer representing current message sender
    Returns:
      names: tuple of strings, (speaker_name, receiver_name, others_names)
      receiver: integer representing validated receiver
    """
    if (receiver >= self.get_game().num_players()
        or speaker >= self.get_game().num_players()):
      logging.info('Warning: rolling receiver/speaker to valid id.')
    receiver = receiver % self.get_game().num_players()
    speaker = speaker % self.get_game().num_players()
    # overwrite speaking to self as speaking to all in header
    receiver_name = ''
    if receiver == speaker:
      if len(self._names) > 2:
        receiver_name = ALL_PLAYERS
        receiver = -1
      else:
        receiver = (receiver + 1) % self.get_game().num_players()
    speaker_name = ''
    others = []
    for idx, name in enumerate(self._names):
      if idx == speaker:
        speaker_name = name
      elif idx == receiver:
        receiver_name = name
      elif receiver > -1:
        others.append(name)
    others_names = ', '.join(others)
    names = (speaker_name, receiver_name, others_names)
    return names, receiver

  def _legal_actions(self, player: int) -> List[int]:
    """Returns a list of legal actions, sorted in ascending order."""
    assert player >= 0
    return list(range(int(np.prod(self._num_actions))))

  def _apply_action(self, action: int):
    """Reply to dialogue (for agents).
    
    Unravel action into to tuple (who to speak to, seed to use, etc.). Then
    simulate action.
    
    Args:
      action: int
    """
    if self.is_chance_node():
      # action is an index into the list of seeds
      # use this to write the message for the previous player
      seed = self._llm_seeds[action]
      assert self._player_action is not None
      self._player_action = self._player_action or 0
      self._played_actions.append(self._player_action)
      speaker_msg = self.action_to_msg(action=self._player_action, seed=seed)
      self._apply_msg(speaker_msg)
      if self.get_game().llm_termination_prompt:
        self._llm_termination = self._llm_is_terminal()
    else:
      # record the action and save it to be played at chance node
      self._player_action = action
      self._current_speaker = int(self._current_player)
      self._num_actions_played += 1

  def _apply_msg(self, speaker_msg: str):
    """Update dialogue history, increment curr player, and update is_terminal.
    
    Args:
      speaker_msg: str
    """
    logging.info('Speaker message:\n%s', speaker_msg)
    self._dialogue.append(speaker_msg)
    self._speakers.append(self._current_player)

    # increment the current player
    self._current_player = (
        self._current_player + 1
    ) % self.get_game().num_players()

    self._player_action = None
    if self.get_game().llm_termination_prompt:
      self._llm_termination = self._llm_is_terminal()

  def apply_msg(self, speaker_msg: str):
    """Reply to dialogue (for human players and interventions).
    
    Args:
      speaker_msg: str
    """
    self._num_actions_played += 1
    self._played_actions.append(-1)  # assign -1 for human messages
    self._apply_msg(speaker_msg)

  def action_to_prompt(self,
                       action: int,
                       seed: int,
                       header: header_utils.Header
                       ) -> Tuple[str, str]:
    """Unravel action int to multidimensional action tuple and construct prompt.
    
    Args:
      action: int, the action taken in the game
      seed: int, llm seed
      header: header_utils.Header, used to format a prompt
    Returns:
      prompt: str, formatted prompt to feed the LLM to generate a new message
      header_plain: str, the formatted header without any private info / actions
    """
    speaker = int(self._current_speaker)
    action_dict = self.unravel_flat_action_to_dict(speaker, action)
    receiver = action_dict['receiver']
    opts = {**action_dict['action'], **action_dict['info']}

    names, _ = self._names_from_validated_receiver(receiver, speaker)
    speaker_name, receiver_name, others_names = names
    header_plain = header.plain.format(sender=speaker_name,
                                       receiver=receiver_name,
                                       others=others_names)

    header_w_opts = header.w_opts.format(sender=speaker_name,
                                         receiver=receiver_name,
                                         others=others_names,
                                         **opts)
    # provide header with opts to llm for response
    logging.info('Generating message (speaker=%d:%s)...',
                 speaker,
                 speaker_name)

    prompt = header.context + '\n\n' + self.dialogue_str + header_w_opts

    return prompt, header_plain

  def action_to_msg(self, action: int, seed: int) -> str:
    """Unravel action int to multidimensional action tuple and construct msg.
    
    Args:
      action: int, the action taken in the game
      seed: int, llm seed
    Returns:
      speaker_msg: str
    """
    header = self.get_game().header
    prompt, header_plain = self.action_to_prompt(action, seed, header)
    logging.info('LLM prompt:\n%s', prompt)

    response = self.get_game().generate_response(
        prompt=prompt,
        seed=seed,
        num_output_tokens=LLM_LENGTH_MESSAGE_TOKENS
    )
    response = response[:LLM_LENGTH_MESSAGE_CHARS]
    logging.info('LLM response:\n%s', response)

    first_special_char = text.first_special_char(
        response, len(response), self.get_game().header.special_chars)
    speaker_msg = header_plain + response[:first_special_char]

    return speaker_msg

  def unravel_flat_action_to_dict(self, speaker: int, action: int
                                  ) -> Dict[str, Any]:
    receiver, *extra_action_idxs = self._unravel_flat_action(action)

    extra_action_strs = [pa[i] for i, pa in zip(extra_action_idxs,
                                                self._prompt_actions.values())]
    action_dict = dict(zip(self.get_game().header.action_keys,
                           extra_action_strs))

    extra_info_strs = [
        pi[speaker] for pi in self._private_info.values()
    ]
    info_dict = dict(zip(self.get_game().header.info_keys, extra_info_strs))

    return {'receiver': receiver,
            'info': info_dict,
            'action': action_dict}

  def compute_rewards(self, dialogue: str) -> np.ndarray:
    """Compute rewards for each player from a given dialogue string.
    
    Args:
      dialogue: str, a single string with the entire dialogue thus far
    Returns:
      rewards: np.ndarray, len-num_players vector of floats
    """
    ct.set_color(logging_utils.GREEN)

    rewards = np.zeros(self.get_game().num_players(), dtype=float)

    if (not self.is_terminal() and
        self.get_game().reward_type == pyspiel.GameType.RewardModel.TERMINAL):
      return rewards

    # gather private info to compute true underlying rewards
    info_prefix = []
    for player, name in enumerate(self._names):
      extra_info_strs = [pi[player] for pi in self._private_info.values()]
      info_prefix_p = [
          f'{k}:\n{v}' for k, v in zip(self.get_game().header.info_keys,
                                       extra_info_strs)
      ]
      info_prefix_p = name + '\n' + '\n'.join(info_prefix_p)
      info_prefix.append(info_prefix_p)
    info_prefix = '\n\n'.join(info_prefix)

    # compute rewards
    for player, name in enumerate(self._names):
      player_payoffs = []
      for p, payoff in enumerate(self.get_game().payoffs):
        if payoff.obs_trans_prefix or payoff.obs_trans_postfix:
          payoff_obs_prompt = (payoff.obs_trans_prefix +
                               dialogue +
                               payoff.obs_trans_postfix)
          logging.info(ct.color('Scoring payoff (speaker=%d:%s)...'),
                       player, name)
          logging.info(ct.color('LLM prompt:\n%s'), payoff_obs_prompt)
          response = self.get_game().generate_response(
              prompt=payoff_obs_prompt,
              seed=DEFAULT_LLM_SEED,
              num_output_tokens=LLM_LENGTH_PAYOFF_OBS_TOKENS
          )
          payoff_obs = response[:LLM_LENGTH_PAYOFF_OBS_CHARS]
        else:
          payoff_obs = dialogue
        payoff_obs = info_prefix + '\n\n' + payoff_obs
        query = self._build_payoff_query(payoff.query, payoff_obs, name)
        logging.info(ct.color('Calculating payoff %d (player=%d:%s)...'),
                     p, player, name)
        logging.info(ct.color('LLM prompt:\n%s'), query)
        response = self.get_game().generate_response(
            prompt=query,
            seed=DEFAULT_LLM_SEED,
            num_output_tokens=LLM_LENGTH_SCORE_TOKENS
            )
        logging.info(ct.color('LLM response:\n%s'), response)

        logging.info(ct.color('Extracting payoff %d (player=%d:%s)...'),
                     p, player, name)
        query = (f'Extract out the final value for {name} as a single ' +
                 'numeric value from the following payoff valuation. Do ' +
                 'NOT show your work:\n\n' +
                 f'{response}\n\nResult: ')
        logging.info(ct.color('LLM prompt:\n%s'), query)
        response = self.get_game().generate_response(
            prompt=query,
            seed=DEFAULT_LLM_SEED,
            num_output_tokens=LLM_LENGTH_SCORE_TOKENS
            )
        logging.info(ct.color('LLM response:\n%s'), response)

        player_payoff = 0  # payoff defaults to 0 if LLM parsing fails
        if text.retrieve_numeric_block(response):
          player_payoff = int(text.retrieve_numeric_block(response))
          player_payoff = min(max(player_payoff, payoff.min), payoff.max)
        else:
          logging.warning(
              ct.color('Payoff extraction from response failed:\n\n%s.'),
              response)
        logging.info(ct.color('Extracted integer payoff (%s): %d'),
                     name, player_payoff)
        player_payoffs.append(player_payoff)
      rewards[player] = self.get_game().aggregate_payoffs(player_payoffs)

    ct.reset()

    return rewards.astype(float)

  def current_player(self) -> int:
    """Returns id of the next player to move, or TERMINAL if game is over."""
    if self.is_terminal():
      return pyspiel.PlayerId.TERMINAL
    elif self._player_action is not None:  # if int, an LLM msg is to be sampled
      return pyspiel.PlayerId.CHANCE
    else:
      return self._current_player

  def is_terminal(self) -> bool:
    """Returns True if the game is over."""
    if ((self._num_actions_played < self.get_game().max_game_length())
        and not self._llm_termination):
      return False
    else:
      return True

  def chance_outcomes(self):
    """Returns the possible chance outcomes and their probabilities."""
    assert self.is_chance_node()
    outcomes = range(self.get_game().num_llm_seeds)
    p = 1.0 / len(outcomes)
    return [(o, p) for o in outcomes]

  def _action_to_string(self, player, action):
    """Action -> string."""
    if player == pyspiel.PlayerId.CHANCE:
      return f'Sampled LLM seed: {action}'
    else:
      action_unraveled = self.unravel_flat_action_to_dict(player, action)
      action_dict = action_unraveled['action']
      return f'Action:\nint: {action}\ndict: {action_dict}'

  def returns(self) -> np.ndarray:
    """Total reward for each player over the course of the game so far."""
    if not self.is_terminal():
      return np.zeros(self.get_game().num_players(), dtype=float)
    else:
      if self._returns is None:
        self._returns = self.compute_rewards(self.dialogue_str)
      return self._returns

  @property
  def dialogue(self) -> List[str]:
    return self._dialogue

  @property
  def dialogue_str(self) -> str:
    return ''.join(self._dialogue)

  @property
  def private_info(self) -> Dict[str, List[str]]:
    return self._private_info

  @property
  def header(self) -> header_utils.Header:
    return self.get_game().header

  @property
  def vectorize(self) -> ...:
    return self.get_game().vectorize

  @property
  def obs(self) -> List[observation_utils.Observation]:
    return self.get_game().obs

  @property
  def names(self) -> List[str]:
    """Returns list of str."""
    return self._names

  @property
  def speakers(self) -> List[int]:
    return self._speakers

  @property
  def played_actions(self) -> List[int]:
    return self._played_actions

  @property
  def num_actions(self) -> Tuple[int, ...]:
    return self._num_actions

  @property
  def prompt_actions(self) -> OrderedDict[str, List[str]]:
    return self._prompt_actions


class ChatGameObserverBase:
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  def __init__(self,
               iig_obs_type: pyspiel.IIGObservationType,
               params: Union[Dict[str, Any], None]):
    """Initializes an empty observation tensor.
    
    Args:
      iig_obs_type: a pyspiel.IIGObservationType
      params: unused
    """
    if params:
      raise ValueError(f'Observation parameters not supported; passed {params}')

    self.iig_obs_type = iig_obs_type
    if self.iig_obs_type.perfect_recall:
      self._str_to_info_state_built = self._build_str_to_info_state()
    else:
      self._str_to_info_state_built = False

    # Determine which observation pieces we want to include.
    pieces = [('player_id', MAX_PLAYERS, (MAX_PLAYERS,))]
    if iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
      if iig_obs_type.perfect_recall:
        pieces.append(('private_info',
                       LLM_LENGTH_MESSAGE_CHARS,
                       (LLM_LENGTH_MESSAGE_CHARS,)))
      else:
        pieces.append(('private_info', VEC_SIZE, (VEC_SIZE,)))
    if iig_obs_type.public_info:
      if iig_obs_type.perfect_recall:
        max_msgs = MAX_PLAYERS * MAX_NUM_REPLIES
        pieces.append(('scenario_prompt',
                       LLM_LENGTH_MESSAGE_CHARS,
                       (LLM_LENGTH_MESSAGE_CHARS)))
        pieces.append(('senders',
                       max_msgs * MAX_PLAYERS,
                       (max_msgs, MAX_PLAYERS)))
        pieces.append(('receivers',
                       max_msgs * MAX_PLAYERS,
                       (max_msgs, MAX_PLAYERS)))
        # record prompt actions as lossless tokenization since we do not know
        # how many actions a game will be defined with. alternatively, we could
        # record the action integer and require the user to unravel the integer
        # on the policy network side. for now, we assume the prompt action is at
        # most LLM_LENGTH_MESSAGE_CHARS subwords. we also assume everyone can
        # see everyone's actions.
        pieces.append(('prompt_actions',
                       max_msgs * LLM_LENGTH_MESSAGE_CHARS,
                       (max_msgs, LLM_LENGTH_MESSAGE_CHARS)))
        pieces.append(('messages',
                       max_msgs * LLM_LENGTH_MESSAGE_CHARS,
                       (max_msgs, LLM_LENGTH_MESSAGE_CHARS)))
      else:
        pieces.append(('dialogue', VEC_SIZE, (VEC_SIZE,)))

    # Build the single flat tensor.
    total_size = sum(size for _, size, _ in pieces)
    self.tensor = np.zeros(total_size, np.float32)

    # Build the named & reshaped views of the bits of the flat tensor.
    self.dict = {}
    index = 0
    for name, size, shape in pieces:
      self.dict[name] = self.tensor[index:index + size].reshape(shape)
      index += size

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

  def set_from(self, state: ChatGameState, player: int):
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
    ct.set_color(logging_utils.PURPLE)

    self.tensor.fill(0)
    self.dict['player_id'][player] = 1

    extra_info_strs = [pi[player] for pi in state.private_info.values()]
    info_prefix = [
        f'{k}:\n{v}' for k, v in zip(state.header.info_keys, extra_info_strs)
    ]
    info_prefix = '\n'.join(info_prefix)
    if 'private_info' in self.dict:
      if self.iig_obs_type.perfect_recall:
        private_info = self._info_state(info_prefix, LLM_LENGTH_MESSAGE_CHARS)
      else:
        private_info = state.vectorize(info_prefix, VEC_SIZE)
      self.dict['private_info'] = private_info

    if self.iig_obs_type.public_info and self.iig_obs_type.perfect_recall:
      self.dict['scenario_prompt'] = self._info_state(state.dialogue[0],
                                                      LLM_LENGTH_MESSAGE_CHARS)
      for i, (speaker, played_action) in enumerate(zip(state.speakers,
                                                       state.played_actions)):
        self.dict['senders'][i][speaker] = 1
        if played_action >= 0:  # played_action = -1 indicates human player
          action_dict = state.unravel_flat_action_to_dict(speaker,
                                                          played_action)
          self.dict['receivers'][i][action_dict['receiver']] = 1
          pa = action_dict['action']
          action_str = '\n'.join([f'{k}: {v}' for k, v in pa.items()])
          self.dict['prompt_actions'][i] = self._info_state(
              action_str, LLM_LENGTH_MESSAGE_CHARS)

        self.dict['messages'][i] = self._info_state(state.dialogue[i + 1],
                                                    LLM_LENGTH_MESSAGE_CHARS)

    if 'dialogue' in self.dict:
      obs_prompt = (state.obs[player].obs_trans_prefix +
                    state.dialogue_str +
                    state.obs[player].obs_trans_postfix)
      logging.info(ct.color('Generating observation (speaker=%d:%s)...'),
                   player,
                   state.names[player])
      logging.info(ct.color('LLM prompt:\n%s'), obs_prompt)
      response = state.get_game().generate_response(
          prompt=obs_prompt,
          seed=DEFAULT_LLM_SEED,
          num_output_tokens=LLM_LENGTH_OBS_TOKENS
          )
      logging.info(ct.color('LLM response:\n%s'), response)
      obs = response[:LLM_LENGTH_OBS_CHARS]

      obs = info_prefix + '\n' + obs

      logging.info(ct.color('Observation (speaker=%d:%s):\n%s'),
                   player,
                   state.names[player],
                   obs)
      logging.info(ct.color('Vectorizing observation...'))
      observation = state.vectorize(obs, VEC_SIZE)
      logging.info(ct.color('Vectorized observation (speaker=%d:%s):\n%s'),
                   player,
                   state.names[player],
                   observation)
      self.dict['dialogue'] = observation

      ct.reset()

  def string_from(self, state: ChatGameState, player: int) -> str:
    """Observation of `state` from the PoV of `player`, as a string."""
    ct.set_color(logging_utils.PURPLE)

    extra_info_strs = [pi[player] for pi in state.private_info.values()]
    info_prefix = [
        f'{k}:\n{v}' for k, v in zip(state.header.info_keys, extra_info_strs)
    ]
    info_prefix = '\n'.join(info_prefix)

    if self.iig_obs_type.perfect_recall:
      return info_prefix + '\n\nFull Dialogue\n\n' + state.dialogue_str
    else:
      obs_prompt = (state.obs[player].obs_trans_prefix +
                    state.dialogue_str +
                    state.obs[player].obs_trans_postfix)
      logging.info(ct.color('Generating observation (speaker=%d:%s)...'),
                   player,
                   state.names[player])
      logging.info(ct.color('LLM prompt:\n%s'), obs_prompt)
      response = state.get_game().generate_response(
          prompt=obs_prompt,
          seed=DEFAULT_LLM_SEED,
          num_output_tokens=LLM_LENGTH_OBS_TOKENS
          )
      logging.info(ct.color('LLM response:\n%s'), response)
      obs = response[:LLM_LENGTH_OBS_CHARS]

      obs = info_prefix + '\n' + obs

      obs_str = 'Observation (speaker={:d}:{:s}):\n{:s}'.format(
          player, state.names[player], obs)

      ct.reset()

      return obs_str


class BaseChatGame(pyspiel.Game):
  """Base Chat game."""

  # pylint:disable=dangerous-default-value
  def __init__(
      self,
      params: Dict[str, Any] = DEFAULT_PARAMS,
  ):
    """Constructor.
    
    BaseChatGame is meant to be inherited from. Do not call its init directly.

    Args:
      params: dict, parameter dict with the following keys

        num_distinct_actions- int, # of actions at each info set
        num_llm_seeds- int, # of seeds to use for generating LLM response
        num_players- int, # of speakers (action: recipient) on the message chain
        players- int, # of speakers (action: recipient) on the message chain
          OPTIONAL. ONLY USED FOR INTERNAL OPEN_SPIEL TESTING!
        min_utility- float, minimum utility any player can attain
        max_utility- float, maximum utility any player can attain
        num_max_replies- int, total # of messages each player can send in an
          episode
    """
    if 'silence_logging' in params and params['silence_logging']:
      logging.set_verbosity(logging.ERROR)  # silence internal game logging
    self._num_distinct_actions = params['num_distinct_actions']
    if params['players'] > 0:
      logging.warning('Only meant for open_spiel testing!')
      num_players = params['players']
      self._num_players = num_players
    else:
      self._num_players = params['num_players']
    self._num_llm_seeds = params['num_llm_seeds']
    self._min_utility = params['min_utility']
    self._max_utility = params['max_utility']
    self._num_max_replies = params['num_max_replies']
    if params['num_max_replies'] > MAX_NUM_REPLIES:
      raise ValueError(
          f'num_max_replies ({self._num_max_replies}) exceeds ' +
          f'MAX_NUM_REPLIES ({MAX_NUM_REPLIES})')

    self._max_game_length = self._num_max_replies * self._num_players

    self._game_info = pyspiel.GameInfo(
        num_distinct_actions=self._num_distinct_actions,
        max_chance_outcomes=self._num_llm_seeds,
        num_players=self._num_players,
        min_utility=self._min_utility,
        max_utility=self._max_utility,
        max_game_length=self._max_game_length)

  def _load_chat_game(self,
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
                      examples_prompt_actions: Union[OrderedDict[str,
                                                                 List[str]],
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
      initial_scenario: Scenario item representing an initial message

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
    self._obs = observations
    self._vectorize = vectorize

    self._header = header

    self._payoffs = payoffs
    self._aggregate_payoffs = aggregate_payoffs
    self._max_score = aggregate_payoffs([p.max for p in payoffs])
    self._reward_type = REWARD_MODEL

    self._given_names = given_names
    self._given_llm_seeds = given_llm_seeds
    self._given_prompt_actions = given_prompt_actions
    self._given_private_info = given_private_info
    self._initial_scenario = initial_scenario

    self._num_names = max(num_names, self._num_players)
    self._num_prompt_actions = num_prompt_actions
    self._num_private_info = num_private_info

    self._examples_names = examples_names
    self._examples_prompt_actions = examples_prompt_actions
    self._examples_private_info = examples_private_info
    self._examples_scenarios = examples_scenarios

    self._llm_list_suffix = llm_list_suffix
    if llm_termination_prompt:
      query = llm_termination_prompt.query
      parsed = next(iter(string.Formatter().parse(query)), '')
      if not parsed or parsed[1] != 'msg':
        raise ValueError('Invalid llm_termination_prompt: ' +
                         f'{query}. It must include a ' +
                         'single formatting kwarg {msg}')
    self._llm_termination_prompt = llm_termination_prompt

    self._rnd = np.random.RandomState(seed)

    if self._given_names:
      if len(self._given_names) != self._num_players:
        raise ValueError('Number of given_names does not match num_players!')
      self._names = self._given_names
      self._names_gen = False
    else:
      retrieve_name = text.retrieve_alpha_block
      self._names = self.generate_prompts('name',
                                          self._examples_names,
                                          self._num_names,
                                          retrieve_name)
      logging.info(ct.color('Generated names:\n%s', logging_utils.YELLOW),
                   '\n'.join(self._names))  # pylint:disable=logging-too-many-args
      if len(self._names) < self._num_players:
        raise ValueError(f'Generated too few names! {len(self._names)} < ' +
                         f'{self._num_players}.')
      self._names_gen = True

    if self._given_llm_seeds:
      if len(self._given_llm_seeds) != self._num_llm_seeds:
        raise ValueError('Number of given_llm_seeds does not match ' +
                         'num_llm_seeds!')
      self._llm_seeds = self._given_llm_seeds
      self._llm_seeds_gen = False
    else:
      self._llm_seeds = list(self._rnd.randint(MIN_RND_SEED, MAX_RND_SEED,
                                               size=self._num_llm_seeds))
      logging.info(ct.color('Generated action seeds:%s', logging_utils.YELLOW),
                   self._llm_seeds)  # pylint:disable=logging-too-many-args
      self._llm_seeds_gen = True

    # loop over every action key in header action keys
    # if action key is in given prompt action, use it and overwrite
    # else, generate it
    def retrieve_prompt(llm_response: str) -> str:
      useless_chars = (' ', '\n')
      special_chars = ITEM_PREFIX
      for char in useless_chars:
        special_chars = special_chars.strip(char)
      special_chars = tuple(special_chars)
      return text.retrieve_special_char_block(llm_response,
                                              special_chars=special_chars,
                                              useless_chars=useless_chars)

    prompt_action_lists = []
    if not self._header.action_keys:
      self._num_prompt_actions = tuple([])
    for i, action_key in enumerate(self._header.action_keys):
      if (self._given_prompt_actions and
          action_key in self._given_prompt_actions):
        action_list = self._given_prompt_actions[action_key]
        if len(action_list) != self._num_prompt_actions[i]:
          logging.info(ct.color(f'Overwriting num_prompt_actions[{i}]=' +
                                f'{self._num_prompt_actions[i]} to reflect ' +
                                f'given len-{len(action_list)} prompt ' +
                                f'action list for action_key={action_key}.',
                                color=logging_utils.YELLOW))
          if isinstance(self._num_prompt_actions, tuple):
            self._num_prompt_actions = list(self._num_prompt_actions)
          self._num_prompt_actions[i] = len(action_list)
      else:
        examples = self._examples_prompt_actions[action_key]
        action_list = self.generate_prompts(action_key,
                                            examples,
                                            self._num_prompt_actions[i],
                                            retrieve_prompt)
        logging.info(ct.color(
            'Generated prompt actions for action key = %s:\n%s',
            color=logging_utils.YELLOW),
                     action_key, '\n-----\n'.join(action_list))
      prompt_action_lists.append(action_list)
    self._prompt_actions = collections.OrderedDict(zip(self._header.action_keys,
                                                       prompt_action_lists))
    if isinstance(self._num_prompt_actions, list):
      self._num_prompt_actions = tuple(self._num_prompt_actions)

    if (self._initial_scenario
        and self._given_private_info
        and tuple(self._given_private_info.keys()) != self._header.info_keys):
      raise ValueError('Must define private info for each player if setting' +
                       ' an initial scenario.')

    private_info_lists = []
    if not self._header.info_keys:
      self._num_private_info = tuple([])
    for i, info_key in enumerate(self._header.info_keys):
      if self._given_private_info and info_key in self._given_private_info:
        info_list = self._given_private_info[info_key]
        if self._initial_scenario:
          if len(info_list) < self._num_players:
            raise ValueError('Must define at least a single private info for ' +
                             'each player if setting an initial scenario. ' +
                             f'Num_players={self._num_players} but only given' +
                             f' len-{len(info_list)} private info list for ' +
                             f'info_key={info_key}.')
          else:
            info_list = info_list[:self._num_players]
        if len(info_list) != self._num_private_info[i]:
          logging.info(ct.color(f'Overwriting num_private_info[{i}]=' +
                                f'{self._num_private_info[i]} to reflect ' +
                                f'given len-{len(info_list)} private info ' +
                                f'list for info_key={info_key}.',
                                color=logging_utils.YELLOW))
          if isinstance(self._num_private_info, tuple):
            self._num_private_info = list(self._num_private_info)
          self._num_private_info[i] = len(info_list)
      else:
        examples = self._examples_private_info[info_key]
        info_list = self.generate_prompts(info_key,
                                          examples,
                                          self._num_private_info[i],
                                          retrieve_prompt)
        logging.info(ct.color('Generated private info for info key = %s:\n%s',
                              color=logging_utils.YELLOW),
                     info_key, '\n-----\n'.join(info_list))
      private_info_lists.append(info_list)
    self._private_info = collections.OrderedDict(zip(self._header.info_keys,
                                                     private_info_lists))
    if isinstance(self._num_private_info, list):
      self._num_private_info = tuple(self._num_private_info)

    if self._examples_scenarios:
      self._meta_query = self._build_meta_query(self._examples_scenarios)
    else:
      self._meta_query = None

    if self._initial_scenario:
      valid = self._initial_scenario_is_valid(self._initial_scenario)
      assert valid, ('Scenario does not match given game spec (names, actions' +
                     ', info, ...')
      self._initial_scenario = self._initial_scenario
    else:
      self._initial_scenario = None

    self._num_actions = (
        self._num_players,
    ) + tuple(self._num_prompt_actions)

    na = int(np.prod(self._num_actions))
    if na != self._num_distinct_actions:
      raise ValueError(f'Size of prompt action space ({na}) does not match ' +
                       f'num_distinct_actions ({self._num_distinct_actions})!')

  def _generate_response(self, prompt: str, seed: int,
                         num_output_tokens: Union[int, None] = None) -> str:
    """Returns LLM generated string given prompt and seed."""
    return ''

  def _generate_bool(self, prompt: str, seed: int) -> bool:
    """Returns LLM generated boolean given prompt and seed."""
    return False

  def _build_meta_query(self, scenarios=List[Tuple]) -> str:
    """Build prompt with several scenarios for generating new scenarios."""
    wrapped_scenarios = []
    for s in scenarios:
      scenario_header_unformatted = self._header.w_opts + s.msg
      s_asdict = dataclasses.asdict(s)
      scenario_header = scenario_header_unformatted.format(**s_asdict,
                                                           others=ALL_PLAYERS)
      wrapped_scenarios.append(scenario_header)
    return ''.join(wrapped_scenarios)

  def _initial_scenario_is_valid(self, scenario: Any) -> bool:
    """Check all components of scenario are well defined and return bool."""
    fields = list(scenario.__dataclass_fields__.keys())

    req_fields = ['sender', 'receiver'] + list(self._header.action_keys)
    req_fields += list(self._header.info_keys)
    valid_fields = True
    for req_field in req_fields:
      valid_fields = (valid_fields and req_field in fields)

    if not valid_fields:
      raise ValueError(f'Scenario must define required fields: {req_fields}. ' +
                       f'Found fields: {fields}')

    valid_players = (scenario.sender in self._names
                     and scenario.receiver in self._names + [ALL_PLAYERS])

    scenario_dict = dataclasses.asdict(scenario)

    valid_actions = True
    for key in self._header.action_keys:
      valid_actions = (valid_actions and
                       key in scenario_dict and
                       scenario_dict[key] in self._prompt_actions[key])

    valid_info = True
    for key in self._header.info_keys:
      # private_info[key][i] is unique to player i
      # initial scenario contains player 0's private info and must match the
      # first item in the list of private information provided
      valid_info = (valid_info and
                    key in scenario_dict and
                    scenario_dict[key] == self._private_info[key][0])

    valid = valid_players and valid_actions and valid_info

    return valid

  def generate_prompts(self, key, examples, num_prompts,
                       retrieve_prompt: Callable[[str], str]) -> List[str]:
    """Generates a list of distinct prompts from an initial list.

    Args:
      key: str, (descriptive) name of prompt type
      examples: list of str, example prompts to seed llm
      num_prompts: int, number of distinct prompts to generate
      retrieve_prompt: function to retrieve example from string

    Returns:
      prompts: list of strings
    """
    ct.set_color(logging_utils.CYAN)

    answers = set()
    num_gen = LLM_LIST_GEN_ATTEMPTS
    prompt = ['#### INSTRUCTIONS #####',
              'Given a list of items from a given category, continue the list' +
              ' and generate an additional item from the same category. The ' +
              f'category is {key}s. Use `{ITEM_PREFIX}` to denote separate ' +
              'items.']
    prompt = '\n'.join(text.wrap(prompt)) + '\n'
    prompt += ('Input:\n' + ITEM_PREFIX +
               ('\n' + ITEM_PREFIX).join(examples) + '\n' +
               self._llm_list_suffix)
    logging.info(ct.color('Generating list of distinct prompts...'))
    logging.info(ct.color('Example prompt:\n%s'), prompt)
    for seed in self._rnd.randint(MIN_RND_SEED, MAX_RND_SEED, size=num_gen):
      logging.info(ct.color('Generating %s (seed=%s)'), key, seed)
      response = self.generate_response(
          prompt=prompt,
          seed=seed,
          num_output_tokens=LLM_LENGTH_LIST_OF_WORDS_TOKENS
          )
      logging.info(ct.color('LLM response\n%s'), response)
      answer = retrieve_prompt(response)
      if answer and answer not in answers:
        answers.add(answer)
      if len(answers) >= num_prompts:
        return list(answers)
    num_distinct = len(answers)
    if len(answers) < num_prompts:
      logging.warning(ct.color(
          'Only %d distinct prompts generated for %d desired:\n%s.'),
                      num_distinct, num_prompts, answers)

    ct.reset()

    return list(answers)

  def generate_scenario(self) -> Tuple[List[str],
                                       OrderedDict[str, List[str]],
                                       Any]:
    """Generates a new game config from examples.
    
    Returns:
      given_names: list of str
      given_private_info: OrderedDict(str: list of str)
      initial_scenario(msg, sender, receiver, **private_info, **prompt_actions)
    """
    player_names = self._rnd.choice(self._names,
                                    size=self._num_players,
                                    replace=False)
    sender, receiver = player_names[:2]
    if self._num_players > 2:
      others = ', '.join(player_names[2:])
    else:
      others = ''

    pa_lists = self._prompt_actions.values()
    prompt_action_vals = [self._rnd.choice(pa_list) for pa_list in pa_lists]
    prompt_actions_header = collections.OrderedDict(zip(
        self._header.action_keys, prompt_action_vals))

    pi_lists = self._private_info.values()
    private_info_vals = [
        self._rnd.choice(pi_list, size=self._num_players)
        for pi_list in pi_lists
    ]
    private_info = collections.OrderedDict(zip(self._header.info_keys,
                                               private_info_vals))
    private_info_vals_player_0 = [piv[0] for piv in private_info_vals]
    private_info_header = collections.OrderedDict(zip(
        self._header.info_keys, private_info_vals_player_0))

    opts = prompt_actions_header
    opts.update(private_info_header)

    # scenarios are generated drawing from a fixed set of personalities
    header = self._header.w_opts.format(sender=sender,
                                        receiver=receiver,
                                        others=others,
                                        **opts)

    # generate a random scenario
    # need to generate new scenario with specific players (i.e. names). Can
    # 1) try to generate multiple scenarios at once and parse output
    # 2) generate a single scenario by varying the LLM seed
    # 3) can rely on the randomness in names and private info to induce new
    #    scenarios
    # we are currently going with option 3)
    logging.info('Generating initial scenario...')
    logging.info('Scenario prompt:\n%s', self._meta_query + header)
    response = self.generate_response(
        prompt=self._meta_query + header,
        seed=DEFAULT_LLM_SEED,
        num_output_tokens=LLM_LENGTH_MESSAGE_TOKENS
        )
    response = response[:LLM_LENGTH_MESSAGE_CHARS]
    logging.info('LLM response:\n%s', response)
    examples = []
    ptr = 0
    i = 0
    augmented_response = header + response
    while ptr < len(augmented_response):
      generated_example = self._header.strip_msg(augmented_response[ptr:],
                                                 sender)
      if not generated_example:
        break
      ptr += len(generated_example)
      generated_example = generated_example.strip('\n')
      logging.info('*Generated Example %d:\n%s', i, generated_example)
      i += 1
      examples.append(generated_example)
    # grab first generated scenario
    scenario_prompt = examples[0]
    logging.info('Example 0 selected')
    actions = collections.OrderedDict(zip(['player_names'],
                                          [player_names]))
    actions.update(self._prompt_actions)

    given_names = player_names
    given_private_info = private_info
    scenario_class = self._examples_scenarios[0].__class__
    initial_scenario = scenario_class(msg=scenario_prompt,
                                      sender=sender,
                                      receiver=receiver,
                                      **opts)

    return (given_names, given_private_info, initial_scenario)

  def new_initial_state_specs(self) -> Tuple[OrderedDict[str, List[str]],
                                             List[int],
                                             str,
                                             OrderedDict[str, List[str]]]:
    """Generates a new dialogue game.
    
    Returns:
      ChatGameState (see ChatGameState class)
    """
    if self._initial_scenario:
      names = self._names
      private_info = self._private_info
      scenario = self._initial_scenario
    else:
      names, private_info, scenario = self.generate_scenario()

    scenario_prompt_unformatted = self._header.plain + scenario.msg
    scenario_prompt = scenario_prompt_unformatted.format(
        sender=scenario.sender,
        receiver=scenario.receiver,
        others=ALL_PLAYERS)
    actions = collections.OrderedDict(zip(['player_names'], [names]))
    actions.update(self._prompt_actions)

    return (actions, self._llm_seeds, scenario_prompt, private_info)

  @property
  def game_info(self) -> pyspiel.GameInfo:
    return self._game_info

  @property
  def obs(self) -> List[observation_utils.Observation]:
    return self._obs

  @property
  def vectorize(self) -> Any:
    return self._vectorize

  @property
  def header(self) -> header_utils.Header:
    return self._header

  @property
  def payoffs(self) -> List[payoff_utils.Payoff]:
    return self._payoffs

  @property
  def aggregate_payoffs(self) -> Callable[[List[int]], float]:
    return self._aggregate_payoffs

  @property
  def reward_type(self) -> pyspiel.GameType.RewardModel:
    return self._reward_type

  @property
  def rnd(self) -> np.random.RandomState:
    return self._rnd

  @property
  def llm_termination_prompt(self) -> Union[term_utils.Termination, None]:
    return self._llm_termination_prompt

  @property
  def llm_seeds(self) -> List[int]:
    return self._llm_seeds

  @property
  def num_llm_seeds(self) -> int:
    return self._num_llm_seeds

  @property
  def given_prompt_actions(self) -> Union[OrderedDict[str, List[str]], None]:
    return self._given_prompt_actions
