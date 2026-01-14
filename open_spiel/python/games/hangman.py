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

"""Hangman, implemented in Python."""

# pylint: disable=g-importing-member

from typing import Any, Mapping

from absl import logging
import numpy as np

from open_spiel.python.observation import IIGObserverForPublicInfoGame
import pyspiel


NUM_DISTINCT_ACTIONS = 26

CORRECT_GUESS_REWARD = 1.0
INCORRECT_GUESS_REWARD = -1.0
WIN_REWARD = 100.0
LOSS_REWARD = -100.0

_MAX_NUM_GUESSES = 26
_DEFAULT_MAX_NUM_INCORRECT_GUESSES = 6
_MIN_WORD_LENGTH = 4

_VALID_LETTERS = "abcdefghijklmnopqrstuvwxyz"
_SKIP_LETTERS = " "  # always show these and do not allow them as guesses
_UNREVEALED_CHARACTER = "_"

_DEFAULT_PARAMS = {
    "word_list_file": "",
    "max_num_incorrect_guesses": _DEFAULT_MAX_NUM_INCORRECT_GUESSES,
}

# Word list that gets used if no word list file is provided.
_DEFAULT_WORD_LIST = [
    "apple", "banana", "cat", "dog", "egg", "fish", "frog", "giraffe",
    "hippo", "jackal", "kangaroo", "lemon", "monkey", "narwhal", "orange",
    "penguin", "pineapple", "pizza", "quail", "rabbit", "snake", "squirrel",
    "tiger", "unicorn", "zebra"
]

_GAME_TYPE = pyspiel.GameType(
    short_name="python_hangman",
    long_name="Python Hangman",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=1,
    min_num_players=1,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification=_DEFAULT_PARAMS,
)


class HangmanGame(pyspiel.Game):
  """A Python version of the Hangman game."""

  _word_list: list[str]
  _word_list_len: int
  _longest_word: int

  def __init__(self, params: Mapping[str, Any]):
    self._max_num_incorrect_guesses = params.get("max_num_incorrect_guesses")
    game_info = pyspiel.GameInfo(
        num_distinct_actions=NUM_DISTINCT_ACTIONS,
        max_chance_outcomes=1,
        num_players=1,
        min_utility=(LOSS_REWARD +
                     self._max_num_incorrect_guesses * INCORRECT_GUESS_REWARD),
        max_utility=WIN_REWARD + _MAX_NUM_GUESSES * CORRECT_GUESS_REWARD,
        max_game_length=_MAX_NUM_GUESSES,
    )
    super().__init__(_GAME_TYPE, game_info, params or dict())
    word_list_file = params.get("word_list_file")
    self._word_list = []
    self._word_list_len = 0
    self._longest_word = 0
    if word_list_file:
      self._word_list, self._longest_word = _load_word_list(word_list_file)
      self._word_list_len = len(self._word_list)
    else:
      logging.warning("No word list file provided; using default word list.")
      self._word_list = _DEFAULT_WORD_LIST[:]
      self._word_list_len = len(self._word_list)
      self._longest_word = max([len(word) for word in self._word_list])
    assert self._word_list
    assert self._word_list_len > 0, "Must set a word list"
    assert self._longest_word > 0, "Longest word must be > 0"

  def new_initial_state(self, word: str | None = None):
    """Returns a state corresponding to the start of a game."""
    assert self._word_list_len > 0, "Must set a word list"
    return HangmanState(self, self._word_list, self._max_num_incorrect_guesses,
                        word)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    if (iig_obs_type is None) or (
        iig_obs_type.public_info and not iig_obs_type.perfect_recall
    ):
      return HangmanObserver(params, self._longest_word)
    else:
      return IIGObserverForPublicInfoGame(iig_obs_type, params)

  @property
  def word_list_len(self):
    return self._word_list_len


class HangmanState(pyspiel.State):
  """A python version of the Hangman state."""

  def __init__(self,
               game: HangmanGame,
               word_list: list[str],
               max_num_incorrect_guesses: int,
               word: str | None = None):
    """Constructor; should only be called by Game.new_initial_state.

    Arguments:
      game: The game object.
      word_list: list of possible words.
      max_num_incorrect_guesses: The maximum number of incorrect guesses
          allowed.
      word: The word to start the game with. If None, a random word is chosen
          via an initial chance node.
    """
    super().__init__(game)
    self._game = game
    self._word_list = word_list
    self._num_guesses = 0
    self._num_incorrect_guesses = 0
    self._max_num_incorrect_guesses = max_num_incorrect_guesses
    self._is_terminal = False
    self._reward = 0
    self._return = 0
    self._total_num_letters_revealed = 0
    self._word = word
    self._letters_guessed = None
    self._letters_revealed = None
    self._cur_player = pyspiel.PlayerId.CHANCE
    if word is not None:
      self._initialize_state(word)

  def _initialize_state(self, word: str):
    self._word = word
    self._letters_guessed = []
    self._letters_revealed = [_UNREVEALED_CHARACTER] * len(self._word)
    self._num_letters_to_guess = 0
    for i in range(len(self._word)):
      if self._word[i] in _SKIP_LETTERS:
        # Show the skip letters (i.e. spaces)
        self._letters_revealed[i] = self._word[i]
      else:
        self._num_letters_to_guess += 1
    assert self._num_letters_to_guess > 0
    self._cur_player = 0

  @property
  def word(self):
    return self._word

  @property
  def letters_guessed(self):
    return self._letters_guessed

  @property
  def letters_revealed(self):
    return self._letters_revealed

  @property
  def num_guesses(self):
    return self._num_guesses

  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    return pyspiel.PlayerId.TERMINAL if self._is_terminal else self._cur_player

  def chance_outcomes(self):
    """Returns the possible chance outcomes and their probabilities."""
    assert self.is_chance_node()
    p = 1.0 / self._game.word_list_len
    return [(o, p) for o in range(self._game.word_list_len)]

  def _legal_actions(self, player):
    """Returns a list of legal actions, sorted in ascending order."""
    assert self.current_player() == player
    if self.is_chance_node():
      actions = [o for o, _ in self.chance_outcomes()]
    else:
      actions = []
      for i, c in enumerate(_VALID_LETTERS):
        if c not in self._letters_guessed:
          actions.append(i)
    return actions

  def _apply_action(self, action):
    """Applies the specified action to the state."""
    if self.is_chance_node():
      assert self._word is None
      assert 0 <= action < len(self._word_list)
      word = self._word_list[action]
      self._initialize_state(word)
    else:
      assert self._letters_guessed is not None
      self._reward = 0
      # 1. Add the guessed letter to the list of letters guessed.
      letter = _VALID_LETTERS[action]
      assert letter not in self._letters_guessed
      self._letters_guessed.append(letter)
      # 2. Iterate over the letters in the word to see if any are revealed.
      num_letters_revealed = 0
      for i, c in enumerate(self._word):
        if self._letters_revealed[i] == _UNREVEALED_CHARACTER and c == letter:
          self._letters_revealed[i] = letter
          num_letters_revealed += 1
      # 3. Update the number of letters revealed and number of guesses.
      self._num_guesses += 1
      self._total_num_letters_revealed += num_letters_revealed
      # 4. Update the reward stepwise reward.
      if num_letters_revealed > 0:
        # Correct guess.
        self._reward = CORRECT_GUESS_REWARD
      else:
        self._num_incorrect_guesses += 1
        self._reward = INCORRECT_GUESS_REWARD
      # 5. Update terminal and bonus rewards (if terminal).
      if (self._num_guesses >= _MAX_NUM_GUESSES or
          self._num_incorrect_guesses >= self._max_num_incorrect_guesses or
          self._total_num_letters_revealed == self._num_letters_to_guess):
        self._is_terminal = True
        # Add the win/loss bonuses.
        if num_letters_revealed > 0:
          self._reward += WIN_REWARD
        else:
          self._reward += LOSS_REWARD
      # 6. Finally, update the return.
      self._return += self._reward

  def action_to_string(self, player, action):
    """Action -> string."""
    if player == pyspiel.PlayerId.CHANCE:
      return f"Chance outcome {action}: {self._word_list[action]}"
    else:
      return f"Guess letter {_VALID_LETTERS[action]}"

  def is_terminal(self):
    """Returns True if the game is over."""
    return self._is_terminal

  def rewards(self):
    return [self._reward]

  def returns(self):
    """Total reward for each player over the course of the game so far."""
    return [self._return]

  def __str__(self) -> str:
    """String for debug purposes. No particular semantics are required."""
    return _hangman_state_to_string(self, show_word=True)


class HangmanObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  def __init__(self, params, longest_word: int):
    """Initializes an empty observation tensor."""
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")
    # The observation should contain a 1-D tensor in `self.tensor` and a
    # dictionary of views onto the tensor, which may be of any shape.
    # Here the observation is indexed `(cell state, row, column)`.
    shape = (longest_word,)
    self.tensor = np.zeros(np.prod(shape), np.float32)
    self.dict = {"observation": np.reshape(self.tensor, shape)}

  def set_from(self, state, player):
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""

  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    del player
    return _hangman_state_to_string(state, show_word=False)


def _hangman_state_to_string(state: HangmanState, show_word: bool = True):
  if state.word:
    assert state.letters_revealed
    letters_revealed_str = "".join(state.letters_revealed)
    assert state.letters_guessed is not None
    letters_guessed_str = "".join(state.letters_guessed)
    word_line = f"Word:             {state.word}\n" if show_word else ""
    return (word_line + f"Letters Revealed: {letters_revealed_str}\n" +
                        f"Letters Guessed:  {letters_guessed_str}\n" +
                        f"Num guesses: {state.num_guesses}\n")
  else:
    return "Not started yet"


def _is_valid_word(word: str):
  if len(word) < _MIN_WORD_LENGTH:
    return False
  for l in word:
    if l not in _VALID_LETTERS and l not in _SKIP_LETTERS:
      return False
  return True


def _load_word_list(word_list_file: str):
  """Returns the word dict."""
  # Get the content of the named resource as a string.
  contents = pyspiel.read_contents_from_file(word_list_file, "r")
  lines = contents.split("\n")
  word_list = []
  max_word_length = 0
  for line in lines:
    if not line:
      continue
    line_len = len(line)
    if _is_valid_word(line):
      word_list.append(line)
      max_word_length = max(max_word_length, line_len)
  assert word_list
  return word_list, max_word_length


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, HangmanGame)
