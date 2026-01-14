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

"""Utils for running tests."""

import dataclasses
import enum

from typing import List

import numpy as np

from open_spiel.python.games.chat_games.envs.comm_substrates import emails


class TestLLM(enum.Enum):
  MOCK = 0


@dataclasses.dataclass(frozen=True)
class MockScore:
  logprob: float


class MockModel():
  """Mock LLM model."""

  def __init__(self, name):
    self.name = name


class MockResponse():
  """Mock LLM response."""

  def __init__(self, text):
    self.text = text


class MockClient():
  """Mock LLM client."""

  def __init__(self):
    # for cycling through mock response options
    self._idxs = {'names': 0,
                  'tones': 0,
                  'examples': 0}

  def sample(self, model: str, length: int, seed: int, prompt: str
             ) -> MockResponse:
    """Returns string responses according to fixed prompt styles."""
    del model, length, seed
    prompt_lower = prompt.lower()
    if 'names' in prompt_lower:
      dummy_names = ['Suzy', 'Bob', 'Alice', 'Doug', 'Arun', 'Maria', 'Zhang']
      dummy_name = dummy_names[self._idxs['names']]
      self._idxs['names'] = (self._idxs['names'] + 1) % len(dummy_names)
      return MockResponse(dummy_name + '\n')
    elif 'tones' in prompt_lower:
      dummy_tones = ['Happy', 'Sad', 'Angry']
      dummy_tone = dummy_tones[self._idxs['tones']]
      self._idxs['tones'] = (self._idxs['tones'] + 1) % len(dummy_tones)
      return MockResponse(dummy_tone + '\n')
    elif 'list of items' in prompt_lower:
      num_examples = 10
      dummy_examples = [f'Example-{i}' for i in range(num_examples)]
      dummy_example = dummy_examples[self._idxs['examples']]
      self._idxs['examples'] = (self._idxs['examples'] + 1) % num_examples
      return MockResponse(dummy_example + '\n')
    elif 'score' in prompt_lower or 'value' in prompt_lower:
      return MockResponse('5\n')
    elif 'summary' in prompt_lower:
      return MockResponse('This is a summary of the dialogue. We are happy.\n')
    elif emails.BLOCK_OPT in prompt:
      return MockResponse('\nThat all sounds good to me.\n')
    else:
      raise ValueError('Prompt not recognized!\n\n' + prompt)

  def score(self, model: str, prompt: str) -> List[MockScore]:
    del model, prompt
    return [MockScore(logprob=-1)]

  def list_models(self) -> List[MockModel]:
    dummy_models = ['dummy_model']
    models = [MockModel(model_name) for model_name in dummy_models]
    return models


class MockLLM():
  """Mock LLM."""

  def __init__(self):
    self.client = MockClient()
    self.model = 'dummy_model'

  def generate_response(self, prompt: str, seed: int,
                        num_output_tokens: int) -> str:
    response = self.client.sample(
        model=self.model,
        length=num_output_tokens,
        seed=seed,
        prompt=prompt
        )
    return response.text

  def generate_bool(self, prompt: str, seed: int) -> bool:
    del seed
    score_true = self.client.score(model=self.model, prompt=prompt + 'Yes')
    score_false = self.client.score(model=self.model, prompt=prompt + 'No')
    if score_true > score_false:
      return True
    else:
      return False


class MockTokenizer():
  """Mock Tokenizer."""

  def to_int(self, text: str) -> np.ndarray:
    return np.zeros(len(text), dtype=np.int32)


class MockVectorizer():
  """Mock Vectorizer."""

  def __init__(self):
    self.tokenizer = MockTokenizer()

  def vectorize(self, text: str, obs_size: int) -> np.ndarray:
    observation = self.tokenizer.to_int(text)[:obs_size]
    num_pad = max(0, obs_size - observation.size)
    observation = np.pad(observation, (0, num_pad))
    return observation
