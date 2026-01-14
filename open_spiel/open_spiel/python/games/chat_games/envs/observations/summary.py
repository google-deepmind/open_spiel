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

"""Prompts useful for inducing LLM-summarization.
"""

from open_spiel.python.games.chat_games.envs.utils import text


prefix = ('You are an assistant designed to summarize conversational ' +
          'dialogue. Please take note of the most import events ' +
          'in the conversation. Especially take note if the final ' +
          'message includes a question. Provide your summary in 100 ' +
          'words or less. Please summarize the following dialogue.')
PREFIX = text.wrap([prefix])[0] + '\n\n'

POSTFIX = '\n\nSummary:\n'
