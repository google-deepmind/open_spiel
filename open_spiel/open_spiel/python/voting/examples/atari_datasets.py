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

"""Helper functions for loading Atari data."""

import logging
from typing import Dict, List
import numpy as np


RAINBOW_TABLE5 = "atari_rainbow_table5.txt"
RAINBOW_TABLE6 = "atari_rainbow_table6.txt"
AGENT57_TABLE = "atari_agent57_table.txt"
MUESLI_TABLE11 = "atari_muesli_table11.txt"


class DataSet:
  """A DataSet container for Atari tables."""

  def __init__(
      self,
      agent_names: List[str],
      game_names: List[str],
      table_data: Dict[str, List[float]],
  ):
    self.agent_names = agent_names
    self.game_names = game_names
    self.table_data = table_data

  def get_column(self, agent_name: str) -> Dict[str, float]:
    column_dict = {}
    agent_idx = self.agent_names.index(agent_name)
    assert 0 <= agent_idx < len(self.agent_names)
    for game_name, scores in self.table_data.items():
      column_dict[game_name] = scores[agent_idx]
    return column_dict

  def delete_column(self, agent_name: str):
    agent_idx = self.agent_names.index(agent_name)
    assert 0 <= agent_idx < len(self.agent_names)
    del self.agent_names[agent_idx]
    for game_name in self.game_names:
      del self.table_data[game_name][agent_idx]

  def delete_game(self, game_name: str):
    assert game_name in self.game_names
    self.game_names.remove(game_name)
    del self.table_data[game_name]

  def add_column(self, column, agent_name):
    """Add a column.

    Args:
        column: a dictionary of game_name -> score,
        agent_name: name for the new agent.

    Note: beware! This can delete rows within this data set, in order to keep
    data complete, i.e. it deletes rows if you don't have this agent's score for
    that game.
    """
    self.agent_names.append(agent_name)
    game_names_copy = self.game_names[:]
    for game_name in game_names_copy:
      if game_name not in column:
        logging.warning("Warning: deleting game {%s}", game_name)
        self.delete_game(game_name)
      else:
        self.table_data[game_name].append(column[game_name])

  def to_task_by_agent_matrix(self) -> np.ndarray:
    num_tasks = len(self.game_names)
    num_agents = len(self.agent_names)
    mat = np.zeros(shape=(num_tasks, num_agents))
    i = 0
    for game_name in self.game_names:
      mat[i] = np.asarray(self.table_data[game_name])
      i += 1
    return mat


def parse_value(val_str: str) -> float:
  """Parse a numerical value from string, dropping ± part."""
  val_str = val_str.replace(",", "")
  val_str = val_str.split("±")[0]
  return float(val_str)


def parse_values(string_values_list: List[str]) -> List[float]:
  """Turn a list of strings into a list of floats."""
  return [parse_value(val) for val in string_values_list]


def delete_agent(dataset: DataSet, agent: str):
  idx = dataset.agent_names.index(agent)
  assert 0 <= idx < len(dataset.agent_names)
  del dataset.agent_names[idx]
  for key in dataset.table_data.keys():
    del dataset.table_data[key][idx]


def make_subset(dataset: DataSet, agent_subset: List[str]):
  for agent in dataset.agent_names:
    if agent not in agent_subset:
      delete_agent(dataset, agent)


def parse_atari_table(filename: str) -> DataSet:
  """Parse an Atari data file.

  The files are created by copy/paste from the papers.

  Args:
    filename: the file that contains the dataset.

  Returns:
      a DataSet object referring to the Atari data.
  """
  with open(filename, "r") as f:
    string_data = f.read()

  # First line is a comment
  # Second line format is column descriptions, e.g.:
  # "# game <agent1 name> <agent2 name> ..."
  # Rest of the lines are copy/paste from the paper tables.
  lines = string_data.split("\n")
  assert lines[1].startswith("# game ")
  agent_names = lines[1].split()[2:]
  num_agents = len(agent_names)
  game_names = []
  table_data = {}
  for i in range(2, len(lines)):
    if lines[i].strip():
      parts = lines[i].split()
      game_name = parts[0]
      game_names.append(game_name)
      str_scores = parts[1:]
      assert len(str_scores) == num_agents, f"Error line: {lines[i]}"
      scores = parse_values(str_scores)
      table_data[game_name] = scores
  return DataSet(agent_names, game_names, table_data)
