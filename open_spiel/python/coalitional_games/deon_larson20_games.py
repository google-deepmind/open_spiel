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

"""Games from D'Eon and Larson '2020.

Testing Axioms Against Human Reward Divisions in Cooperative Games.
https://www.ifaamas.org/Proceedings/aamas2020/pdfs/p312.pdf
"""

from open_spiel.python.coalitional_games import basic_games


SHAPLEY_VALUES = {
    # Experiment 1
    "1-Worse-Solo": [25, 25, 10],
    "1-Worse-Both": [25, 25, 10],
    "1-Worse-Pair": [25, 25, 10],
    "1-Better-Solo": [30, 15, 15],
    "1-Better-Both": [30, 15, 15],
    "1-Better-Pair": [30, 15, 15],
    "Distinct-Solo": [30, 20, 10],
    "Distinct-Both": [30, 20, 10],
    "Distinct-Pair": [30, 20, 10],
    "Additive": [30, 20, 10],
    # Experiment 2
    "1-Worse-Zeros2": [25, 25, 10],
    "1-Worse-Zeros5": [25, 25, 10],
    "1-Worse-Zeros10": [25, 25, 10],
    "1-Worse-Sum30": [25, 25, 10],
    "1-Worse-Sum45": [25, 25, 10],
    "1-Worse-Sum60": [25, 25, 10],
    "1-Better-Zeros2": [30, 15, 15],
    "1-Better-Zeros5": [30, 15, 15],
    "1-Better-Zeros10": [30, 15, 15],
    "1-Better-Sum30": [30, 15, 15],
    "1-Better-Sum45": [30, 15, 15],
    "1-Better-Sum60": [30, 15, 15],
    "1-Null-Zeros": [40, 20, 0],
    "1-Null-Sum40": [40, 20, 0],
    "1-Null-Sum50": [40, 20, 0],
    "1-Null-Sum60": [40, 20, 0],
}


def make_game(name: str) -> basic_games.TabularGame:
  """Returns a game from D'Eon and Larson '2020.

  Testing Axioms Against Human Reward Divisions in Cooperative Games.
  https://www.ifaamas.org/Proceedings/aamas2020/pdfs/p312.pdf

  Args:
    name: the name of the game, as in Table 1 of the paper.

  Raises:
    RuntimeError: when the name of the game is not known.
  """

  if name == "1-Worse-Solo":
    # A  B  C  AB AC BC
    # 40 40 10 60 60 60
    return basic_games.TabularGame({
        (0, 0, 0): 0.0,
        (1, 0, 0): 40.0,
        (0, 1, 0): 40.0,
        (0, 0, 1): 10.0,
        (1, 1, 0): 60.0,
        (1, 0, 1): 60.0,
        (0, 1, 1): 60.0,
        (1, 1, 1): 60.0,
    })
  elif name == "1-Worse-Both":
    # A  B  C  AB AC BC
    # 15 15 0  45 30 30
    return basic_games.TabularGame({
        (0, 0, 0): 0.0,
        (1, 0, 0): 15.0,
        (0, 1, 0): 15.0,
        (0, 0, 1): 0.0,
        (1, 1, 0): 45.0,
        (1, 0, 1): 30.0,
        (0, 1, 1): 30.0,
        (1, 1, 1): 60.0,
    })
  elif name == "1-Worse-Pair":
    # A  B  C  AB AC BC
    # 0  0  0  45 15 15
    return basic_games.TabularGame({
        (0, 0, 0): 0.0,
        (1, 0, 0): 0.0,
        (0, 1, 0): 0.0,
        (0, 0, 1): 0.0,
        (1, 1, 0): 45.0,
        (1, 0, 1): 15.0,
        (0, 1, 1): 15.0,
        (1, 1, 1): 60.0,
    })
  elif name == "1-Better-Solo":
    # A  B  C  AB AC BC
    # 40 10 10 60 60 60
    return basic_games.TabularGame({
        (0, 0, 0): 0.0,
        (1, 0, 0): 40.0,
        (0, 1, 0): 10.0,
        (0, 0, 1): 10.0,
        (1, 1, 0): 60.0,
        (1, 0, 1): 60.0,
        (0, 1, 1): 60.0,
        (1, 1, 1): 60.0,
    })
  elif name == "1-Better-Both":
    # A  B  C  AB AC BC
    # 15 0  0  45 45 30
    return basic_games.TabularGame({
        (0, 0, 0): 0.0,
        (1, 0, 0): 15.0,
        (0, 1, 0): 0.0,
        (0, 0, 1): 0.0,
        (1, 1, 0): 45.0,
        (1, 0, 1): 45.0,
        (0, 1, 1): 30.0,
        (1, 1, 1): 60.0,
    })
  elif name == "1-Better-Pair":
    # A  B  C  AB AC BC
    # 0  0  0  45 45 15
    return basic_games.TabularGame({
        (0, 0, 0): 0.0,
        (1, 0, 0): 0.0,
        (0, 1, 0): 0.0,
        (0, 0, 1): 0.0,
        (1, 1, 0): 45.0,
        (1, 0, 1): 45.0,
        (0, 1, 1): 15.0,
        (1, 1, 1): 60.0,
    })
  elif name == "Distinct-Solo":
    # A  B  C  AB AC BC
    # 40 20 0  60 60 60
    return basic_games.TabularGame({
        (0, 0, 0): 0.0,
        (1, 0, 0): 40.0,
        (0, 1, 0): 20.0,
        (0, 0, 1): 0.0,
        (1, 1, 0): 60.0,
        (1, 0, 1): 60.0,
        (0, 1, 1): 60.0,
        (1, 1, 1): 60.0,
    })
  elif name == "Distinct-Both":
    # A  B  C  AB AC BC
    # 20 10 0  60 50 40
    return basic_games.TabularGame({
        (0, 0, 0): 0.0,
        (1, 0, 0): 20.0,
        (0, 1, 0): 10.0,
        (0, 0, 1): 0.0,
        (1, 1, 0): 60.0,
        (1, 0, 1): 50.0,
        (0, 1, 1): 40.0,
        (1, 1, 1): 60.0,
    })
  elif name == "Distinct-Pair":
    # A  B  C  AB AC BC
    # 0  0  0  60 40 20
    return basic_games.TabularGame({
        (0, 0, 0): 0.0,
        (1, 0, 0): 0.0,
        (0, 1, 0): 0.0,
        (0, 0, 1): 0.0,
        (1, 1, 0): 60.0,
        (1, 0, 1): 40.0,
        (0, 1, 1): 20.0,
        (1, 1, 1): 60.0,
    })
  elif name == "Additive":
    # A  B  C  AB AC BC
    # 30 20 10 50 40 30
    return basic_games.TabularGame({
        (0, 0, 0): 0.0,
        (1, 0, 0): 30.0,
        (0, 1, 0): 20.0,
        (0, 0, 1): 10.0,
        (1, 1, 0): 50.0,
        (1, 0, 1): 40.0,
        (0, 1, 1): 30.0,
        (1, 1, 1): 60.0,
    })
  elif name == "1-Worse-Zeros2":
    # A  B  C  AB AC BC
    # 2  0  0  40 10 12
    return basic_games.TabularGame({
        (0, 0, 0): 0.0,
        (1, 0, 0): 2.0,
        (0, 1, 0): 0.0,
        (0, 0, 1): 0.0,
        (1, 1, 0): 40.0,
        (1, 0, 1): 10.0,
        (0, 1, 1): 12.0,
        (1, 1, 1): 60.0,
    })
  elif name == "1-Worse-Zeros5":
    # A  B  C  AB AC BC
    # 5  0  0  40 10 15
    return basic_games.TabularGame({
        (0, 0, 0): 0.0,
        (1, 0, 0): 5.0,
        (0, 1, 0): 0.0,
        (0, 0, 1): 0.0,
        (1, 1, 0): 40.0,
        (1, 0, 1): 10.0,
        (0, 1, 1): 15.0,
        (1, 1, 1): 60.0,
    })
  elif name == "1-Worse-Zeros10":
    # A  B  C  AB AC BC
    # 10 0  0  40 10 20
    return basic_games.TabularGame({
        (0, 0, 0): 0.0,
        (1, 0, 0): 10.0,
        (0, 1, 0): 0.0,
        (0, 0, 1): 0.0,
        (1, 1, 0): 40.0,
        (1, 0, 1): 10.0,
        (0, 1, 1): 20.0,
        (1, 1, 1): 60.0,
    })
  elif name == "1-Worse-Sum30":
    # A  B  C  AB AC BC
    # 20 5  5  60 30 45
    return basic_games.TabularGame({
        (0, 0, 0): 0.0,
        (1, 0, 0): 20.0,
        (0, 1, 0): 5.0,
        (0, 0, 1): 5.0,
        (1, 1, 0): 60.0,
        (1, 0, 1): 30.0,
        (0, 1, 1): 45.0,
        (1, 1, 1): 60.0,
    })
  elif name == "1-Worse-Sum45":
    # A  B  C  AB AC BC
    # 25 10 10 60 30 45
    return basic_games.TabularGame({
        (0, 0, 0): 0.0,
        (1, 0, 0): 25.0,
        (0, 1, 0): 10.0,
        (0, 0, 1): 10.0,
        (1, 1, 0): 60.0,
        (1, 0, 1): 30.0,
        (0, 1, 1): 45.0,
        (1, 1, 1): 60.0,
    })
  elif name == "1-Worse-Sum60":
    # A  B  C  AB AC BC
    # 30 15 15 60 30 45
    return basic_games.TabularGame({
        (0, 0, 0): 0.0,
        (1, 0, 0): 30.0,
        (0, 1, 0): 15.0,
        (0, 0, 1): 15.0,
        (1, 1, 0): 60.0,
        (1, 0, 1): 30.0,
        (0, 1, 1): 45.0,
        (1, 1, 1): 60.0,
    })
  elif name == "1-Better-Zeros2":
    # A  B  C  AB AC BC
    # 2  2  0  38 40 10
    return basic_games.TabularGame({
        (0, 0, 0): 0.0,
        (1, 0, 0): 2.0,
        (0, 1, 0): 2.0,
        (0, 0, 1): 0.0,
        (1, 1, 0): 38.0,
        (1, 0, 1): 40.0,
        (0, 1, 1): 10.0,
        (1, 1, 1): 60.0,
    })
  elif name == "1-Better-Zeros5":
    # A  B  C  AB AC BC
    # 5  5  0  35 40 10
    return basic_games.TabularGame({
        (0, 0, 0): 0.0,
        (1, 0, 0): 5.0,
        (0, 1, 0): 5.0,
        (0, 0, 1): 0.0,
        (1, 1, 0): 35.0,
        (1, 0, 1): 40.0,
        (0, 1, 1): 10.0,
        (1, 1, 1): 60.0,
    })
  elif name == "1-Better-Zeros10":
    # A  B  C  AB AC BC
    # 10 10 0  30 40 10
    return basic_games.TabularGame({
        (0, 0, 0): 0.0,
        (1, 0, 0): 10.0,
        (0, 1, 0): 10.0,
        (0, 0, 1): 0.0,
        (1, 1, 0): 30.0,
        (1, 0, 1): 40.0,
        (0, 1, 1): 10.0,
        (1, 1, 1): 60.0,
    })
  elif name == "1-Better-Sum30":
    # A  B  C  AB AC BC
    # 15 15 0  45 60 30
    return basic_games.TabularGame({
        (0, 0, 0): 0.0,
        (1, 0, 0): 15.0,
        (0, 1, 0): 15.0,
        (0, 0, 1): 0.0,
        (1, 1, 0): 45.0,
        (1, 0, 1): 60.0,
        (0, 1, 1): 30.0,
        (1, 1, 1): 60.0,
    })
  elif name == "1-Better-Sum45":
    # A  B  C  AB AC BC
    # 20 20 5  45 60 30
    return basic_games.TabularGame({
        (0, 0, 0): 0.0,
        (1, 0, 0): 20.0,
        (0, 1, 0): 20.0,
        (0, 0, 1): 5.0,
        (1, 1, 0): 45.0,
        (1, 0, 1): 60.0,
        (0, 1, 1): 30.0,
        (1, 1, 1): 60.0,
    })
  elif name == "1-Better-Sum60":
    # A  B  C  AB AC BC
    # 25 25 10 45 60 30
    return basic_games.TabularGame({
        (0, 0, 0): 0.0,
        (1, 0, 0): 25.0,
        (0, 1, 0): 25.0,
        (0, 0, 1): 10.0,
        (1, 1, 0): 45.0,
        (1, 0, 1): 60.0,
        (0, 1, 1): 30.0,
        (1, 1, 1): 60.0,
    })
  elif name == "1-Null-Zeros":
    # A  B  C  AB AC BC
    # 20 0  0  60 20 0
    return basic_games.TabularGame({
        (0, 0, 0): 0.0,
        (1, 0, 0): 20.0,
        (0, 1, 0): 0.0,
        (0, 0, 1): 0.0,
        (1, 1, 0): 60.0,
        (1, 0, 1): 20.0,
        (0, 1, 1): 0.0,
        (1, 1, 1): 60.0,
    })
  elif name == "1-Null-Sum40":
    # A  B  C  AB AC BC
    # 30 10 0  60 30 10
    return basic_games.TabularGame({
        (0, 0, 0): 0.0,
        (1, 0, 0): 30.0,
        (0, 1, 0): 10.0,
        (0, 0, 1): 0.0,
        (1, 1, 0): 60.0,
        (1, 0, 1): 30.0,
        (0, 1, 1): 10.0,
        (1, 1, 1): 60.0,
    })
  elif name == "1-Null-Sum50":
    # A  B  C  AB AC BC
    # 35 15 0  60 35 15
    return basic_games.TabularGame({
        (0, 0, 0): 0.0,
        (1, 0, 0): 35.0,
        (0, 1, 0): 15.0,
        (0, 0, 1): 0.0,
        (1, 1, 0): 60.0,
        (1, 0, 1): 35.0,
        (0, 1, 1): 15.0,
        (1, 1, 1): 60.0,
    })
  elif name == "1-Null-Sum60":
    # A  B  C  AB AC BC
    # 40 20 0  60 40 20
    return basic_games.TabularGame({
        (0, 0, 0): 0.0,
        (1, 0, 0): 40.0,
        (0, 1, 0): 20.0,
        (0, 0, 1): 0.0,
        (1, 1, 0): 60.0,
        (1, 0, 1): 40.0,
        (0, 1, 1): 20.0,
        (1, 1, 1): 60.0,
    })
  else:
    raise RuntimeError("unknown game")
