// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/matrix_game.h"

// Some sample matrix games.

namespace open_spiel {

using matrix_game::MatrixGame;

// Matching Pennies.
namespace matching_pennies {
// Facts about the game
const GameType kGameType{
    /*short_name=*/"matrix_mp",
    /*long_name=*/"Matching Pennies",
    GameType::Dynamics::kSimultaneous,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kOneShot,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{}  // no parameters
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(
      new MatrixGame(kGameType, params, {"Heads", "Tails"}, {"Heads", "Tails"},
                     {1, -1, -1, 1}, {-1, 1, 1, -1}));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace matching_pennies

// Rock, Paper, Scissors.
namespace rock_paper_scissors {
// Facts about the game
const GameType kGameType{
    /*short_name=*/"matrix_rps",
    /*long_name=*/"Rock, Paper, Scissors",
    GameType::Dynamics::kSimultaneous,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kOneShot,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{}  // no parameters
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new MatrixGame(
      kGameType, params, {"Rock", "Paper", "Scissors"},
      {"Rock", "Paper", "Scissors"}, {0, -1, 1, 1, 0, -1, -1, 1, 0},
      {0, 1, -1, -1, 0, 1, 1, -1, 0}));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace rock_paper_scissors

// Rock, Paper, Scissors, Water: a variant of RPS by Martin Schmid which adds
// an action to both players that always gives, adding a pure equilibrium to the
// game.
namespace rock_paper_scissors_water {
// Facts about the game
const GameType kGameType{
    /*short_name=*/"matrix_rpsw",
    /*long_name=*/"Rock, Paper, Scissors, Water",
    GameType::Dynamics::kSimultaneous,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kOneShot,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{}  // no parameters
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(
      new MatrixGame(kGameType, params, {"Rock", "Paper", "Scissors", "Water"},
                     {"Rock", "Paper", "Scissors", "Water"},
                     {0, -1, 1, 0, 1, 0, -1, 0, -1, 1, 0, 0, 0, 0, 0, 0},
                     {0, 1, -1, 0, -1, 0, 1, 0, 1, -1, 0, 0, 0, 0, 0, 0}));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace rock_paper_scissors_water

// A general-sum variant of Rock, Paper, Scissors. Often used as a
// counter-example for certain learning dynamics, such as ficitions play.
// See Chapter 7 of (Shoham and Leyton-Brown, Multiagent Systems Algorithmic,
// Game-Theoretic, and Logical Foundations, 2009) for detailed examples.
namespace shapleys_game {
// Facts about the game
const GameType kGameType{
    /*short_name=*/"matrix_shapleys_game",
    /*long_name=*/"Shapley's Game",
    GameType::Dynamics::kSimultaneous,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kOneShot,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{}  // no parameters
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(
      new MatrixGame(kGameType, params, {"Rock", "Paper", "Scissors"},
                     {"Rock", "Paper", "Scissors"}, {0, 0, 1, 1, 0, 0, 0, 1, 0},
                     {0, 1, 0, 0, 0, 1, 1, 0, 0}));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace shapleys_game

// Prisoner's Dilemma.
namespace prisoners_dilemma {
const GameType kGameType{
    /*short_name=*/"matrix_pd",
    /*long_name=*/"Prisoner's Dilemma",
    GameType::Dynamics::kSimultaneous,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kOneShot,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{}  // no parameters
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(
      new MatrixGame(kGameType, params, {"Cooperate", "Defect"},
                     {"Cooperate", "Defect"}, {5, 0, 10, 1}, {5, 10, 0, 1}));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace prisoners_dilemma

// Stag Hunt.
namespace stag_hunt {
const GameType kGameType{
    /*short_name=*/"matrix_sh",
    /*long_name=*/"Stag Hunt",
    GameType::Dynamics::kSimultaneous,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kOneShot,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{}  // no parameters
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(
      new MatrixGame(kGameType, params, {"Stag", "Hare"}, {"Stag", "Hare"},
                     {2, 0, 1, 1}, {2, 1, 0, 1}));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace stag_hunt

// Coordination.
namespace coordination {
const GameType kGameType{
    /*short_name=*/"matrix_coordination",
    /*long_name=*/"Coordination",
    GameType::Dynamics::kSimultaneous,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kOneShot,
    GameType::Utility::kIdentical,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{}  // no parameters
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(
      new MatrixGame(kGameType, params, {"Left", "Right"}, {"Left", "Right"},
                     {1, 0, 0, 1}, {1, 0, 0, 1}));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace coordination

// Chicken-Dare game.
// As described in https://web.stanford.edu/~saberi/lecture6.pdf
namespace chicken_dare {
const GameType kGameType{
    /*short_name=*/"matrix_cd",
    /*long_name=*/"Chicken-Dare",
    GameType::Dynamics::kSimultaneous,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kOneShot,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{}  // no parameters
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(
      new MatrixGame(kGameType, params, {"Dare", "Chicken"},
                     {"Dare", "Chicken"}, {0, 4, 1, 3}, {0, 1, 4, 3}));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace chicken_dare

}  // namespace open_spiel
