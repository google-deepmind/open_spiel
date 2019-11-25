// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include <unordered_map>

#include "open_spiel/algorithms/best_response.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/algorithms/cfr_br.h"
#include "open_spiel/algorithms/evaluate_bots.h"
#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/algorithms/matrix_game_utils.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/algorithms/trajectories.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"
#include "open_spiel/matrix_game.h"
#include "open_spiel/normal_form_game.h"
#include "open_spiel/policy.h"
#include "open_spiel/query.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"
#include "open_spiel/spiel_utils.h"
#include "pybind11/include/pybind11/pybind11.h"
#include "pybind11/include/pybind11/stl.h"

namespace open_spiel {
namespace {

using ::open_spiel::algorithms::Evaluator;
using ::open_spiel::algorithms::Exploitability;
using ::open_spiel::algorithms::NashConv;
using ::open_spiel::algorithms::TabularBestResponse;
using ::open_spiel::matrix_game::MatrixGame;

namespace py = ::pybind11;

// This exception class is used to forward errors from Spiel to Python.
// Do not create exceptions of this type directly! Instead, call
// SpielFatalError, which will raise a Python exception when called from
// Python, and exit the process otherwise.
class SpielException : public std::exception {
 public:
  explicit SpielException(std::string message) : message_(message) {}
  const char* what() const noexcept override { return message_.c_str(); }

 private:
  std::string message_;
};

// Trampoline helper class to allow implementing Bots in Python. See
// https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python
class PyBot : public Bot {
 public:
  // We need the bot constructor
  using Bot::Bot;

  // Override the bot's action choice.
  using step_retval_t = std::pair<ActionsAndProbs, open_spiel::Action>;
  void ApplyAction(Action action) override {
    PYBIND11_OVERLOAD_NAME(
        void,  // Return type (must be a simple token for macro parser)
        Bot,   // Parent class
        "apply_action",  // Name of function in Python
        ApplyAction,     // Name of function in C++
        action           // Arguments
    );
  }

  // Choose and execute an action in a game. The bot should return its
  // distribution over actions and also its selected action.
  step_retval_t Step(const State& state) override {
    PYBIND11_OVERLOAD_PURE_NAME(
        step_retval_t,  // Return type (must be a simple token for macro parser)
        Bot,            // Parent class
        "step",         // Name of function in Python
        Step,           // Name of function in C++
        state           // Arguments
    );
  }

  // Restart at the specified state.
  void Restart(const State& state) override {
    PYBIND11_OVERLOAD_NAME(
        void,       // Return type (must be a simple token for macro parser)
        Bot,        // Parent class
        "restart",  // Name of function in Python
        Restart,    // Name of function in C++
        state       // Arguments
    );
  }
};

// Definintion of our Python module.
PYBIND11_MODULE(pyspiel, m) {
  m.doc() = "Open Spiel";

  py::enum_<open_spiel::GameParameter::Type>(m, "GameParameterType")
      .value("UNSET", open_spiel::GameParameter::Type::kUnset)
      .value("INT", open_spiel::GameParameter::Type::kInt)
      .value("DOUBLE", open_spiel::GameParameter::Type::kDouble)
      .value("STRING", open_spiel::GameParameter::Type::kString)
      .value("BOOL", open_spiel::GameParameter::Type::kBool);

  py::class_<GameParameter> game_parameter(m, "GameParameter");
  game_parameter.def(py::init<double>())
      .def(py::init<std::string>())
      .def(py::init<bool>())
      .def(py::init<int>())
      .def(py::init<GameParameters>())
      .def("is_mandatory", &GameParameter::is_mandatory)
      .def("__str__", &GameParameter::ToString)
      .def("__repr__", &GameParameter::ToReprString);

  py::enum_<open_spiel::StateType>(m, "StateType")
      .value("TERMINAL", open_spiel::StateType::kTerminal)
      .value("CHANCE", open_spiel::StateType::kChance)
      .value("DECISION", open_spiel::StateType::kDecision)
      .export_values();

  py::class_<GameType> game_type(m, "GameType");
  game_type
      .def(py::init<std::string, std::string, GameType::Dynamics,
                    GameType::ChanceMode, GameType::Information,
                    GameType::Utility, GameType::RewardModel, int, int, bool,
                    bool, bool, bool,
                    std::map<std::string, GameParameter>>())
      .def_readonly("short_name", &GameType::short_name)
      .def_readonly("long_name", &GameType::long_name)
      .def_readonly("dynamics", &GameType::dynamics)
      .def_readonly("chance_mode", &GameType::chance_mode)
      .def_readonly("information", &GameType::information)
      .def_readonly("utility", &GameType::utility)
      .def_readonly("reward_model", &GameType::reward_model)
      .def_readonly("max_num_players", &GameType::max_num_players)
      .def_readonly("min_num_players", &GameType::min_num_players)
      .def_readonly("provides_information_state",
                    &GameType::provides_information_state)
      .def_readonly("provides_information_state_as_normalized_vector",
                    &GameType::provides_information_state_as_normalized_vector)
      .def_readonly("provides_observation", &GameType::provides_observation)
      .def_readonly("provides_observation_as_normalized_vector",
                    &GameType::provides_observation_as_normalized_vector)
      .def_readonly("parameter_specification",
                    &GameType::parameter_specification)
      .def("__repr__", [](const GameType& gt) {
        return "<GameType '" + gt.short_name + "'>";
      });

  py::enum_<GameType::Dynamics>(game_type, "Dynamics")
      .value("SEQUENTIAL", GameType::Dynamics::kSequential)
      .value("SIMULTANEOUS", GameType::Dynamics::kSimultaneous);

  py::enum_<GameType::ChanceMode>(game_type, "ChanceMode")
      .value("DETERMINISTIC", GameType::ChanceMode::kDeterministic)
      .value("EXPLICIT_STOCHASTIC", GameType::ChanceMode::kExplicitStochastic)
      .value("SAMPLED_STOCHASTIC", GameType::ChanceMode::kSampledStochastic);

  py::enum_<GameType::Information>(game_type, "Information")
      .value("ONE_SHOT", GameType::Information::kOneShot)
      .value("PERFECT_INFORMATION", GameType::Information::kPerfectInformation)
      .value("IMPERFECT_INFORMATION",
             GameType::Information::kImperfectInformation);

  py::enum_<GameType::Utility>(game_type, "Utility")
      .value("ZERO_SUM", GameType::Utility::kZeroSum)
      .value("CONSTANT_SUM", GameType::Utility::kConstantSum)
      .value("GENERAL_SUM", GameType::Utility::kGeneralSum)
      .value("IDENTICAL", GameType::Utility::kIdentical);

  py::enum_<GameType::RewardModel>(game_type, "RewardModel")
      .value("REWARDS", GameType::RewardModel::kRewards)
      .value("TERMINAL", GameType::RewardModel::kTerminal);

  py::enum_<open_spiel::PlayerId>(m, "PlayerId")
      .value("INVALID", open_spiel::kInvalidPlayer)
      .value("TERMINAL", open_spiel::kTerminalPlayerId)
      .value("CHANCE", open_spiel::kChancePlayerId)
      .value("SIMULTANEOUS", open_spiel::kSimultaneousPlayerId);

  m.attr("INVALID_ACTION") = py::int_(open_spiel::kInvalidAction);

  py::class_<State> state(m, "State");
  state.def("current_player", &State::CurrentPlayer)
      .def("apply_action", &State::ApplyAction)
      .def("legal_actions",
           (std::vector<open_spiel::Action>(State::*)(int) const) &
               State::LegalActions)
      .def("legal_actions",
           (std::vector<open_spiel::Action>(State::*)(void) const) &
               State::LegalActions)
      .def("legal_actions_mask",
           (std::vector<int>(State::*)(int) const) & State::LegalActionsMask)
      .def("legal_actions_mask",
           (std::vector<int>(State::*)(void) const) & State::LegalActionsMask)
      .def("action_to_string", (std::string(State::*)(Player, Action) const) &
                                   State::ActionToString)
      .def("action_to_string",
           (std::string(State::*)(Action) const) & State::ActionToString)
      .def("string_to_action",
           (Action(State::*)(Player, const std::string&) const) &
               State::StringToAction)
      .def("string_to_action",
           (Action(State::*)(const std::string&) const) & State::StringToAction)
      .def("__str__", &State::ToString)
      .def("is_terminal", &State::IsTerminal)
      .def("rewards", &State::Rewards)
      .def("returns", &State::Returns)
      .def("player_reward", &State::PlayerReward)
      .def("player_return", &State::PlayerReturn)
      .def("is_chance_node", &State::IsChanceNode)
      .def("is_simultaneous_node", &State::IsSimultaneousNode)
      .def("history", &State::History)
      .def("history_str", &State::HistoryString)
      .def("information_state",
           (std::string(State::*)(int) const) & State::InformationState)
      .def("information_state",
           (std::string(State::*)() const) & State::InformationState)
      .def("information_state_as_normalized_vector",
           (std::vector<double>(State::*)(int) const) &
               State::InformationStateAsNormalizedVector)
      .def("information_state_as_normalized_vector",
           (std::vector<double>(State::*)() const) &
               State::InformationStateAsNormalizedVector)
      .def("observation",
           (std::string(State::*)(int) const) & State::Observation)
      .def("observation", (std::string(State::*)() const) & State::Observation)
      .def("observation_as_normalized_vector",
           (std::vector<double>(State::*)(int) const) &
               State::ObservationAsNormalizedVector)
      .def("observation_as_normalized_vector",
           (std::vector<double>(State::*)() const) &
               State::ObservationAsNormalizedVector)
      .def("clone", &State::Clone)
      .def("child", &State::Child)
      .def("undo_action", &State::UndoAction)
      .def("apply_actions", &State::ApplyActions)
      .def("num_distinct_actions", &State::NumDistinctActions)
      .def("num_players", &State::NumPlayers)
      .def("chance_outcomes", &State::ChanceOutcomes)
      .def("get_type", &State::GetType)
      .def("serialize", &State::Serialize)
      .def(py::pickle(              // Pickle support
          [](const State& state) {  // __getstate__
            return SerializeGameAndState(*state.GetGame(), state);
          },
          [](const std::string& data) {  // __setstate__
            std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
                game_and_state = DeserializeGameAndState(data);
            return std::move(game_and_state.second);
          }));

  py::class_<Game, std::shared_ptr<Game>> game(m, "Game");
  game.def("num_distinct_actions", &Game::NumDistinctActions)
      .def("new_initial_state", &Game::NewInitialState)
      .def("max_chance_outcomes", &Game::MaxChanceOutcomes)
      .def("get_parameters", &Game::GetParameters)
      .def("num_players", &Game::NumPlayers)
      .def("min_utility", &Game::MinUtility)
      .def("max_utility", &Game::MaxUtility)
      .def("get_type", &Game::GetType)
      .def("utility_sum", &Game::UtilitySum)
      .def("information_state_normalized_vector_shape",
           &Game::InformationStateNormalizedVectorShape)
      .def("information_state_normalized_vector_size",
           &Game::InformationStateNormalizedVectorSize)
      .def("observation_normalized_vector_shape",
           &Game::ObservationNormalizedVectorShape)
      .def("observation_normalized_vector_size",
           &Game::ObservationNormalizedVectorSize)
      .def("deserialize_state", &Game::DeserializeState)
      .def("max_game_length", &Game::MaxGameLength)
      .def("__str__", &Game::ToString)
      .def(py::pickle(                            // Pickle support
          [](std::shared_ptr<const Game> game) {  // __getstate__
            return game->ToString();
          },
          [](const std::string& data) {  // __setstate__
            // Have to remove the const here for this to compile, presumably
            // because the holder type is non-const. But seems like you can't
            // set the holder type to std::shared_ptr<const Game> either.
            return std::const_pointer_cast<Game>(LoadGame(data));
          }));

  py::class_<NormalFormGame, std::shared_ptr<NormalFormGame>> normal_form_game(
      m, "NormalFormGame", game);
  normal_form_game.def(py::pickle(                      // Pickle support
      [](std::shared_ptr<const NormalFormGame> game) {  // __getstate__
        return game->ToString();
      },
      [](const std::string& data) {  // __setstate__
        // Have to remove the const here for this to compile, presumably
        // because the holder type is non-const. But seems like you can't
        // set the holder type to std::shared_ptr<const Game> either.
        return std::const_pointer_cast<NormalFormGame>(
            std::static_pointer_cast<const NormalFormGame>(LoadGame(data)));
      }));

  py::class_<MatrixGame, std::shared_ptr<MatrixGame>> matrix_game(
      m, "MatrixGame", normal_form_game);
  matrix_game
      .def(py::init<GameType, GameParameters, std::vector<std::string>,
                    std::vector<std::string>, std::vector<double>,
                    std::vector<double>>())
      .def(py::init<GameType, GameParameters, std::vector<std::string>,
                    std::vector<std::string>,
                    const std::vector<std::vector<double>>&,
                    const std::vector<std::vector<double>>&>())
      .def("num_rows", &MatrixGame::NumRows)
      .def("num_cols", &MatrixGame::NumCols)
      .def("row_utility", &MatrixGame::RowUtility)
      .def("col_utility", &MatrixGame::ColUtility)
      .def("player_utility", &MatrixGame::PlayerUtility)
      .def("row_action_name", &MatrixGame::RowActionName)
      .def("col_action_name", &MatrixGame::ColActionName)
      .def(py::pickle(                                  // Pickle support
          [](std::shared_ptr<const MatrixGame> game) {  // __getstate__
            return game->ToString();
          },
          [](const std::string& data) {  // __setstate__
            // Have to remove the const here for this to compile, presumably
            // because the holder type is non-const. But seems like you can't
            // set the holder type to std::shared_ptr<const Game> either.
            return std::const_pointer_cast<MatrixGame>(
                algorithms::LoadMatrixGame(data));
          }));

  py::class_<Bot, PyBot> bot(m, "Bot");
  bot.def(py::init<const Game&, int>())
      .def("player_id", &Bot::PlayerId)
      .def("step", &Bot::Step)
      .def("apply_action", &Bot::ApplyAction)
      .def("restart", &Bot::Restart);

  py::class_<algorithms::Evaluator> mcts_evaluator(m, "Evaluator");
  py::class_<algorithms::RandomRolloutEvaluator, algorithms::Evaluator>(
      m, "RandomRolloutEvaluator")
      .def(py::init<int, int>(), py::arg("n_rollouts"), py::arg("seed"));

  py::enum_<algorithms::ChildSelectionPolicy>(m, "ChildSelectionPolicy")
      .value("UCT", algorithms::ChildSelectionPolicy::UCT)
      .value("PUCT", algorithms::ChildSelectionPolicy::PUCT);

  py::class_<algorithms::MCTSBot, Bot>(m, "MCTSBot")
      .def(py::init<const Game&, Player, Evaluator*, double, int, int64_t,
                    bool, int, bool,
                    ::open_spiel::algorithms::ChildSelectionPolicy>(),
           py::arg("game"), py::arg("player"), py::arg("evaluator"),
           py::arg("uct_c"), py::arg("max_simulations"),
           py::arg("max_memory_mb"), py::arg("solve"), py::arg("seed"),
           py::arg("verbose"),
           py::arg("child_selection_policy") =
               algorithms::ChildSelectionPolicy::UCT)
      .def("step", &algorithms::MCTSBot::Step)
      .def("mcts_search", &algorithms::MCTSBot::MCTSearch);

  py::class_<TabularBestResponse>(m, "TabularBestResponse")
      .def(py::init<const open_spiel::Game&, int,
                    const std::unordered_map<std::string,
                                             open_spiel::ActionsAndProbs>&>())
      .def(py::init<const open_spiel::Game&, int, const open_spiel::Policy*>())
      .def("value", &TabularBestResponse::Value)
      .def("get_best_response_policy",
           &TabularBestResponse::GetBestResponsePolicy)
      .def("get_best_response_actions",
           &TabularBestResponse::GetBestResponseActions)
      .def("set_policy", py::overload_cast<const std::unordered_map<
                             std::string, open_spiel::ActionsAndProbs>&>(
                             &TabularBestResponse::SetPolicy))
      .def("set_policy",
           py::overload_cast<const Policy*>(&TabularBestResponse::SetPolicy));

  py::class_<open_spiel::Policy>(m, "Policy")
      .def("action_probabilities",
           (std::unordered_map<Action, double>(open_spiel::Policy::*)(
               const open_spiel::State&) const) &
               open_spiel::Policy::GetStatePolicyAsMap)
      .def("get_state_policy_as_map",
           (std::unordered_map<Action, double>(open_spiel::Policy::*)(
               const std::string&) const) &
               open_spiel::Policy::GetStatePolicyAsMap);

  // A tabular policy represented internally as a map. Note that this
  // implementation is not directly compatible with the Python TabularPolicy
  // implementation; the latter is implemented as a table of size
  // [num_states, num_actions], while this is implemented as a map. It is
  // non-trivial to convert between the two, but we have a function that does so
  // in the open_spiel/python/policy.py file.
  py::class_<open_spiel::TabularPolicy, open_spiel::Policy>(m, "TabularPolicy")
      .def(py::init<const std::unordered_map<std::string, ActionsAndProbs>&>())
      .def("get_state_policy", &open_spiel::TabularPolicy::GetStatePolicy)
      .def("policy_table", &open_spiel::TabularPolicy::PolicyTable);
  m.def("UniformRandomPolicy", &open_spiel::GetUniformPolicy);

  py::class_<open_spiel::algorithms::CFRSolver>(m, "CFRSolver")
      .def(py::init<const Game&>())
      .def("evaluate_and_update_policy",
           &open_spiel::algorithms::CFRSolver::EvaluateAndUpdatePolicy)
      .def("current_policy", &open_spiel::algorithms::CFRSolver::CurrentPolicy)
      .def("average_policy", &open_spiel::algorithms::CFRSolver::AveragePolicy);

  py::class_<open_spiel::algorithms::CFRPlusSolver>(m, "CFRPlusSolver")
      .def(py::init<const Game&>())
      .def("evaluate_and_update_policy",
           &open_spiel::algorithms::CFRPlusSolver::EvaluateAndUpdatePolicy)
      .def("current_policy", &open_spiel::algorithms::CFRSolver::CurrentPolicy)
      .def("average_policy",
           &open_spiel::algorithms::CFRPlusSolver::AveragePolicy);

  py::class_<open_spiel::algorithms::CFRBRSolver>(m, "CFRBRSolver")
      .def(py::init<const Game&>())
      .def("evaluate_and_update_policy",
           &open_spiel::algorithms::CFRPlusSolver::EvaluateAndUpdatePolicy)
      .def("current_policy", &open_spiel::algorithms::CFRSolver::CurrentPolicy)
      .def("average_policy",
           &open_spiel::algorithms::CFRPlusSolver::AveragePolicy);

  py::class_<open_spiel::algorithms::TrajectoryRecorder>(m,
                                                         "TrajectoryRecorder")
      .def(py::init<const Game&, const std::unordered_map<std::string, int>&,
                    int>())
      .def("record_batch",
           &open_spiel::algorithms::TrajectoryRecorder::RecordBatch);

  m.def("create_matrix_game",
        py::overload_cast<const std::string&, const std::string&,
                          const std::vector<std::string>&,
                          const std::vector<std::string>&,
                          const std::vector<std::vector<double>>&,
                          const std::vector<std::vector<double>>&>(
            &open_spiel::matrix_game::CreateMatrixGame),
        "Creates an arbitrary matrix game from named rows/cols and utilities.");

  m.def("create_matrix_game",
        py::overload_cast<const std::vector<std::vector<double>>&,
                          const std::vector<std::vector<double>>&>(
            &open_spiel::matrix_game::CreateMatrixGame),
        "Creates an arbitrary matrix game from dimensions and utilities.");

  m.def("load_game",
        py::overload_cast<const std::string&>(&open_spiel::LoadGame),
        "Returns a new game object for the specified short name using default "
        "parameters");

  m.def("load_game",
        py::overload_cast<const std::string&, const GameParameters&>(
            &open_spiel::LoadGame),
        "Returns a new game object for the specified short name using given "
        "parameters");

  m.def("load_game_as_turn_based",
        py::overload_cast<const std::string&>(&open_spiel::LoadGameAsTurnBased),
        "Converts a simultaneous game into an turn-based game with infosets.");

  m.def("load_game_as_turn_based",
        py::overload_cast<const std::string&, const GameParameters&>(
            &open_spiel::LoadGameAsTurnBased),
        "Converts a simultaneous game into an turn-based game with infosets.");

  m.def("load_matrix_game", open_spiel::algorithms::LoadMatrixGame,
        "Loads a game as a matrix game (will fail if not a matrix game.");

  m.def("extensive_to_matrix_game",
        open_spiel::algorithms::ExtensiveToMatrixGame,
        "Converts a two-player extensive-game to its equivalent matrix game, "
        "which is exponentially larger. Use only with small games.");

  m.def("registered_names", GameRegister::RegisteredNames,
        "Returns the names of all available games.");

  m.def("registered_games", GameRegister::RegisteredGames,
        "Returns the details of all available games.");

  m.def("evaluate_bots", open_spiel::EvaluateBots,
        "Plays a single game with the given bots and returns the final "
        "utilities.");

  m.def("make_uniform_random_bot", open_spiel::MakeUniformRandomBot,
        "A uniform random bot, for test purposes.");

  m.def("serialize_game_and_state", open_spiel::SerializeGameAndState,
        "A general implementation of game and state serialization.");

  m.def("deserialize_game_and_state", open_spiel::DeserializeGameAndState,
        "A general implementation of deserialization of a game and state "
        "string serialized by serialize_game_and_state.");

  m.def("exploitability",
        py::overload_cast<const Game&, const Policy&>(&Exploitability),
        "Returns the sum of the utility that a best responder wins when when "
        "playing against 1) the player 0 policy contained in `policy` and 2) "
        "the player 1 policy contained in `policy`."
        "This only works for two player, zero- or constant-sum sequential "
        "games, and raises a SpielFatalError if an incompatible game is passed "
        "to it.");

  m.def("exploitability",
        py::overload_cast<const Game&,
                    const std::unordered_map<std::string, ActionsAndProbs>&>(
            &Exploitability),
        "Returns the sum of the utility that a best responder wins when when "
        "playing against 1) the player 0 policy contained in `policy` and 2) "
        "the player 1 policy contained in `policy`."
        "This only works for two player, zero- or constant-sum sequential "
        "games, and raises a SpielFatalError if an incompatible game is passed "
        "to it.");

  m.def("nash_conv", py::overload_cast<const Game&, const Policy&>(&NashConv),
        "Returns the sum of the utility that a best responder wins when when "
        "playing against 1) the player 0 policy contained in `policy` and 2) "
        "the player 1 policy contained in `policy`."
        "This only works for two player, zero- or constant-sum sequential "
        "games, and raises a SpielFatalError if an incompatible game is passed "
        "to it.");

  m.def("nash_conv",
        py::overload_cast<
            const Game&,
            const std::unordered_map<std::string, ActionsAndProbs>&>(&NashConv),
        "Calculates a measure of how far the given policy is from a Nash "
        "equilibrium by returning the sum of the improvements in the value "
        "that each player could obtain by unilaterally changing their strategy "
        "while the opposing player maintains their current strategy (which "
        "for a Nash equilibrium, this value is 0).");

  m.def("convert_to_turn_based", open_spiel::ConvertToTurnBased,
        "Returns a turn-based version of the given game.");

  m.def("expected_returns",
        py::overload_cast<const State&, const std::vector<const Policy*>&, int>(
            &open_spiel::algorithms::ExpectedReturns),
        "Computes the undiscounted expected returns from a depth-limited "
        "search.");

  py::class_<open_spiel::algorithms::BatchedTrajectory>(m, "BatchedTrajectory")
      .def(py::init<int>())
      .def_readwrite("observations",
                     &open_spiel::algorithms::BatchedTrajectory::observations)
      .def_readwrite("state_indices",
                     &open_spiel::algorithms::BatchedTrajectory::state_indices)
      .def_readwrite("legal_actions",
                     &open_spiel::algorithms::BatchedTrajectory::legal_actions)
      .def_readwrite("actions",
                     &open_spiel::algorithms::BatchedTrajectory::actions)
      .def_readwrite(
          "player_policies",
          &open_spiel::algorithms::BatchedTrajectory::player_policies)
      .def_readwrite("player_ids",
                     &open_spiel::algorithms::BatchedTrajectory::player_ids)
      .def_readwrite("rewards",
                     &open_spiel::algorithms::BatchedTrajectory::rewards)
      .def_readwrite("valid", &open_spiel::algorithms::BatchedTrajectory::valid)
      .def_readwrite(
          "next_is_terminal",
          &open_spiel::algorithms::BatchedTrajectory::next_is_terminal)
      .def_readwrite("batch_size",
                     &open_spiel::algorithms::BatchedTrajectory::batch_size)
      .def_readwrite(
          "max_trajectory_length",
          &open_spiel::algorithms::BatchedTrajectory::max_trajectory_length)
      .def("resize_fields",
           &open_spiel::algorithms::BatchedTrajectory::ResizeFields);

  m.def("record_batched_trajectories",
        py::overload_cast<
            const Game&, const std::vector<open_spiel::TabularPolicy>&,
            const std::unordered_map<std::string, int>&, int, bool, int, int>(
            &open_spiel::algorithms::RecordBatchedTrajectory),
        "Records a batch of trajectories.");

  // Game-Specific Query API.
  m.def("negotiation_item_pool", &open_spiel::query::NegotiationItemPool);
  m.def("negotiation_agent_utils",
        &open_spiel::query::NegotiationAgentUtils);

  // Set an error handler that will raise exceptions. These exceptions are for
  // the Python interface only. When used from C++, OpenSpiel will never raise
  // exceptions - the process will be terminated instead.
  open_spiel::SetErrorHandler(
      [](const std::string& string) { throw SpielException(string); });
}

}  // namespace
}  // namespace open_spiel
