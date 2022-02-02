// Copyright 2021 DeepMind Technologies Limited
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

#include <memory>
#include <unordered_map>

#include "open_spiel/algorithms/matrix_game_utils.h"
#include "open_spiel/algorithms/nfg_writer.h"
#include "open_spiel/algorithms/tensor_game_utils.h"
#include "open_spiel/canonical_game_strings.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/efg_game.h"
#include "open_spiel/games/efg_game_data.h"
#include "open_spiel/games/nfg_game.h"
#include "open_spiel/matrix_game.h"
#include "open_spiel/normal_form_game.h"
#include "open_spiel/observer.h"
#include "open_spiel/python/pybind11/algorithms_corr_dist.h"
#include "open_spiel/python/pybind11/algorithms_trajectories.h"
#include "open_spiel/python/pybind11/bots.h"
#include "open_spiel/python/pybind11/game_transforms.h"
#include "open_spiel/python/pybind11/games_backgammon.h"
#include "open_spiel/python/pybind11/games_bridge.h"
#include "open_spiel/python/pybind11/games_chess.h"
#include "open_spiel/python/pybind11/games_kuhn_poker.h"
#include "open_spiel/python/pybind11/games_leduc_poker.h"
#include "open_spiel/python/pybind11/games_negotiation.h"
#include "open_spiel/python/pybind11/games_tarok.h"
#include "open_spiel/python/pybind11/observer.h"
#include "open_spiel/python/pybind11/policy.h"
#include "open_spiel/python/pybind11/pybind11.h"
#include "open_spiel/python/pybind11/python_games.h"
#include "open_spiel/python/pybind11/referee.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

// List of optional python submodules.
#if OPEN_SPIEL_BUILD_WITH_GAMUT
#include "open_spiel/games/gamut/gamut_pybind11.h"
#endif
#if OPEN_SPIEL_BUILD_WITH_XINXIN
#include "open_spiel/bots/xinxin/xinxin_pybind11.h"
#endif

// This file contains OpenSpiel's Python API. The best place to see an overview
// of the API is to refer to python/examples/example.py. Generally, all the core
// functions are exposed as snake case in Python (i.e. CurrentPlayer becomes
// current_player, ApplyAction becomes apply_action, etc.) but otherwise the
// functions and their effect remain the same. For a more detailed documentation
// of each of the core API functions, please see spiel.h.

namespace open_spiel {
namespace {

using ::open_spiel::matrix_game::MatrixGame;
using ::open_spiel::tensor_game::TensorGame;

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

// Definintion of our Python module.
PYBIND11_MODULE(pyspiel, m) {
  m.doc() = "Open Spiel";

  m.def("game_parameters_from_string", GameParametersFromString,
        "Parses a string as a GameParameter dictionary.");

  m.def("game_parameters_to_string", GameParametersToString,
        "Converts a GameParameter dictionary to string.");

  py::enum_<PrivateInfoType>(m, "PrivateInfoType")
      .value("ALL_PLAYERS", PrivateInfoType::kAllPlayers)
      .value("NONE", PrivateInfoType::kNone)
      .value("SINGLE_PLAYER", PrivateInfoType::kSinglePlayer)
      .export_values();

  py::class_<IIGObservationType>(m, "IIGObservationType")
      .def(py::init<bool, bool, PrivateInfoType>(),
           py::arg("public_info") = true, py::arg("perfect_recall"),
           py::arg("private_info") = PrivateInfoType::kSinglePlayer)
      .def_readonly("public_info", &IIGObservationType::public_info)
      .def_readonly("perfect_recall", &IIGObservationType::perfect_recall)
      .def_readonly("private_info", &IIGObservationType::private_info);

  py::class_<UniformProbabilitySampler> uniform_sampler(
      m, "UniformProbabilitySampler");
  uniform_sampler.def(py::init<double, double>())
      .def(py::init<int, double, double>())
      .def("__call__", &UniformProbabilitySampler::operator());

  py::enum_<open_spiel::StateType>(m, "StateType")
      .value("TERMINAL", open_spiel::StateType::kTerminal)
      .value("CHANCE", open_spiel::StateType::kChance)
      .value("DECISION", open_spiel::StateType::kDecision)
      .value("MEAN_FIELD", open_spiel::StateType::kMeanField)
      .export_values();

  py::class_<GameType> game_type(m, "GameType");
  game_type
      .def(py::init<std::string, std::string, GameType::Dynamics,
                    GameType::ChanceMode, GameType::Information,
                    GameType::Utility, GameType::RewardModel, int, int, bool,
                    bool, bool, bool, GameParameters,
                    bool, bool>(),
           py::arg("short_name"), py::arg("long_name"), py::arg("dynamics"),
           py::arg("chance_mode"), py::arg("information"), py::arg("utility"),
           py::arg("reward_model"), py::arg("max_num_players"),
           py::arg("min_num_players"),
           py::arg("provides_information_state_string"),
           py::arg("provides_information_state_tensor"),
           py::arg("provides_observation_string"),
           py::arg("provides_observation_tensor"),
           py::arg("parameter_specification") =
               GameParameters(),
           py::arg("default_loadable") = true,
           py::arg("provides_factored_observation_string") = false)
      .def(py::init<const GameType&>())
      .def_readonly("short_name", &GameType::short_name)
      .def_readonly("long_name", &GameType::long_name)
      .def_readonly("dynamics", &GameType::dynamics)
      .def_readonly("chance_mode", &GameType::chance_mode)
      .def_readonly("information", &GameType::information)
      .def_readonly("utility", &GameType::utility)
      .def_readonly("reward_model", &GameType::reward_model)
      .def_readonly("max_num_players", &GameType::max_num_players)
      .def_readonly("min_num_players", &GameType::min_num_players)
      .def_readonly("provides_information_state_string",
                    &GameType::provides_information_state_string)
      .def_readonly("provides_information_state_tensor",
                    &GameType::provides_information_state_tensor)
      .def_readonly("provides_observation_string",
                    &GameType::provides_observation_string)
      .def_readonly("provides_observation_tensor",
                    &GameType::provides_observation_tensor)
      .def_readonly("parameter_specification",
                    &GameType::parameter_specification)
      .def_readonly("default_loadable", &GameType::default_loadable)
      .def_readonly("provides_factored_observation_string",
                    &GameType::provides_factored_observation_string)
      .def("pretty_print",
           [](const GameType& value) { return GameTypeToString(value); })
      .def("__repr__",
           [](const GameType& gt) {
             return "<GameType '" + gt.short_name + "'>";
           })
      .def("__eq__",
           [](const GameType& value, GameType* value2) {
             return value2 &&
                    GameTypeToString(value) == GameTypeToString(*value2);
           })
      .def(py::pickle(                     // Pickle support
          [](const GameType& game_type) {  // __getstate__
            return GameTypeToString(game_type);
          },
          [](const std::string& data) {  // __setstate__
            return GameTypeFromString(data);
          }));

  py::enum_<GameType::Dynamics>(game_type, "Dynamics")
      .value("SEQUENTIAL", GameType::Dynamics::kSequential)
      .value("MEAN_FIELD",
             GameType::Dynamics::kMeanField)
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
      .value("DEFAULT_PLAYER_ID", open_spiel::kDefaultPlayerId)
      .value("INVALID", open_spiel::kInvalidPlayer)
      .value("TERMINAL", open_spiel::kTerminalPlayerId)
      .value("CHANCE", open_spiel::kChancePlayerId)
      .value("MEAN_FIELD", open_spiel::kMeanFieldPlayerId)
      .value("SIMULTANEOUS", open_spiel::kSimultaneousPlayerId);

  py::class_<GameInfo> game_info(m, "GameInfo");
  game_info
      .def(py::init<int, int, int, double, double, double, int>(),
           py::arg("num_distinct_actions"), py::arg("max_chance_outcomes"),
           py::arg("num_players"), py::arg("min_utility"),
           py::arg("max_utility"), py::arg("utility_sum") = 0,
           py::arg("max_game_length"))
      .def(py::init<const GameInfo&>())
      .def_readonly("num_distinct_actions", &GameInfo::num_distinct_actions)
      .def_readonly("max_chance_outcomes", &GameInfo::max_chance_outcomes)
      .def_readonly("num_players", &GameInfo::num_players)
      .def_readonly("min_utility", &GameInfo::min_utility)
      .def_readonly("max_utility", &GameInfo::max_utility)
      .def_readonly("utility_sum", &GameInfo::utility_sum)
      .def_readonly("max_game_length", &GameInfo::max_game_length);

  m.attr("INVALID_ACTION") = py::int_(open_spiel::kInvalidAction);

  py::enum_<open_spiel::TensorLayout>(m, "TensorLayout")
      .value("HWC", open_spiel::TensorLayout::kHWC)
      .value("CHW", open_spiel::TensorLayout::kCHW);

  py::class_<State::PlayerAction> player_action(m, "PlayerAction");
  player_action.def_readonly("player", &State::PlayerAction::player)
      .def_readonly("action", &State::PlayerAction::action);

  // https://github.com/pybind/pybind11/blob/smart_holder/README_smart_holder.rst
  py::classh<State, PyState> state(m, "State");
  state.def(py::init<std::shared_ptr<const Game>>())
      .def("current_player", &State::CurrentPlayer)
      .def("apply_action", &State::ApplyAction)
      .def("apply_action_with_legality_check",
           py::overload_cast<Action>(
               &State::ApplyActionWithLegalityCheck))
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
      .def("__repr__", &State::ToString)
      .def("is_terminal", &State::IsTerminal)
      .def("is_initial_state", &State::IsInitialState)
      .def("move_number", &State::MoveNumber)
      .def("rewards", &State::Rewards)
      .def("returns", &State::Returns)
      .def("player_reward", &State::PlayerReward)
      .def("player_return", &State::PlayerReturn)
      .def("is_chance_node", &State::IsChanceNode)
      .def("is_mean_field_node", &State::IsMeanFieldNode)
      .def("is_simultaneous_node", &State::IsSimultaneousNode)
      .def("is_player_node", &State::IsPlayerNode)
      .def("history", &State::History)
      .def("history_str", &State::HistoryString)
      .def("full_history", &State::FullHistory)
      .def("information_state_string",
           (std::string(State::*)(int) const) & State::InformationStateString)
      .def("information_state_string",
           (std::string(State::*)() const) & State::InformationStateString)
      .def("information_state_tensor",
           (std::vector<float>(State::*)(int) const) &
               State::InformationStateTensor)
      .def("information_state_tensor", (std::vector<float>(State::*)() const) &
                                           State::InformationStateTensor)
      .def("observation_string",
           (std::string(State::*)(int) const) & State::ObservationString)
      .def("observation_string",
           (std::string(State::*)() const) & State::ObservationString)
      .def("observation_tensor",
           (std::vector<float>(State::*)(int) const) & State::ObservationTensor)
      .def("observation_tensor",
           (std::vector<float>(State::*)() const) & State::ObservationTensor)
      .def("clone", &State::Clone)
      .def("child", &State::Child)
      .def("undo_action", &State::UndoAction)
      .def("apply_actions", &State::ApplyActions)
      .def("apply_actions_with_legality_checks",
           &State::ApplyActionsWithLegalityChecks)
      .def("num_distinct_actions", &State::NumDistinctActions)
      .def("num_players", &State::NumPlayers)
      .def("chance_outcomes", &State::ChanceOutcomes)
      .def("get_game", &State::GetGame)
      .def("get_type", &State::GetType)
      .def("serialize", &State::Serialize)
      .def("resample_from_infostate", &State::ResampleFromInfostate)
      .def(py::pickle(              // Pickle support
          [](const State& state) {  // __getstate__
            return SerializeGameAndState(*state.GetGame(), state);
          },
          [](const std::string& data) {  // __setstate__
            auto state = DeserializeGameAndState(data).second;
            auto pydict = PyDict(*state);
            return std::make_pair(std::move(state), pydict);
          }))
      .def("distribution_support", &State::DistributionSupport)
      .def("update_distribution", &State::UpdateDistribution)
      .def("mean_field_population", &State::MeanFieldPopulation);

  py::classh<Game, PyGame> game(m, "Game");
  game.def(py::init<GameType, GameInfo, GameParameters>())
      .def("num_distinct_actions", &Game::NumDistinctActions)
      .def("new_initial_states", &Game::NewInitialStates)
      .def("new_initial_state",
           [](const Game* self) { return self->NewInitialState(); })
      .def("new_initial_state",
           [](const Game* self, const std::string& s) {
             return self->NewInitialState(s);
           })
      .def("new_initial_state_for_population",
           &Game::NewInitialStateForPopulation)
      .def("max_chance_outcomes", &Game::MaxChanceOutcomes)
      .def("get_parameters", &Game::GetParameters)
      .def("num_players", &Game::NumPlayers)
      .def("min_utility", &Game::MinUtility)
      .def("max_utility", &Game::MaxUtility)
      .def("get_type", &Game::GetType)
      .def("utility_sum", &Game::UtilitySum)
      .def("information_state_tensor_shape", &Game::InformationStateTensorShape)
      .def("information_state_tensor_layout",
           &Game::InformationStateTensorLayout)
      .def("information_state_tensor_size", &Game::InformationStateTensorSize)
      .def("observation_tensor_shape", &Game::ObservationTensorShape)
      .def("observation_tensor_layout", &Game::ObservationTensorLayout)
      .def("observation_tensor_size", &Game::ObservationTensorSize)
      .def("policy_tensor_shape", &Game::PolicyTensorShape)
      .def("deserialize_state", &Game::DeserializeState)
      .def("max_game_length", &Game::MaxGameLength)
      .def("action_to_string", &Game::ActionToString)
      .def("max_chance_nodes_in_history", &Game::MaxChanceNodesInHistory)
      .def("max_move_number", &Game::MaxMoveNumber)
      .def("max_history_length", &Game::MaxHistoryLength)
      .def("make_observer",
           [](const Game& game, IIGObservationType iig_obs_type,
              const GameParameters& params) {
             return game.MakeObserver(iig_obs_type, params);
           })
      .def("make_observer",
           [](const Game& game, const GameParameters& params) {
             return game.MakeObserver(absl::nullopt, params);
           })
      .def("__str__", &Game::ToString)
      .def("__repr__", &Game::ToString)
      .def("__eq__",
           [](const Game& value, Game* value2) {
             return value2 && value.ToString() == value2->ToString();
           })
      .def(py::pickle(                            // Pickle support
          [](std::shared_ptr<const Game> game) {  // __getstate__
            return game->Serialize();
          },
          [](const std::string& data) {  // __setstate__
            // We must remove the const for this to compile.
            return std::shared_ptr<Game>(
                std::const_pointer_cast<Game>(DeserializeGame(data)));
          }));

  py::classh<NormalFormGame> normal_form_game(m, "NormalFormGame", game);
  normal_form_game.def("get_utilities", &NormalFormGame::GetUtilities)
      .def("get_utility", &NormalFormGame::GetUtility)
      .def(py::pickle(                      // Pickle support
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

  py::classh<MatrixGame> matrix_game(m, "MatrixGame", normal_form_game);
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
      .def("row_utilities",
           [](const MatrixGame& game) {
             const std::vector<double>& row_utilities = game.RowUtilities();
             return py::array_t<double>({game.NumRows(), game.NumCols()},
                                        &row_utilities[0]);
           })
      .def("col_utilities",
           [](const MatrixGame& game) {
             const std::vector<double>& col_utilities = game.ColUtilities();
             return py::array_t<double>({game.NumRows(), game.NumCols()},
                                        &col_utilities[0]);
           })
      .def("player_utilities",
           [](const MatrixGame& game, const Player player) {
             const std::vector<double>& player_utilities =
                 game.PlayerUtilities(player);
             return py::array_t<double>({game.NumRows(), game.NumCols()},
                                        &player_utilities[0]);
           })
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

  py::classh<TensorGame> tensor_game(m, "TensorGame", normal_form_game);
  tensor_game
      .def(py::init<GameType, GameParameters,
                    std::vector<std::vector<std::string>>,
                    std::vector<std::vector<double>>>())
      .def("shape", &TensorGame::Shape)
      .def("player_utility", &TensorGame::PlayerUtility)
      .def("player_utilities",
           [](const TensorGame& game, const Player player) {
             const std::vector<double>& utilities =
                 game.PlayerUtilities(player);
             return py::array_t<double>(game.Shape(), &utilities[0]);
           })
      .def("action_name", &TensorGame::ActionName)
      .def(py::pickle(                                  // Pickle support
          [](std::shared_ptr<const TensorGame> game) {  // __getstate__
            return game->ToString();
          },
          [](const std::string& data) {  // __setstate__
            // Have to remove the const here for this to compile, presumably
            // because the holder type is non-const. But seems like you can't
            // set the holder type to std::shared_ptr<const Game> either.
            return std::const_pointer_cast<TensorGame>(
                algorithms::LoadTensorGame(data));
          }));

  m.def("hulh_game_string", &open_spiel::HulhGameString);
  m.def("hunl_game_string", &open_spiel::HunlGameString);
  m.def("turn_based_goofspiel_game_string",
        &open_spiel::TurnBasedGoofspielGameString);

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

  m.def("create_tensor_game",
        py::overload_cast<const std::string&, const std::string&,
                          const std::vector<std::vector<std::string>>&,
                          const std::vector<std::vector<double>>&>(
            &open_spiel::tensor_game::CreateTensorGame),
        "Creates an arbitrary tensor game from named actions and utilities.");

  m.def("create_matrix_game",
        py::overload_cast<const std::vector<std::vector<double>>&,
                          const std::vector<std::vector<double>>&>(
            &open_spiel::matrix_game::CreateMatrixGame),
        "Creates an arbitrary matrix game from dimensions and utilities.");

  m.def("create_tensor_game",
        py::overload_cast<const std::vector<std::vector<double>>&,
                          const std::vector<int>&>(
            &open_spiel::tensor_game::CreateTensorGame),
        "Creates an arbitrary matrix game from dimensions and utilities.");

  m.def(
      "create_tensor_game",
      [](const std::vector<py::array_t<
             double, py::array::c_style | py::array::forcecast>>& utilities) {
        const int num_players = utilities.size();
        const std::vector<int> shape(
            utilities[0].shape(), utilities[0].shape() + utilities[0].ndim());
        std::vector<std::vector<double>> flat_utilities;
        for (const auto& player_utilities : utilities) {
          SPIEL_CHECK_EQ(player_utilities.ndim(), num_players);
          SPIEL_CHECK_TRUE(
              std::equal(shape.begin(), shape.end(), player_utilities.shape()));
          flat_utilities.push_back(std::vector<double>(
              player_utilities.data(),
              player_utilities.data() + player_utilities.size()));
        }
        return open_spiel::tensor_game::CreateTensorGame(flat_utilities, shape);
      },
      "Creates an arbitrary matrix game from dimensions and utilities.");

  m.def("game_to_nfg_string", open_spiel::GameToNFGString,
        "Get the Gambit .nfg text for a normal-form game.");

  m.def("load_game",
        py::overload_cast<const std::string&>(&open_spiel::LoadGame),
        "Returns a new game object for the specified short name using default "
        "parameters");

  m.def("load_game",
        py::overload_cast<const std::string&, const GameParameters&>(
            &open_spiel::LoadGame),
        "Returns a new game object for the specified short name using given "
        "parameters");

  m.def("load_matrix_game", open_spiel::algorithms::LoadMatrixGame,
        "Loads a game as a matrix game (will fail if not a matrix game.");

  m.def("load_tensor_game", open_spiel::algorithms::LoadTensorGame,
        "Loads a game as a tensor game (will fail if not a tensor game.");

  m.def("load_efg_game", open_spiel::efg_game::LoadEFGGame,
        "Load a gambit extensive form game (.efg) from string data.");
  m.def("get_sample_efg_data", open_spiel::efg_game::GetSampleEFGData,
        "Get Kuhn poker EFG data.");
  m.def("get_kuhn_poker_efg_data", open_spiel::efg_game::GetKuhnPokerEFGData,
        "Get sample EFG data.");

  m.def("load_nfg_game", open_spiel::nfg_game::LoadNFGGame,
        "Load a gambit normal form game (.nfg) from string data.");

  m.def("extensive_to_matrix_game",
        open_spiel::algorithms::ExtensiveToMatrixGame,
        "Converts a two-player extensive-game to its equivalent matrix game, "
        "which is exponentially larger. Use only with small games.");

  m.def("registered_names", GameRegisterer::RegisteredNames,
        "Returns the names of all available games.");

  m.def("registered_games", GameRegisterer::RegisteredGames,
        "Returns the details of all available games.");

  m.def("serialize_game_and_state", open_spiel::SerializeGameAndState,
        "A general implementation of game and state serialization.");

  m.def(
      "deserialize_game_and_state",
      [](const std::string& data) {
        auto rv = open_spiel::DeserializeGameAndState(data);
        return std::make_pair(rv.first, std::move(rv.second));
      },
      "A general implementation of deserialization of a game and state "
      "string serialized by serialize_game_and_state.");

  m.def("register_game", RegisterPyGame,
        "Register a Python game implementation");

  m.def("random_sim_test", testing::RandomSimTest, py::arg("game"),
        py::arg("num_sims"), py::arg("serialize"), py::arg("verbose"),
        py::arg("mask_test") = true,
        py::arg("state_checker_fn") =
            py::cpp_function(&testing::DefaultStateChecker),
        py::arg("mean_field_population") = -1, "Run the C++ tests on a game");

  // Set an error handler that will raise exceptions. These exceptions are for
  // the Python interface only. When used from C++, OpenSpiel will never raise
  // exceptions - the process will be terminated instead.
  open_spiel::SetErrorHandler([](const std::string& string) {
    std::cerr << "OpenSpiel exception: " << string << std::endl << std::flush;
    throw SpielException(string);
  });
  py::register_exception<SpielException>(m, "SpielError", PyExc_RuntimeError);

  // Register other bits of the API.
  init_pyspiel_bots(m);                   // Bots and bot-related algorithms.
  init_pyspiel_policy(m);           // Policies and policy-related algorithms.
  init_pyspiel_algorithms_corr_dist(m);     // Correlated eq. distance funcs
  init_pyspiel_algorithms_trajectories(m);  // Trajectories.
  init_pyspiel_game_transforms(m);          // Game transformations.
  init_pyspiel_games_backgammon(m);         // Backgammon game.
  init_pyspiel_games_bridge(m);  // Game-specific functions for bridge.
  init_pyspiel_games_chess(m);   // Chess game.
  init_pyspiel_games_kuhn_poker(m);   // Kuhn Poker game.
  init_pyspiel_games_leduc_poker(m);  // Leduc poker game.
  init_pyspiel_games_negotiation(m);  // Negotiation game.
  init_pyspiel_games_tarok(m);   // Game-specific functions for tarok.
  init_pyspiel_observer(m);      // Observers and observations.

  // List of optional python submodules.
#if OPEN_SPIEL_BUILD_WITH_GAMUT
  init_pyspiel_gamut(m);
#endif
#if OPEN_SPIEL_BUILD_WITH_XINXIN
  init_pyspiel_xinxin(m);
#endif
#if OPEN_SPIEL_BUILD_WITH_HIGC
  init_pyspiel_referee(m);
#endif
}

}  // namespace
}  // namespace open_spiel
