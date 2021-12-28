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

#include "jlcxx/jlcxx.hpp"
#include "jlcxx/stl.hpp"
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
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"
#include "open_spiel/spiel_utils.h"

namespace jlcxx {
template <>
struct SuperType<open_spiel::SimMoveGame> {
  typedef open_spiel::Game type;
};
template <>
struct SuperType<open_spiel::NormalFormGame> {
  typedef open_spiel::SimMoveGame type;
};
template <>
struct SuperType<open_spiel::matrix_game::MatrixGame> {
  typedef open_spiel::NormalFormGame type;
};

template <>
struct SuperType<open_spiel::algorithms::RandomRolloutEvaluator> {
  typedef open_spiel::algorithms::Evaluator type;
};

template <>
struct SuperType<open_spiel::TabularPolicy> {
  typedef open_spiel::Policy type;
};

template <>
struct SuperType<open_spiel::algorithms::MCTSBot> {
  typedef open_spiel::Bot type;
};

template <>
struct SuperType<open_spiel::algorithms::CFRSolver> {
  typedef open_spiel::algorithms::CFRSolverBase type;
};
template <>
struct SuperType<open_spiel::algorithms::CFRBRSolver> {
  typedef open_spiel::algorithms::CFRSolverBase type;
};
template <>
struct SuperType<open_spiel::algorithms::CFRPlusSolver> {
  typedef open_spiel::algorithms::CFRSolverBase type;
};
}  // namespace jlcxx

template <>
struct jlcxx::IsMirroredType<open_spiel::GameParameter::Type> : std::true_type {
};
template <>
struct jlcxx::IsMirroredType<open_spiel::StateType> : std::true_type {};
template <>
struct jlcxx::IsMirroredType<open_spiel::GameType::Dynamics> : std::true_type {
};
template <>
struct jlcxx::IsMirroredType<open_spiel::GameType::ChanceMode>
    : std::true_type {};
template <>
struct jlcxx::IsMirroredType<open_spiel::GameType::Information>
    : std::true_type {};
template <>
struct jlcxx::IsMirroredType<open_spiel::GameType::Utility> : std::true_type {};
template <>
struct jlcxx::IsMirroredType<open_spiel::GameType::RewardModel>
    : std::true_type {};
template <>
struct jlcxx::IsMirroredType<open_spiel::PlayerId> : std::true_type {};
template <>
struct jlcxx::IsMirroredType<open_spiel::algorithms::ChildSelectionPolicy>
    : std::true_type {};

template <>
struct jlcxx::IsMirroredType<std::pair<open_spiel::Action, double>>
    : std::true_type {};

template <typename K, typename V>
struct jlcxx::julia_type_factory<std::pair<K, V>> {
  static jl_datatype_t* julia_type() {
    return (jl_datatype_t*)apply_type(
        jlcxx::julia_type("Pair"),
        jl_svec2(julia_base_type<K>(), julia_base_type<V>()));
  }
};

JLCXX_MODULE define_julia_module(jlcxx::Module& mod) {
  jlcxx::stl::apply_stl<std::pair<open_spiel::Action, double>>(mod);
  jlcxx::stl::apply_stl<std::vector<std::pair<open_spiel::Action, double>>>(
      mod);
  jlcxx::stl::apply_stl<std::vector<double>>(mod);
  jlcxx::stl::apply_stl<std::vector<std::vector<double>>>(mod);
  jlcxx::stl::apply_stl<std::vector<int>>(mod);
  jlcxx::stl::apply_stl<std::vector<std::vector<int>>>(mod);
  jlcxx::stl::apply_stl<std::vector<open_spiel::Action>>(mod);

  mod.add_bits<open_spiel::GameParameter::Type>("GameParameterStateType",
                                                jlcxx::julia_type("CppEnum"));
  mod.set_const("UNSET", open_spiel::GameParameter::Type::kUnset);
  mod.set_const("INT", open_spiel::GameParameter::Type::kInt);
  mod.set_const("DOUBLE", open_spiel::GameParameter::Type::kDouble);
  mod.set_const("STRING", open_spiel::GameParameter::Type::kString);
  mod.set_const("BOOL", open_spiel::GameParameter::Type::kBool);

  mod.add_type<open_spiel::GameParameter>("GameParameter")
      .constructor<const std::string&>()
      .constructor<const std::string&, const bool&>()
      .constructor<const bool&>()
      .constructor<const bool&, const bool&>()
      .constructor<const int&>()
      .constructor<const int&, const bool&>()
      .constructor<const double&>()
      .constructor<const double&, const bool&>()
      .constructor<const open_spiel::GameParameter::Type&>()
      .constructor<const open_spiel::GameParameter::Type&, const bool&>()
      .method("is_mandatory", &open_spiel::GameParameter::is_mandatory)
      .method("to_string", &open_spiel::GameParameter::ToString)
      .method("to_repr_string", &open_spiel::GameParameter::ToReprString);

  // !!! not a good pratice to do so
  mod.add_type<jlcxx::Parametric<jlcxx::TypeVar<1>, jlcxx::TypeVar<2>>>(
         "StdMap", jlcxx::julia_type("AbstractDict", "Base"))
      .apply<std::map<std::string, open_spiel::GameParameter>,
             std::unordered_map<open_spiel::Action, double>,
             std::unordered_map<std::string, open_spiel::Action>,
             std::unordered_map<std::string, open_spiel::ActionsAndProbs>,
             std::unordered_map<std::string, int>>([](auto wrapped) {
        typedef typename decltype(wrapped)::type WrappedT;
        typedef typename WrappedT::key_type WrappedKey;
        typedef typename WrappedT::mapped_type WrappedVal;

        wrapped.module().set_override_module(jl_base_module);

        wrapped.method("length", &WrappedT::size);
        wrapped.method("getindex",
                       [](WrappedT& w, WrappedKey& k) { return w[k]; });
        wrapped.method("setindex!", [](WrappedT& w, WrappedVal& v,
                                       WrappedKey& k) { return w[k] = v; });
        wrapped.method("keys", [](WrappedT ps) {
          std::vector<WrappedKey> keys;
          keys.reserve(ps.size());
          for (auto const& it : ps) {
            keys.emplace_back(it.first);
          }
          return keys;
        });

        wrapped.module().unset_override_module();
      });

  mod.add_bits<open_spiel::StateType>("StateType",
                                      jlcxx::julia_type("CppEnum"));
  mod.set_const("TERMINAL_STATE", open_spiel::StateType::kTerminal);
  mod.set_const("CHANCE_STATE", open_spiel::StateType::kChance);
  mod.set_const("DECISION_STATE", open_spiel::StateType::kDecision);

  mod.add_bits<open_spiel::GameType::Dynamics>("Dynamics",
                                               jlcxx::julia_type("CppEnum"));
  mod.set_const("SEQUENTIAL", open_spiel::GameType::Dynamics::kSequential);
  mod.set_const("SIMULTANEOUS", open_spiel::GameType::Dynamics::kSimultaneous);

  mod.add_bits<open_spiel::GameType::ChanceMode>("ChanceMode",
                                                 jlcxx::julia_type("CppEnum"));
  mod.set_const("DETERMINISTIC",
                open_spiel::GameType::ChanceMode::kDeterministic);
  mod.set_const("EXPLICIT_STOCHASTIC",
                open_spiel::GameType::ChanceMode::kExplicitStochastic);
  mod.set_const("SAMPLED_STOCHASTIC",
                open_spiel::GameType::ChanceMode::kSampledStochastic);

  mod.add_bits<open_spiel::GameType::Information>("Information",
                                                  jlcxx::julia_type("CppEnum"));
  mod.set_const("ONE_SHOT", open_spiel::GameType::Information::kOneShot);
  mod.set_const("PERFECT_INFORMATION",
                open_spiel::GameType::Information::kPerfectInformation);
  mod.set_const("IMPERFECT_INFORMATION",
                open_spiel::GameType::Information::kImperfectInformation);

  mod.add_bits<open_spiel::GameType::Utility>("Utility",
                                              jlcxx::julia_type("CppEnum"));
  mod.set_const("ZERO_SUM", open_spiel::GameType::Utility::kZeroSum);
  mod.set_const("CONSTANT_SUM", open_spiel::GameType::Utility::kConstantSum);
  mod.set_const("GENERAL_SUM", open_spiel::GameType::Utility::kGeneralSum);
  mod.set_const("IDENTICAL", open_spiel::GameType::Utility::kIdentical);

  mod.add_bits<open_spiel::GameType::RewardModel>("RewardModel",
                                                  jlcxx::julia_type("CppEnum"));
  mod.set_const("REWARDS", open_spiel::GameType::RewardModel::kRewards);
  mod.set_const("TERMINAL", open_spiel::GameType::RewardModel::kTerminal);

  mod.add_type<open_spiel::GameType>("GameType")
      .method("short_name",
              [](const open_spiel::GameType& gt) { return gt.short_name; })
      .method("long_name",
              [](const open_spiel::GameType& gt) { return gt.long_name; })
      .method("dynamics",
              [](const open_spiel::GameType& gt) { return gt.dynamics; })
      .method("chance_mode",
              [](const open_spiel::GameType& gt) { return gt.chance_mode; })
      .method("information",
              [](const open_spiel::GameType& gt) { return gt.information; })
      .method("utility",
              [](const open_spiel::GameType& gt) { return gt.utility; })
      .method("reward_model",
              [](const open_spiel::GameType& gt) { return gt.reward_model; })
      .method("max_num_players",
              [](const open_spiel::GameType& gt) { return gt.max_num_players; })
      .method("min_num_players",
              [](const open_spiel::GameType& gt) { return gt.min_num_players; })
      .method("default_loadable",
              [](const open_spiel::GameType& gt) {
                return gt.default_loadable;
              })
      .method("provides_information_state_string",
              [](const open_spiel::GameType& gt) {
                return gt.provides_information_state_string;
              })
      .method("provides_information_state_tensor",
              [](const open_spiel::GameType& gt) {
                return gt.provides_information_state_tensor;
              })
      .method("provides_observation_string",
              [](const open_spiel::GameType& gt) {
                return gt.provides_observation_string;
              })
      .method("provides_observation_tensor",
              [](const open_spiel::GameType& gt) {
                return gt.provides_observation_tensor;
              })
      .method("provides_factored_observation_string",
              [](const open_spiel::GameType& gt) {
                return gt.provides_factored_observation_string;
              })
      .method("parameter_specification", [](const open_spiel::GameType& gt) {
        return gt.parameter_specification;
      });

  mod.add_bits<open_spiel::PlayerId>("PlayerId", jlcxx::julia_type("CppEnum"));
  mod.set_const("INVALID_PLAYER", open_spiel::kInvalidPlayer);
  mod.set_const("TERMINAL_PLAYER", open_spiel::kTerminalPlayerId);
  mod.set_const("CHANCE_PLAYER", open_spiel::kChancePlayerId);
  mod.set_const("SIMULTANEOUS_PLAYER", open_spiel::kSimultaneousPlayerId);

  mod.set_const("INVALID_ACTION", open_spiel::kInvalidAction);

  mod.add_type<open_spiel::State>("State")
      .method("current_player", &open_spiel::State::CurrentPlayer)
      .method("apply_action", &open_spiel::State::ApplyAction)
      .method("legal_actions",
              [](open_spiel::State& s) { return s.LegalActions(); })
      .method("legal_actions",
              [](open_spiel::State& s, open_spiel::Player p) {
                return s.LegalActions(p);
              })
      .method("legal_actions_mask",
              [](open_spiel::State& s) { return s.LegalActionsMask(); })
      .method("legal_actions_mask",
              [](open_spiel::State& s, open_spiel::Player p) {
                return s.LegalActionsMask(p);
              })
      .method("action_to_string",
              [](open_spiel::State& s, open_spiel::Player p,
                 open_spiel::Action a) { return s.ActionToString(p, a); })
      .method("action_to_string",
              [](open_spiel::State& s, open_spiel::Action a) {
                return s.ActionToString(a);
              })
      .method("string_to_action",
              [](open_spiel::State& s, open_spiel::Player p,
                 const std::string& action_str) {
                return s.StringToAction(p, action_str);
              })
      .method("string_to_action",
              [](open_spiel::State& s, const std::string& action_str) {
                return s.StringToAction(action_str);
              })
      .method("to_string", &open_spiel::State::ToString)
      .method("is_terminal", &open_spiel::State::IsTerminal)
      .method("rewards", &open_spiel::State::Rewards)
      .method("returns", &open_spiel::State::Returns)
      .method("player_reward", &open_spiel::State::PlayerReward)
      .method("player_return", &open_spiel::State::PlayerReturn)
      .method("is_chance_node", &open_spiel::State::IsChanceNode)
      .method("is_simultaneous_node", &open_spiel::State::IsSimultaneousNode)
      .method("is_mean_field_node", &open_spiel::State::IsMeanFieldNode)
      .method("is_player_node", &open_spiel::State::IsPlayerNode)
      .method("history", &open_spiel::State::History)
      .method("history_str", &open_spiel::State::HistoryString)
      .method("information_state_string",
              [](open_spiel::State& s, open_spiel::Player p) {
                return s.InformationStateString(p);
              })
      .method("information_state_string",
              [](open_spiel::State& s) { return s.InformationStateString(); })
      .method("information_state_tensor",
              [](open_spiel::State& s) { return s.InformationStateTensor(); })
      .method("information_state_tensor",
              [](open_spiel::State& s, open_spiel::Player p) {
                return s.InformationStateTensor(p);
              })
      .method("information_state_as_normalized_vector",
              [](open_spiel::State& s, open_spiel::Player p,
                 std::vector<float> data) {
                return s.InformationStateTensor(p, &data);
              })
      .method("observation_string",
              [](open_spiel::State& s) { return s.ObservationString(); })
      .method("observation_string",
              [](open_spiel::State& s, open_spiel::Player p) {
                return s.ObservationString(p);
              })
      .method("observation_tensor",
              [](open_spiel::State& s) { return s.ObservationTensor(); })
      .method("observation_tensor",
              [](open_spiel::State& s, open_spiel::Player p) {
                return s.ObservationTensor(p);
              })
      .method("clone", &open_spiel::State::Clone)
      .method("child", &open_spiel::State::Child)
      .method("undo_action", &open_spiel::State::UndoAction)
      .method("apply_actions",
              [](open_spiel::State& s, std::vector<open_spiel::Action> data) {
                return s.ApplyActions(data);
              })
      .method("num_distinct_actions", &open_spiel::State::NumDistinctActions)
      .method("num_players", &open_spiel::State::NumPlayers)
      .method("chance_outcomes", &open_spiel::State::ChanceOutcomes)
      .method("get_type", &open_spiel::State::GetType)
      .method("serialize", &open_spiel::State::Serialize)
      .method("distribution_support", &open_spiel::State::DistributionSupport)
      .method("update_distribution",
              [](open_spiel::State& s, std::vector<double> distribution) {
                return s.UpdateDistribution(distribution);
              });

  mod.add_type<open_spiel::Game>("Game")
      .method("num_distinct_actions", &open_spiel::Game::NumDistinctActions)
      .method("new_initial_state",
              [](open_spiel::Game& g) { return g.NewInitialState(); })
      .method("new_initial_state_from_string",
              [](open_spiel::Game& g, const std::string& s) {
                return g.NewInitialState(s);
              })
      .method("max_chance_outcomes", &open_spiel::Game::MaxChanceOutcomes)
      .method("get_parameters", &open_spiel::Game::GetParameters)
      .method("num_players", &open_spiel::Game::NumPlayers)
      .method("min_utility", &open_spiel::Game::MinUtility)
      .method("max_utility", &open_spiel::Game::MaxUtility)
      .method("get_type", &open_spiel::Game::GetType)
      .method("utility_sum", &open_spiel::Game::UtilitySum)
      .method("information_state_tensor_shape",
              &open_spiel::Game::InformationStateTensorShape)
      .method("information_state_tensor_size",
              &open_spiel::Game::InformationStateTensorSize)
      .method("observation_tensor_shape",
              &open_spiel::Game::ObservationTensorShape)
      .method("observation_tensor_size",
              &open_spiel::Game::ObservationTensorSize)
      .method("deserialize_state", &open_spiel::Game::DeserializeState)
      .method("max_game_length", &open_spiel::Game::MaxGameLength)
      .method("to_string", &open_spiel::Game::ToString);

  mod.add_type<open_spiel::SimMoveGame>("SimMoveGame");
  mod.add_type<open_spiel::NormalFormGame>("NormalFormGame");

  mod.add_type<open_spiel::matrix_game::MatrixGame>(
         "MatrixGame", jlcxx::julia_base_type<open_spiel::Game>())
      .constructor<open_spiel::GameType, open_spiel::GameParameters,
                   std::vector<std::string>, std::vector<std::string>,
                   std::vector<double>, std::vector<double>>()
      .constructor<open_spiel::GameType, open_spiel::GameParameters,
                   std::vector<std::string>, std::vector<std::string>,
                   const std::vector<std::vector<double>>&,
                   const std::vector<std::vector<double>>&>()
      .method("num_rows", &open_spiel::matrix_game::MatrixGame::NumRows)
      .method("num_cols", &open_spiel::matrix_game::MatrixGame::NumCols)
      .method("row_utility", &open_spiel::matrix_game::MatrixGame::RowUtility)
      .method("col_utility", &open_spiel::matrix_game::MatrixGame::ColUtility)
      .method("player_utility",
              &open_spiel::matrix_game::MatrixGame::PlayerUtility)
      .method("row_action_name",
              &open_spiel::matrix_game::MatrixGame::RowActionName)
      .method("col_action_name",
              &open_spiel::matrix_game::MatrixGame::ColActionName);

  mod.method(
      "create_matrix_game",
      [](const std::string& a, const std::string& b,
         const std::vector<std::string>& c, const std::vector<std::string>& d,
         const std::vector<std::vector<double>>& e,
         const std::vector<std::vector<double>>& f) {
        return open_spiel::matrix_game::CreateMatrixGame(a, b, c, d, e, f);
      });
  mod.method("create_matrix_game",
             [](const std::vector<std::vector<double>>& a,
                const std::vector<std::vector<double>>& b) {
               return open_spiel::matrix_game::CreateMatrixGame(a, b);
             });
  mod.method("_load_game",
             [](const std::string& s) { return open_spiel::LoadGame(s); });
  mod.method("_load_game",
             [](const std::string& s, const open_spiel::GameParameters& ps) {
               return open_spiel::LoadGame(s, ps);
             });
  mod.method("_load_game_as_turn_based", [](const std::string& s) {
    return open_spiel::LoadGameAsTurnBased(s);
  });
  mod.method("_load_game_as_turn_based",
             [](const std::string& s, const open_spiel::GameParameters& ps) {
               return open_spiel::LoadGameAsTurnBased(s, ps);
             });
  mod.method("load_matrix_game", &open_spiel::algorithms::LoadMatrixGame);
  mod.method("extensive_to_matrix_game",
             &open_spiel::algorithms::ExtensiveToMatrixGame);
  mod.method("registered_names", &open_spiel::GameRegisterer::RegisteredNames);
  mod.method("registered_games", &open_spiel::GameRegisterer::RegisteredGames);

  mod.add_type<open_spiel::Bot>("Bot")
      .method("step", &open_spiel::Bot::Step)
      .method("restart", &open_spiel::Bot::Restart)
      .method("restart_at", &open_spiel::Bot::RestartAt)
      .method("provides_force_action", &open_spiel::Bot::ProvidesForceAction)
      .method("force_action", &open_spiel::Bot::ForceAction)
      .method("provides_policy", &open_spiel::Bot::ProvidesPolicy)
      .method("get_policy", &open_spiel::Bot::GetPolicy)
      .method("step_with_policy", &open_spiel::Bot::StepWithPolicy);

  jlcxx::stl::apply_stl<open_spiel::Bot*>(mod);

  mod.add_type<open_spiel::Policy>("Policy")
      .method("get_state_policy_as_parallel_vectors",
              [](open_spiel::Policy p, const open_spiel::State& state) {
                return p.GetStatePolicyAsParallelVectors(state);
              })
      .method("get_state_policy_as_parallel_vectors",
              [](open_spiel::Policy p, const std::string state) {
                return p.GetStatePolicyAsParallelVectors(state);
              })
      .method("get_state_policy_as_map",
              [](open_spiel::Policy p, const open_spiel::State& state) {
                return p.GetStatePolicyAsMap(state);
              })
      .method("get_state_policy_as_map",
              [](open_spiel::Policy p, const std::string state) {
                return p.GetStatePolicyAsMap(state);
              })
      .method("get_state_policy",
              [](open_spiel::Policy p, const open_spiel::State& state) {
                return p.GetStatePolicy(state);
              })
      .method("get_state_policy",
              [](open_spiel::Policy p, const std::string state) {
                return p.GetStatePolicy(state);
              });

  jlcxx::stl::apply_stl<const open_spiel::Policy*>(mod);

  mod.add_type<open_spiel::TabularPolicy>(
         "TabularPolicy", jlcxx::julia_base_type<open_spiel::Policy>())
      .constructor<const open_spiel::Game&>()
      .constructor<
          const std::unordered_map<std::string, open_spiel::ActionsAndProbs>&>()
      .method("get_state_policy", &open_spiel::TabularPolicy::GetStatePolicy)
      .method("policy_table",
              [](open_spiel::TabularPolicy p) { return p.PolicyTable(); })
      .method("get_state_policy",
              [](open_spiel::TabularPolicy p, const open_spiel::State& state) {
                return p.GetStatePolicy(state.InformationStateString());
              })
      .method("get_state_policy",
              [](open_spiel::TabularPolicy p, const std::string& state) {
                return p.GetStatePolicy(state);
              });

  jlcxx::stl::apply_stl<open_spiel::TabularPolicy>(mod);

  mod.method("get_empty_tabular_policy", &open_spiel::GetEmptyTabularPolicy);
  mod.method("get_uniform_policy", &open_spiel::GetUniformPolicy);
  mod.method("get_random_policy", &open_spiel::GetRandomPolicy);
  mod.method("get_first_action_policy", &open_spiel::GetFirstActionPolicy);

  // !!! Bots below are not exported directly in c++
  // !!! which makes it hard to dispatch overriden methods
  // mod.method("make_uniform_random_bot", &open_spiel::MakeUniformRandomBot);
  // mod.method("make_fixed_action_preference_bot",
  // &open_spiel::MakeFixedActionPreferenceBot); mod.method("make_policy_bot",
  // [](const open_spiel::Game& game, open_spiel::Player pid, int seed,
  // open_spiel::Policy policy) { return open_spiel::MakePolicyBot(game, pid,
  // seed, std::make_unique<open_spiel::Policy>(policy)); });

  // !!! just a workaround here
  mod.add_type<std::pair<std::shared_ptr<const open_spiel::Game>,
                         std::unique_ptr<open_spiel::State>>>("GameStatePair")
      .method("first",
              [](std::pair<std::shared_ptr<const open_spiel::Game>,
                           std::unique_ptr<open_spiel::State>>& p) {
                return p.first;
              })
      .method("last", [](std::pair<std::shared_ptr<const open_spiel::Game>,
                                   std::unique_ptr<open_spiel::State>>& p) {
        return std::move(p.second);
      });

  mod.method("serialize_game_and_state", &open_spiel::SerializeGameAndState);
  mod.method("_deserialize_game_and_state",
             &open_spiel::DeserializeGameAndState);

  mod.add_type<open_spiel::algorithms::Evaluator>("Evaluator");

  mod.add_type<open_spiel::algorithms::RandomRolloutEvaluator>(
         "RandomRolloutEvaluator",
         jlcxx::julia_base_type<open_spiel::algorithms::Evaluator>())
      .constructor<int, int>()
      .method("evaluate", &open_spiel::algorithms::Evaluator::Evaluate)
      .method("prior", &open_spiel::algorithms::Evaluator::Prior);

  mod.method("random_rollout_evaluator_factory", [](int rollouts, int seed) {
    return std::shared_ptr<open_spiel::algorithms::Evaluator>(
        new open_spiel::algorithms::RandomRolloutEvaluator(rollouts, seed));
  });

  mod.add_bits<open_spiel::algorithms::ChildSelectionPolicy>(
      "ChildSelectionPolicy", jlcxx::julia_type("CppEnum"));
  mod.set_const("UCT", open_spiel::algorithms::ChildSelectionPolicy::UCT);
  mod.set_const("PUCT", open_spiel::algorithms::ChildSelectionPolicy::PUCT);

  mod.add_type<open_spiel::algorithms::SearchNode>("SearchNode")
      .constructor<open_spiel::Action, open_spiel::Player, double>()
      .method("UCTValue", &open_spiel::algorithms::SearchNode::UCTValue)
      .method("PUCTValue", &open_spiel::algorithms::SearchNode::PUCTValue)
      .method("compare_final",
              &open_spiel::algorithms::SearchNode::CompareFinal)
      .method("best_child", &open_spiel::algorithms::SearchNode::BestChild)
      .method("to_string", &open_spiel::algorithms::SearchNode::ToString)
      .method("children_str", &open_spiel::algorithms::SearchNode::ChildrenStr)
      // TODO(author11): https://github.com/JuliaInterop/CxxWrap.jl/issues/90
      .method("get_action",
              [](open_spiel::algorithms::SearchNode& sn) { return sn.action; })
      .method("get_prior",
              [](open_spiel::algorithms::SearchNode& sn) { return sn.prior; })
      .method("get_player",
              [](open_spiel::algorithms::SearchNode& sn) { return sn.player; })
      .method("get_explore_count",
              [](open_spiel::algorithms::SearchNode& sn) {
                return sn.explore_count;
              })
      .method("get_total_reward",
              [](open_spiel::algorithms::SearchNode& sn) {
                return sn.total_reward;
              })
      .method("get_outcome",
              [](open_spiel::algorithms::SearchNode& sn) { return sn.outcome; })
      .method("set_action!",
              [](open_spiel::algorithms::SearchNode& sn,
                 open_spiel::Action action) { sn.action = action; })
      .method("set_prior!", [](open_spiel::algorithms::SearchNode& sn,
                               double prior) { sn.prior = prior; })
      .method("set_player!",
              [](open_spiel::algorithms::SearchNode& sn,
                 open_spiel::Player player) { sn.player = player; })
      .method("set_explore_count!",
              [](open_spiel::algorithms::SearchNode& sn, int explore_count) {
                sn.explore_count = explore_count;
              })
      .method("set_total_reward!",
              [](open_spiel::algorithms::SearchNode& sn, double total_reward) {
                sn.total_reward = total_reward;
              })
      .method("set_outcome!",
              [](open_spiel::algorithms::SearchNode& sn,
                 std::vector<double> outcome) { sn.outcome = outcome; });

  jlcxx::stl::apply_stl<open_spiel::algorithms::SearchNode>(mod);

  mod.method("get_children", [](open_spiel::algorithms::SearchNode& sn) {
    return sn.children;
  });
  mod.method("set_children!",
             [](open_spiel::algorithms::SearchNode& sn,
                std::vector<open_spiel::algorithms::SearchNode> children) {
               sn.children = children;
             });

  mod.add_type<open_spiel::algorithms::MCTSBot>(
         "MCTSBot", jlcxx::julia_base_type<open_spiel::Bot>())
      .constructor<const open_spiel::Game&,
                   std::shared_ptr<open_spiel::algorithms::Evaluator>,
                   double, int, int64_t, bool, int, bool,
                   open_spiel::algorithms::ChildSelectionPolicy, double,
                   double>()
      .method("restart", &open_spiel::algorithms::MCTSBot::Restart)
      .method("restart_at", &open_spiel::algorithms::MCTSBot::RestartAt)
      .method("step", &open_spiel::algorithms::MCTSBot::Step)
      .method("step_with_policy",
              &open_spiel::algorithms::MCTSBot::StepWithPolicy)
      .method("mcts_search", &open_spiel::algorithms::MCTSBot::MCTSearch);

  jlcxx::stl::apply_stl<open_spiel::algorithms::MCTSBot*>(mod);

  mod.add_type<open_spiel::algorithms::TabularBestResponse>(
         "TabularBestResponse")
      .constructor<const open_spiel::Game&, open_spiel::Player,
                   const open_spiel::Policy*>()
      .constructor<
          const open_spiel::Game&, open_spiel::Player,
          const std::unordered_map<std::string, open_spiel::ActionsAndProbs>&>()
      .method("best_response_action",
              [](open_spiel::algorithms::TabularBestResponse& t,
                 const std::string& infostate) {
                return t.BestResponseAction(infostate);
              })
      .method(
          "get_best_response_actions",
          &open_spiel::algorithms::TabularBestResponse::GetBestResponseActions)
      .method(
          "get_best_response_policy",
          &open_spiel::algorithms::TabularBestResponse::GetBestResponsePolicy)
      .method("value",
              [](open_spiel::algorithms::TabularBestResponse& t,
                 const std::string& history) {
                return t.Value(history);
              })
      .method("set_policy",
              [](open_spiel::algorithms::TabularBestResponse& t,
                 const open_spiel::Policy* p) { return t.SetPolicy(p); })
      .method(
          "set_policy",
          [](open_spiel::algorithms::TabularBestResponse& t,
             std::unordered_map<std::string, open_spiel::ActionsAndProbs>& p) {
            return t.SetPolicy(p);
          });

  mod.add_type<open_spiel::algorithms::CFRSolverBase>("CFRSolverBase")
      .method("evaluate_and_update_policy",
              &open_spiel::algorithms::CFRSolver::EvaluateAndUpdatePolicy)
      .method("current_policy",
              &open_spiel::algorithms::CFRSolver::CurrentPolicy)
      .method("average_policy",
              &open_spiel::algorithms::CFRSolver::AveragePolicy);

  mod.add_type<open_spiel::algorithms::CFRSolver>(
         "CFRSolver",
         jlcxx::julia_base_type<open_spiel::algorithms::CFRSolverBase>())
      .constructor<const open_spiel::Game&>();

  mod.add_type<open_spiel::algorithms::CFRPlusSolver>(
         "CFRPlusSolver",
         jlcxx::julia_base_type<open_spiel::algorithms::CFRSolverBase>())
      .constructor<const open_spiel::Game&>();

  mod.add_type<open_spiel::algorithms::CFRBRSolver>(
         "CFRBRSolver",
         jlcxx::julia_base_type<open_spiel::algorithms::CFRSolverBase>())
      .constructor<const open_spiel::Game&>()
      .method("evaluate_and_update_policy",
              &open_spiel::algorithms::CFRSolver::EvaluateAndUpdatePolicy);

  mod.add_type<open_spiel::algorithms::TrajectoryRecorder>("TrajectoryRecorder")
      .constructor<const open_spiel::Game&,
                   const std::unordered_map<std::string, int>&, int>();

  mod.method("evaluate_bots", [](open_spiel::State* state,
                                 const std::vector<open_spiel::Bot*>& bots,
                                 int seed) {
    return open_spiel::EvaluateBots(state, bots, seed);
  });
  mod.method("exploitability", [](const open_spiel::Game& game,
                                  const open_spiel::Policy& policy) {
    return open_spiel::algorithms::Exploitability(game, policy);
  });
  mod.method(
      "exploitability",
      [](const open_spiel::Game& game,
         const std::unordered_map<std::string, open_spiel::ActionsAndProbs>&
             policy) {
        return open_spiel::algorithms::Exploitability(game, policy);
      });
  mod.method("nash_conv", [](const open_spiel::Game& game,
                             const open_spiel::Policy& policy) {
    return open_spiel::algorithms::NashConv(game, policy);
  });
  mod.method(
      "nash_conv",
      [](const open_spiel::Game& game,
         const std::unordered_map<std::string, open_spiel::ActionsAndProbs>&
             policy) {
        return open_spiel::algorithms::NashConv(game, policy);
      });
  mod.method("convert_to_turn_based", &open_spiel::ConvertToTurnBased);
  mod.method("expected_returns",
             [](const open_spiel::State& state,
                const std::vector<const open_spiel::Policy*> policies,
                int depth_limit) {
               return open_spiel::algorithms::ExpectedReturns(state, policies,
                                                              depth_limit);
             });
  mod.method("expected_returns",
             [](const open_spiel::State& state,
                const open_spiel::Policy& joint_policy, int depth_limit) {
               return open_spiel::algorithms::ExpectedReturns(
                   state, joint_policy, depth_limit);
             });

  mod.add_type<open_spiel::algorithms::BatchedTrajectory>("BatchedTrajectory")
      .constructor<int>()
      .method("observations",
              [](open_spiel::algorithms::BatchedTrajectory bt) {
                return bt.observations;
              })
      .method("state_indices",
              [](open_spiel::algorithms::BatchedTrajectory bt) {
                return bt.state_indices;
              })
      .method("legal_actions",
              [](open_spiel::algorithms::BatchedTrajectory bt) {
                return bt.legal_actions;
              })
      .method("actions",
              [](open_spiel::algorithms::BatchedTrajectory bt) {
                return bt.actions;
              })
      .method("player_policies",
              [](open_spiel::algorithms::BatchedTrajectory bt) {
                return bt.player_policies;
              })
      .method("player_ids",
              [](open_spiel::algorithms::BatchedTrajectory bt) {
                return bt.player_ids;
              })
      .method("rewards",
              [](open_spiel::algorithms::BatchedTrajectory bt) {
                return bt.rewards;
              })
      .method(
          "valid",
          [](open_spiel::algorithms::BatchedTrajectory bt) { return bt.valid; })
      .method("next_is_terminal",
              [](open_spiel::algorithms::BatchedTrajectory bt) {
                return bt.next_is_terminal;
              })
      .method("max_trajectory_length",
              [](open_spiel::algorithms::BatchedTrajectory bt) {
                return bt.max_trajectory_length;
              })
      .method("resize_fields",
              &open_spiel::algorithms::BatchedTrajectory::ResizeFields);

  mod.method("record_batched_trajectories",
             [](const open_spiel::Game& game,
                const std::vector<open_spiel::TabularPolicy>& policies,
                const std::unordered_map<std::string, int>& state_to_index,
                int batch_size, bool include_full_observations, int seed,
                int max_unroll_length) {
               return open_spiel::algorithms::RecordBatchedTrajectory(
                   game, policies, state_to_index, batch_size,
                   include_full_observations, seed, max_unroll_length);
             });
}  // NOLINT(readability/fn_size)
