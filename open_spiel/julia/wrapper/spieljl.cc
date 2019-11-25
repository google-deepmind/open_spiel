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
#include "open_spiel/query.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"
#include "open_spiel/spiel_utils.h"

// ??? how to map the `std::map<K, V>` to `Dict{K, V}`

template<typename K, typename V>
std::map<K, V> keys_vals_to_map(jlcxx::ArrayRef<K, 1> keys, jlcxx::ArrayRef<V, 1> vals)
{
  assert(keys.size() == vals.size());

  std::map<K, V> m;

  for (int i=0; i<keys.size(); i++)
  {
    m[keys[i]] = vals[i];
  }

  return m;
}

template<typename K, typename V>
std::tuple<std::vector<K>, std::vector<V>> map_to_keys_vals(std::map<K, V> m)
{
  std::vector<K> keys;
  std::vector<V> vals;

  for (auto const& [key, val] : m)
  {
    keys.push_back(key);
    vals.push_back(val);
  }

  return std::make_tuple(keys, vals);
}

template<> struct jlcxx::IsMirroredType<open_spiel::GameParameter::Type> : std::true_type {};
template<> struct jlcxx::IsMirroredType<open_spiel::StateType> : std::true_type {};
template<> struct jlcxx::IsMirroredType<open_spiel::GameType::Dynamics> : std::true_type {};
template<> struct jlcxx::IsMirroredType<open_spiel::GameType::ChanceMode> : std::true_type {};
template<> struct jlcxx::IsMirroredType<open_spiel::GameType::Information> : std::true_type {};
template<> struct jlcxx::IsMirroredType<open_spiel::GameType::Utility> : std::true_type {};
template<> struct jlcxx::IsMirroredType<open_spiel::GameType::RewardModel> : std::true_type {};
template<> struct jlcxx::IsMirroredType<open_spiel::PlayerId> : std::true_type {};

template<> struct jlcxx::IsMirroredType<std::pair<open_spiel::Action, double>> : std::true_type {};

template<> struct jlcxx::julia_type_factory<std::pair<open_spiel::Action, double>>
{
  static jl_datatype_t* julia_type()
  {
    return (jl_datatype_t*)apply_type(jlcxx::julia_type("Pair"), jl_svec2(julia_base_type<open_spiel::Action>(), julia_base_type<double>()));
  }
};

// TODO: use stl once the enum error got fixed in CxxWrap@v0.8.2
JLCXX_MODULE define_julia_module(jlcxx::Module& mod)
{
  jlcxx::stl::apply_stl<std::pair<open_spiel::Action, double>>(mod);

  mod.add_bits<open_spiel::GameParameter::Type>("GameParameterStateType", jlcxx::julia_type("CppEnum"));
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
    .constructor<const double&>()
    .constructor<const double&, const bool&>()
    .method("is_mandatory", &open_spiel::GameParameter::is_mandatory);

  mod.add_bits<open_spiel::StateType>("StateType", jlcxx::julia_type("CppEnum"));
  mod.set_const("TERMINAL_STATE", open_spiel::StateType::kTerminal);
  mod.set_const("CHANCE_STATE", open_spiel::StateType::kChance);
  mod.set_const("DECISION_STATE", open_spiel::StateType::kDecision);

  mod.add_type<open_spiel::GameType>("GameType");

  mod.add_bits<open_spiel::GameType::Dynamics>("Dynamics", jlcxx::julia_type("CppEnum"));
  mod.set_const("SEQUENTIAL", open_spiel::GameType::Dynamics::kSequential);
  mod.set_const("SIMULTANEOUS", open_spiel::GameType::Dynamics::kSimultaneous);

  mod.add_bits<open_spiel::GameType::ChanceMode>("ChanceMode", jlcxx::julia_type("CppEnum"));
  mod.set_const("DETERMINISTIC", open_spiel::GameType::ChanceMode::kDeterministic);
  mod.set_const("EXPLICIT_STOCHASTIC", open_spiel::GameType::ChanceMode::kExplicitStochastic);
  mod.set_const("SAMPLED_STOCHASTIC", open_spiel::GameType::ChanceMode::kSampledStochastic);

  mod.add_bits<open_spiel::GameType::Information>("Information", jlcxx::julia_type("CppEnum"));
  mod.set_const("ONE_SHOT", open_spiel::GameType::Information::kOneShot);
  mod.set_const("PERFECT_INFORMATION", open_spiel::GameType::Information::kPerfectInformation);
  mod.set_const("IMPERFECT_INFORMATION", open_spiel::GameType::Information::kImperfectInformation);

  mod.add_bits<open_spiel::GameType::Utility>("Utility", jlcxx::julia_type("CppEnum"));
  mod.set_const("ZERO_SUM", open_spiel::GameType::Utility::kZeroSum);
  mod.set_const("CONSTANT_SUM", open_spiel::GameType::Utility::kConstantSum);
  mod.set_const("GENERAL_SUM", open_spiel::GameType::Utility::kGeneralSum);
  mod.set_const("IDENTICAL", open_spiel::GameType::Utility::kIdentical);

  mod.add_bits<open_spiel::GameType::RewardModel>("RewardModel", jlcxx::julia_type("CppEnum"));
  mod.set_const("REWARDS", open_spiel::GameType::RewardModel::kRewards);
  mod.set_const("TERMINAL", open_spiel::GameType::RewardModel::kTerminal);

  mod.add_bits<open_spiel::PlayerId>("PlayerId", jlcxx::julia_type("CppEnum"));
  mod.set_const("INVALID_PLAYER", open_spiel::kInvalidPlayer);
  mod.set_const("TERMINAL_PLAYER", open_spiel::kTerminalPlayerId);
  mod.set_const("CHANCE_PLAYER", open_spiel::kChancePlayerId);
  mod.set_const("SIMULTANEOUS_PLAYER", open_spiel::kSimultaneousPlayerId);

  mod.set_const("INVALID_ACTION", open_spiel::kInvalidAction);

  mod.add_type<open_spiel::State>("State")
    .method("current_player", &open_spiel::State::CurrentPlayer)
    .method("apply_action", &open_spiel::State::ApplyAction)
    .method("legal_actions", [](open_spiel::State &s) { return s.LegalActions(); })
    .method("legal_actions", [](open_spiel::State &s, open_spiel::Player p) { return s.LegalActions(p); })
    .method("legal_actions_mask", [](open_spiel::State &s) { return s.LegalActionsMask();})
    .method("legal_actions_mask", [](open_spiel::State &s, open_spiel::Player p) { return s.LegalActionsMask(p);})
    .method("action_to_string", [](open_spiel::State &s, open_spiel::Player p, open_spiel::Action a) { return s.ActionToString(p, a);})
    .method("action_to_string", [](open_spiel::State &s, open_spiel::Action a) { return s.ActionToString(a);})
    .method("string_to_action", [](open_spiel::State &s, open_spiel::Player p, const std::string& action_str) { return s.StringToAction(p, action_str);})
    .method("string_to_action", [](open_spiel::State &s, const std::string& action_str) { return s.StringToAction(action_str);})
    .method("string", &open_spiel::State::ToString)
    .method("is_terminal", &open_spiel::State::IsTerminal)
    .method("rewards", &open_spiel::State::Rewards)
    .method("returns", &open_spiel::State::Returns)
    .method("player_reward", &open_spiel::State::PlayerReward)
    .method("player_return", &open_spiel::State::PlayerReturn)
    .method("is_chance_node", &open_spiel::State::IsChanceNode)
    .method("is_simultaneous_node", &open_spiel::State::IsSimultaneousNode)
    .method("history", &open_spiel::State::History)
    .method("history_str", &open_spiel::State::HistoryString)
    .method("information_state", [](open_spiel::State &s, open_spiel::Player p) { return s.InformationState(p);})
    .method("information_state", [](open_spiel::State &s) { return s.InformationState();})
    .method("information_state_as_normalized_vector", [](open_spiel::State &s) { return s.InformationStateAsNormalizedVector();})
    .method("information_state_as_normalized_vector", [](open_spiel::State &s, open_spiel::Player p) { return s.InformationStateAsNormalizedVector(p);})
    .method("information_state_as_normalized_vector", [](open_spiel::State &s, open_spiel::Player p, jlcxx::ArrayRef<double, 1> values) {
      std::vector<double> data(values.begin(), values.end());
      return s.InformationStateAsNormalizedVector(p, &data);
      })
    .method("observation", [](open_spiel::State &s) { return s.Observation();})
    .method("observation", [](open_spiel::State &s, open_spiel::Player p) { return s.Observation(p);})
    .method("observation_as_normalized_vector", [](open_spiel::State &s) { return s.ObservationAsNormalizedVector();})
    .method("observation_as_normalized_vector", [](open_spiel::State &s, open_spiel::Player p) { return s.ObservationAsNormalizedVector(p);})
    .method("observation_as_normalized_vector", [](open_spiel::State &s, open_spiel::Player p, jlcxx::ArrayRef<double, 1> values) {
      std::vector<double> data(values.begin(), values.end());
      return s.ObservationAsNormalizedVector(p, &data);
      })
    .method("clone", &open_spiel::State::Clone)
    .method("child", &open_spiel::State::Child)
    .method("undo_action", &open_spiel::State::UndoAction)
    .method("apply_actions", [](open_spiel::State &s, jlcxx::ArrayRef<open_spiel::Action, 1> values) {
      std::vector<open_spiel::Action> data(values.begin(), values.end());
      return s.ApplyActions(data);
      })
    .method("num_distinct_actions", &open_spiel::State::NumDistinctActions)
    .method("num_players", &open_spiel::State::NumPlayers)
    .method("chance_outcomes", &open_spiel::State::ChanceOutcomes)
    .method("get_type", &open_spiel::State::GetType)
    .method("serialize", &open_spiel::State::Serialize);
  
  mod.add_type<open_spiel::Game>("Game")
    .method("num_distinct_actions", &open_spiel::Game::NumDistinctActions)
    .method("new_initial_state", &open_spiel::Game::NewInitialState)
    .method("max_chance_outcomes", &open_spiel::Game::MaxChanceOutcomes)
    .method("get_parameters", [](open_spiel::Game &g) { return map_to_keys_vals(g.GetParameters());} )
    .method("num_players", &open_spiel::Game::NumPlayers)
    .method("min_utility", &open_spiel::Game::MinUtility)
    .method("max_utility", &open_spiel::Game::MaxUtility)
    .method("get_type", &open_spiel::Game::GetType)
    .method("utility_sum", &open_spiel::Game::UtilitySum)
    .method("information_state_normalized_vector_shape", &open_spiel::Game::InformationStateNormalizedVectorShape)
    .method("information_state_normalized_vector_size", &open_spiel::Game::InformationStateNormalizedVectorSize)
    .method("observation_normalized_vector_shape", &open_spiel::Game::ObservationNormalizedVectorShape)
    .method("observation_normalized_vector_size", &open_spiel::Game::ObservationNormalizedVectorSize)
    .method("deserialize_state", &open_spiel::Game::DeserializeState)
    .method("max_game_length", &open_spiel::Game::MaxGameLength)
    .method("string", &open_spiel::Game::ToString);

  // mod.add_type<open_spiel::maxtix_game::MatrixGame>("MatrixGame")
  //   .constructor<open_spiel::GameType, jlcxx::ArrayRef<>()
  //   .method("")

  mod.method("load_game", [](const std::string & s) { return open_spiel::LoadGame(s); });
  mod.method("load_game_as_turn_based", [](const std::string &s) { return open_spiel::LoadGameAsTurnBased(s); });

  // mod.method("load_matrix_game", &open_spiel::algorithms::LoadMatrixGame);
  // mod.method("extensive_to_matrix_game", &open_spiel::algorithms::ExtensiveToMatrixGame);
  mod.method("registered_names", &open_spiel::GameRegister::RegisteredNames);
  mod.method("registered_games", &open_spiel::GameRegister::RegisteredGames);
}