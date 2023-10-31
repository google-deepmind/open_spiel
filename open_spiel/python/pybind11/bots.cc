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

#include "open_spiel/python/pybind11/bots.h"

#include <stdint.h>

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/algorithms/evaluate_bots.h"
#include "open_spiel/algorithms/is_mcts.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/bots/gin_rummy/simple_gin_rummy_bot.h"
#include "open_spiel/bots/uci/uci_bot.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/python/pybind11/pybind11.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"
#include "open_spiel/spiel_utils.h"
#include "pybind11/include/pybind11/cast.h"
#include "pybind11/include/pybind11/detail/common.h"
#include "pybind11/include/pybind11/pybind11.h"
#include "pybind11/include/pybind11/pytypes.h"
#include "pybind11/include/pybind11/smart_holder.h"

// Optional headers.
#if OPEN_SPIEL_BUILD_WITH_ROSHAMBO
#include "open_spiel/bots/roshambo/roshambo_bot.h"
#endif

namespace open_spiel {
namespace {

using ::open_spiel::algorithms::Evaluator;
using ::open_spiel::algorithms::SearchNode;

namespace py = ::pybind11;

}  // namespace

void init_pyspiel_bots(py::module& m) {
  py::classh<Bot, PyBot<Bot>> bot(m, "Bot");
  bot.def(py::init<>())
      .def("step", &Bot::Step)
      .def("restart", &Bot::Restart)
      .def("restart_at", &Bot::RestartAt)
      .def("provides_force_action", &Bot::ProvidesForceAction)
      .def("force_action", &Bot::ForceAction)
      .def("inform_action", &Bot::InformAction)
      .def("inform_actions", &Bot::InformActions)
      .def("provides_policy", &Bot::ProvidesPolicy)
      .def("get_policy", &Bot::GetPolicy)
      .def("step_with_policy", &Bot::StepWithPolicy)
      .def("is_clonable", &Bot::IsClonable)
      .def("clone", &Bot::Clone);

  m.def(
      "load_bot",
      py::overload_cast<const std::string&, const std::shared_ptr<const Game>&,
                        Player>(&open_spiel::LoadBot),
      py::arg("bot_name"), py::arg("game"), py::arg("player"),
      "Returns a new bot object for the specified bot name using default "
      "parameters");
  m.def(
      "load_bot",
      py::overload_cast<const std::string&, const std::shared_ptr<const Game>&,
                        Player, const GameParameters&>(&open_spiel::LoadBot),
      py::arg("bot_name"), py::arg("game"), py::arg("player"),
      py::arg("params"),
      "Returns a new bot object for the specified bot name using given "
      "parameters");
  m.def("is_bot_registered", &IsBotRegistered,
        "Checks if a bot under the given name is registered.");
  m.def("registered_bots", &RegisteredBots,
        "Returns a list of registered bot names.");
  m.def(
      "bots_that_can_play_game",
      [](std::shared_ptr<const Game> game, int player) {
        return BotsThatCanPlayGame(*game, player);
      },
      py::arg("game"), py::arg("player"),
      "Returns a list of bot names that can play specified game for the "
      "given player.");
  m.def(
      "bots_that_can_play_game",
      [](std::shared_ptr<const Game> game) {
        return BotsThatCanPlayGame(*game);
      },
      py::arg("game"),
      "Returns a list of bot names that can play specified game for any "
      "player.");

  py::class_<algorithms::Evaluator,
             std::shared_ptr<algorithms::Evaluator>> mcts_evaluator(
                 m, "Evaluator");
  py::class_<algorithms::RandomRolloutEvaluator,
             algorithms::Evaluator,
             std::shared_ptr<algorithms::RandomRolloutEvaluator>>(
                 m, "RandomRolloutEvaluator")
      .def(py::init<int, int>(), py::arg("n_rollouts"), py::arg("seed"));

  py::enum_<algorithms::ChildSelectionPolicy>(m, "ChildSelectionPolicy")
      .value("UCT", algorithms::ChildSelectionPolicy::UCT)
      .value("PUCT", algorithms::ChildSelectionPolicy::PUCT);

  py::class_<SearchNode> search_node(m, "SearchNode");
  search_node.def_readonly("action", &SearchNode::action)
      .def_readonly("prior", &SearchNode::prior)
      .def_readonly("player", &SearchNode::player)
      .def_readonly("explore_count", &SearchNode::explore_count)
      .def_readonly("total_reward", &SearchNode::total_reward)
      .def_readonly("outcome", &SearchNode::outcome)
      .def_readonly("children", &SearchNode::children)
      .def("best_child", &SearchNode::BestChild)
      .def("to_string", &SearchNode::ToString)
      .def("children_str", &SearchNode::ChildrenStr);

  py::classh<algorithms::MCTSBot, Bot>(m, "MCTSBot")
      .def(
          py::init([](std::shared_ptr<const Game> game,
                      std::shared_ptr<Evaluator> evaluator, double uct_c,
                      int max_simulations, int64_t max_memory_mb, bool solve,
                      int seed, bool verbose,
                      algorithms::ChildSelectionPolicy child_selection_policy) {
            return new algorithms::MCTSBot(
                *game, evaluator, uct_c, max_simulations, max_memory_mb, solve,
                seed, verbose, child_selection_policy);
          }),
          py::arg("game"), py::arg("evaluator"), py::arg("uct_c"),
          py::arg("max_simulations"), py::arg("max_memory_mb"),
          py::arg("solve"), py::arg("seed"), py::arg("verbose"),
          py::arg("child_selection_policy") =
              algorithms::ChildSelectionPolicy::UCT)
      .def("step", &algorithms::MCTSBot::Step)
      .def("mcts_search", &algorithms::MCTSBot::MCTSearch);

  py::enum_<algorithms::ISMCTSFinalPolicyType>(m, "ISMCTSFinalPolicyType")
      .value("NORMALIZED_VISIT_COUNT",
             algorithms::ISMCTSFinalPolicyType::kNormalizedVisitCount)
      .value("MAX_VISIT_COUNT",
             algorithms::ISMCTSFinalPolicyType::kMaxVisitCount)
      .value("MAX_VALUE", algorithms::ISMCTSFinalPolicyType::kMaxValue);

  py::classh<algorithms::ISMCTSBot, Bot>(m, "ISMCTSBot")
      .def(py::init<int, std::shared_ptr<Evaluator>, double, int, int,
                    algorithms::ISMCTSFinalPolicyType, bool, bool>(),
           py::arg("seed"), py::arg("evaluator"), py::arg("uct_c"),
           py::arg("max_simulations"),
           py::arg("max_world_samples") = algorithms::kUnlimitedNumWorldSamples,
           py::arg("final_policy_type") =
               algorithms::ISMCTSFinalPolicyType::kNormalizedVisitCount,
           py::arg("use_observation_string") = false,
           py::arg("allow_inconsistent_action_sets") = false)
      .def("step", &algorithms::ISMCTSBot::Step)
      .def("provides_policy", &algorithms::MCTSBot::ProvidesPolicy)
      .def("get_policy", &algorithms::ISMCTSBot::GetPolicy)
      .def("step_with_policy", &algorithms::ISMCTSBot::StepWithPolicy)
      .def("restart", &algorithms::ISMCTSBot::Restart)
      .def("restart_at", &algorithms::ISMCTSBot::RestartAt);

  m.def("evaluate_bots",
        py::overload_cast<State*, const std::vector<Bot*>&, int>(
            open_spiel::EvaluateBots),
        py::arg("state"), py::arg("bots"), py::arg("seed"),
        "Plays a single game with the given bots and returns the final "
        "utilities.");

  m.def("make_uniform_random_bot", open_spiel::MakeUniformRandomBot,
        "A uniform random bot, for test purposes.");

  m.def("make_stateful_random_bot", open_spiel::MakeStatefulRandomBot,
        "A stateful random bot, for test purposes.");
  m.def(
      "make_policy_bot",
      [](std::shared_ptr<const Game> game, Player player_id, int seed,
         std::shared_ptr<Policy> policy) {
        return MakePolicyBot(*game, player_id, seed, policy);
      },
      "A bot that samples from a policy.");

#ifndef _WIN32
  m.def("make_uci_bot", open_spiel::uci::MakeUCIBot, py::arg("bot_binary_path"),
      py::arg("move_time"), py::arg("ponder"), py::arg("options"),
      "Bot that can play chess using UCI chess engine.");
#endif


#if OPEN_SPIEL_BUILD_WITH_ROSHAMBO
  m.attr("ROSHAMBO_NUM_THROWS") = py::int_(open_spiel::roshambo::kNumThrows);
  m.attr("ROSHAMBO_NUM_BOTS") = py::int_(open_spiel::roshambo::kNumBots);
  // no arguments; returns vector of strings
  m.def("roshambo_bot_names", open_spiel::roshambo::RoshamboBotNames);
  // args: player_int (int), bot name (string), num throws (int), returns bot
  m.def("make_roshambo_bot", open_spiel::roshambo::MakeRoshamboBot,
        py::arg("player_id"), py::arg("bot_name"),
        py::arg("num_throws") = open_spiel::roshambo::kNumThrows);
#endif

  m.def(
      "make_simple_gin_rummy_bot",
      [](const GameParameters& params,
         int player_id) -> std::unique_ptr<open_spiel::Bot> {
        return std::make_unique<gin_rummy::SimpleGinRummyBot>(params,
                                                              player_id);
      },
      py::arg("params"), py::arg("player_id"));
}
}  // namespace open_spiel
