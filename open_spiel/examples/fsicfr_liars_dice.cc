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

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/algorithms/fsicfr.h"
#include "open_spiel/algorithms/tabular_best_response_mdp.h"
#include "open_spiel/games/liars_dice.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace {

constexpr const int kSeed = 1873561;

using liars_dice::LiarsDiceState;

void BuildGraph(FSICFRGraph* graph, const State& state, Action chance_id_0,
                Action chance_id_1, int max_predecessors, int parent_node_id,
                Action parent_action, int parent_other_player_chance_id) {
  if (state.IsTerminal()) {
    const auto& ld_state = static_cast<const LiarsDiceState&>(state);
    std::string terminal_key =
        absl::StrCat("terminal ", ld_state.dice_outcome(0, 0), " ",
                     ld_state.dice_outcome(1, 0), " ",
                     ld_state.calling_player(), " ", ld_state.last_bid());
    FSICFRNode* node = graph->GetOrCreateTerminalNode(
        terminal_key, state.PlayerReturn(0), max_predecessors);
    FSICFRNode* parent_node = graph->GetNode(parent_node_id);
    SPIEL_CHECK_TRUE(parent_node != nullptr);
    // Connect to the parent.
    parent_node->AddChild(parent_action, parent_other_player_chance_id, node);
  } else if (state.IsChanceNode()) {
    std::vector<Action> legal_actions = state.LegalActions();
    for (Action outcome : legal_actions) {
      Action next_chance_id_0 = chance_id_0;
      Action next_chance_id_1 = chance_id_1;
      if (chance_id_0 == kInvalidAction) {
        next_chance_id_0 = outcome;
      } else {
        next_chance_id_1 = outcome;
      }
      std::unique_ptr<State> next_state = state.Child(outcome);
      BuildGraph(graph, *next_state, next_chance_id_0, next_chance_id_1,
                 max_predecessors, parent_node_id, parent_action,
                 parent_other_player_chance_id);
    }
  } else {
    std::string info_state_string = state.InformationStateString();
    Player player = state.CurrentPlayer();
    int my_chance_id = player == 0 ? chance_id_0 : chance_id_1;
    int other_chance_id = player == 0 ? chance_id_1 : chance_id_0;
    std::vector<Action> legal_actions = state.LegalActions();

    FSICFRNode* node =
        graph->GetOrCreateDecisionNode(legal_actions, info_state_string, player,
                                       max_predecessors, my_chance_id);
    int next_max_predecessors = node->max_predecessors + 1;
    int node_id = node->id;

    node->max_predecessors = std::max(max_predecessors, node->max_predecessors);

    FSICFRNode* parent_node = graph->GetNode(parent_node_id);

    // Connect it to the parent.
    if (parent_node != nullptr) {
      parent_node->AddChild(parent_action, parent_other_player_chance_id, node);
    }

    // Recrusively build the graph from the children.
    for (Action action : legal_actions) {
      std::unique_ptr<State> next_state = state.Child(action);
      BuildGraph(graph, *next_state, chance_id_0, chance_id_1,
                 next_max_predecessors, node_id, action, other_chance_id);
    }
  }
}

void RunFSICFR() {
  std::unique_ptr<FSICFRGraph> graph = std::make_unique<FSICFRGraph>();
  std::shared_ptr<const Game> game = LoadGame("liars_dice_ir");
  std::unique_ptr<State> initial_state = game->NewInitialState();
  std::cout << "Building the graph." << std::endl;
  BuildGraph(graph.get(), *initial_state, kInvalidAction, kInvalidAction, 0, -1,
             kInvalidAction, -1);
  std::cout << "Graph has " << graph->size() << " nodes." << std::endl;
  std::cout << "Topologically sorting the nodes." << std::endl;
  graph->TopSort();
  FSICFRSolver solver(*game, kSeed, {6, 6}, graph.get());

  std::cout << "Running iterations" << std::endl;
  int max_iterations = 1000000;
  int total_iterations = 0;
  int num_iterations = 0;

  // solver.RunIteration();
  // std::exit(-1);

  while (total_iterations < max_iterations) {
    solver.RunIterations(num_iterations);
    // Must use the best response MDP since it supports imperfect recall.
    TabularPolicy average_policy = solver.GetAveragePolicy();
    TabularBestResponseMDP tbr(*game, average_policy);
    TabularBestResponseMDPInfo br_info = tbr.NashConv();
    total_iterations += num_iterations;
    std::cout << total_iterations << " " << br_info.nash_conv << std::endl;
    num_iterations = (num_iterations == 0 ? 10 : total_iterations);
  }
}

}  // namespace
}  // namespace algorithms
}  // namespace open_spiel

int main(int argc, char** argv) { open_spiel::algorithms::RunFSICFR(); }
