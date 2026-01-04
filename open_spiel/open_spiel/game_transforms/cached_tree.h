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

#ifndef OPEN_SPIEL_GAME_TRANSFORMS_CACHED_TREE_H_
#define OPEN_SPIEL_GAME_TRANSFORMS_CACHED_TREE_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_transforms/game_wrapper.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/spiel_globals.h"

// A tree built dynamically built and cached in memory. This wrapper can be used
// to speed up the traversals of the game tree and corresponding functions like
// information state keys and tensors for games whose tree is not too large.

namespace open_spiel {
namespace cached_tree {

class CachedTreeState;
class CachedTreeGame;

// A node corresponds to a state in the game.
struct Node {
  Player current_player = kInvalidPlayer;
  std::unique_ptr<State> state;
  absl::optional<std::string> to_string;
  ActionsAndProbs chance_outcomes;
  std::vector<double> returns;
  std::vector<double> rewards;
  absl::optional<bool> terminal;
  absl::flat_hash_map<Action, Node*> children;
  absl::flat_hash_map<std::vector<Action>, Node*> joint_action_children;
  absl::flat_hash_map<std::pair<Player, Action>, std::string> action_to_string;
  absl::flat_hash_map<Player, std::vector<Action>> legal_actions;
  absl::flat_hash_map<Player, std::string> information_state_string;
  absl::flat_hash_map<Player, std::string> observation_string;
  absl::flat_hash_map<Action, std::vector<Action>>
      legal_actions_consistent_with_information_from;
};


class CachedTreeState : public WrappedState {
 public:
  CachedTreeState(std::shared_ptr<const Game> game, Node* node);
  CachedTreeState(const CachedTreeState& other);

  // Note: overridden to use the wrapped state inside the node.
  const State& GetWrappedState() const override;

  // Must override all the methods of the WrappedState. This is because this
  // wrapper bypasses using the state_ pointer inside WrappedState.
  Player CurrentPlayer() const override;
  std::vector<Action> LegalActions(Player player) const override;
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Rewards() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action action) override;
  ActionsAndProbs ChanceOutcomes() const override;
  std::vector<Action> LegalChanceOutcomes() const override;
  std::vector<Action> ActionsConsistentWithInformationFrom(
      Action action) const override;

 protected:
  void DoApplyAction(Action action_id) override;
  void DoApplyActions(const std::vector<Action>& actions) override;

 private:
  const CachedTreeGame& parent_game_;
  Node* node_ = nullptr;
};

class CachedTreeGame : public WrappedGame {
 public:
  explicit CachedTreeGame(std::shared_ptr<const Game> game);
  std::unique_ptr<State> NewInitialState() const override;
  double MinUtility() const override;
  double MaxUtility() const override;

  Node* CreateChildNode(Node* parent, const CachedTreeState* state,
                        Action action) const;
  Node* CreateChildNode(Node* parent, const CachedTreeState* state,
                        const std::vector<Action>& joint_action) const;


 private:
  // protected member game_ is inherited from WrappedGame.
  mutable absl::optional<double> min_utility_;
  mutable absl::optional<double> max_utility_;
  mutable Node* root_ = nullptr;
  mutable std::vector<std::unique_ptr<Node>> nodes_;
};

// Helper function to convert
std::shared_ptr<const Game> ConvertToCachedTree(const Game& game);
std::shared_ptr<const Game> LoadGameAsCachedTree(const std::string& name);
std::shared_ptr<const Game> LoadGameAsCachedTree(const std::string& name,
                                                 const GameParameters& params);


}  // namespace cached_tree
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAME_TRANSFORMS_CACHED_TREE_H_

