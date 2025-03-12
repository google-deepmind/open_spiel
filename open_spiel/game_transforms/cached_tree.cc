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

#include "open_spiel/game_transforms/cached_tree.h"

#include <memory>
#include <string>
#include <vector>
#include <utility>

#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/game_transforms/game_wrapper.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace cached_tree {

namespace {
// These parameters reflect the most-general game, with the maximum
// API coverage. The actual game may be simpler and might not provide
// all the interfaces.
// This is used as a placeholder for game registration. The actual instantiated
// game will have more accurate information.
const GameType kGameType{/*short_name=*/"cached_tree",
                         /*long_name=*/"Cached Tree Game Transform",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kSampledStochastic,
                         GameType::Information::kImperfectInformation,
                         GameType::Utility::kGeneralSum,
                         GameType::RewardModel::kRewards,
                         /*max_num_players=*/100,
                         /*min_num_players=*/1,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/true,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         {{"game", GameParameter(GameParameter::Type::kGame,
                                                 /*is_mandatory=*/true)}},
                         /*default_loadable=*/false,
                         /*provides_factored_observation_string=*/false,
                         /*is_concrete=*/false};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return ConvertToCachedTree(*LoadGame(params.at("game").game_value()));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

GameType ConvertType(GameType type) {
  type.dynamics = GameType::Dynamics::kSequential;
  type.information = GameType::Information::kImperfectInformation;
  type.short_name = kGameType.short_name;
  type.long_name = "Turn-based " + type.long_name;
  type.parameter_specification = kGameType.parameter_specification;
  return type;
}

GameParameters ConvertParams(const GameType& type, GameParameters params) {
  params["name"] = GameParameter(type.short_name);
  GameParameters new_params{{"game", GameParameter{params}}};
  return new_params;
}

}  // namespace

// Note: overridden to use the wrapped state inside the node.
const State& CachedTreeState::GetWrappedState() const {
  return *(node_->state);
}

CachedTreeState::CachedTreeState(std::shared_ptr<const Game> game, Node* node)
    : WrappedState(game, nullptr),
      parent_game_(down_cast<const CachedTreeGame&>(*game)),
      node_(node) {}

CachedTreeState::CachedTreeState(const CachedTreeState& other)
    : WrappedState(other, nullptr),
      parent_game_(other.parent_game_),
      node_(other.node_) {}

void CachedTreeState::DoApplyAction(Action action_id) {
  auto iter = node_->children.find(action_id);
  if (iter != node_->children.end()) {
    node_ = iter->second;
    return;
  }

  // If we get here, the child does not exist. Create it and connect it.
  node_ = parent_game_.CreateChildNode(node_, this, action_id);
}

void CachedTreeState::DoApplyActions(const std::vector<Action>& actions) {
  auto iter = node_->joint_action_children.find(actions);
  if (iter != node_->joint_action_children.end()) {
    node_ = iter->second;
    return;
  }

  // If we get here, the child does not exist. Create it and connect it.
  node_ = parent_game_.CreateChildNode(node_, this, actions);
}

std::unique_ptr<State> CachedTreeState::Clone() const {
  return std::make_unique<CachedTreeState>(*this);
}

Player CachedTreeState::CurrentPlayer() const {
  if (node_->current_player == kInvalidPlayer) {
    node_->current_player = node_->state->CurrentPlayer();
  }
  return node_->current_player;
}

std::vector<Action> CachedTreeState::LegalActions(Player player) const {
  auto iter = node_->legal_actions.find(player);
  if (iter != node_->legal_actions.end()) {
    return iter->second;
  }
  std::vector<Action> legal_actions = node_->state->LegalActions(player);
  node_->legal_actions[player] = legal_actions;
  return legal_actions;
}

std::vector<Action> CachedTreeState::LegalActions() const {
  return LegalActions(CurrentPlayer());
}

std::string CachedTreeState::ActionToString(Player player,
                                            Action action_id) const {
  auto key = std::make_pair(player, action_id);
  auto iter = node_->action_to_string.find(key);
  if (iter != node_->action_to_string.end()) {
    return iter->second;
  }
  std::string action_string = node_->state->ActionToString(player, action_id);
  node_->action_to_string[key] = action_string;
  return action_string;
}

std::string CachedTreeState::ToString() const {
  if (node_->to_string.has_value()) {
    return node_->to_string.value();
  }
  node_->to_string = node_->state->ToString();
  return node_->to_string.value();
}

bool CachedTreeState::IsTerminal() const {
  if (node_->terminal.has_value()) {
    return node_->terminal.value();
  }
  node_->terminal = node_->state->IsTerminal();
  return node_->terminal.value();
}

std::vector<double> CachedTreeState::Rewards() const {
  if (node_->rewards.empty()) {
    node_->rewards = node_->state->Rewards();
  }
  return node_->rewards;
}

std::vector<double> CachedTreeState::Returns() const {
  if (node_->returns.empty()) {
    node_->returns = node_->state->Returns();
  }
  return node_->returns;
}

std::string CachedTreeState::InformationStateString(Player player) const {
  auto iter = node_->information_state_string.find(player);
  if (iter != node_->information_state_string.end()) {
    return iter->second;
  }
  std::string information_state_string =
      node_->state->InformationStateString(player);
  node_->information_state_string[player] = information_state_string;
  return information_state_string;
}

void CachedTreeState::InformationStateTensor(Player player,
                                             absl::Span<float> values) const {
  node_->state->InformationStateTensor(player, values);
}

std::string CachedTreeState::ObservationString(Player player) const {
  auto iter = node_->observation_string.find(player);
  if (iter != node_->observation_string.end()) {
    return iter->second;
  }
  std::string observation_string = node_->state->ObservationString(player);
  node_->observation_string[player] = observation_string;
  return observation_string;
}

void CachedTreeState::ObservationTensor(Player player,
                                        absl::Span<float> values) const {
  node_->state->ObservationTensor(player, values);
}

void CachedTreeState::UndoAction(Player player, Action action) {
  node_->state->UndoAction(player, action);
  history_.pop_back();
}

ActionsAndProbs CachedTreeState::ChanceOutcomes() const {
  if (node_->chance_outcomes.empty()) {
    node_->chance_outcomes = node_->state->ChanceOutcomes();
  }
  return node_->chance_outcomes;
}

std::vector<Action> CachedTreeState::LegalChanceOutcomes() const {
  return LegalActions(kChancePlayerId);
}

std::vector<Action> CachedTreeState::ActionsConsistentWithInformationFrom(
      Action action) const {
  auto iter =
      node_->legal_actions_consistent_with_information_from.find(action);
  if (iter != node_->legal_actions_consistent_with_information_from.end()) {
    return iter->second;
  }
  std::vector<Action> legal_actions_consistent_with_information_from =
      node_->state->ActionsConsistentWithInformationFrom(action);
  node_->legal_actions_consistent_with_information_from[action] =
      legal_actions_consistent_with_information_from;
  return legal_actions_consistent_with_information_from;
}

Node* CachedTreeGame::CreateChildNode(Node* parent,
                                      const CachedTreeState* state,
                                      Action action) const {
  SPIEL_CHECK_TRUE(parent != nullptr);
  SPIEL_CHECK_TRUE(state != nullptr);
  SPIEL_CHECK_TRUE(action != kInvalidAction);
  nodes_.push_back(std::make_unique<Node>());
  Node* child_node = nodes_.back().get();
  child_node->state = parent->state->Child(action);
  parent->children[action] = child_node;
  return child_node;
}

Node* CachedTreeGame::CreateChildNode(
    Node* parent,
    const CachedTreeState* state,
    const std::vector<Action>& joint_action) const {
  SPIEL_CHECK_TRUE(parent != nullptr);
  SPIEL_CHECK_TRUE(state != nullptr);
  SPIEL_CHECK_FALSE(joint_action.empty());
  nodes_.push_back(std::make_unique<Node>());
  Node* child_node = nodes_.back().get();
  auto actual_child_state = parent->state->Clone();
  actual_child_state->ApplyActions(joint_action);
  child_node->state = std::move(actual_child_state);
  parent->joint_action_children[joint_action] = child_node;
  return child_node;
}

std::unique_ptr<State> CachedTreeGame::NewInitialState() const {
  if (root_ == nullptr) {
    SPIEL_CHECK_EQ(nodes_.size(), 0);
    nodes_.push_back(std::make_unique<Node>());
    root_ = nodes_.back().get();
    root_->state = game_->NewInitialState();
  }
  return std::make_unique<CachedTreeState>(shared_from_this(), root_);
}

double CachedTreeGame::MinUtility() const {
  if (!min_utility_.has_value()) {
    min_utility_ = game_->MinUtility();
  }
  return min_utility_.value();
}

double CachedTreeGame::MaxUtility() const {
  if (!max_utility_.has_value()) {
    max_utility_ = game_->MaxUtility();
  }
  return max_utility_.value();
}

CachedTreeGame::CachedTreeGame(std::shared_ptr<const Game> game)
    : WrappedGame(game, ConvertType(game->GetType()),
                  ConvertParams(game->GetType(), game->GetParameters())) {}

std::shared_ptr<const Game> ConvertToCachedTree(const Game& game) {
  return std::shared_ptr<const CachedTreeGame>(
      new CachedTreeGame(game.shared_from_this()));
}

std::shared_ptr<const Game> LoadGameAsCachedTree(const std::string& name) {
  auto game = LoadGame(name);
  return ConvertToCachedTree(*game);
}

std::shared_ptr<const Game> LoadGameAsCachedTree(const std::string& name,
                                                const GameParameters& params) {
  auto game = LoadGame(name, params);
  return ConvertToCachedTree(*game);
}

}  // namespace cached_tree
}  // namespace open_spiel

