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

#ifndef OPEN_SPIEL_GAMES_EFG_GAME_H_
#define OPEN_SPIEL_GAMES_EFG_GAME_H_

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// A parser for the .efg format used by Gambit:
// http://www.gambit-project.org/gambit14/formats.html
//
// Parameters:
//       "filename"   string     name of a file containing the data
//
// Note: not the full EFG is supported as stated on that page. In particular:
//   - Payoffs / outcomes at non-terminal nodes are not supported
//   - Player nodes and chance nodes must each have one child
//

namespace open_spiel {
namespace efg_game {

enum class NodeType {
  kChance,
  kPlayer,
  kTerminal,
};

// A node object that represent a subtree of the game.
struct Node {
  Node* parent;
  NodeType type;
  int id;
  std::string name;
  int infoset_number;  // Must starting at 1 for each player.
  int player_number;   // The EFG player numbers (starting at 1 rather than 0).
  std::string infoset_name;
  std::string outcome_name;
  int outcome_number;
  std::vector<std::string> actions;
  std::vector<Action> action_ids;
  std::vector<Node*> children;
  std::vector<double> probs;
  std::vector<double> payoffs;
};

// A function to load an EFG directly from string data. Note: games loaded
// using this function will not be serializable (nor will their states). Use
// the general LoadGame with the filename argument if serialization is required.
std::shared_ptr<const Game> LoadEFGGame(const std::string& data);

// Helper function to construct a tabular policy explicitly. The game must be
// an EFG game. The map uses is
// (player, information ste strings) -> vector of (action string, prob), e.g.:
//     { {{0, "infoset1"}, {{"actionA, prob1"}, {"actionB", prob2}}},
//       {{1, "infoset2"}, {{"actionC, prob1"}, {"actionD", prob2}}} }
TabularPolicy EFGGameTabularPolicy(
    std::shared_ptr<const Game> game,
    const absl::flat_hash_map<std::pair<Player, std::string>,
                              std::vector<std::pair<std::string, double>>>&
        policy_map);

class EFGState : public State {
 public:
  explicit EFGState(std::shared_ptr<const Game> game, const Node* root);
  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action action) override;
  std::vector<Action> LegalActions() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;

 protected:
  void DoApplyAction(Action action) override;

 private:
  int ActionIdx(Action action) const;
  const Node* cur_node_;
};

class EFGGame : public Game {
 public:
  explicit EFGGame(const GameParameters& params);
  explicit EFGGame(const std::string& data);
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(
        new EFGState(shared_from_this(), nodes_[0].get()));
  }

  int MaxChanceOutcomes() const override;
  int NumDistinctActions() const override;
  int NumPlayers() const override;
  double MinUtility() const override;
  double UtilitySum() const override;
  double MaxUtility() const override;
  int MaxGameLength() const override;
  int MaxChanceNodesInHistory() const override;
  int MaxMoveNumber() const override;
  int MaxHistoryLength() const override;
  std::vector<int> InformationStateTensorShape() const override;

  // Gets the player / decision node action associated to this label.
  Action GetAction(const std::string& label) const {
    auto iter = action_ids_.find(label);
    SPIEL_CHECK_TRUE(iter != action_ids_.end());
    return iter->second;
  }

  // Gets the chance node action associated to this label.
  Action GetChanceAction(const std::string& label) const {
    auto iter = chance_action_ids_.find(label);
    SPIEL_CHECK_TRUE(iter != chance_action_ids_.end());
    return iter->second;
  }

  Action AddOrGetAction(const std::string& label) {
    auto iter = action_ids_.find(label);
    if (iter != action_ids_.end()) {
      return iter->second;
    }
    Action new_action = action_ids_.size();
    action_ids_[label] = new_action;
    return new_action;
  }

  Action AddOrGetChanceOutcome(const std::string& label) {
    auto iter = chance_action_ids_.find(label);
    if (iter != chance_action_ids_.end()) {
      return iter->second;
    }
    Action new_action = chance_action_ids_.size();
    chance_action_ids_[label] = new_action;
    return new_action;
  }

  // Get the information state strings by names or numbers.
  // Note: since the names of the information sets are not required to be
  // unique, if the same name is used for different infoset numbers in the file
  // then the information set number may not be the correct one. Only use
  // GetInformationStateStringByName if the names are unique and there is a
  // one-to-one correspondence with infoset numbers!
  std::string GetInformationStateStringByName(Player player,
                                              const std::string& name) const;
  std::string GetInformationStateStringByNumber(Player player,
                                                int number) const;

  // Return the number of information states for the specified player.
  int NumInfoStates(Player player) const {
    return infoset_num_to_states_count_[player].size();
  }

 private:
  std::unique_ptr<Node> NewNode() const;
  void ParseGame();
  void ParsePrologue();
  std::string NextToken();
  void AdvancePosition();
  std::string GetLine(int line) const;
  bool ParseDoubleValue(const std::string& str, double* value) const;
  bool IsWhiteSpace(char c) const;
  bool IsNodeToken(char c) const;
  void UpdateAndCheckInfosetMaps(const Node* node);
  void ParseChanceNode(Node* parent, Node* child, int depth);
  void ParsePlayerNode(Node* parent, Node* child, int depth);
  void ParseTerminalNode(Node* parent, Node* child, int depth);
  void RecParseSubtree(Node* parent, Node* child, int depth);
  std::string PrettyTree(const Node* node, const std::string& indent) const;

  std::string filename_;
  std::string string_data_;
  int pos_;
  int line_ = 1;
  std::vector<std::unique_ptr<Node>> nodes_;
  std::string name_;
  std::string description_;
  std::vector<std::string> player_names_;
  int num_chance_nodes_;
  int num_players_;
  int max_actions_;
  int max_depth_;
  absl::optional<double> util_sum_;
  absl::optional<double> max_util_;
  absl::optional<double> min_util_;
  bool constant_sum_;
  bool identical_payoffs_;
  bool general_sum_;
  bool perfect_information_;

  // Maintains a map of infoset number -> count of states in the infoset
  // (one for each player).
  std::vector<absl::flat_hash_map<int, int>> infoset_num_to_states_count_;

  // Maintains a (player, infoset number) -> infoset name mapping and vice
  // versa, for retrieval of information set strings externally
  // (GetInformationStateStringByName and GetInformationStateStringByNumber).
  absl::flat_hash_map<std::pair<Player, int>, std::string>
      infoset_player_num_to_name_;
  absl::flat_hash_map<std::string, std::pair<Player, int>>
      infoset_name_to_player_num_;

  // Action label -> action id mapping. Note that chance actions are excluded.
  absl::flat_hash_map<std::string, Action> action_ids_;

  // Outcome label -> action id mapping for chance nodes.
  absl::flat_hash_map<std::string, Action> chance_action_ids_;
};

}  // namespace efg_game
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_EFG_GAME_H_
