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

#include "open_spiel/games/efg_game.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/file.h"

namespace open_spiel {
namespace efg_game {
namespace {

constexpr int kBuffSize = 1024;

// Facts about the game. These are defaults that will differ depending on the
// game's descriptions. Using dummy defaults just to register the game.
const GameType kGameType{/*short_name=*/"efg_game",
                         /*long_name=*/"efg_game",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kDeterministic,
                         GameType::Information::kPerfectInformation,
                         GameType::Utility::kGeneralSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/2,
                         /*min_num_players=*/2,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/true,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/false,
                         /*parameter_specification=*/
                         {{"filename", GameParameter(std::string(""))}},
                         /*default_loadable=*/false};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new EFGGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

std::string NodeToString(const Node* node) {
  std::string str = "";
  if (node->type == NodeType::kTerminal) {
    absl::StrAppend(&str, "Terminal: ", node->name, " ", node->outcome_name);
    for (double payoff : node->payoffs) {
      absl::StrAppend(&str, " ", payoff);
    }
    absl::StrAppend(&str, "\n");
  } else if (node->type == NodeType::kChance) {
    absl::StrAppend(&str, "Chance: ", node->name, " ", node->infoset_number,
                    " ", node->infoset_name);
    for (int i = 0; i < node->children.size(); ++i) {
      absl::StrAppend(&str, " ", node->actions[i], " ", node->probs[i]);
    }
    absl::StrAppend(&str, "\n");
  } else if (node->type == NodeType::kPlayer) {
    absl::StrAppend(&str, "Player: ", node->name, " ", node->player_number, " ",
                    node->infoset_number, " ", node->infoset_name);
    for (int i = 0; i < node->children.size(); ++i) {
      absl::StrAppend(&str, " ", node->actions[i]);
    }
    absl::StrAppend(&str, "\n");
  }
  return str;
}

std::string EFGInformationStateString(Player owner, Player observer, int number,
                                      const std::string& name) {
  return absl::StrCat(owner, "-", observer, "-", number, "-", name);
}
}  // namespace

EFGState::EFGState(std::shared_ptr<const Game> game, const Node* root)
    : State(game), cur_node_(root) {}

Player EFGState::CurrentPlayer() const {
  if (cur_node_->type == NodeType::kChance) {
    return kChancePlayerId;
  } else if (cur_node_->type == NodeType::kTerminal) {
    return kTerminalPlayerId;
  } else {
    // Gambit player numbers are between 1 and num_players
    SPIEL_CHECK_GE(cur_node_->player_number, 1);
    SPIEL_CHECK_LE(cur_node_->player_number, num_players_);
    return cur_node_->player_number - 1;
  }
}

std::string EFGState::ActionToString(Player player, Action action) const {
  int action_idx = ActionIdx(action);
  SPIEL_CHECK_GE(action_idx, 0);
  SPIEL_CHECK_LT(action_idx, cur_node_->actions.size());
  return cur_node_->actions[action_idx];
}

std::string EFGState::ToString() const {
  return absl::StrCat(cur_node_->id, ": ", NodeToString(cur_node_));
}

bool EFGState::IsTerminal() const {
  return cur_node_->type == NodeType::kTerminal;
}

std::vector<double> EFGState::Returns() const {
  if (cur_node_->type == NodeType::kTerminal) {
    SPIEL_CHECK_EQ(cur_node_->payoffs.size(), num_players_);
    return cur_node_->payoffs;
  } else {
    return std::vector<double>(num_players_, 0);
  }
}

std::string EFGState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  // The information set number has to uniquely identify the infoset, whereas
  // the names are optional. But the numbers are unique per player, so must
  // add the player number.
  return EFGInformationStateString(cur_node_->player_number - 1, player,
                                   cur_node_->infoset_number,
                                   cur_node_->infoset_name);
}

void EFGState::InformationStateTensor(Player player,
                                      absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  std::fill(values.begin(), values.end(), 0.0);
  int offset = 0;
  int index = 0;

  // Current player, or terminal.
  if (cur_node_->type == NodeType::kTerminal) {
    index = offset + num_players_;
  } else {
    index = offset + cur_node_->player_number - 1;
  }
  SPIEL_CHECK_GE(index, 0);
  SPIEL_CHECK_LT(index, values.size());
  values[index] = 1.0;
  offset += num_players_ + 1;

  // Observing player.
  index = offset + player;
  SPIEL_CHECK_GE(index, 0);
  SPIEL_CHECK_LT(index, values.size());
  values[index] = 1.0;
  offset += num_players_;

  // Infostate number.
  index = offset + cur_node_->infoset_number - 1;
  SPIEL_CHECK_GE(index, 0);
  SPIEL_CHECK_LT(index, values.size());
  values[index] = 1.0;

  offset += static_cast<const EFGGame*>(game_.get())->NumInfoStates(player);
  SPIEL_CHECK_LE(offset, values.size());
}

std::string EFGState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return absl::StrCat(cur_node_->player_number - 1, "-", player, "-",
                      cur_node_->infoset_number, "-", cur_node_->infoset_name);
}

std::unique_ptr<State> EFGState::Clone() const {
  return std::unique_ptr<State>(new EFGState(*this));
}

void EFGState::UndoAction(Player player, Action action) {
  SPIEL_CHECK_TRUE(cur_node_->parent != nullptr);
  cur_node_ = cur_node_->parent;
}

int EFGState::ActionIdx(Action action) const {
  int action_idx = -1;
  for (int i = 0; i < cur_node_->action_ids.size(); ++i) {
    if (action == cur_node_->action_ids[i]) {
      action_idx = i;
      break;
    }
  }
  return action_idx;
}

void EFGState::DoApplyAction(Action action) {
  // Actions in these games are just indices into the legal actions.
  SPIEL_CHECK_FALSE(cur_node_->type == NodeType::kTerminal);
  SPIEL_CHECK_GE(action, 0);
  if (IsChanceNode()) {
    SPIEL_CHECK_LT(action, game_->MaxChanceOutcomes());
  } else {
    SPIEL_CHECK_LT(action, game_->NumDistinctActions());
  }
  int action_idx = ActionIdx(action);
  SPIEL_CHECK_NE(action_idx, -1);
  SPIEL_CHECK_FALSE(cur_node_->children[action_idx] == nullptr);
  cur_node_ = cur_node_->children[action_idx];
}

std::vector<Action> EFGState::LegalActions() const {
  return cur_node_->action_ids;
}

std::vector<std::pair<Action, double>> EFGState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  SPIEL_CHECK_TRUE(cur_node_->type == NodeType::kChance);
  std::vector<std::pair<Action, double>> outcomes(cur_node_->children.size());
  for (int i = 0; i < cur_node_->children.size(); ++i) {
    outcomes[i].first = cur_node_->action_ids[i];
    outcomes[i].second = cur_node_->probs[i];
  }
  return outcomes;
}

int EFGGame::MaxChanceOutcomes() const { return chance_action_ids_.size(); }

int EFGGame::NumDistinctActions() const { return action_ids_.size(); }

int EFGGame::NumPlayers() const { return num_players_; }

double EFGGame::MinUtility() const { return min_util_.value(); }

double EFGGame::UtilitySum() const { return util_sum_.value(); }

double EFGGame::MaxUtility() const { return max_util_.value(); }

int EFGGame::MaxGameLength() const { return max_depth_; }

int EFGGame::MaxChanceNodesInHistory() const { return num_chance_nodes_; }

int EFGGame::MaxMoveNumber() const { return max_depth_; }

int EFGGame::MaxHistoryLength() const { return max_depth_; }

std::vector<int> EFGGame::InformationStateTensorShape() const {
  int max_player_infosets = 0;
  for (Player p = 0; p < num_players_; ++p) {
    max_player_infosets = std::max<int>(max_player_infosets,
                                        infoset_num_to_states_count_[p].size());
  }

  return {
    num_players_ + 1 +    // Current player (plus special for terminal).
    num_players_ +        // Current observing player.
    max_player_infosets   // Information set number (for the current player).
  };
}

EFGGame::EFGGame(const GameParameters& params)
    : Game(kGameType, params),
      filename_(ParameterValue<std::string>("filename")),
      string_data_(file::ReadContentsFromFile(filename_, "r")),
      pos_(0),
      num_chance_nodes_(0),
      max_actions_(0),
      max_depth_(0),
      constant_sum_(true),
      identical_payoffs_(true),
      general_sum_(true),
      perfect_information_(true) {
  SPIEL_CHECK_GT(string_data_.size(), 0);

  // Now parse the string data into a data structure.
  ParseGame();
}

EFGGame::EFGGame(const std::string& data)
    : Game(kGameType, {}),
      string_data_(data),
      pos_(0),
      num_chance_nodes_(0),
      max_actions_(0),
      max_depth_(0),
      constant_sum_(true),
      identical_payoffs_(true),
      general_sum_(true),
      perfect_information_(true) {
  ParseGame();
}

std::shared_ptr<const Game> LoadEFGGame(const std::string& data) {
  return std::shared_ptr<const Game>(new EFGGame(data));
}

bool EFGGame::IsWhiteSpace(char c) const {
  return (c == ' ' || c == '\r' || c == '\n');
}

bool EFGGame::IsNodeToken(char c) const {
  return (c == 'c' || c == 'p' || c == 't');
}

std::unique_ptr<Node> EFGGame::NewNode() const {
  std::unique_ptr<Node> new_node = std::make_unique<Node>();
  new_node->id = nodes_.size();
  return new_node;
}

// Let's use custom parser macros, so that we can print the line
// and an error about what happened while parsing the gambit file.

#define SPIEL_EFG_PARSE_CHECK_OP(x_exp, op, y_exp)                   \
  do {                                                               \
    auto x = x_exp;                                                  \
    auto y = y_exp;                                                  \
    if (!((x)op(y)))                                                 \
      open_spiel::SpielFatalError(open_spiel::internal::SpielStrCat( \
          __FILE__, ":", __LINE__, " ", #x_exp " " #op " " #y_exp,   \
          "\n" #x_exp, " = ", x, ", " #y_exp " = ", y, "\n",         \
          " while parsing line #", line_, ":\n", GetLine(line_)));   \
  } while (false)

#define SPIEL_EFG_PARSE_CHECK_GE(x, y) SPIEL_EFG_PARSE_CHECK_OP(x, >=, y)
#define SPIEL_EFG_PARSE_CHECK_GT(x, y) SPIEL_EFG_PARSE_CHECK_OP(x, >, y)
#define SPIEL_EFG_PARSE_CHECK_LE(x, y) SPIEL_EFG_PARSE_CHECK_OP(x, <=, y)
#define SPIEL_EFG_PARSE_CHECK_LT(x, y) SPIEL_EFG_PARSE_CHECK_OP(x, <, y)
#define SPIEL_EFG_PARSE_CHECK_EQ(x, y) SPIEL_EFG_PARSE_CHECK_OP(x, ==, y)
#define SPIEL_EFG_PARSE_CHECK_NE(x, y) SPIEL_EFG_PARSE_CHECK_OP(x, !=, y)

#define SPIEL_EFG_PARSE_CHECK_TRUE(x)                            \
  while (!(x))                                                   \
  open_spiel::SpielFatalError(open_spiel::internal::SpielStrCat( \
      __FILE__, ":", __LINE__, " CHECK_TRUE(", #x, ")\n",        \
      " while parsing line #", line_, ":\n", GetLine(line_)))

#define SPIEL_EFG_PARSE_CHECK_FALSE(x)                           \
  while (x)                                                      \
  open_spiel::SpielFatalError(open_spiel::internal::SpielStrCat( \
      __FILE__, ":", __LINE__, " CHECK_FALSE(", #x, ")\n",       \
      " while parsing line #", line_, ":\n", GetLine(line_)))

bool EFGGame::ParseDoubleValue(const std::string& str, double* value) const {
  if (str.find('/') != std::string::npos) {
    // Check for rational number of the form X/Y
    std::vector<std::string> parts = absl::StrSplit(str, '/');
    SPIEL_EFG_PARSE_CHECK_EQ(parts.size(), 2);
    int numerator = 0, denominator = 0;
    bool success = absl::SimpleAtoi(parts[0], &numerator);
    if (!success) {
      return false;
    }
    success = absl::SimpleAtoi(parts[1], &denominator);
    if (!success) {
      return false;
    }
    SPIEL_EFG_PARSE_CHECK_FALSE(denominator == 0);
    *value = static_cast<double>(numerator) / denominator;
    return true;
  } else {
    // Otherwise, parse as a double.
    return absl::SimpleAtod(str, value);
  }
}

std::string EFGGame::NextToken() {
  std::string str = "";
  bool reading_quoted_string = false;

  if (string_data_.at(pos_) == '"') {
    reading_quoted_string = true;
    AdvancePosition();
  }

  while (true) {
    // Check stopping condition:
    if (pos_ >= string_data_.length() ||
        (reading_quoted_string && string_data_.at(pos_) == '"') ||
        (!reading_quoted_string && IsWhiteSpace(string_data_.at(pos_)))) {
      break;
    }

    str.push_back(string_data_.at(pos_));
    AdvancePosition();
  }

  if (reading_quoted_string) {
    SPIEL_EFG_PARSE_CHECK_EQ(string_data_.at(pos_), '"');
  }
  AdvancePosition();

  // Advance the position to the next token.
  while (pos_ < string_data_.length() && IsWhiteSpace(string_data_.at(pos_))) {
    AdvancePosition();
  }

  return str;
}

void EFGGame::AdvancePosition() {
  pos_++;
  if (string_data_[pos_] == '\n') line_++;
}

std::string EFGGame::GetLine(int line) const {
  SPIEL_CHECK_GE(line, 1);

  int cur_line = 1;
  int pos = 0;
  int len = string_data_.size();
  std::string buf;
  do {
    if (cur_line == line) buf.push_back(string_data_[pos]);
    if (string_data_[pos] == '\n') cur_line++;
    pos++;
  } while (cur_line != line + 1 && pos < len);

  return buf;
}

/*
EFG 2 R "General Bayes game, one stage" { "Player 1" "Player 2" }
c "ROOT" 1 "(0,1)" { "1G" 0.500000 "1B" 0.500000 } 0
c "" 2 "(0,2)" { "2g" 0.500000 "2b" 0.500000 } 0
p "" 1 1 "(1,1)" { "H" "L" } 0
p "" 2 1 "(2,1)" { "h" "l" } 0
t "" 1 "Outcome 1" { 10.000000 2.000000 }
t "" 2 "Outcome 2" { 0.000000 10.000000 }
p "" 2 1 "(2,1)" { "h" "l" } 0
t "" 3 "Outcome 3" { 2.000000 4.000000 }
t "" 4 "Outcome 4" { 4.000000 0.000000 }
p "" 1 1 "(1,1)" { "H" "L" } 0
p "" 2 2 "(2,2)" { "h" "l" } 0
t "" 5 "Outcome 5" { 10.000000 2.000000 }
t "" 6 "Outcome 6" { 0.000000 10.000000 }
p "" 2 2 "(2,2)" { "h" "l" } 0
t "" 7 "Outcome 7" { 2.000000 4.000000 }
t "" 8 "Outcome 8" { 4.000000 0.000000 }
c "" 3 "(0,3)" { "2g" 0.500000 "2b" 0.500000 } 0
p "" 1 2 "(1,2)" { "H" "L" } 0
p "" 2 1 "(2,1)" { "h" "l" } 0
t "" 9 "Outcome 9" { 4.000000 2.000000 }
t "" 10 "Outcome 10" { 2.000000 10.000000 }
p "" 2 1 "(2,1)" { "h" "l" } 0
t "" 11 "Outcome 11" { 0.000000 4.000000 }
t "" 12 "Outcome 12" { 10.000000 2.000000 }
p "" 1 2 "(1,2)" { "H" "L" } 0
p "" 2 2 "(2,2)" { "h" "l" } 0
t "" 13 "Outcome 13" { 4.000000 2.000000 }
t "" 14 "Outcome 14" { 2.000000 10.000000 }
p "" 2 2 "(2,2)" { "h" "l" } 0
t "" 15 "Outcome 15" { 0.000000 4.000000 }
t "" 16 "Outcome 16" { 10.000000 0.000000 }
*/
void EFGGame::ParsePrologue() {
  // Parse the first part of the header "EFG 2 R "
  SPIEL_EFG_PARSE_CHECK_TRUE(NextToken() == "EFG");
  SPIEL_EFG_PARSE_CHECK_LT(pos_, string_data_.length());
  SPIEL_EFG_PARSE_CHECK_TRUE(NextToken() == "2");
  SPIEL_EFG_PARSE_CHECK_LT(pos_, string_data_.length());
  SPIEL_EFG_PARSE_CHECK_TRUE(NextToken() == "R");
  SPIEL_EFG_PARSE_CHECK_LT(pos_, string_data_.length());
  SPIEL_EFG_PARSE_CHECK_EQ(string_data_.at(pos_), '"');
  name_ = NextToken();
  std::string token = NextToken();
  SPIEL_EFG_PARSE_CHECK_TRUE(token == "{");
  SPIEL_EFG_PARSE_CHECK_EQ(string_data_.at(pos_), '"');
  token = NextToken();
  while (token != "}") {
    player_names_.push_back(token);
    token = NextToken();
  }
  num_players_ = player_names_.size();
  infoset_num_to_states_count_.resize(num_players_, {});
  if (string_data_.at(pos_) == '"') {
    description_ = NextToken();
  }
  SPIEL_EFG_PARSE_CHECK_LT(pos_, string_data_.length());
  SPIEL_EFG_PARSE_CHECK_TRUE(IsNodeToken(string_data_.at(pos_)));
}

void EFGGame::ParseChanceNode(Node* parent, Node* child, int depth) {
  // a text string, giving the name of the node
  // a positive integer specifying the information set number
  // (optional) the name of the information set
  // (optional) a list of actions at the information set with their
  //      corresponding probabilities
  // a nonnegative integer specifying the outcome
  // (optional)the payoffs to each player for the outcome
  //
  // c "ROOT" 1 "(0,1)" { "1G" 0.500000 "1B" 0.500000 } 0
  SPIEL_EFG_PARSE_CHECK_TRUE(NextToken() == "c");
  num_chance_nodes_++;
  max_depth_ = std::max(max_depth_, depth);
  child->type = NodeType::kChance;
  child->parent = parent;
  SPIEL_EFG_PARSE_CHECK_EQ(string_data_.at(pos_), '"');
  child->name = NextToken();
  SPIEL_EFG_PARSE_CHECK_FALSE(string_data_.at(pos_) == '"');
  SPIEL_EFG_PARSE_CHECK_TRUE(
      absl::SimpleAtoi(NextToken(), &child->infoset_number));
  if (string_data_.at(pos_) == '"') {
    child->infoset_name = NextToken();
  }
  // I do not understand how the list of children can be optional.
  SPIEL_EFG_PARSE_CHECK_TRUE(NextToken() == "{");
  int chance_outcomes = 0;
  double prob_sum = 0.0;
  while (string_data_.at(pos_) == '"') {
    std::string action_str = NextToken();
    child->actions.push_back(action_str);
    Action action = AddOrGetChanceOutcome(action_str);
    child->action_ids.push_back(action);
    double prob = -1;
    SPIEL_EFG_PARSE_CHECK_TRUE(ParseDoubleValue(NextToken(), &prob));
    SPIEL_EFG_PARSE_CHECK_GE(prob, 0.0);
    SPIEL_EFG_PARSE_CHECK_LE(prob, 1.0);
    prob_sum += prob;
    child->probs.push_back(prob);
    nodes_.push_back(NewNode());
    child->children.push_back(nodes_.back().get());
    chance_outcomes++;
  }
  SPIEL_EFG_PARSE_CHECK_GT(child->actions.size(), 0);
  absl::c_sort(child->action_ids);
  SPIEL_EFG_PARSE_CHECK_TRUE(Near(prob_sum, 1.0));
  SPIEL_EFG_PARSE_CHECK_TRUE(NextToken() == "}");
  SPIEL_EFG_PARSE_CHECK_TRUE(
      absl::SimpleAtoi(NextToken(), &child->outcome_number));
  // Do not support optional payoffs here for now.

  // Now, recurse:
  for (Node* grand_child : child->children) {
    RecParseSubtree(child, grand_child, depth + 1);
  }
}

void EFGGame::UpdateAndCheckInfosetMaps(const Node* node) {
  // If the infoset name is not empty:
  //   1. ensure mapping from infoset (player,num) -> name is consistent, adding
  //      it if it doesn't exist.
  //   2. Add also name -> (player, num) to a hash map
  Player player = node->player_number - 1;
  if (!node->infoset_name.empty()) {
    std::pair<Player, int> key = {player, node->infoset_number};
    const auto& iter1 = infoset_player_num_to_name_.find(key);
    if (iter1 != infoset_player_num_to_name_.end()) {
      if (iter1->second != node->infoset_name) {
        SpielFatalError(absl::StrCat(
            "Inconsistent infoset (player, num) -> name: ",
            static_cast<int>(player), ",", node->infoset_number, " ",
            node->infoset_name, " ", iter1->second, "\nfilename: ", filename_,
            "\nstring data:\n", string_data_));
      }
    } else {
      std::pair<Player, int> key = {player, node->infoset_number};
      infoset_player_num_to_name_[key] = node->infoset_name;
    }

    // Name -> infoset number is not required to be unique in .efg so we don't
    // check it. So these may overlap unless the mapping is unique in the file.
    infoset_name_to_player_num_[node->infoset_name] = {player,
                                                       node->infoset_number};
  }
}

void EFGGame::ParsePlayerNode(Node* parent, Node* child, int depth) {
  // a text string, giving the name of the node
  // a positive integer specifying the player who owns the node
  // a positive integer specifying the information set
  // (optional) the name of the information set
  // (optional) a list of action names for the information set
  // a nonnegative integer specifying the outcome
  // (optional) the name of the outcome
  // the payoffs to each player for the outcome
  //
  // p "" 1 1 "(1,1)" { "H" "L" } 0
  SPIEL_EFG_PARSE_CHECK_TRUE(NextToken() == "p");
  max_depth_ = std::max(max_depth_, depth);
  child->type = NodeType::kPlayer;
  child->parent = parent;
  SPIEL_EFG_PARSE_CHECK_EQ(string_data_.at(pos_), '"');
  child->name = NextToken();
  SPIEL_EFG_PARSE_CHECK_FALSE(string_data_.at(pos_) == '"');
  SPIEL_EFG_PARSE_CHECK_TRUE(
      absl::SimpleAtoi(NextToken(), &child->player_number));
  SPIEL_EFG_PARSE_CHECK_TRUE(
      absl::SimpleAtoi(NextToken(), &child->infoset_number));
  infoset_num_to_states_count_[child->player_number - 1]
                              [child->infoset_number]++;
  if (infoset_num_to_states_count_[child->player_number - 1]
                                  [child->infoset_number] > 1) {
    perfect_information_ = false;
  }
  child->infoset_name = "";
  if (string_data_.at(pos_) == '"') {
    child->infoset_name = NextToken();
  }
  UpdateAndCheckInfosetMaps(child);
  // Do not understand how the list of actions can be optional.
  SPIEL_EFG_PARSE_CHECK_TRUE(NextToken() == "{");
  int actions = 0;
  while (string_data_.at(pos_) == '"') {
    std::string action_str = NextToken();
    child->actions.push_back(action_str);
    Action action = AddOrGetAction(action_str);
    child->action_ids.push_back(action);
    nodes_.push_back(NewNode());
    child->children.push_back(nodes_.back().get());
    actions++;
  }
  SPIEL_EFG_PARSE_CHECK_GT(child->actions.size(), 0);
  absl::c_sort(child->action_ids);
  max_actions_ = std::max(max_actions_, actions);
  SPIEL_EFG_PARSE_CHECK_TRUE(NextToken() == "}");
  SPIEL_EFG_PARSE_CHECK_TRUE(
      absl::SimpleAtoi(NextToken(), &child->outcome_number));
  // Do not support optional payoffs here for now.

  // Now, recurse:
  for (Node* grand_child : child->children) {
    RecParseSubtree(child, grand_child, depth + 1);
  }
}

void EFGGame::ParseTerminalNode(Node* parent, Node* child, int depth) {
  // a text string, giving the name of the node
  // a nonnegative integer specifying the outcome
  // (optional) the name of the outcome
  // the payoffs to each player for the outcome
  //
  // t "" 1 "Outcome 1" { 10.000000 2.000000 }
  SPIEL_EFG_PARSE_CHECK_TRUE(NextToken() == "t");
  max_depth_ = std::max(max_depth_, depth);
  child->type = NodeType::kTerminal;
  child->parent = parent;
  SPIEL_EFG_PARSE_CHECK_EQ(string_data_.at(pos_), '"');
  child->name = NextToken();
  SPIEL_EFG_PARSE_CHECK_TRUE(
      absl::SimpleAtoi(NextToken(), &child->outcome_number));
  if (string_data_.at(pos_) == '"') {
    child->outcome_name = NextToken();
  }
  SPIEL_EFG_PARSE_CHECK_TRUE(NextToken() == "{");

  int idx = 0;
  double util_sum = 0;
  bool identical = true;
  while (string_data_.at(pos_) != '}') {
    double utility = 0;
    SPIEL_EFG_PARSE_CHECK_TRUE(ParseDoubleValue(NextToken(), &utility));
    child->payoffs.push_back(utility);
    util_sum += utility;
    if (!min_util_.has_value()) {
      min_util_ = utility;
    }
    if (!max_util_.has_value()) {
      max_util_ = utility;
    }
    min_util_ = std::min(min_util_.value(), utility);
    max_util_ = std::max(max_util_.value(), utility);

    if (identical && idx >= 1 &&
        Near(child->payoffs[idx - 1], child->payoffs[idx])) {
      identical = true;
    } else {
      identical = false;
    }

    idx++;
  }
  SPIEL_EFG_PARSE_CHECK_EQ(child->payoffs.size(), num_players_);
  SPIEL_EFG_PARSE_CHECK_TRUE(NextToken() == "}");

  // Inspect the utilities to classify the utility type for this game.
  if (!util_sum_.has_value()) {
    util_sum_ = util_sum;
  }

  if (constant_sum_ && Near(util_sum_.value(), util_sum)) {
    constant_sum_ = true;
  } else {
    constant_sum_ = false;
  }

  if (identical_payoffs_ && identical) {
    identical_payoffs_ = true;
  } else {
    identical_payoffs_ = false;
  }
}

void EFGGame::RecParseSubtree(Node* parent, Node* child, int depth) {
  switch (string_data_.at(pos_)) {
    case 'c':
      ParseChanceNode(parent, child, depth);
      break;
    case 'p':
      ParsePlayerNode(parent, child, depth);
      break;
    case 't':
      ParseTerminalNode(parent, child, depth);
      break;
    default:
      SpielFatalError(absl::StrCat("Unexpected character at pos ", pos_, ": ",
                                   string_data_.substr(pos_, 1)));
  }
}

std::string EFGGame::PrettyTree(const Node* node,
                                const std::string& indent) const {
  std::string str = indent + NodeToString(node);
  for (Node* child : node->children) {
    str += PrettyTree(child, indent + "  ");
  }
  return str;
}

std::string EFGGame::GetInformationStateStringByName(
    Player player, const std::string& name) const {
  const auto& iter = infoset_name_to_player_num_.find(name);
  if (iter == infoset_name_to_player_num_.end()) {
    SpielFatalError(absl::StrCat("Information state not found: ", name));
  }
  if (iter->second.first != player) {
    SpielFatalError(absl::StrCat("Player mismatch in lookup by name: ", name,
                                 " ", player, " ", iter->second.first));
  }
  return EFGInformationStateString(player, player, iter->second.second, name);
}

std::string EFGGame::GetInformationStateStringByNumber(Player player,
                                                       int number) const {
  const auto& iter = infoset_player_num_to_name_.find({player, number});
  if (iter == infoset_player_num_to_name_.end()) {
    SpielFatalError(
        absl::StrCat("Information state not found: ", player, ",", number));
  }
  return EFGInformationStateString(player, player, number, iter->second);
}

void EFGGame::ParseGame() {
  // Skip any initial whitespace.
  while (IsWhiteSpace(string_data_.at(pos_))) {
    AdvancePosition();
  }
  SPIEL_EFG_PARSE_CHECK_LT(pos_, string_data_.length());

  ParsePrologue();
  nodes_.push_back(NewNode());
  RecParseSubtree(nullptr, nodes_[0].get(), 0);
  SPIEL_EFG_PARSE_CHECK_GE(pos_, string_data_.length());

  // Modify the game type.
  if (num_chance_nodes_ > 0) {
    game_type_.chance_mode = GameType::ChanceMode::kExplicitStochastic;
  }

  if (perfect_information_) {
    game_type_.information = GameType::Information::kPerfectInformation;
  } else {
    game_type_.information = GameType::Information::kImperfectInformation;
  }

  if (constant_sum_ && Near(util_sum_.value(), 0.0)) {
    game_type_.utility = GameType::Utility::kZeroSum;
  } else if (constant_sum_) {
    game_type_.utility = GameType::Utility::kConstantSum;
  } else if (identical_payoffs_) {
    game_type_.utility = GameType::Utility::kIdentical;
  } else {
    game_type_.utility = GameType::Utility::kGeneralSum;
  }

  game_type_.max_num_players = num_players_;
  game_type_.min_num_players = num_players_;

  // Check infoset number consistency. Currently they must start at 1 and end
  // at n_i for each player i. The InformationStateTensor currently requires
  // this to implement a proper info state tensor.
  for (Player p = 0; p < num_players_; ++p) {
    int max_value = 0;
    for (const auto& number_and_count : infoset_num_to_states_count_[p]) {
      SPIEL_CHECK_GE(number_and_count.first, 1);
      SPIEL_CHECK_LE(number_and_count.first,
                     infoset_num_to_states_count_[p].size());
      max_value = std::max<int>(max_value, number_and_count.first);
    }
    SPIEL_CHECK_EQ(max_value, infoset_num_to_states_count_[p].size());
  }
}

TabularPolicy EFGGameTabularPolicy(
    std::shared_ptr<const Game> game,
    const absl::flat_hash_map<std::pair<Player, std::string>,
                              std::vector<std::pair<std::string, double>>>&
        policy_map) {
  const EFGGame* efg_game = dynamic_cast<const EFGGame*>(game.get());
  SPIEL_CHECK_TRUE(efg_game != nullptr);

  TabularPolicy policy;
  for (const auto& outer_iter : policy_map) {
    Player player = outer_iter.first.first;
    std::string infoset_label = outer_iter.first.second;
    std::string infoset_str =
        efg_game->GetInformationStateStringByName(player, infoset_label);

    ActionsAndProbs state_policy;
    state_policy.reserve(outer_iter.second.size());
    for (const auto& inner_iter : outer_iter.second) {
      std::string action_label = inner_iter.first;
      double prob = inner_iter.second;
      Action action = efg_game->GetAction(action_label);
      state_policy.push_back({action, prob});
    }

    policy.SetStatePolicy(infoset_str, state_policy);
  }

  return policy;
}


}  // namespace efg_game
}  // namespace open_spiel
