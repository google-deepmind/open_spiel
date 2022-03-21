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

#include "open_spiel/games/nfg_game.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>

#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/matrix_game.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tensor_game.h"
#include "open_spiel/utils/file.h"

namespace open_spiel {
namespace nfg_game {
namespace {
using std::shared_ptr;

constexpr int kBuffSize = 1024;

// Facts about the game. These are defaults that will differ depending on the
// game's descriptions. Using dummy defaults just to register the game.
const GameType kGameType{/*short_name=*/"nfg_game",
                         /*long_name=*/"nfg_game",
                         GameType::Dynamics::kSimultaneous,
                         GameType::ChanceMode::kDeterministic,
                         GameType::Information::kOneShot,
                         GameType::Utility::kGeneralSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/100,
                         /*min_num_players=*/2,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/false,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/false,
                         /*parameter_specification=*/
                         {{"filename", GameParameter(std::string(""))}},
                         /*default_loadable=*/false};

class NFGGameParser {
 public:
  explicit NFGGameParser(const std::string& data)
      : string_data_(data), pos_(0) {}

  shared_ptr<const Game> ParseGame() {
    // Skip any initial whitespace.
    while (IsWhiteSpace(string_data_.at(pos_))) {
      AdvancePosition();
    }
    SPIEL_CHECK_LT(pos_, string_data_.length());

    ParsePrologue();
    InitializeMetaInformation();
    ParseUtilities();

    if (num_players_ == 2) {
      return matrix_game::CreateMatrixGame(
          "matrix_nfg", name_, matrix_row_action_names_,
          matrix_col_action_names_, matrix_row_utilities_,
          matrix_col_utilities_);
    } else {
      return tensor_game::CreateTensorGame(
          "tensor_nfg", name_, tensor_action_names_, tensor_utilities_);
    }
  }

 private:
  void ParsePrologue() {
    // Parse the first part of the header "NFG 1 R "
    SPIEL_CHECK_TRUE(NextToken() == "NFG");
    SPIEL_CHECK_TRUE(NextToken() == "1");
    // Older versions of .nfg format use D
    std::string data_type = NextToken();
    SPIEL_CHECK_TRUE(data_type == "R" || data_type == "D");
    SPIEL_CHECK_EQ(string_data_.at(pos_), '"');
    name_ = NextToken();
    // Player names
    std::string token = NextToken();
    SPIEL_CHECK_TRUE(token == "{");
    SPIEL_CHECK_EQ(string_data_.at(pos_), '"');
    token = NextToken();
    while (token != "}") {
      player_names_.push_back(token);
      token = NextToken();
    }
    num_players_ = player_names_.size();
    // Number of actions
    token = NextToken();
    SPIEL_CHECK_TRUE(token == "{");
    token = NextToken();
    while (token != "}") {
      int num = 0;
      SPIEL_CHECK_TRUE(absl::SimpleAtoi(token, &num));
      num_actions_.push_back(num);
      token = NextToken();
    }
    SPIEL_CHECK_EQ(num_actions_.size(), num_players_);
  }

  void InitializeMetaInformation() {
    total_entries_ = std::accumulate(num_actions_.begin(), num_actions_.end(),
                                     1ULL, std::multiplies<uint64_t>());
    SPIEL_CHECK_GT(total_entries_, 0);

    // Fill some of the meta information.
    if (num_players_ == 2) {
      matrix_row_action_names_.reserve(num_actions_[0]);
      matrix_col_action_names_.reserve(num_actions_[1]);
      matrix_row_utilities_ = std::vector<double>(total_entries_, 0);
      matrix_col_utilities_ = std::vector<double>(total_entries_, 0);
      for (int a = 0; a < num_actions_[0]; ++a) {
        matrix_row_action_names_.push_back(absl::StrCat("", a));
      }
      for (int a = 0; a < num_actions_[1]; ++a) {
        matrix_col_action_names_.push_back(absl::StrCat("", a));
      }
    } else {
      tensor_action_names_.reserve(num_players_);
      tensor_utilities_.reserve(num_players_);
      for (int p = 0; p < num_players_; ++p) {
        tensor_utilities_.push_back(std::vector<double>(total_entries_, 0));
        tensor_action_names_.push_back({});
        tensor_action_names_.back().reserve(num_actions_[p]);
        for (int a = 0; a < num_actions_[p]; ++a) {
          tensor_action_names_[p].push_back(absl::StrCat("", a));
        }
      }
    }
  }

  int RowMajorIndex(const std::vector<int>& num_actions,
                    const std::vector<int>& actions) {
    int index = 0;
    int base_value = 1;
    for (int p = actions.size() - 1; p >= 0; --p) {
      if (p + 1 < actions.size()) {
        base_value *= num_actions[p + 1];
      }
      index += actions[p] * base_value;
    }
    return index;
  }

  void ParseUtilities() {
    // Parse all the utilities.
    std::string token;
    std::vector<int> actions(num_players_, 0);
    for (uint64_t entry = 0; entry < total_entries_; ++entry) {
      double value = 0;
      int row_major_index = RowMajorIndex(num_actions_, actions);
      for (int p = 0; p < num_players_; ++p) {
        // Check that the position has not reached the end for every value we
        // read, except the very last one.
        bool check_end = entry != total_entries_ - 1 && p != num_players_ - 1;
        std::string token = NextToken(check_end);
        ParseDoubleValue(token, &value);

        if (num_players_ == 2) {
          if (p == 0) {
            matrix_row_utilities_[row_major_index] = value;
          } else {
            matrix_col_utilities_[row_major_index] = value;
          }
        } else {
          tensor_utilities_[p][row_major_index] = value;
        }
      }

      // next action indices, in column-major order.
      for (int i = 0; i < actions.size(); ++i) {
        if (++actions[i] < num_actions_[i]) {
          break;
        } else {
          actions[i] = 0;
        }
      }
    }

    // After reading all the utilities, we should reach the end of the file.
    SPIEL_CHECK_EQ(pos_, string_data_.length());
  }

  bool ParseDoubleValue(const std::string& str, double* value) const {
    if (str.find('/') != std::string::npos) {
      // Check for rational number of the form X/Y
      std::vector<std::string> parts = absl::StrSplit(str, '/');
      SPIEL_CHECK_EQ(parts.size(), 2);
      int numerator = 0, denominator = 0;
      bool success = absl::SimpleAtoi(parts[0], &numerator);
      if (!success) {
        return false;
      }
      success = absl::SimpleAtoi(parts[1], &denominator);
      if (!success) {
        return false;
      }
      SPIEL_CHECK_FALSE(denominator == 0);
      *value = static_cast<double>(numerator) / denominator;
      return true;
    } else {
      // Otherwise, parse as a double.
      return absl::SimpleAtod(str, value);
    }
  }

  bool IsWhiteSpace(char c) const {
    return (c == ' ' || c == '\r' || c == '\n');
  }

  void AdvancePosition() { pos_++; }

  // Get the next token, and then advance the position to the start of the next
  // token. If check_not_end is true, then a check is done to ensure that the
  // position has not reached the end of the string.
  std::string NextToken(bool check_not_end = true) {
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
      SPIEL_CHECK_EQ(string_data_.at(pos_), '"');
    }
    AdvancePosition();

    // Advance the position to the next token.
    while (pos_ < string_data_.length() &&
           IsWhiteSpace(string_data_.at(pos_))) {
      AdvancePosition();
    }

    if (check_not_end) {
      SPIEL_CHECK_LT(pos_, string_data_.length());
    }

    return str;
  }

  const std::string& string_data_;
  int pos_;
  int num_players_;
  std::string name_;
  std::vector<std::string> player_names_;
  std::vector<int> num_actions_;
  std::vector<std::vector<double>> utilities_;

  // Information needed to construct the matrix / tensor games.
  uint64_t total_entries_;
  // MatrixGame case.
  std::vector<std::string> matrix_row_action_names_;
  std::vector<std::string> matrix_col_action_names_;
  std::vector<double> matrix_row_utilities_;
  std::vector<double> matrix_col_utilities_;
  // TensorGame case.
  std::vector<std::vector<std::string>> tensor_action_names_;
  std::vector<std::vector<double>> tensor_utilities_;
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  // return std::shared_ptr<const Game>(new EFGGame(params));
  std::string filename = params.at("filename").string_value();
  std::string string_data = file::ReadContentsFromFile(filename, "r");

  SPIEL_CHECK_GT(string_data.size(), 0);
  NFGGameParser parser(string_data);
  return parser.ParseGame();
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace

std::shared_ptr<const Game> LoadNFGGame(const std::string& data) {
  NFGGameParser parser(data);
  return parser.ParseGame();
}

}  // namespace nfg_game
}  // namespace open_spiel
