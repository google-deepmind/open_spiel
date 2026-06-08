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

#include "open_spiel/games/crossword/crossword.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/strings/ascii.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/crossword/crossword_board.h"
#include "open_spiel/games/crossword/crossword_default_puzzle.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/file.h"
#include "open_spiel/utils/status.h"

namespace open_spiel {
namespace crossword {
namespace {

inline constexpr char kEmptyCell = ' ';


// Facts about the game
const GameType kGameType{/*short_name=*/"crossword",
                         /*long_name=*/"crossword",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kImperfectInformation,
                         GameType::Utility::kGeneralSum,
                         GameType::RewardModel::kRewards,
                         /*max_num_players=*/kNumPlayers,
                         /*min_num_players=*/kNumPlayers,
                         /*provides_information_state_string=*/false,
                         /*provides_information_state_tensor=*/false,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/
                         {{"puzzles_root",
                           GameParameter(kDefaultPuzzlesRoot)},
                           {"word_list_file",
                            GameParameter(kDefaultWordListFile)}},
                         /*default_loadable*/true,
                         /*provides_factored_observation_string*/false,
                         /*is_concrete*/true,
                         /*action_structs_only*/true};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new CrosswordGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);
}  // namespace

CrosswordState::CrosswordState(std::shared_ptr<const Game> game)
    : State(game),
      parent_game_(down_cast<const CrosswordGame&>(*game)),
      num_actions_(0),
      puzzle_loading_attempts_(0),
      current_player_(kChancePlayerId),
      reward_(0),
      return_(0) {
  if (parent_game_.num_puzzles() > 0) {
    // If there is a database of puzzles, we start with a chance node which
    // will select a puzzle.
    current_player_ = kChancePlayerId;
  } else {
    // Otherwise, we start with a default preset puzzle.
    StatusWithValue<CrosswordBoard> result =
        ParseCrosswordFromString(kDefaultPuzzleContents);
    SPIEL_CHECK_TRUE(result.ok());
    current_player_ = 0;
    board_ = result.value();
    InitializeState();
  }
}

void CrosswordState::InitializeState() {
  // Set all the cells to empty or blocked.
  for (int r = 0; r < board_.num_rows(); ++r) {
    for (int c = 0; c < board_.num_cols(); ++c) {
      if (board_.CharacterAt(r, c) == kBlockedCell) {
        board_state_.push_back(kBlockedCell);
      } else {
        board_state_.push_back(kEmptyCell);
      }
    }
  }

  // Build up cell location to cid needed for ToString().
  for (const auto& [cid, cell] : board_.cid_to_cells_map()) {
    cell_to_cid_[cell] = cid;
  }

  // All clues are initially unsolved.
  clue_solved_.resize(board_.num_clues(), 0);

  num_actions_ = 0;
  return_ = 0;

  // Set the answer set for legality checks.
  for (const std::string& answer : board_.Answers()) {
    answer_set_.insert(absl::AsciiStrToUpper(answer));
  }
}

int CrosswordState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  } else {
    return current_player_;
  }
}

void CrosswordState::DoApplyAction(Action action) {
  SPIEL_CHECK_FALSE(IsTerminal());
  if (IsChanceNode()) {
    SPIEL_CHECK_GE(action, 0);
    SPIEL_CHECK_LT(action, parent_game_.num_puzzles());
    std::string puzzle_file = parent_game_.crossword_file(action);
    StatusWithValue<CrosswordBoard> result =
        ParseCrosswordFromFile(puzzle_file);
    if (!result.ok()) {
      puzzle_loading_attempts_++;
      std::cerr << absl::StrCat(
          "Warning: [Attempt ", puzzle_loading_attempts_,
          "] Failed to load puzzle: ", puzzle_file, ".\n",
          "Error message: \n", result.message(), "\n");
    } else {
      current_player_ = 0;
      board_ = result.value();
      InitializeState();
    }
  } else {
    SpielFatalError("DoApplyAction unimplemented for non-chance node. "
                    "This game uses action structs only.");
  }
}

Status CrosswordState::ValidateActionStruct(
    const ActionStruct& action_struct) const {
  const auto& crossword_action =
      down_cast<const CrosswordActionStruct&>(action_struct);
  if (crossword_action.word.empty()) {
    return ErrorStatus(
        "Action word cannot be empty.");
  }

  // If the word list is non-empty: check word list union answer set to see if
  // the action word is present. If the word list is empty, then any word is
  // allowed.
  if (!parent_game_.word_set().empty()) {
    if (!(answer_set_.contains(absl::AsciiStrToUpper(crossword_action.word)) ||
          parent_game_.word_set().contains(
              absl::AsciiStrToUpper(crossword_action.word)))) {
      return ErrorStatus(
          absl::StrCat("Action word ", crossword_action.word,
                       " not found in word list or answer set."));
    }
  }

  const auto& cid_to_cells_map = board_.cid_to_cells_map();
  const std::string& cid = crossword_action.clue_id;
  const Clue& clue = board_.clue(cid);
  int clue_index = board_.clue_index(clue);

  if (clue_solved_[clue_index] == 1) {
    // Can't replay clues that have already been solved.
    return ErrorStatus(
        absl::StrCat("Clue ", cid, " has already been solved."));
  }

  const std::string& answer = board_.answer(cid);

  // The word has to be the same length as the answer.
  if (crossword_action.word.length() != answer.length()) {
    return ErrorStatus(
        absl::StrCat("Action word ", crossword_action.word,
                     " is not the correct length for clue ", cid));
  }

  auto iter = cid_to_cells_map.find(cid);
  if (iter == cid_to_cells_map.end()) {
    return ErrorStatus(
        absl::StrCat("Clue id ", cid, " not found in board."));
  }

  auto [row, col] = iter->second;
  for (int i = 0; i < crossword_action.word.size(); ++i) {
    int r = row + (clue.direction == Direction::kAcross ? 0 : i);
    int c = col + (clue.direction == Direction::kAcross ? i : 0);
    if (!board_.InBounds(r, c)) {
      return ErrorStatus(
          absl::StrCat("Action word ", crossword_action.word,
                       " goes out of bounds at position ", i));
    }
    char board_char = CharAt(r, c);
    if (board_char != kEmptyCell &&
        absl::ascii_toupper(board_char) !=
            absl::ascii_toupper(crossword_action.word[i])) {
      std::string error_message = absl::StrCat(
          "Action word ", crossword_action.word,
          " conflicts with board character ") + board_char +
          absl::StrCat(" at position ", i);
      return ErrorStatus(error_message);
    }
  }

  return OkStatus();
}

void CrosswordState::SetRewardAndIncrementReturn(int reward) {
  reward_ = reward;
  return_ += reward;
}

Status CrosswordState::ApplyActionStruct(const ActionStruct& action_struct) {
  auto status = ValidateActionStruct(action_struct);
  if (!status.ok()) {
    return status;
  }
  const CrosswordActionStruct& crossword_action =
      down_cast<const CrosswordActionStruct&>(action_struct);
  const auto& cid_to_cells_map = board_.cid_to_cells_map();
  const std::string& cid = crossword_action.clue_id;
  const Clue& clue = board_.clue(cid);
  auto [row, col] = cid_to_cells_map.at(cid);
  // If it's the wrong answer, it's a legal action, but we add a penalty.
  if (!board_.MatchesCorrectLetters(row, col, clue.direction,
                                    crossword_action.word)) {
    num_actions_++;
    SetRewardAndIncrementReturn(kIncorrectCluePenalty);
    return OkStatus();
  }
  // Correct answer, update the board and clue solved status.
  for (int i = 0; i < crossword_action.word.size(); ++i) {
    int r = row + (clue.direction == Direction::kAcross ? 0 : i);
    int c = col + (clue.direction == Direction::kAcross ? i : 0);
    board_state_[r * board_.num_cols() + c] = crossword_action.word[i];
  }
  int clue_index = board_.clue_index(clue);
  SPIEL_CHECK_GE(clue_index, 0);
  SPIEL_CHECK_LT(clue_index, clue_solved_.size());
  clue_solved_[clue_index] = 1;
  int reward = kCorrectClueReward + (IsSolved() ? kSolvedPuzzleReward : 0);
  SetRewardAndIncrementReturn(reward);
  num_actions_++;
  return OkStatus();
}


std::string CrosswordState::ActionToString(Player player,
                                           Action action) const {
  if (player == kChancePlayerId) {
    return absl::StrCat("Chance node, sample puzzle: ", action);
  } else {
    SpielFatalError("ActionToString unimplemented for non-chance node. "
                    "This game uses action structs only.");
  }
}

std::unique_ptr<ActionStruct>
CrosswordActionStructSampler::SampleActionStruct() {
  const CrosswordGame* parent_game =
      down_cast<const CrosswordGame*>(state_->GetGame().get());
  SPIEL_CHECK_TRUE(parent_game != nullptr);
  const auto& word_set = parent_game->word_set();
  if (word_set.empty()) {
    std::cerr << "Word set is empty, cannot sample action. "
              << "Returning nullptr.\n";
    return nullptr;
  }

  const auto* crossword_state = down_cast<const CrosswordState*>(state_);
  const std::vector<int>& clue_solved = crossword_state->clue_solved();
  const auto& answer_set = crossword_state->answer_set();

  while (true) {
    std::vector<int> unsolved_clue_indices;
    for (int i = 0; i < clue_solved.size(); ++i) {
      if (clue_solved[i] == 0) {
        unsolved_clue_indices.push_back(i);
      }
    }
    int sampled_idx = absl::Uniform<int>(rng_, 0, unsolved_clue_indices.size());
    int clue_index = unsolved_clue_indices[sampled_idx];
    const Clue& clue = crossword_state->board().clue(clue_index);
    std::string cid = ClueId(clue);
    std::string answer = crossword_state->board().answer(cid);
    std::string sampled_word = "";

    // For all words in the word set and in the answer set, sample one.
    std::vector<std::string> possible_words;
    for (const std::string& word : word_set) {
      if (word.length() == answer.length()) {
        CrosswordActionStruct candidate_action(cid, word);
        if (state_->ValidateActionStruct(candidate_action).ok()) {
          possible_words.push_back(word);
        }
      }
    }
    for (const std::string& word : answer_set) {
      if (word.length() == answer.length()) {
        CrosswordActionStruct candidate_action(cid, word);
        if (state_->ValidateActionStruct(candidate_action).ok()) {
          possible_words.push_back(word);
        }
      }
    }

    if (!possible_words.empty()) {
      sampled_word =
          possible_words[absl::Uniform<int>(rng_, 0, possible_words.size())];
      return std::make_unique<CrosswordActionStruct>(cid, sampled_word);
    }
  }
}

std::vector<std::pair<Action, double>> CrosswordState::ChanceOutcomes() const {
  int num_puzzles = parent_game_.num_puzzles();
  SPIEL_CHECK_GT(num_puzzles, 0);
  std::vector<std::pair<Action, double>> outcomes;
  outcomes.reserve(num_puzzles);
  for (int i = 0; i < num_puzzles; ++i) {
    outcomes.push_back({i, 1.0 / num_puzzles});
  }
  return outcomes;
}

std::vector<Action> CrosswordState::LegalActions() const {
  if (IsTerminal()) {
    return {};
  } else if (IsChanceNode()) {
    return LegalChanceOutcomes();
  } else {
    SpielFatalError("LegalActions unimplemented for non-chance node. "
                    "This game uses action structs only.");
  }
}

std::string CrosswordState::ToString() const {
  std::string result = "";
  if (current_player_ == kChancePlayerId) {
    absl::StrAppend(&result, "Chance node, selecting a puzzle.");
  } else {
    absl::StrAppend(&result, "rows: ", board_.num_rows(),
                    ", cols: ", board_.num_cols(), "\n");
    absl::StrAppend(&result, "num_actions: ", num_actions_, "\n");
    absl::StrAppend(&result, "return: ", return_, "\n\n");
    for (int r = 0; r < board_.num_rows(); ++r) {
      for (int c = 0; c < board_.num_cols(); ++c) {
        if (CharAt(r, c) == kBlockedCell) {
          absl::StrAppend(&result, "## ");
          continue;
        }
        auto iter = cell_to_cid_.find({r, c});
        if (iter == cell_to_cid_.end()) {
          // No clue starts at this cell.
          absl::StrAppend(&result, "   ");
        } else {
          std::string number = iter->second.substr(1);
          absl::StrAppend(&result, number);
          int num_spaces = 3 - number.size();
          if (num_spaces > 0) {
            for (int i = 0; i < num_spaces; ++i) {
              absl::StrAppend(&result, " ");
            }
          }
        }
      }
      absl::StrAppend(&result, "\n");
      for (int c = 0; c < board_.num_cols(); ++c) {
        if (CharAt(r, c) == kBlockedCell) {
          absl::StrAppend(&result, "## ");
        } else if (CharAt(r, c) == kEmptyCell) {
          absl::StrAppend(&result, "   ");
          continue;
        } else {
          absl::StrAppend(&result, " ");
          result.push_back(absl::ascii_toupper(CharAt(r, c)));
          absl::StrAppend(&result, " ");
        }
      }
      absl::StrAppend(&result, "\n");
    }
  }
  return result;
}

bool CrosswordState::IsSolved() const {
  // Note: this can be true before all clues are marked as solved. E.g. a
  // crossword can end even if only the Across clues are solved, filling up the
  // board with letters. Before this happens, the unsolved clues that still have
  // all their letters filled on the board are still valid actions.
  return (!board_state_.empty() &&
          std::find(board_state_.begin(), board_state_.end(), kEmptyCell) ==
          board_state_.end());
}

bool CrosswordState::IsTerminal() const {
  return (puzzle_loading_attempts_ >= kMaxPuzzleLoadingAttempts ||
          num_actions_ >= kMaxGameLength || IsSolved());
}

std::vector<double> CrosswordState::Rewards() const {
  return {static_cast<double>(reward_)};
}

std::vector<double> CrosswordState::Returns() const {
  return {static_cast<double>(return_)};
}

std::string CrosswordState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void CrosswordState::ObservationTensor(Player player,
                                       absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
}

std::unique_ptr<State> CrosswordState::Clone() const {
  return std::unique_ptr<State>(new CrosswordState(*this));
}

Status CrosswordGame::ParseWordList() {
  if (word_list_file_.empty()) {
    return OkStatus();
  }

  std::string contents = file::ReadContentsFromFile(word_list_file_, "r");
  std::vector<std::string> words = absl::StrSplit(contents, '\n');
  for (const std::string& word : words) {
    auto trimmed = absl::StripAsciiWhitespace(word);
    if (!trimmed.empty()) {
      word_set_.insert(absl::AsciiStrToUpper(trimmed));
    }
  }
  return OkStatus();
}

Status CrosswordGame::BuildPuzzleFileList() {
  if (puzzles_root_.empty()) {
    return OkStatus();
  }

  std::vector<std::string> all_files = file::ListDir(puzzles_root_,
                                                     /*recurse*/true);
  for (const std::string& file : all_files) {
    if (file.ends_with(".xd")) {
      puzzle_files_.push_back(file);
    }
  }
  return OkStatus();
}

CrosswordGame::CrosswordGame(const GameParameters& params)
    : Game(kGameType, params),
      puzzles_root_(ParameterValue<std::string>("puzzles_root",
                                                kDefaultPuzzlesRoot)),
      word_list_file_(ParameterValue<std::string>("word_list_file",
                                                  kDefaultWordListFile)) {
  ParseWordList();
  BuildPuzzleFileList();
}

int CrosswordGame::NumDistinctActions() const {
  return 0;
}

}  // namespace crossword
}  // namespace open_spiel

