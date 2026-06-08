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

#ifndef OPEN_SPIEL_GAMES_CROSSWORD_CROSSWORD_H_
#define OPEN_SPIEL_GAMES_CROSSWORD_CROSSWORD_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/container/flat_hash_set.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/games/crossword/crossword_board.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/status.h"

// A simple crossword game.
//
// IMPORTANT NOTE: this game uses actions structs only. In particular, it does
// not support flat (integer) actions and ApplyActionStruct must be used to
// apply actions to the state. See the documentation for these types of games in
// spiel.h, under the `action_structs_only` field of the GameType).
//
// The words list is optional. If not specified, any word is legal. The word
// list is a file with one valid word per line. Puzzles are stored in xd format.
// The puzzles_root parameter points to the directory containing the puzzles.
// If this is set, the game starts with a chance node allowing the selection of
// any puzzle in the directory. If it is unset, then a single default puzzle is
// used (see crossword_default_puzzle.h).
//
// There are many crossword puzzles in xd format freely available online, see
// https://github.com/century-arcade/xd for more information.
//
// Parameters:
//   puzzles_root      string   The directory containing the puzzles. If
//                              empty, uses a single default puzzle.
//                              Default: "".
//   word_list_file    string   The file containing the legal word list. If
//                              empty, all words are legal. Default: "".

namespace open_spiel {
namespace crossword {

// Constants
inline constexpr int kNumPlayers = 1;
inline constexpr const char* kDefaultPuzzlesRoot = "";
inline constexpr const char* kDefaultWordListFile = "";
inline constexpr int kMaxGameLength = 1000;

// The maximum number of times to try loading a puzzle from the puzzle files,
// i.e. after 10 failures, we just give up terminate the game.
inline constexpr int kMaxPuzzleLoadingAttempts = 10;

// Rewards
inline constexpr double kIncorrectCluePenalty = -1.0;
inline constexpr double kCorrectClueReward = 1.0;
inline constexpr double kSolvedPuzzleReward = 100.0;

class CrosswordGame;

struct CrosswordActionStruct : public ActionStruct {
  std::string clue_id;  // "A1", "D2", etc.
  std::string word;
  CrosswordActionStruct(const std::string& _clue_id, const std::string& _word)
       : clue_id(_clue_id), word(_word) {}
  std::string ToString() const {
    return absl::StrCat("CrosswordActionStruct(clue_id=", clue_id,
                        ", word=", word, ")");
  }
  SPIEL_STRUCT_BOILERPLATE(CrosswordActionStruct, clue_id, word);
};

class CrosswordActionStructSampler : public ActionStructSampler {
 public:
  explicit CrosswordActionStructSampler(const State* state, int rng_seed)
      : ActionStructSampler(state, rng_seed) {}
  std::unique_ptr<ActionStruct> SampleActionStruct() override;
};

class CrosswordState : public State {
 public:
  explicit CrosswordState(std::shared_ptr<const Game> game);
  CrosswordState(const CrosswordState&) = default;

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Rewards() const override;
  std::vector<double> Returns() const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions() const override;

  // For random sampling of a puzzle from a list of puzzle files.
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;

  Status ValidateActionStruct(
      const ActionStruct& action_struct) const override;

  Status ApplyActionStruct(const ActionStruct& action_struct) override;

  std::unique_ptr<ActionStructSampler> GetActionStructSampler(
      int seed) const override {
    return std::make_unique<CrosswordActionStructSampler>(this, seed);
  }

  char CharAt(int row, int col) const {
    return board_state_[row * board_.num_cols() + col];
  }

  const std::vector<int>& clue_solved() const { return clue_solved_; }
  const CrosswordBoard& board() const { return board_; }
  const absl::flat_hash_set<std::string>& answer_set() const {
    return answer_set_;
  }

 protected:
  void DoApplyAction(Action action_id) override;

 private:
  void InitializeState();
  void SetRewardAndIncrementReturn(int reward);
  bool IsSolved() const;

  const CrosswordGame& parent_game_;
  CrosswordBoard board_;

  int num_actions_ = 0;
  int puzzle_loading_attempts_ = 0;
  Player current_player_ = 0;
  int reward_ = 0;  // per-step reward
  int return_ = 0;  // cumulative reward
  std::vector<char> board_state_;
  absl::flat_hash_map<std::pair<int, int>, std::string> cell_to_cid_;
  std::vector<int> clue_solved_;  // 1 if solved, 0 otherwise.
  absl::flat_hash_set<std::string> answer_set_;  // Used for legality checks.
};

class CrosswordGame : public Game {
 public:
  explicit CrosswordGame(const GameParameters& params);

  int NumDistinctActions() const override;
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1.0; }
  double MaxUtility() const override { return 1.0; }
  int MaxGameLength() const override { return kMaxGameLength; }

  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new CrosswordState(shared_from_this()));
  }

  void AddWordToWordSet(const std::string& word) {
    word_set_.insert(word);
  }

  int num_puzzles() const { return puzzle_files_.size(); }
  int num_words() const { return word_set_.size(); }
  std::string crossword_file(int index) const {
    return puzzle_files_[index];
  }

  const absl::flat_hash_set<std::string>& word_set() const {
    return word_set_;
  }

 private:
  Status ParseWordList();
  Status BuildPuzzleFileList();

  std::string puzzles_root_;
  std::string word_list_file_;
  absl::flat_hash_set<std::string> word_set_;
  std::vector<std::string> puzzle_files_;
};

}  // namespace crossword
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_CROSSWORD_CROSSWORD_H_
