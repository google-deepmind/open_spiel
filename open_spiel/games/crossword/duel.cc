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

#include "open_spiel/games/crossword/duel.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/strings/ascii.h"
#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/games/crossword/duel_board.h"
#include "open_spiel/games/crossword/duel_default_board.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/file.h"

namespace open_spiel {
namespace crossword_duel {
namespace {

constexpr absl::string_view kChancePrefix = "ChanceOutcome: ";
constexpr absl::string_view kActionPrefix = "DuelAction: ";

// Used when the game parameter `word_list_file` is empty. Kept short and
// well within the tile bag's letter budget for the default board.
const std::vector<std::string> kDefaultWordList = {
    "CAT", "DOG", "BIRD", "FISH", "LION",
    "TIGER", "BEAR", "WOLF", "DEER", "GOAT",
};

const GameType kGameType{
    /*short_name=*/"crossword_duel",
    /*long_name=*/"Crossword Duel",
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
    {{"word_list_file", GameParameter(kDefaultWordListFile)}},
    /*default_loadable*/ true,
    /*provides_factored_observation_string*/ false,
    /*is_concrete*/ true,
    /*action_structs_only*/ true};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new CrosswordDuelGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

std::string DirectionToString(Direction direction) {
  return direction == Direction::kAcross ? "across" : "down";
}

}  // namespace

CrosswordDuelState::CrosswordDuelState(std::shared_ptr<const Game> game)
    : State(game),
      parent_game_(down_cast<const CrosswordDuelGame&>(*game)),
      board_(parent_game_.board()) {
  board_state_.assign(board_.num_rows() * board_.num_cols(), kEmptyCell);
  racks_.assign(kNumPlayers, {});
  scores_.assign(kNumPlayers, 0);
  reward_.assign(kNumPlayers, 0);
  bag_ = board_.BuildTileBag();

  drawing_player_ = 0;
  tiles_needed_ = std::min<int>(kRackSize, bag_.size());
  if (tiles_needed_ == 0) {
    dealing_initial_ = false;
    current_player_ = 0;
  } else {
    current_player_ = kChancePlayerId;
  }
}

Player CrosswordDuelState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  }
  return current_player_;
}

bool CrosswordDuelState::IsBoardEmpty() const {
  for (char c : board_state_) {
    if (c != kEmptyCell) return false;
  }
  return true;
}

int CrosswordDuelState::RackValue(Player player) const {
  int total = 0;
  for (char c : racks_[player]) {
    total += board_.LetterValue(c);
  }
  return total;
}

std::vector<std::pair<Action, double>> CrosswordDuelState::ChanceOutcomes()
    const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  std::array<int, 26> counts{};
  int total = 0;
  for (char c : bag_) {
    counts[c - 'A']++;
    total++;
  }
  SPIEL_CHECK_GT(total, 0);
  std::vector<std::pair<Action, double>> outcomes;
  for (int i = 0; i < 26; ++i) {
    if (counts[i] > 0) {
      outcomes.push_back({i, static_cast<double>(counts[i]) / total});
    }
  }
  return outcomes;
}

void CrosswordDuelState::FinishDraws() {
  if (dealing_initial_) {
    if (drawing_player_ == 0) {
      drawing_player_ = 1;
      tiles_needed_ = std::min<int>(kRackSize, bag_.size());
      if (tiles_needed_ == 0) {
        dealing_initial_ = false;
        current_player_ = 0;
      }
      // Otherwise stay at a chance node to deal to player 1.
    } else {
      dealing_initial_ = false;
      current_player_ = 0;
    }
    return;
  }
  // This was a mid-game refill following `drawing_player_`'s move.
  AdvanceOrEnd(drawing_player_);
}

void CrosswordDuelState::AdvanceOrEnd(Player mover) {
  Player other = 1 - mover;
  if (racks_[mover].empty() && bag_.empty()) {
    int bonus = RackValue(other);
    scores_[mover] += bonus;
    reward_[mover] += bonus;
    scores_[other] -= bonus;
    reward_[other] -= bonus;
    game_over_ = true;
  } else {
    current_player_ = other;
  }
}

void CrosswordDuelState::DoApplyAction(Action action) {
  SPIEL_CHECK_FALSE(IsTerminal());
  SPIEL_CHECK_TRUE(IsChanceNode());
  SPIEL_CHECK_GE(action, 0);
  SPIEL_CHECK_LT(action, 26);

  reward_[0] = 0;
  reward_[1] = 0;

  char letter = 'A' + static_cast<char>(action);
  auto it = std::find(bag_.begin(), bag_.end(), letter);
  SPIEL_CHECK_TRUE(it != bag_.end());
  bag_.erase(it);
  racks_[drawing_player_].push_back(letter);
  tiles_needed_--;

  serialized_history_.push_back(absl::StrCat(kChancePrefix, action));

  if (tiles_needed_ == 0) {
    FinishDraws();
  }
}

std::string CrosswordDuelState::ComputeCrossWord(int row, int col,
                                                  Direction main_direction,
                                                  char new_letter,
                                                  int* new_letter_index) const {
  Direction cross_dir = (main_direction == Direction::kAcross)
                            ? Direction::kDown
                            : Direction::kAcross;
  int dr = (cross_dir == Direction::kDown) ? 1 : 0;
  int dc = (cross_dir == Direction::kAcross) ? 1 : 0;

  int start_r = row;
  int start_c = col;
  while (board_.InBounds(start_r - dr, start_c - dc) &&
        CharAt(start_r - dr, start_c - dc) != kEmptyCell) {
    start_r -= dr;
    start_c -= dc;
  }

  std::string result;
  int r = start_r;
  int c = start_c;
  while (board_.InBounds(r, c) &&
        ((r == row && c == col) || CharAt(r, c) != kEmptyCell)) {
    if (r == row && c == col && new_letter_index != nullptr) {
      *new_letter_index = result.size();
    }
    result.push_back((r == row && c == col) ? new_letter : CharAt(r, c));
    r += dr;
    c += dc;
  }
  return result;
}

Status CrosswordDuelState::ValidateActionStruct(
    const ActionStruct& action_struct) const {
  if (IsChanceNode() || IsTerminal()) {
    return ErrorStatus(
        "Cannot apply an action struct at a chance or terminal node.");
  }
  const auto& a = down_cast<const CrosswordDuelActionStruct&>(action_struct);
  Player mover = current_player_;

  if (a.pass) {
    return OkStatus();
  }

  std::string word = absl::AsciiStrToUpper(a.word);
  if (word.size() < kMinWordLength) {
    return ErrorStatus(absl::StrCat("Word must be at least ", kMinWordLength,
                                    " letters: ", a.word));
  }

  std::string direction_str = absl::AsciiStrToLower(a.direction);
  Direction direction;
  if (direction_str == "across") {
    direction = Direction::kAcross;
  } else if (direction_str == "down") {
    direction = Direction::kDown;
  } else {
    return ErrorStatus(absl::StrCat("Unknown direction: ", a.direction));
  }

  int row = a.row;
  int col = a.col;
  int word_len = word.size();

  // The word must be maximal: no existing letter immediately before the
  // start or immediately after the end (in the placement direction).
  int before_r = row - (direction == Direction::kDown ? 1 : 0);
  int before_c = col - (direction == Direction::kAcross ? 1 : 0);
  if (board_.InBounds(before_r, before_c) &&
      CharAt(before_r, before_c) != kEmptyCell) {
    return ErrorStatus(
        "Word is not maximal: a letter immediately precedes the start.");
  }
  int end_r = row + (direction == Direction::kDown ? word_len : 0);
  int end_c = col + (direction == Direction::kAcross ? word_len : 0);
  if (board_.InBounds(end_r, end_c) && CharAt(end_r, end_c) != kEmptyCell) {
    return ErrorStatus(
        "Word is not maximal: a letter immediately follows the end.");
  }

  bool touches_existing = false;
  int num_new_letters = 0;
  absl::flat_hash_map<char, int> needed_from_rack;

  for (int i = 0; i < word_len; ++i) {
    int r = row + (direction == Direction::kDown ? i : 0);
    int c = col + (direction == Direction::kAcross ? i : 0);
    if (!board_.InBounds(r, c)) {
      return ErrorStatus(absl::StrCat("Word ", word, " goes out of bounds."));
    }
    char existing = CharAt(r, c);
    char letter = word[i];
    if (existing != kEmptyCell) {
      if (existing != letter) {
        return ErrorStatus(absl::StrCat(
            "Word ", word, " conflicts with the board at position ", i));
      }
      touches_existing = true;
    } else {
      num_new_letters++;
      needed_from_rack[letter]++;

      int pr1 = r + (direction == Direction::kAcross ? -1 : 0);
      int pc1 = c + (direction == Direction::kAcross ? 0 : -1);
      int pr2 = r + (direction == Direction::kAcross ? 1 : 0);
      int pc2 = c + (direction == Direction::kAcross ? 0 : 1);
      if ((board_.InBounds(pr1, pc1) && CharAt(pr1, pc1) != kEmptyCell) ||
          (board_.InBounds(pr2, pc2) && CharAt(pr2, pc2) != kEmptyCell)) {
        touches_existing = true;
      }
    }
  }

  if (num_new_letters == 0) {
    return ErrorStatus("Move must place at least one new tile from the rack.");
  }

  if (IsBoardEmpty()) {
    bool covers_start = false;
    for (int i = 0; i < word_len; ++i) {
      int r = row + (direction == Direction::kDown ? i : 0);
      int c = col + (direction == Direction::kAcross ? i : 0);
      if (board_.IsStartCell(r, c)) {
        covers_start = true;
        break;
      }
    }
    if (!covers_start) {
      return ErrorStatus("The first word must cover the start cell.");
    }
  } else if (!touches_existing) {
    return ErrorStatus("Word must connect to an existing tile on the board.");
  }

  absl::flat_hash_map<char, int> rack_counts;
  for (char c : racks_[mover]) {
    rack_counts[c]++;
  }
  for (const auto& [letter, needed] : needed_from_rack) {
    if (rack_counts[letter] < needed) {
      return ErrorStatus(absl::StrCat("Not enough '", std::string(1, letter),
                                      "' tiles in rack for word ", word));
    }
  }

  if (!parent_game_.word_set().empty()) {
    if (!parent_game_.word_set().contains(word)) {
      return ErrorStatus(absl::StrCat("Word not in word list: ", word));
    }
    for (int i = 0; i < word_len; ++i) {
      int r = row + (direction == Direction::kDown ? i : 0);
      int c = col + (direction == Direction::kAcross ? i : 0);
      if (CharAt(r, c) != kEmptyCell) {
        continue;  // Not a new letter; any cross word here already existed.
      }
      std::string cross = ComputeCrossWord(r, c, direction, word[i]);
      if (cross.size() > 1 && !parent_game_.word_set().contains(cross)) {
        return ErrorStatus(absl::StrCat("Cross word not in word list: ",
                                        cross));
      }
    }
  }

  return OkStatus();
}

int CrosswordDuelState::ComputeScoreAndApply(
    const CrosswordDuelActionStruct& a, Player mover) {
  std::string word = absl::AsciiStrToUpper(a.word);
  Direction direction = (absl::AsciiStrToLower(a.direction) == "across")
                            ? Direction::kAcross
                            : Direction::kDown;
  int row = a.row;
  int col = a.col;

  struct NewLetterPos {
    int row;
    int col;
    char letter;
  };
  std::vector<NewLetterPos> new_positions;

  int word_multiplier = 1;
  int main_word_score = 0;
  int cross_words_score = 0;
  int num_new_letters = 0;

  for (int i = 0; i < word.size(); ++i) {
    int r = row + (direction == Direction::kDown ? i : 0);
    int c = col + (direction == Direction::kAcross ? i : 0);
    char letter = word[i];
    int letter_value = board_.LetterValue(letter);
    bool is_new = (CharAt(r, c) == kEmptyCell);

    if (!is_new) {
      main_word_score += letter_value;
      continue;
    }

    num_new_letters++;
    BonusType bonus = board_.BonusAt(r, c);
    int letter_mult = 1;
    int cell_word_mult = 1;
    switch (bonus) {
      case BonusType::kDoubleLetter:
        letter_mult = 2;
        break;
      case BonusType::kTripleLetter:
        letter_mult = 3;
        break;
      case BonusType::kDoubleWord:
        word_multiplier *= 2;
        cell_word_mult = 2;
        break;
      case BonusType::kTripleWord:
        word_multiplier *= 3;
        cell_word_mult = 3;
        break;
      case BonusType::kNone:
        break;
    }
    main_word_score += letter_value * letter_mult;

    int new_letter_index = -1;
    std::string cross = ComputeCrossWord(r, c, direction, letter,
                                         &new_letter_index);
    if (cross.size() > 1) {
      int cross_score = 0;
      for (int k = 0; k < cross.size(); ++k) {
        cross_score += (k == new_letter_index)
                          ? letter_value * letter_mult
                          : board_.LetterValue(cross[k]);
      }
      cross_words_score += cross_score * cell_word_mult;
    }

    new_positions.push_back({r, c, letter});
  }

  main_word_score *= word_multiplier;
  int total = main_word_score + cross_words_score;
  if (num_new_letters == kRackSize) {
    total += kBingoBonus;
  }

  for (const NewLetterPos& np : new_positions) {
    board_state_[np.row * board_.num_cols() + np.col] = np.letter;
    auto it = std::find(racks_[mover].begin(), racks_[mover].end(),
                        np.letter);
    SPIEL_CHECK_TRUE(it != racks_[mover].end());
    racks_[mover].erase(it);
  }

  return total;
}

Status CrosswordDuelState::ApplyActionStruct(
    const ActionStruct& action_struct) {
  Player mover = current_player_;
  Status status = ValidateActionStruct(action_struct);
  if (!status.ok()) {
    return status;
  }
  const auto& a = down_cast<const CrosswordDuelActionStruct&>(action_struct);

  reward_[0] = 0;
  reward_[1] = 0;
  num_actions_++;
  serialized_history_.push_back(absl::StrCat(kActionPrefix, a.ToJson()));

  if (a.pass) {
    consecutive_passes_++;
    if (consecutive_passes_ >= kNumPlayers) {
      for (Player p = 0; p < kNumPlayers; ++p) {
        int penalty = RackValue(p);
        scores_[p] -= penalty;
        reward_[p] -= penalty;
      }
      game_over_ = true;
    } else {
      current_player_ = 1 - mover;
    }
    return OkStatus();
  }

  int score = ComputeScoreAndApply(a, mover);
  scores_[mover] += score;
  reward_[mover] = score;
  consecutive_passes_ = 0;

  int rack_size = racks_[mover].size();
  int to_draw = std::min<int>(kRackSize - rack_size,
                              static_cast<int>(bag_.size()));
  if (to_draw > 0) {
    drawing_player_ = mover;
    tiles_needed_ = to_draw;
    current_player_ = kChancePlayerId;
  } else {
    AdvanceOrEnd(mover);
  }
  return OkStatus();
}

std::string CrosswordDuelState::ActionToString(Player player,
                                                Action action) const {
  if (player == kChancePlayerId) {
    char letter = 'A' + static_cast<char>(action);
    return absl::StrCat("Draw tile: ", std::string(1, letter));
  }
  SpielFatalError("ActionToString unimplemented for non-chance node. "
                  "This game uses action structs only.");
}

std::unique_ptr<ActionStruct>
CrosswordDuelActionStructSampler::SampleActionStruct() {
  const auto* duel_state = down_cast<const CrosswordDuelState*>(state_);
  const CrosswordDuelGame* parent_game =
      down_cast<const CrosswordDuelGame*>(state_->GetGame().get());
  const auto& word_set = parent_game->word_set();
  const CrosswordDuelBoard& board = parent_game->board();
  Player mover = state_->CurrentPlayer();

  if (word_set.empty()) {
    std::cerr << "Word set is empty, cannot sample action. "
              << "Passing.\n";
    auto pass = std::make_unique<CrosswordDuelActionStruct>(
        0, 0, "across", "");
    pass->pass = true;
    return pass;
  }

  std::vector<std::pair<int, int>> anchors;
  bool board_empty = duel_state->IsBoardEmpty();
  if (board_empty) {
    for (int r = 0; r < board.num_rows(); ++r) {
      for (int c = 0; c < board.num_cols(); ++c) {
        if (board.IsStartCell(r, c)) anchors.push_back({r, c});
      }
    }
  } else {
    static const int dr[4] = {-1, 1, 0, 0};
    static const int dc[4] = {0, 0, -1, 1};
    for (int r = 0; r < board.num_rows(); ++r) {
      for (int c = 0; c < board.num_cols(); ++c) {
        if (duel_state->CharAt(r, c) != kEmptyCell) continue;
        for (int k = 0; k < 4; ++k) {
          int nr = r + dr[k];
          int nc = c + dc[k];
          if (board.InBounds(nr, nc) &&
              duel_state->CharAt(nr, nc) != kEmptyCell) {
            anchors.push_back({r, c});
            break;
          }
        }
      }
    }
  }

  // Pre-filter the word list by rough rack feasibility before validating
  // candidates -- with real dictionaries (10^5+ words) a plain uniform
  // sample almost never lands on a word the rack can actually spell, so the
  // sampler would degenerate into passing every turn. `excess` is the number
  // of a word's letters (by multiset) that are NOT covered by the rack; on
  // an empty board every letter must come from the rack (excess must be 0),
  // while on a non-empty board some letters may be supplied by tiles already
  // on the board, so a small slack is allowed.
  constexpr int kMaxExcessLetters = 6;
  const std::vector<char>& rack = duel_state->rack(mover);
  std::array<int, 26> rack_counts{};
  for (char c : rack) rack_counts[c - 'A']++;
  int max_excess = board_empty ? 0 : kMaxExcessLetters;

  std::vector<std::string> candidate_words;
  std::array<int, 26> word_counts{};
  for (const std::string& w : word_set) {
    if (w.size() < kMinWordLength || w.size() > board.num_rows()) continue;
    if (board_empty && w.size() > rack.size()) continue;

    word_counts.fill(0);
    for (char c : w) word_counts[c - 'A']++;
    int excess = 0;
    for (int i = 0; i < 26 && excess <= max_excess; ++i) {
      if (word_counts[i] > rack_counts[i]) {
        excess += word_counts[i] - rack_counts[i];
      }
    }
    if (excess <= max_excess) {
      candidate_words.push_back(w);
    }
  }

  auto make_pass = []() {
    auto pass = std::make_unique<CrosswordDuelActionStruct>(0, 0, "across",
                                                            "");
    pass->pass = true;
    return pass;
  };

  if (anchors.empty() || candidate_words.empty()) {
    return make_pass();
  }

  constexpr int kMaxAttempts = 3000;
  for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
    const auto& [anchor_row, anchor_col] =
        anchors[absl::Uniform<int>(rng_, 0, anchors.size())];
    Direction direction = absl::Uniform<int>(rng_, 0, 2) == 0
                              ? Direction::kAcross
                              : Direction::kDown;
    const std::string& word =
        candidate_words[absl::Uniform<int>(rng_, 0, candidate_words.size())];
    int offset = absl::Uniform<int>(rng_, 0, static_cast<int>(word.size()));
    int row =
        anchor_row - (direction == Direction::kDown ? offset : 0);
    int col =
        anchor_col - (direction == Direction::kAcross ? offset : 0);

    auto candidate = std::make_unique<CrosswordDuelActionStruct>(
        row, col, DirectionToString(direction), word);
    if (state_->ValidateActionStruct(*candidate).ok()) {
      return candidate;
    }
  }
  return make_pass();
}

std::vector<Action> CrosswordDuelState::LegalActions() const {
  if (IsTerminal()) {
    return {};
  } else if (IsChanceNode()) {
    return LegalChanceOutcomes();
  }
  SpielFatalError("LegalActions unimplemented for non-chance node. "
                  "This game uses action structs only.");
}

std::string CrosswordDuelState::ToString() const {
  std::string result;
  absl::StrAppend(&result, "Scores: P0=", scores_[0], " P1=", scores_[1],
                  "\n");
  absl::StrAppend(&result, "Bag remaining: ", bag_.size(), "\n");
  if (IsChanceNode()) {
    absl::StrAppend(&result, "Chance node, drawing tile for player ",
                    drawing_player_, " (", tiles_needed_, " left)\n");
  } else if (!IsTerminal()) {
    absl::StrAppend(&result, "Current player: ", current_player_, "\n");
  } else {
    absl::StrAppend(&result, "Game over.\n");
  }
  absl::StrAppend(&result, "\nBoard:\n");
  for (int r = 0; r < board_.num_rows(); ++r) {
    for (int c = 0; c < board_.num_cols(); ++c) {
      char ch = CharAt(r, c);
      result.push_back(ch == kEmptyCell ? '.' : ch);
      result.push_back(' ');
    }
    absl::StrAppend(&result, "\n");
  }
  absl::StrAppend(&result, "\nRack P0: ",
                  std::string(racks_[0].begin(), racks_[0].end()));
  absl::StrAppend(&result, "\nRack P1: ",
                  std::string(racks_[1].begin(), racks_[1].end()));
  absl::StrAppend(&result, "\n");
  return result;
}

bool CrosswordDuelState::IsTerminal() const {
  return game_over_ || num_actions_ >= kMaxGameLength;
}

std::vector<double> CrosswordDuelState::Rewards() const {
  return {static_cast<double>(reward_[0]), static_cast<double>(reward_[1])};
}

std::vector<double> CrosswordDuelState::Returns() const {
  return {static_cast<double>(scores_[0]), static_cast<double>(scores_[1])};
}

std::string CrosswordDuelState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  std::string result;
  absl::StrAppend(&result, "Scores: P0=", scores_[0], " P1=", scores_[1],
                  "\n");
  absl::StrAppend(&result, "Bag remaining: ", bag_.size(), "\n\nBoard:\n");
  for (int r = 0; r < board_.num_rows(); ++r) {
    for (int c = 0; c < board_.num_cols(); ++c) {
      char ch = CharAt(r, c);
      result.push_back(ch == kEmptyCell ? '.' : ch);
      result.push_back(' ');
    }
    absl::StrAppend(&result, "\n");
  }
  absl::StrAppend(&result, "\nYour rack: ",
                  std::string(racks_[player].begin(), racks_[player].end()));
  absl::StrAppend(&result, "\nOpponent rack size: ",
                  racks_[1 - player].size(), "\n");
  return result;
}

void CrosswordDuelState::ObservationTensor(Player player,
                                           absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
}

std::unique_ptr<State> CrosswordDuelState::Clone() const {
  return std::unique_ptr<State>(new CrosswordDuelState(*this));
}

std::string CrosswordDuelState::Serialize() const {
  std::string result;
  for (const std::string& line : serialized_history_) {
    absl::StrAppend(&result, line, "\n");
  }
  return result;
}

std::unique_ptr<State> CrosswordDuelGame::DeserializeState(
    const std::string& str) const {
  std::unique_ptr<State> state = NewInitialState();
  std::vector<std::string> lines = absl::StrSplit(str, '\n');
  for (const std::string& line : lines) {
    if (line.empty()) continue;
    if (absl::StartsWith(line, kChancePrefix)) {
      int outcome = kInvalidAction;
      SPIEL_CHECK_TRUE(
          absl::SimpleAtoi(line.substr(kChancePrefix.size()), &outcome));
      state->ApplyAction(outcome);
    } else if (absl::StartsWith(line, kActionPrefix)) {
      std::string json_str = std::string(line.substr(kActionPrefix.size()));
      CrosswordDuelActionStruct action_struct(json_str);
      state->ApplyActionStruct(action_struct);
    }
  }
  return state;
}

Status CrosswordDuelGame::ParseWordList() {
  if (word_list_file_.empty()) {
    for (const std::string& word : kDefaultWordList) {
      word_set_.insert(absl::AsciiStrToUpper(word));
    }
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

CrosswordDuelGame::CrosswordDuelGame(const GameParameters& params)
    : Game(kGameType, params),
      word_list_file_(ParameterValue<std::string>("word_list_file",
                                                   kDefaultWordListFile)) {
  StatusWithValue<CrosswordDuelBoard> result =
      ParseDuelBoardFromString(kDuelDefaultBoard);
  SPIEL_CHECK_TRUE(result.ok());
  board_ = result.value();
  ParseWordList();
}

CrosswordDuelGame::CrosswordDuelGame(const GameParameters& params,
                                     const std::string& board_string)
    : Game(kGameType, params),
      word_list_file_(ParameterValue<std::string>("word_list_file",
                                                   kDefaultWordListFile)) {
  StatusWithValue<CrosswordDuelBoard> result =
      ParseDuelBoardFromString(board_string);
  SPIEL_CHECK_TRUE(result.ok());
  board_ = result.value();
  ParseWordList();
}

std::shared_ptr<const Game> CreateCrosswordDuelGame(
    const std::string& board, const std::string& word_list_file) {
  GameParameters params;
  if (!word_list_file.empty()) {
    params["word_list_file"] = GameParameter(word_list_file);
  }
  return std::shared_ptr<const Game>(new CrosswordDuelGame(params, board));
}

}  // namespace crossword_duel
}  // namespace open_spiel
