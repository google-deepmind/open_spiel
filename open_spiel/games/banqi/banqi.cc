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

#include "open_spiel/games/banqi/banqi.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace banqi {
namespace {

const GameType kGameType{
    /*short_name=*/"banqi",
    /*long_name=*/"Banqi",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    // Note: Banqi appears to be a game of imperfect information due to the
    // hidden pieces. However, unlike e.g. Stratego, the identity of the
    // pieces is determined randomly, not by one of the players. So there is
    // never any private information known to one player but not the other. So,
    // game-theoretically it's in the same class as Backgammon: a perfect
    // information game with a chance component.
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    {
        {"max_no_progress", GameParameter(kDefaultMaxNoProgress)},
    }};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new BanqiGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

// Display characters for pieces.
// Red: G=General A=Advisor E=Elephant R=Chariot(Rook) H=Horse C=Cannon
// S=Soldier Black: lowercase equivalents.
const char kRedChars[] = "GAERHCS";
const char kBlackChars[] = "gaerhcs";

std::string PieceToString(int piece) {
  if (piece == kEmpty) return ".";
  if (piece == kFaceDown) return "?";
  int color = PieceColor(piece);
  int type = PieceType(piece);
  return std::string(1, color == kRed ? kRedChars[type] : kBlackChars[type]);
}

int SquareRow(int sq) { return sq / kNumCols; }
int SquareCol(int sq) { return sq % kNumCols; }
int SquareIndex(int row, int col) { return row * kNumCols + col; }

const int kDirs[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

}  // namespace

// --- BanqiState ---

BanqiState::BanqiState(std::shared_ptr<const Game> game, int max_no_progress)
    : State(game), max_no_progress_(max_no_progress) {
  board_.fill(kFaceDown);
  for (int i = 0; i < kNumDistinctPieces; ++i) {
    remaining_pieces_[i] = kInitialPieceCounts[i];
  }
  player_color_.fill(kNoColor);
}

Player BanqiState::CurrentPlayer() const {
  if (IsTerminal()) return kTerminalPlayerId;
  if (pending_flip_square_ >= 0) return kChancePlayerId;
  return current_player_;
}

bool BanqiState::CanCapture(int attacker, int defender) const {
  SPIEL_CHECK_GE(attacker, 0);
  SPIEL_CHECK_GE(defender, 0);
  int att_type = PieceType(attacker);
  int def_type = PieceType(defender);

  if (PieceColor(attacker) == PieceColor(defender)) return false;

  // Cannons cannot capture by adjacency (they must jump).
  if (att_type == kCannon) return false;

  // Soldier cannot capture cannon.
  if (att_type == kSoldier && def_type == kCannon) return false;

  // General cannot capture soldier (circular weakness).
  if (att_type == kGeneral && def_type == kSoldier) return false;

  // Soldier can capture general (circular strength).
  if (att_type == kSoldier && def_type == kGeneral) return true;

  // Standard hierarchy: lower type index = higher rank.
  return att_type <= def_type;
}

void BanqiState::AddCannonCaptures(int source,
                                   std::vector<Action>* moves) const {
  int cannon_color = PieceColor(board_[source]);
  int sr = SquareRow(source), sc = SquareCol(source);

  for (const auto& dir : kDirs) {
    int screens = 0;
    int r = sr + dir[0], c = sc + dir[1];
    while (r >= 0 && r < kNumRows && c >= 0 && c < kNumCols) {
      int sq = SquareIndex(r, c);
      if (board_[sq] != kEmpty) {
        if (screens == 0) {
          screens = 1;
        } else {
          // Second non-empty square: capture if opponent's face-up piece.
          if (board_[sq] >= 0 && PieceColor(board_[sq]) != cannon_color) {
            moves->push_back(kNumFlipActions + source * kNumSquares + sq);
          }
          break;
        }
      }
      r += dir[0];
      c += dir[1];
    }
  }
}

bool BanqiState::HasLegalMovesForPlayer(Player player) const {
  if (num_face_down_ > 0) return true;
  int color = player_color_[player];
  if (color == kNoColor) return false;

  for (int sq = 0; sq < kNumSquares; ++sq) {
    int piece = board_[sq];
    if (piece < 0 || PieceColor(piece) != color) continue;

    int r = SquareRow(sq), c = SquareCol(sq);
    for (const auto& dir : kDirs) {
      int nr = r + dir[0], nc = c + dir[1];
      if (nr < 0 || nr >= kNumRows || nc < 0 || nc >= kNumCols) continue;
      int dest = SquareIndex(nr, nc);
      if (board_[dest] == kEmpty) return true;
      if (PieceType(piece) != kCannon && board_[dest] >= 0 &&
          CanCapture(piece, board_[dest])) {
        return true;
      }
    }

    // Check cannon jump captures.
    if (PieceType(piece) == kCannon) {
      for (const auto& dir : kDirs) {
        int screens = 0;
        int cr = r + dir[0], cc = c + dir[1];
        while (cr >= 0 && cr < kNumRows && cc >= 0 && cc < kNumCols) {
          int csq = SquareIndex(cr, cc);
          if (board_[csq] != kEmpty) {
            if (screens == 0) {
              screens = 1;
            } else {
              if (board_[csq] >= 0 && PieceColor(board_[csq]) != color) {
                return true;
              }
              break;
            }
          }
          cr += dir[0];
          cc += dir[1];
        }
      }
    }
  }
  return false;
}

std::vector<Action> BanqiState::LegalActions() const {
  if (IsTerminal()) return {};

  if (CurrentPlayer() == kChancePlayerId) {
    return LegalChanceOutcomes();
  }

  std::vector<Action> actions;
  int my_color = player_color_[current_player_];

  // Flip actions: any face-down square.
  for (int sq = 0; sq < kNumSquares; ++sq) {
    if (board_[sq] == kFaceDown) {
      actions.push_back(sq);
    }
  }

  // Move and capture for the current player's face-up pieces.
  if (my_color != kNoColor) {
    for (int sq = 0; sq < kNumSquares; ++sq) {
      int piece = board_[sq];
      if (piece < 0 || PieceColor(piece) != my_color) continue;

      int r = SquareRow(sq), c = SquareCol(sq);

      // One-square orthogonal moves (all pieces including cannon).
      for (const auto& dir : kDirs) {
        int nr = r + dir[0], nc = c + dir[1];
        if (nr < 0 || nr >= kNumRows || nc < 0 || nc >= kNumCols) continue;
        int dest = SquareIndex(nr, nc);
        if (board_[dest] == kEmpty) {
          actions.push_back(kNumFlipActions + sq * kNumSquares + dest);
        } else if (PieceType(piece) != kCannon && board_[dest] >= 0 &&
                   CanCapture(piece, board_[dest])) {
          actions.push_back(kNumFlipActions + sq * kNumSquares + dest);
        }
      }

      // Cannon jump captures.
      if (PieceType(piece) == kCannon) {
        AddCannonCaptures(sq, &actions);
      }
    }
  }

  std::sort(actions.begin(), actions.end());
  return actions;
}

std::vector<std::pair<Action, double>> BanqiState::ChanceOutcomes() const {
  SPIEL_CHECK_GE(pending_flip_square_, 0);
  std::vector<std::pair<Action, double>> outcomes;
  for (int i = 0; i < kNumDistinctPieces; ++i) {
    if (remaining_pieces_[i] > 0) {
      outcomes.push_back(
          {i, static_cast<double>(remaining_pieces_[i]) / num_face_down_});
    }
  }
  return outcomes;
}

void BanqiState::DoApplyAction(Action action) {
  if (CurrentPlayer() == kChancePlayerId) {
    // Resolve a pending flip.
    SPIEL_CHECK_GE(pending_flip_square_, 0);
    SPIEL_CHECK_GE(action, 0);
    SPIEL_CHECK_LT(action, kNumDistinctPieces);
    SPIEL_CHECK_GT(remaining_pieces_[action], 0);

    undo_stack_.push_back(
        {kFaceDown, no_progress_count_, player_color_, is_draw_, outcome_});

    board_[pending_flip_square_] = action;
    remaining_pieces_[action]--;
    num_face_down_--;

    // First flip determines player colors.
    if (player_color_[0] == kNoColor) {
      int revealed_color = PieceColor(action);
      player_color_[player_before_chance_] = revealed_color;
      player_color_[1 - player_before_chance_] = 1 - revealed_color;
    }

    no_progress_count_ = 0;
    current_player_ = 1 - player_before_chance_;
    pending_flip_square_ = -1;
    player_before_chance_ = kInvalidPlayer;
    total_moves_++;
    CheckTerminal();
  } else if (action < kNumFlipActions) {
    // Flip initiation: enter chance node.
    SPIEL_CHECK_EQ(board_[action], kFaceDown);
    undo_stack_.push_back(
        {kEmpty, no_progress_count_, player_color_, is_draw_, outcome_});
    pending_flip_square_ = action;
    player_before_chance_ = current_player_;
  } else {
    // Move or capture.
    int encoded = action - kNumFlipActions;
    int source = encoded / kNumSquares;
    int dest = encoded % kNumSquares;
    SPIEL_CHECK_GE(board_[source], 0);

    int captured = board_[dest];
    undo_stack_.push_back(
        {captured, no_progress_count_, player_color_, is_draw_, outcome_});

    board_[dest] = board_[source];
    board_[source] = kEmpty;

    if (captured >= 0) {
      no_progress_count_ = 0;
    } else {
      no_progress_count_++;
    }

    current_player_ = 1 - current_player_;
    total_moves_++;
    CheckTerminal();
  }
}

void BanqiState::CheckTerminal() {
  // Draw by no progress.
  if (no_progress_count_ >= max_no_progress_) {
    is_draw_ = true;
    return;
  }

  // Check if current player has any legal moves.
  if (player_color_[0] != kNoColor &&
      !HasLegalMovesForPlayer(current_player_)) {
    outcome_ = 1 - current_player_;
  }
}

bool BanqiState::IsTerminal() const {
  return is_draw_ || outcome_ != kInvalidPlayer;
}

std::vector<double> BanqiState::Returns() const {
  if (is_draw_) return {0.0, 0.0};
  if (outcome_ == Player{0}) return {1.0, -1.0};
  if (outcome_ == Player{1}) return {-1.0, 1.0};
  return {0.0, 0.0};
}

std::string BanqiState::ActionToString(Player player, Action action_id) const {
  if (player == kChancePlayerId) {
    return absl::StrCat("Reveal:", PieceToString(action_id));
  }
  if (action_id < kNumFlipActions) {
    return absl::StrCat("Flip(", SquareRow(action_id), ",",
                        SquareCol(action_id), ")");
  }
  int encoded = action_id - kNumFlipActions;
  int source = encoded / kNumSquares;
  int dest = encoded % kNumSquares;
  std::string src_str =
      absl::StrCat(PieceToString(board_[source]), "(", SquareRow(source), ",",
                   SquareCol(source), ")");
  if (board_[dest] >= 0) {
    return absl::StrCat(src_str, "x(", SquareRow(dest), ",", SquareCol(dest),
                        ")");
  }
  return absl::StrCat(src_str, "-(", SquareRow(dest), ",", SquareCol(dest),
                      ")");
}

std::string BanqiState::ToString() const {
  std::string result;
  for (int r = 0; r < kNumRows; ++r) {
    for (int c = 0; c < kNumCols; ++c) {
      if (c > 0) absl::StrAppend(&result, " ");
      absl::StrAppend(&result, PieceToString(board_[SquareIndex(r, c)]));
    }
    absl::StrAppend(&result, "\n");
  }
  absl::StrAppend(&result, "Player: ", current_player_);
  if (player_color_[0] != kNoColor) {
    absl::StrAppend(&result,
                    " (P0=", player_color_[0] == kRed ? "Red" : "Black", ")");
  }
  return result;
}

std::string BanqiState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void BanqiState::ObservationTensor(Player player,
                                   absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  SPIEL_CHECK_EQ(static_cast<int>(values.size()), kObsTensorSize);
  std::fill(values.begin(), values.end(), 0.0f);

  int offset = 0;
  for (int sq = 0; sq < kNumSquares; ++sq) {
    int piece = board_[sq];
    int channel;
    if (piece == kEmpty) {
      channel = kNumDistinctPieces;  // 14
    } else if (piece == kFaceDown) {
      channel = kNumDistinctPieces + 1;  // 15
    } else {
      channel = piece;  // 0-13
    }
    values[offset + channel * kNumSquares + sq] = 1.0f;
  }
  offset += kNumObsChannels * kNumSquares;

  values[offset + current_player_] = 1.0f;
  offset += kNumPlayers;

  SPIEL_CHECK_EQ(offset, kObsTensorSize);
}

void BanqiState::UndoAction(Player player, Action action) {
  SPIEL_CHECK_FALSE(undo_stack_.empty());
  UndoInfo info = undo_stack_.back();
  undo_stack_.pop_back();

  // Restore common state.
  no_progress_count_ = info.no_progress_before;
  player_color_ = info.player_color_before;
  is_draw_ = info.was_draw;
  outcome_ = info.outcome_before;

  if (player == kChancePlayerId) {
    // Undo chance resolution: restore the flipped square to face-down.
    SPIEL_CHECK_GE(history_.size(), 2);
    int flip_sq = history_[history_.size() - 2].action;
    board_[flip_sq] = kFaceDown;
    remaining_pieces_[action]++;
    num_face_down_++;
    pending_flip_square_ = flip_sq;
    // Restore current_player_ to who initiated the flip.
    // After chance, current_player_ was set to 1 - player_before_chance_.
    // So player_before_chance_ = 1 - current_player_.
    player_before_chance_ = 1 - current_player_;
    current_player_ = player_before_chance_;
    total_moves_--;
  } else if (action < kNumFlipActions) {
    // Undo flip initiation.
    pending_flip_square_ = -1;
    player_before_chance_ = kInvalidPlayer;
    current_player_ = player;
    // total_moves_ was not incremented during flip initiation.
  } else {
    // Undo move/capture.
    int encoded = action - kNumFlipActions;
    int source = encoded / kNumSquares;
    int dest = encoded % kNumSquares;
    board_[source] = board_[dest];
    board_[dest] = info.captured_piece;
    current_player_ = player;
    total_moves_--;
  }

  history_.pop_back();
  --move_number_;
}

std::unique_ptr<State> BanqiState::Clone() const {
  return std::unique_ptr<State>(new BanqiState(*this));
}

// --- BanqiGame ---

BanqiGame::BanqiGame(const GameParameters& params)
    : Game(kGameType, params),
      max_no_progress_(ParameterValue<int>("max_no_progress")) {}

int BanqiGame::MaxGameLength() const {
  // 32 flips x 2 actions each (flip + chance) = 64
  // Up to 32 captures, each allowing max_no_progress moves before the next.
  return 2 * kNumSquares + kNumSquares * max_no_progress_;
}

}  // namespace banqi
}  // namespace open_spiel
