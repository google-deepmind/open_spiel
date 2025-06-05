// Copyright 2025 DeepMind Technologies Limited
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

#include "open_spiel/games/azul/azul.h"

#include <algorithm>
#include <memory>
#include <random>
#include <vector>
#include <ctime>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace azul {
namespace {

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"azul",
    /*long_name=*/"Azul",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/kMaxNumPlayers,
    /*min_num_players=*/kMinNumPlayers,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{{"num_players", GameParameter(kDefaultNumPlayers)},
                                {"rng_seed", GameParameter(kDefaultSeed)}}
};

std::shared_ptr<const Game> GameFactory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new AzulGame(params));
}

REGISTER_SPIEL_GAME(kGameType, GameFactory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

// Floor line penalty points
const std::vector<int> kFloorPenalties = {-1, -1, -2, -2, -2, -3, -3};

}  // namespace

std::string TileColorToString(TileColor color) {
  switch (color) {
    case TileColor::kBlue: return "B";
    case TileColor::kYellow: return "Y";  
    case TileColor::kRed: return "R";
    case TileColor::kBlack: return "K";
    case TileColor::kWhite: return "W";
    case TileColor::kFirstPlayer: return "F";
    default: return "?";
  }
}

TileColor StringToTileColor(const std::string& str) {
  if (str == "B" || str == "Blue") return TileColor::kBlue;
  if (str == "Y" || str == "Yellow") return TileColor::kYellow;
  if (str == "R" || str == "Red") return TileColor::kRed;
  if (str == "K" || str == "Black") return TileColor::kBlack;
  if (str == "W" || str == "White") return TileColor::kWhite;
  if (str == "F" || str == "FirstPlayer") return TileColor::kFirstPlayer;
  return TileColor::kBlue;  // Default to a valid color instead of kEmpty
}

AzulState::AzulState(std::shared_ptr<const Game> game) 
    : State(game), 
      num_players_(kDefaultNumPlayers),
      current_player_(0),
      factories_(GetNumFactories()),
      player_boards_(num_players_),
      first_player_tile_available_(true),
      first_player_next_round_(0),
      game_ended_(false),
      round_number_(1),
      needs_bag_shuffle_(false) {
  
  // Initialize bag with tiles in a deterministic but balanced order
  for (int i = 0; i < kTotalTilesPerColor; ++i) {
    for (int color = 0; color < kNumTileColors; ++color) {
      bag_.push_back(static_cast<TileColor>(color));
    }
  }
  // Reset RNG to original seed to ensure consistent initial state
  const AzulGame* azul_game = static_cast<const AzulGame*>(GetGame().get());
  azul_game->GetRNG().seed(azul_game->GetOriginalSeed());
  // Shuffle bag deterministically using game's RNG 
  std::shuffle(bag_.begin(), bag_.end(), azul_game->GetRNG());
  SetupNewRound();
}

AzulState::AzulState(std::shared_ptr<const Game> game, int num_players)
    : State(game),
      num_players_(num_players),
      current_player_(0),
      factories_(GetNumFactories()),
      player_boards_(num_players),
      first_player_tile_available_(true),
      first_player_next_round_(0),
      game_ended_(false),
      round_number_(1),
      needs_bag_shuffle_(false) {
  
  
  // Initialize bag with tiles in a deterministic but balanced order
  for (int i = 0; i < kTotalTilesPerColor; ++i) {
    for (int color = 0; color < kNumTileColors; ++color) {
      bag_.push_back(static_cast<TileColor>(color));
    }
  }
    
  // Reset RNG to original seed to ensure consistent initial state
  const AzulGame* azul_game = static_cast<const AzulGame*>(GetGame().get());
  azul_game->GetRNG().seed(azul_game->GetOriginalSeed());
  // Shuffle bag deterministically using game's RNG 
  std::shuffle(bag_.begin(), bag_.end(), azul_game->GetRNG());

  SetupNewRound();
}

Player AzulState::CurrentPlayer() const {
  if (IsTerminal()) return kTerminalPlayerId;
  if (needs_bag_shuffle_) return kChancePlayerId;
  return current_player_;
}

void AzulState::SetupNewRound() {
  // Assert that factories and center pile are already empty
  // (they should be empty after a properly completed round)
  for (const auto& factory : factories_) {
    for (int color = 0; color < kNumTileColors; ++color) {
      SPIEL_CHECK_EQ(factory.tiles[color], 0);
    }
  }
  for (int color = 0; color < kNumTileColors; ++color) {
    SPIEL_CHECK_EQ(center_pile_.tiles[color], 0);
  }
  
  first_player_tile_available_ = true;
  
  FillFactories();
}

void AzulState::FillFactories() {
  // Fill each factory with 4 tiles from the bag
  for (auto& factory : factories_) {
    for (int i = 0; i < kTilesPerFactory && !bag_.empty(); ++i) {
      TileColor color = bag_.back();
      bag_.pop_back();
      factory.tiles[static_cast<int>(color)]++;
    }
  }
  
  // If bag is empty but we need more tiles, set up chance node for refilling
  if (bag_.empty() && !discard_pile_.empty()) {
    bool need_more_tiles = false;
    for (auto& factory : factories_) {
      if (factory.TotalTiles() < kTilesPerFactory) {
        need_more_tiles = true;
        break;
      }
    }
    
    if (need_more_tiles) {
      needs_bag_shuffle_ = true;
      return;  // Wait for chance outcome to shuffle and continue
    }
  }
}

std::vector<Action> AzulState::LegalActions() const {
  if (IsTerminal()) return {};
  
  // Handle chance node for bag shuffling
  if (needs_bag_shuffle_) {
    return {0};  // Single chance outcome for shuffling
  }
  
  std::vector<Action> actions;
  
  // Actions from factories
  for (int factory_id = 0; factory_id < static_cast<int>(factories_.size()); ++factory_id) {
    const azul::Factory& factory = factories_[factory_id];
    if (factory.IsEmpty()) continue;
    
    for (int color = 0; color < kNumTileColors; ++color) {
      if (factory.tiles[color] == 0) continue;
      
      TileColor tile_color = static_cast<TileColor>(color);
      
      // Can go to pattern lines
      for (int line = 0; line < kNumPatternLines; ++line) {
        const auto& pattern_line = player_boards_[current_player_].pattern_lines[line];
        
        // Check if pattern line can accept this color
        if (pattern_line.CanAccept(tile_color, line)) {
          // Also check that corresponding wall position is empty
          int wall_col = -1;
          for (int col = 0; col < kWallSize; ++col) {
            if (kWallPattern[line][col] == tile_color) {
              wall_col = col;
              break;
            }
          }
          
          if (wall_col != -1 && !player_boards_[current_player_].wall[line][wall_col]) {
            actions.push_back(EncodeAction(false, factory_id, tile_color, line));
          }
        }
      }
      
      // Can always go to floor
      actions.push_back(EncodeAction(false, factory_id, tile_color, -1));
    }
  }
  
  // Actions from center pile
  if (!center_pile_.IsEmpty()) {
    for (int color = 0; color < kNumTileColors; ++color) {
      if (center_pile_.tiles[color] == 0) continue;
      
      TileColor tile_color = static_cast<TileColor>(color);
      
      // Can go to pattern lines
      for (int line = 0; line < kNumPatternLines; ++line) {
        const auto& pattern_line = player_boards_[current_player_].pattern_lines[line];
        
        if (pattern_line.CanAccept(tile_color, line)) {
          int wall_col = -1;
          for (int col = 0; col < kWallSize; ++col) {
            if (kWallPattern[line][col] == tile_color) {
              wall_col = col;
              break;
            }
          }
          
          if (wall_col != -1 && !player_boards_[current_player_].wall[line][wall_col]) {
            actions.push_back(EncodeAction(true, -1, tile_color, line));
          }
        }
      }
      
      // Can always go to floor
      actions.push_back(EncodeAction(true, -1, tile_color, -1));
    }
  }
  
  // Sort actions to ensure they are in ascending order
  std::sort(actions.begin(), actions.end());
  
  return actions;
}

void AzulState::DoApplyAction(Action action) {
  // Handle chance node for bag shuffling
  if (needs_bag_shuffle_) {
    // Refill bag from discard pile and shuffle
    bag_ = discard_pile_;
    discard_pile_.clear();
    // Reset RNG to original seed to ensure consistent initial state
    const AzulGame* azul_game = static_cast<const AzulGame*>(GetGame().get());
    azul_game->GetRNG().seed(azul_game->GetOriginalSeed());
    // Shuffle bag deterministically using game's RNG 
    std::shuffle(bag_.begin(), bag_.end(), azul_game->GetRNG());
    needs_bag_shuffle_ = false;
    
    // Continue filling factories
    for (auto& factory : factories_) {
      while (factory.TotalTiles() < kTilesPerFactory && !bag_.empty()) {
        TileColor color = bag_.back();
        bag_.pop_back();
        factory.tiles[static_cast<int>(color)]++;
      }
    }
    return;
  }
  
  DecodedAction decoded = DecodeAction(action);
  
  // Take tiles from source
  std::vector<TileColor> taken_tiles;
  if (decoded.from_center) {
    // Take all tiles of this color from center
    int count = center_pile_.tiles[static_cast<int>(decoded.color)];
    for (int i = 0; i < count; ++i) {
      taken_tiles.push_back(decoded.color);
    }
    center_pile_.tiles[static_cast<int>(decoded.color)] = 0;
    
    // Take first player tile if available
    if (first_player_tile_available_) {
      player_boards_[current_player_].floor_line.push_back(TileColor::kFirstPlayer);
      first_player_tile_available_ = false;
      first_player_next_round_ = current_player_;
    }
  } else {
    // Take tiles from factory
    Factory& factory = factories_[decoded.factory_id];
    int count = factory.tiles[static_cast<int>(decoded.color)];
    for (int i = 0; i < count; ++i) {
      taken_tiles.push_back(decoded.color);
    }
    factory.tiles[static_cast<int>(decoded.color)] = 0;
    
    // Move remaining tiles to center
    for (int color = 0; color < kNumTileColors; ++color) {
      center_pile_.tiles[color] += factory.tiles[color];
      factory.tiles[color] = 0;
    }
  }
  
  // Place tiles
  PlayerBoard& board = player_boards_[current_player_];
  if (decoded.destination == -1) {
    // Place all tiles on floor line
    for (TileColor tile : taken_tiles) {
      board.floor_line.push_back(tile);
    }
  } else {
    // Place tiles in pattern line
    int line = decoded.destination;
    auto& pattern_line = board.pattern_lines[line];
    
    // Set color if line is empty
    if (pattern_line.IsEmpty()) {
      pattern_line.color = decoded.color;
    }
    
    // Place as many as possible in pattern line
    int max_capacity = line + 1;
    for (TileColor tile : taken_tiles) {
      if (pattern_line.count < max_capacity) {
        pattern_line.count++;
      } else {
        // Overflow goes to floor line
        board.floor_line.push_back(tile);
      }
    }
  }
  
  // Check if round is over (all factories and center empty)
  bool round_over = center_pile_.IsEmpty();
  for (const auto& factory : factories_) {
    if (!factory.IsEmpty()) {
      round_over = false;
      break;
    }
  }
  
  if (round_over) {
    EndRoundScoring();
    
    // Check if game is over
    bool game_over = false;
    for (int player = 0; player < num_players_; ++player) {
      if (IsWallComplete(player)) {
        game_over = true;
        break;
      }
    }
    
    if (game_over) {
      game_ended_ = true;
    } else {
      // Setup next round
      current_player_ = first_player_next_round_;
      round_number_++;
      SetupNewRound();
      return;
    }
  } else {
    // Next player
    current_player_ = (current_player_ + 1) % num_players_;
  }
}

void AzulState::EndRoundScoring() {
  for (int player = 0; player < num_players_; ++player) {
    PlayerBoard& board = player_boards_[player];
    
    // Score pattern lines that are complete
    for (int line = 0; line < kNumPatternLines; ++line) {
      auto& pattern_line = board.pattern_lines[line];
      int tiles_needed = line + 1;
      
      if (pattern_line.count == tiles_needed && !pattern_line.IsEmpty()) {
        // Place one tile on wall
        int wall_col = -1;
        for (int col = 0; col < kWallSize; ++col) {
          if (kWallPattern[line][col] == pattern_line.color) {
            wall_col = col;
            break;
          }
        }
        
        if (wall_col != -1) {
          board.wall[line][wall_col] = true;
          
          // Calculate score for this tile
          int score = 1;
          
          // Horizontal adjacent tiles
          int left = wall_col - 1;
          int right = wall_col + 1;
          while (left >= 0 && board.wall[line][left]) {
            score++;
            left--;
          }
          while (right < kWallSize && board.wall[line][right]) {
            score++;
            right++;
          }
          
          // Vertical adjacent tiles (only if horizontal score > 1)
          int vertical_score = 1;
          int up = line - 1;
          int down = line + 1;
          while (up >= 0 && board.wall[up][wall_col]) {
            vertical_score++;
            up--;
          }
          while (down < kWallSize && board.wall[down][wall_col]) {
            vertical_score++;
            down++;
          }
          
          if (vertical_score > 1) {
            score += vertical_score - 1;
          }
          
          board.score += score;
        }
        
        // Discard excess tiles (all except the one placed on wall)
        for (int i = 0; i < pattern_line.count - 1; ++i) {
          discard_pile_.push_back(pattern_line.color);
        }
        
        // Clear pattern line (color becomes irrelevant when count is 0)
        pattern_line.count = 0;
      }
    }
    
    // Apply floor line penalties
    int penalty = 0;
    for (size_t i = 0; i < board.floor_line.size() && i < kFloorPenalties.size(); ++i) {
      penalty += kFloorPenalties[i];
    }
    board.score = std::max(0, board.score + penalty);
    
    // Discard floor line tiles (except first player tile)
    for (TileColor tile : board.floor_line) {
      if (tile != TileColor::kFirstPlayer) {
        discard_pile_.push_back(tile);
      }
    }
    board.floor_line.clear();
  }
}

bool AzulState::IsWallComplete(Player player) const {
  const auto& wall = player_boards_[player].wall;
  for (int row = 0; row < kWallSize; ++row) {
    bool row_complete = true;
    for (int col = 0; col < kWallSize; ++col) {
      if (!wall[row][col]) {
        row_complete = false;
        break;
      }
    }
    if (row_complete) return true;
  }
  return false;
}

int AzulState::CalculateScore(Player player) const {
  const PlayerBoard& board = player_boards_[player];
  int final_score = board.score;
  
  // Bonus points for complete rows
  for (int row = 0; row < kWallSize; ++row) {
    bool complete = true;
    for (int col = 0; col < kWallSize; ++col) {
      if (!board.wall[row][col]) {
        complete = false;
        break;
      }
    }
    if (complete) final_score += 2;
  }
  
  // Bonus points for complete columns
  for (int col = 0; col < kWallSize; ++col) {
    bool complete = true;
    for (int row = 0; row < kWallSize; ++row) {
      if (!board.wall[row][col]) {
        complete = false;
        break;
      }
    }
    if (complete) final_score += 7;
  }
  
  // Bonus points for complete colors
  for (int color = 0; color < kNumTileColors; ++color) {
    bool complete = true;
    for (int row = 0; row < kWallSize; ++row) {
      bool found = false;
      for (int col = 0; col < kWallSize; ++col) {
        if (kWallPattern[row][col] == static_cast<TileColor>(color) && 
            board.wall[row][col]) {
          found = true;
          break;
        }
      }
      if (!found) {
        complete = false;
        break;
      }
    }
    if (complete) final_score += 10;
  }
  
  return final_score;
}

AzulState::DecodedAction AzulState::DecodeAction(Action action) const {
  DecodedAction decoded;
  
  // Linear encoding scheme to match kNumDistinctActions calculation
  // Actions are ordered as:
  // 1. Factory to pattern line: factory_id * kNumTileColors * kNumPatternLines + color * kNumPatternLines + line
  // 2. Center to pattern line: (max_factories * kNumTileColors * kNumPatternLines) + color * kNumPatternLines + line  
  // 3. Factory to floor: (max_factories * kNumTileColors * kNumPatternLines) + (kNumTileColors * kNumPatternLines) + factory_id * kNumTileColors + color
  // 4. Center to floor: (max_factories * kNumTileColors * kNumPatternLines) + (kNumTileColors * kNumPatternLines) + (max_factories * kNumTileColors) + color
  
  int max_factories = GetNumFactories();
  int factory_pattern_actions = max_factories * kNumTileColors * kNumPatternLines;
  int center_pattern_actions = kNumTileColors * kNumPatternLines;
  int factory_floor_actions = max_factories * kNumTileColors;
  
  if (action < factory_pattern_actions) {
    // Factory to pattern line
    decoded.from_center = false;
    decoded.factory_id = action / (kNumTileColors * kNumPatternLines);
    int remainder = action % (kNumTileColors * kNumPatternLines);
    decoded.color = static_cast<TileColor>(remainder / kNumPatternLines);
    decoded.destination = remainder % kNumPatternLines;
  } else if (action < factory_pattern_actions + center_pattern_actions) {
    // Center to pattern line
    decoded.from_center = true;
    decoded.factory_id = -1;
    int remainder = action - factory_pattern_actions;
    decoded.color = static_cast<TileColor>(remainder / kNumPatternLines);
    decoded.destination = remainder % kNumPatternLines;
  } else if (action < factory_pattern_actions + center_pattern_actions + factory_floor_actions) {
    // Factory to floor
    decoded.from_center = false;
    int remainder = action - factory_pattern_actions - center_pattern_actions;
    decoded.factory_id = remainder / kNumTileColors;
    decoded.color = static_cast<TileColor>(remainder % kNumTileColors);
    decoded.destination = -1;
  } else {
    // Center to floor
    decoded.from_center = true;
    decoded.factory_id = -1;
    int remainder = action - factory_pattern_actions - center_pattern_actions - factory_floor_actions;
    decoded.color = static_cast<TileColor>(remainder);
    decoded.destination = -1;
  }
  
  return decoded;
}

Action AzulState::EncodeAction(bool from_center, int factory_id, TileColor color, int destination) const {
  int max_factories = GetNumFactories();
  int factory_pattern_actions = max_factories * kNumTileColors * kNumPatternLines;
  int center_pattern_actions = kNumTileColors * kNumPatternLines;
  int factory_floor_actions = max_factories * kNumTileColors;
  
  if (!from_center && destination >= 0) {
    // Factory to pattern line
    return factory_id * kNumTileColors * kNumPatternLines + 
           static_cast<int>(color) * kNumPatternLines + destination;
  } else if (from_center && destination >= 0) {
    // Center to pattern line
    return factory_pattern_actions + 
           static_cast<int>(color) * kNumPatternLines + destination;
  } else if (!from_center && destination == -1) {
    // Factory to floor
    return factory_pattern_actions + center_pattern_actions + 
           factory_id * kNumTileColors + static_cast<int>(color);
  } else {
    // Center to floor
    return factory_pattern_actions + center_pattern_actions + factory_floor_actions + 
           static_cast<int>(color);
  }
}

std::string AzulState::ActionToString(Player player, Action action) const {
  DecodedAction decoded = DecodeAction(action);
  std::string source = decoded.from_center ? "Center" : 
                      absl::StrCat("Factory", decoded.factory_id);
  std::string destination = decoded.destination == -1 ? "Floor" :
                           absl::StrCat("Line", decoded.destination);
  return absl::StrCat("Take ", TileColorToString(decoded.color), 
                     " from ", source, " to ", destination);
}

std::string AzulState::ToString() const {
  std::string str = absl::StrCat("Round: ", round_number_, "\n");
  str += absl::StrCat("Current Player: ", current_player_, "\n");
  
  // Show factories
  for (size_t i = 0; i < factories_.size(); ++i) {
    str += absl::StrCat("Factory ", i, ": ");
    for (int color = 0; color < kNumTileColors; ++color) {
      int count = factories_[i].tiles[color];
      for (int j = 0; j < count; ++j) {
        str += TileColorToString(static_cast<TileColor>(color));
      }
    }
    str += "\n";
  }
  
  // Show center
  str += "Center: ";
  if (first_player_tile_available_) str += "F";
  for (int color = 0; color < kNumTileColors; ++color) {
    int count = center_pile_.tiles[color];
    for (int j = 0; j < count; ++j) {
      str += TileColorToString(static_cast<TileColor>(color));
    }
  }
  str += "\n";
  
  // Show player boards
  for (int player = 0; player < num_players_; ++player) {
    const PlayerBoard& board = player_boards_[player];
    str += absl::StrCat("Player ", player, " (Score: ", board.score, "):\n");
    
    // Pattern lines
    for (int line = 0; line < kNumPatternLines; ++line) {
      str += "  ";
      
      // Pattern line - tiles are right-aligned within each line's capacity
      const auto& pattern_line = board.pattern_lines[line];
      int line_capacity = line + 1;
      
      // Calculate leading spaces (for lines that aren't at the bottom)
      int leading_spaces = kNumPatternLines - line_capacity;
      
      // Add leading spaces for visual alignment
      for (int i = 0; i < leading_spaces; ++i) {
        str += " ";
      }
      
      // Add empty spaces within the line capacity for unfilled positions
      for (int i = 0; i < line_capacity - pattern_line.count; ++i) {
        str += " ";
      }
      
      // Add the actual tiles
      for (int j = 0; j < pattern_line.count; ++j) {
        str += TileColorToString(pattern_line.color);
      }
      
      str += " | ";
      
      // Wall
      for (int col = 0; col < kWallSize; ++col) {
        if (board.wall[line][col]) {
          str += TileColorToString(kWallPattern[line][col]);
        } else {
          str += ".";
        }
      }
      str += "\n";
    }
    
    // Floor line
    str += "Floor: ";
    for (TileColor tile : board.floor_line) {
      str += TileColorToString(tile);
    }
    str += "\n";
  }
  
  return str;
}

bool AzulState::IsTerminal() const {
  return game_ended_;
}

std::vector<double> AzulState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>(num_players_, 0.0);
  }
  
  std::vector<double> returns(num_players_);
  std::vector<int> scores(num_players_);
  std::vector<int> completed_rows(num_players_);
  int max_score = -1000;
  
  // Calculate all scores and completed rows, find maximum score
  for (int player = 0; player < num_players_; ++player) {
    scores[player] = CalculateScore(player);
    max_score = std::max(max_score, scores[player]);
    
    // Count completed horizontal lines (rows)
    completed_rows[player] = 0;
    const PlayerBoard& board = player_boards_[player];
    for (int row = 0; row < kWallSize; ++row) {
      bool complete = true;
      for (int col = 0; col < kWallSize; ++col) {
        if (!board.wall[row][col]) {
          complete = false;
          break;
        }
      }
      if (complete) completed_rows[player]++;
    }
  }
  
  // Find players with max score
  std::vector<int> score_tied_players;
  for (int player = 0; player < num_players_; ++player) {
    if (scores[player] == max_score) {
      score_tied_players.push_back(player);
    }
  }
  
  // Apply tiebreaker: among score-tied players, find max completed rows
  int max_completed_rows = -1;
  for (int player : score_tied_players) {
    max_completed_rows = std::max(max_completed_rows, completed_rows[player]);
  }
  
  // Final winners: max score AND max completed rows among score-tied players
  std::vector<int> final_winners;
  for (int player : score_tied_players) {
    if (completed_rows[player] == max_completed_rows) {
      final_winners.push_back(player);
    }
  }
  
  int num_winners = final_winners.size();
  
  // Zero-sum assignment: 
  // If all players tie, everyone gets 0
  // Otherwise, winners split +1, losers split -1
  if (num_winners == num_players_) {
    // All players tied - everyone gets 0
    for (int player = 0; player < num_players_; ++player) {
      returns[player] = 0.0;
    }
  } else {
    // Some players won, others lost
    double winner_utility = 1.0 / num_winners;
    int num_losers = num_players_ - num_winners;
    double loser_utility = -1.0 / num_losers;
    
    // Initialize all as losers
    for (int player = 0; player < num_players_; ++player) {
      returns[player] = loser_utility;
    }
    
    // Set winners
    for (int winner : final_winners) {
      returns[winner] = winner_utility;
    }
  }
  
  return returns;
}

std::string AzulState::InformationStateString(Player player) const {
  return ToString();
}

std::string AzulState::ObservationString(Player player) const {
  return ToString();
}

void AzulState::ObservationTensor(Player player, absl::Span<float> values) const {
  SPIEL_CHECK_EQ(values.size(), game_->ObservationTensorSize());
  std::fill(values.begin(), values.end(), 0.0f);
  
  int offset = 0;
  
  // Factories: 9 factories * 5 colors = 45 values
  int max_factories = 2 * kMaxNumPlayers + 1;  // 9 for max players
  for (int f = 0; f < max_factories; ++f) {
    for (int c = 0; c < kNumTileColors; ++c) {
      if (f < static_cast<int>(factories_.size())) {
        values[offset] = static_cast<float>(factories_[f].tiles[c]);
      }
      offset++;
    }
  }
  
  // Center pile: 5 colors = 5 values  
  for (int c = 0; c < kNumTileColors; ++c) {
    values[offset] = static_cast<float>(center_pile_.tiles[c]);
    offset++;
  }
  
  // First player tile available: 1 value
  values[offset] = first_player_tile_available_ ? 1.0f : 0.0f;
  offset++;
  
  // Round number: 1 value (normalized)
  values[offset] = static_cast<float>(round_number_) / 10.0f;  // Normalize assuming max ~10 rounds
  offset++;
  
  // Current player (one-hot): 4 values
  for (int p = 0; p < kMaxNumPlayers; ++p) {
    values[offset] = (p == current_player_) ? 1.0f : 0.0f;
    offset++;
  }
  
  // Tile supply: bag contents (5 colors) + discard pile contents (5 colors) = 10 values
  std::array<int, kNumTileColors> bag_counts = {};
  std::array<int, kNumTileColors> discard_counts = {};
  
  // Count tiles in bag by color
  for (TileColor tile : bag_) {
    if (tile >= TileColor::kBlue && tile <= TileColor::kWhite) {
      bag_counts[static_cast<int>(tile)]++;
    }
  }
  
  // Count tiles in discard pile by color
  for (TileColor tile : discard_pile_) {
    if (tile >= TileColor::kBlue && tile <= TileColor::kWhite) {
      discard_counts[static_cast<int>(tile)]++;
    }
  }
  
  // Bag contents (normalized by total tiles per color)
  for (int c = 0; c < kNumTileColors; ++c) {
    values[offset] = static_cast<float>(bag_counts[c]) / static_cast<float>(kTotalTilesPerColor);
    offset++;
  }
  
  // Discard pile contents (normalized by total tiles per color)
  for (int c = 0; c < kNumTileColors; ++c) {
    values[offset] = static_cast<float>(discard_counts[c]) / static_cast<float>(kTotalTilesPerColor);
    offset++;
  }
  
  // For each player: pattern lines, wall, floor, score
  for (int p = 0; p < kMaxNumPlayers; ++p) {
    if (p < num_players_) {
      const PlayerBoard& board = player_boards_[p];
      
      // Pattern lines: 5 lines * (5 color one-hot + 1 count) = 30 values
      for (int line = 0; line < kNumPatternLines; ++line) {
        const auto& pattern_line = board.pattern_lines[line];
        
        // Color (one-hot encoding)
        for (int c = 0; c < kNumTileColors; ++c) {
          values[offset] = (!pattern_line.IsEmpty() && 
                          pattern_line.color == static_cast<TileColor>(c)) ? 1.0f : 0.0f;
          offset++;
        }
        
        // Count (normalized by line capacity)
        float normalized_count = static_cast<float>(pattern_line.count) / static_cast<float>(line + 1);
        values[offset] = normalized_count;
        offset++;
      }
      
      // Wall: 5x5 = 25 values
      for (int row = 0; row < kWallSize; ++row) {
        for (int col = 0; col < kWallSize; ++col) {
          values[offset] = board.wall[row][col] ? 1.0f : 0.0f;
          offset++;
        }
      }
      
      // Floor line: 7 positions = 7 values (preserves order for penalty calculation)
      for (int pos = 0; pos < 7; ++pos) {
        if (pos < static_cast<int>(board.floor_line.size())) {
          TileColor tile = board.floor_line[pos];
          if (tile == TileColor::kFirstPlayer) {
            values[offset] = 6.0f;  // First player marker
          } else {
            values[offset] = static_cast<float>(tile) + 1.0f;  // Colors 1-5
          }
        } else {
          values[offset] = 0.0f;  // Empty position
        }
        offset++;
      }
      
      // Score (normalized)
      values[offset] = static_cast<float>(board.score) / 100.0f;  // Normalize assuming max ~100 points
      offset++;
      
    } else {
      // Pad with zeros for non-existent players
      int player_size = kNumPatternLines * (kNumTileColors + 1) + // Pattern lines: 30
                       kWallSize * kWallSize +                    // Wall: 25
                       7 +                                        // Floor line positions: 7
                       1;                                         // Score: 1
      for (int i = 0; i < player_size; ++i) {
        values[offset] = 0.0f;
        offset++;
      }
    }
  }
  
  SPIEL_CHECK_EQ(offset, values.size());
}

std::unique_ptr<State> AzulState::Clone() const {
  return std::make_unique<AzulState>(*this);
}

void AzulState::UndoAction(Player player, Action action) {
  // Undo is complex in Azul due to random bag shuffling
  // For now, we'll mark it as not implemented
  SpielFatalError("UndoAction not implemented for Azul");
}

std::vector<std::pair<Action, double>> AzulState::ChanceOutcomes() const {
  if (needs_bag_shuffle_) {
    // For bag shuffling, we have a single outcome with probability 1.0
    // The actual shuffling is handled deterministically by the RNG
    return {{0, 1.0}};
  }
  return {};
}

AzulGame::AzulGame(const GameParameters& params) 
    : Game(kGameType, params),
      rng_(ParameterValue<int>("rng_seed") == -1 
           ? std::time(0) 
           : ParameterValue<int>("rng_seed")) {
  num_players_ = ParameterValue<int>("num_players", kDefaultNumPlayers);
  SPIEL_CHECK_GE(num_players_, kMinNumPlayers);
  SPIEL_CHECK_LE(num_players_, kMaxNumPlayers);
  
  // Store the original seed so we can reset RNG state if needed
  original_seed_ = ParameterValue<int>("rng_seed") == -1 
                   ? std::time(0) 
                   : ParameterValue<int>("rng_seed");
}

std::vector<int> AzulGame::ObservationTensorShape() const {
  // Calculate the exact tensor size:
  // Factories: 9 factories * 5 colors = 45
  // Center: 5 colors = 5  
  // First player tile: 1
  // Round number: 1
  // Current player (one-hot): 4
  // Tile supply: bag (5 colors) + discard pile (5 colors) = 10
  // Per player (4 players): 
  //   Pattern lines: 5 * (5 color one-hot + 1 count) = 30
  //   Wall: 5*5 = 25
  //   Floor positions: 7 (preserves order for penalty calculation)
  //   Score: 1
  //   Total per player: 30 + 25 + 7 + 1 = 63
  // Total: 45 + 5 + 1 + 1 + 4 + 10 + (63 * 4) = 318
  
  int max_factories = 2 * kMaxNumPlayers + 1;  // 9
  int factories_size = max_factories * kNumTileColors;  // 45
  int center_size = kNumTileColors;  // 5
  int global_info_size = 1 + 1 + kMaxNumPlayers;  // first_player + round + current_player
  int tile_supply_size = 2 * kNumTileColors;  // bag + discard pile = 10
  
  int per_player_size = kNumPatternLines * (kNumTileColors + 1) +  // Pattern lines: 30
                       kWallSize * kWallSize +                     // Wall: 25  
                       7 +                                         // Floor positions: 7
                       1;                                          // Score: 1
  
  int total_size = factories_size + center_size + global_info_size + tile_supply_size + 
                  (per_player_size * kMaxNumPlayers);
  
  return {total_size};
}

std::string ActionToString(Action action, int num_players) {
  // This is a helper function - actual ActionToString is in the state
  return absl::StrCat("Action ", action);
}

}  // namespace azul
}  // namespace open_spiel
 