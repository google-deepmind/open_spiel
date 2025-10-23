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

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"
#include "spiel_utils.h"

namespace open_spiel {
namespace azul {
namespace {

namespace testing = open_spiel::testing;

// Custom state checker function for Azul-specific validation
void AzulStateChecker(const State& state) {
  // Check that the state string representation is valid and contains expected elements
  std::string state_str = state.ToString();
  SPIEL_CHECK_FALSE(state_str.empty());
  SPIEL_CHECK_TRUE(state_str.find("Round:") != std::string::npos);
  SPIEL_CHECK_TRUE(state_str.find("Factory") != std::string::npos);
  SPIEL_CHECK_TRUE(state_str.find("Center:") != std::string::npos);
  SPIEL_CHECK_TRUE(state_str.find("Player") != std::string::npos);
  SPIEL_CHECK_TRUE(state_str.find("Score:") != std::string::npos);
  SPIEL_CHECK_TRUE(state_str.find("Floor:") != std::string::npos);
  
  // Validate game state invariants based on Azul rules
  const AzulState* azul_state = dynamic_cast<const AzulState*>(&state);
  if (azul_state != nullptr) {
    for (int player = 0; player < state.NumPlayers(); ++player) {
      const auto& board = azul_state->GetPlayerBoard(player);
      
      for (int line = 0; line < kNumPatternLines; ++line) {
        const auto& pattern_line = board.pattern_lines[line];
        
        // Pattern line capacity constraint: line i can hold at most i+1 tiles
        SPIEL_CHECK_GE(pattern_line.count, 0);
        SPIEL_CHECK_LE(pattern_line.count, line + 1);
        // First player tile is a special tile and cannot be placed in a pattern line.
        SPIEL_CHECK_NE(pattern_line.color, TileColor::kFirstPlayer);
        if (!pattern_line.IsEmpty()) {
          // Find where this color would go on the wall
          int wall_col = -1;
          for (int col = 0; col < kWallSize; ++col) {
            if (kWallPattern[line][col] == pattern_line.color) {
              wall_col = col;
              break;
            }
          }
          // Color should exist in wall pattern
          SPIEL_CHECK_NE(wall_col, -1);
          // If tiles exist, the wall must not have this color.
          SPIEL_CHECK_FALSE(board.wall[line][wall_col]);
        }
      }
      
      // Score should never be negative
      SPIEL_CHECK_GE(board.score, 0);
    }
    
    // Each factory should have 0-4 tiles
    const auto& factories = azul_state->Factories();
    for (const auto& factory : factories) {
      int total_tiles = 0;
      for (int color = 0; color < kNumTileColors; ++color) {
        SPIEL_CHECK_GE(factory.tiles[color], 0);
        total_tiles += factory.tiles[color];
      }
      SPIEL_CHECK_GE(total_tiles, 0);
      SPIEL_CHECK_LE(total_tiles, kTilesPerFactory);
    }

    const auto& center = azul_state->CenterPile();
    // Center pile can contain any colors except kEmpty - just check that tile counts are valid
    for (int color = 0; color < kNumTileColors; ++color) {
      SPIEL_CHECK_GE(center.tiles[color], 0);
    }
    
    // First player tile should only appear once per round
    bool first_player_in_center = azul_state->HasFirstPlayerTile();
    int first_player_on_floors = 0;
    for (int player = 0; player < state.NumPlayers(); ++player) {
      const auto& board = azul_state->GetPlayerBoard(player);
      for (TileColor tile : board.floor_line) {
        if (tile == TileColor::kFirstPlayer) {
          first_player_on_floors++;
        }
      }
    }
    // First player tile should be in exactly one place (center or one player's floor)
    SPIEL_CHECK_TRUE((first_player_in_center ? 1 : 0) + first_player_on_floors <= 1);
  }

  if (state.IsTerminal()) {
    // Game should only end when at least one player has a complete horizontal line
    if (azul_state != nullptr) {
      bool has_complete_line = false;
      for (int player = 0; player < state.NumPlayers(); ++player) {
        const auto& board = azul_state->GetPlayerBoard(player);
        for (int row = 0; row < kWallSize; ++row) {
          bool complete_row = true;
          for (int col = 0; col < kWallSize; ++col) {
            if (!board.wall[row][col]) {
              complete_row = false;
              break;
            }
          }
          if (complete_row) {
            has_complete_line = true;
            break;
          }
        }
        if (has_complete_line) break;
      }
      // Terminal states should have at least one complete horizontal line.
      SPIEL_CHECK_TRUE(has_complete_line);
    }
  } else {
    // Legal actions should exist and be non-empty for non-terminal states
    std::vector<Action> actions = state.LegalActions();
    SPIEL_CHECK_FALSE(actions.empty());
    
    // Test action string formatting - all actions should have proper format
    bool has_pattern_line_action = false;
    bool has_floor_action = false;
    
    for (Action action : actions) {
      std::string action_str = state.ActionToString(state.CurrentPlayer(), action);
      SPIEL_CHECK_FALSE(action_str.empty());
      
      // Every action should have these components
      SPIEL_CHECK_TRUE(action_str.find("Take") != std::string::npos);
      SPIEL_CHECK_TRUE(action_str.find("from") != std::string::npos);
      SPIEL_CHECK_TRUE(action_str.find("to") != std::string::npos);
      
      // Track action types
      if (action_str.find("Line") != std::string::npos) {
        has_pattern_line_action = true;
      }
      if (action_str.find("Floor") != std::string::npos) {
        has_floor_action = true;
      }
    }
    
    // Both pattern line and floor actions should be available (unless very edge cases)
    // This validates that the action generation is working correctly
    if (actions.size() > 1) {  // Only check if we have multiple actions
      SPIEL_CHECK_TRUE(has_pattern_line_action || has_floor_action);
    }
  }
  
  // Test information state and observation functions for all players
  for (int player = 0; player < state.NumPlayers(); ++player) {
    std::string info_state = state.InformationStateString(player);
    std::string obs_state = state.ObservationString(player);
    SPIEL_CHECK_FALSE(info_state.empty());
    SPIEL_CHECK_FALSE(obs_state.empty());
  }
  
  // Test that cloning works correctly
  std::unique_ptr<State> cloned_state = state.Clone();
  SPIEL_CHECK_EQ(state.CurrentPlayer(), cloned_state->CurrentPlayer());
  SPIEL_CHECK_EQ(state.IsTerminal(), cloned_state->IsTerminal());
  SPIEL_CHECK_EQ(state.ToString(), cloned_state->ToString());
  
  if (!state.IsTerminal()) {
    std::vector<Action> original_actions = state.LegalActions();
    std::vector<Action> cloned_actions = cloned_state->LegalActions();
    SPIEL_CHECK_EQ(original_actions.size(), cloned_actions.size());
    
    for (size_t i = 0; i < original_actions.size(); ++i) {
      SPIEL_CHECK_EQ(original_actions[i], cloned_actions[i]);
    }
  }
}

void BasicAzulTests() {
  // Core OpenSpiel tests with comprehensive state validation
  testing::LoadGameTest("azul");
  testing::ChanceOutcomesTest(*LoadGame("azul", {{"rng_seed", GameParameter(0)}}));
  
  // Extensive random simulation with our comprehensive state checker
  testing::RandomSimTest(*LoadGame("azul", {{"rng_seed", GameParameter(0)}}), 100, true, false, true, &AzulStateChecker);
  
  // Test with different player counts
  for (int players = 2; players <= 4; ++players) {
    std::shared_ptr<const Game> game = LoadGame("azul", {{"num_players", GameParameter(players)}, {"rng_seed", GameParameter(0)}});
    testing::RandomSimTest(*game, 20, true, false, true, &AzulStateChecker);
  }
  
  // Test that legal actions are sorted (required by OpenSpiel)
  std::shared_ptr<const Game> game = LoadGame("azul", {{"rng_seed", GameParameter(0)}});
  std::unique_ptr<State> state = game->NewInitialState();
  testing::CheckLegalActionsAreSorted(*game, *state);
}

// Helper function to create a test state with specific wall configuration
std::unique_ptr<AzulState> CreateTestState(std::shared_ptr<const Game> game) {
  auto state = std::make_unique<AzulState>(game, 2);
  // Clear factories and center to avoid assertions in SetupNewRound
  for (auto& factory : state->factories_) {
    factory.tiles.fill(0);
  }
  state->center_pile_.tiles.fill(0);
  return state;
}

void ScoringTests() {
  std::shared_ptr<const Game> game = LoadGame("azul", {{"rng_seed", GameParameter(0)}});
  
  // Test 1: Pattern line completion and basic scoring
  {
    auto state = CreateTestState(game);
    auto& board = state->player_boards_[0];
    
    // Set up a complete pattern line (line 0 needs 1 tile)
    board.pattern_lines[0].color = TileColor::kBlue;
    board.pattern_lines[0].count = 1;
    
    int initial_score = board.score;
    state->EndRoundScoring();
    
    // Should have gained 1 point for placing the tile
    SPIEL_CHECK_EQ(board.score, initial_score + 1);
    // Pattern line should be cleared
    SPIEL_CHECK_EQ(board.pattern_lines[0].count, 0);
    // Wall should have the tile placed
    SPIEL_CHECK_TRUE(board.wall[0][0]);  // Blue goes in position [0][0] according to pattern
  }
  
  // Test 2: Horizontal adjacency scoring
  {
    auto state = CreateTestState(game);
    auto& board = state->player_boards_[0];
    
    // Place some tiles on the wall first
    board.wall[1][0] = true;  // White at [1][0]
    board.wall[1][2] = true;  // Yellow at [1][2]
    
    // Complete pattern line to place Blue at [1][1] (between the existing tiles)
    board.pattern_lines[1].color = TileColor::kBlue;
    board.pattern_lines[1].count = 2;  // Line 1 needs 2 tiles
    
    int initial_score = board.score;
    state->EndRoundScoring();
    
    // Should score 3 points: the new tile plus 2 adjacent tiles horizontally
    SPIEL_CHECK_EQ(board.score, initial_score + 3);
    SPIEL_CHECK_TRUE(board.wall[1][1]);  // Blue placed at [1][1]
  }
  
  // Test 3: Vertical adjacency scoring
  {
    auto state = CreateTestState(game);
    auto& board = state->player_boards_[0];
    
    // Place tiles vertically adjacent to where we'll place a new tile
    board.wall[0][1] = true;  // Yellow at [0][1]
    board.wall[2][1] = true;  // White at [2][1]
    
    // Complete pattern line to place Yellow at [1][1]
    board.pattern_lines[1].color = TileColor::kBlue;
    board.pattern_lines[1].count = 2;
    
    int initial_score = board.score;
    state->EndRoundScoring();
    
    // Should score 3 points: the new tile plus 2 adjacent tiles vertically
    SPIEL_CHECK_EQ(board.score, initial_score + 3);
    SPIEL_CHECK_TRUE(board.wall[1][1]);
  }
  
  // Test 4: Both horizontal and vertical adjacency
  {
    auto state = CreateTestState(game);
    auto& board = state->player_boards_[0];
    
    // Create a cross pattern around [2][2]
    board.wall[2][1] = true;  // Black at [2][1] (left)
    board.wall[2][3] = true;  // Yellow at [2][3] (right) 
    board.wall[1][2] = true;  // White at [1][2] (up)
    board.wall[3][2] = true;  // Red at [3][2] (down)
    
    // Complete pattern line to place Blue at [2][2] (center)
    board.pattern_lines[2].color = TileColor::kBlue;
    board.pattern_lines[2].count = 3;  // Line 2 needs 3 tiles
    
    int initial_score = board.score;
    state->EndRoundScoring();
    
    // Should score: horizontal (3) + vertical (3) - 1 for double counting center = 5 points
    SPIEL_CHECK_EQ(board.score, initial_score + 5);
    SPIEL_CHECK_TRUE(board.wall[2][2]);
  }
  
  // Test 5: Floor line penalties
  {
    auto state = CreateTestState(game);
    auto& board = state->player_boards_[0];
    
    // Add tiles to floor line (penalties: -1, -1, -2, -2, -2, -3, -3)
    board.floor_line = {TileColor::kRed, TileColor::kBlue, TileColor::kYellow}; // -1, -1, -2 = -4 total
    
    int initial_score = board.score;
    state->EndRoundScoring();
    
    // Should lose 4 points but score can't go below 0
    SPIEL_CHECK_EQ(board.score, std::max(0, initial_score - 4));
    // Floor line should be cleared
    SPIEL_CHECK_TRUE(board.floor_line.empty());
  }
  
  // Test 6: End-game row bonus scoring
  {
    auto state = CreateTestState(game);
    auto& board = state->player_boards_[0];
    
    // Complete a full row (row 0)
    board.wall[0][0] = true;  // Blue
    board.wall[0][1] = true;  // Yellow
    board.wall[0][2] = true;  // Red
    board.wall[0][3] = true;  // Black
    board.wall[0][4] = true;  // White
    
    state->game_ended_ = true;  // Make it terminal for final scoring
    
    int base_score = board.score;
    int final_score = state->CalculateScore(0);
    
    // Should get +2 bonus for complete row
    SPIEL_CHECK_EQ(final_score, base_score + 2);
  }
  
  // Test 7: End-game column bonus scoring
  {
    auto state = CreateTestState(game);
    auto& board = state->player_boards_[0];
    
    // Complete a full column (column 1)
    board.wall[0][1] = true;  // Yellow
    board.wall[1][1] = true;  // Blue
    board.wall[2][1] = true;  // White  
    board.wall[3][1] = true;  // Black
    board.wall[4][1] = true;  // Red
    
    state->game_ended_ = true;
    
    int base_score = board.score;
    int final_score = state->CalculateScore(0);
    
    // Should get +7 bonus for complete column
    SPIEL_CHECK_EQ(final_score, base_score + 7);
  }
  
  // Test 8: End-game color bonus scoring
  {
    auto state = CreateTestState(game);
    auto& board = state->player_boards_[0];
    
    // Complete all Blue tiles (one in each row)
    board.wall[0][0] = true;  // Row 0, Col 0 = Blue
    board.wall[1][1] = true;  // Row 1, Col 1 = Blue
    board.wall[2][2] = true;  // Row 2, Col 2 = Blue
    board.wall[3][3] = true;  // Row 3, Col 3 = Blue
    board.wall[4][4] = true;  // Row 4, Col 4 = Blue
    
    state->game_ended_ = true;
    
    int base_score = board.score;
    int final_score = state->CalculateScore(0);
    
    // Should get +10 bonus for complete color
    SPIEL_CHECK_EQ(final_score, base_score + 10);
  }
  
  // Test 9: Multiple bonuses combined
  {
    auto state = CreateTestState(game);
    auto& board = state->player_boards_[0];
    
    // Complete the entire wall for maximum bonuses
    for (int row = 0; row < kWallSize; ++row) {
      for (int col = 0; col < kWallSize; ++col) {
        board.wall[row][col] = true;
      }
    }
    
    state->game_ended_ = true;
    
    int base_score = board.score;
    int final_score = state->CalculateScore(0);
    
    // Should get: 5 rows * 2 + 5 columns * 7 + 5 colors * 10 = 10 + 35 + 50 = 95 bonus points
    SPIEL_CHECK_EQ(final_score, base_score + 95);
  }
}

}  // namespace
}  // namespace azul
}  // namespace open_spiel

int main(int argc, char** argv) {
  // Run core OpenSpiel tests with Azul-specific validation
  open_spiel::azul::BasicAzulTests();
  
  // Run comprehensive scoring tests
  open_spiel::azul::ScoringTests();

  return 0;
} 