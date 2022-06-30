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

// This generates strategically interesting instances of Colored Trails
// according to the criteria of Sec 5 of Jong et al', 2011, Metastrategies in
// the Colored Trails Game.
// https://www.ifaamas.org/Proceedings/aamas2011/papers/C4_R57.pdf

#include <random>
#include <string>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/games/colored_trails.h"
#include "open_spiel/utils/file.h"
#include "open_spiel/utils/init.h"

ABSL_FLAG(int, seed, 0, "Seed to use");
ABSL_FLAG(int, num_boards, 10000, "Number of boards to generate.");
ABSL_FLAG(std::string, filename, "/tmp/boards.txt", "File to save boards to.");

namespace open_spiel {
namespace colored_trails {
namespace {

constexpr int kBaseScoreEpsilon = 20;

bool CheckBoard(const Board& board) {
  std::vector<int> base_scores(board.num_players);
  int min_score = board.size * 100;
  int max_score = board.size * -100;

  for (Player player = 0; player < board.num_players; ++player) {
    std::pair<int, bool> score_and_solved = Score(player, board);
    if (score_and_solved.second) {
      // Cannot be solvable without negotiation.
      return false;
    }
    base_scores[player] = score_and_solved.first;
    min_score = std::min(min_score, base_scores[player]);
    max_score = std::max(max_score, base_scores[player]);
  }

  if (max_score - min_score > kBaseScoreEpsilon) {
    return false;
  }

  // Now check that there exist two trades:
  // - one between player 0 and 2, such that both can reach the goal
  // - one between player 1 and 2, such that both can reach the goal
  for (int proposer : {0, 1}) {
    bool found_trade = false;
    ChipComboIterator iter1(board.chips[proposer]);
    while (!found_trade && !iter1.IsFinished()) {
      std::vector<int> combo1 = iter1.Next();
      ChipComboIterator iter2(board.chips[2]);
      while (!found_trade && !iter2.IsFinished()) {
        std::vector<int> combo2 = iter2.Next();
        // Do the trade and check if both can reach the goal.
        Board board_copy = board;
        Trade trade(combo1, combo2);
        board_copy.ApplyTrade({proposer, 2}, trade);
        std::pair<int, bool> prop_score_and_goal = Score(proposer, board_copy);
        if (prop_score_and_goal.second) {
          std::pair<int, bool> rec_score_and_goal = Score(2, board_copy);
          if (rec_score_and_goal.second) {
            found_trade = true;
          }
        }
      }
    }
    if (!found_trade) {
      return false;
    }
  }

  return true;
}

std::string GenerateBoard(std::mt19937* rng) {
  bool valid_board = false;
  std::string board_string;

  while (!valid_board) {
    Board board;
    // Generate the player's chips.
    int width = kNumChipsUpperBound - kNumChipsLowerBound + 1;
    for (int p = 0; p < board.num_players; ++p) {
      // First their number of chips.
      board.num_chips[p] =
          kNumChipsLowerBound + absl::Uniform<int>(*rng, 0, width);
      // Then, their chips
      for (int i = 0; i < board.num_chips[p]; ++i) {
        int chip = absl::Uniform<int>(*rng, 0, board.num_colors);
        board.chips[p][chip]++;
      }
    }

    // Now, the board.
    for (int r = 0; r < board.size; ++r) {
      for (int c = 0; c < board.size; ++c) {
        int idx = r * board.size + c;
        board.board[idx] = absl::Uniform<int>(*rng, 0, board.num_colors);
      }
    }

    // Now the player positions.
    // The flag position is the last one, hence positions.size() here
    for (int p = 0; p < board.positions.size(); ++p) {
      int candidate = -1;
      while (absl::c_find(board.positions, candidate) !=
             board.positions.end()) {
        candidate = absl::Uniform<int>(*rng, 0, board.size * board.size);
      }
      board.positions[p] = candidate;
    }

    // Check the board.
    valid_board = CheckBoard(board);
    board_string = board.ToString();
  }

  return board_string;
}

void GenerateBoards(int num) {
  std::string filename = absl::GetFlag(FLAGS_filename);
  int seed = absl::GetFlag(FLAGS_seed);
  std::mt19937 rng(seed);

  std::cout << "Starting." << std::endl;
  TradeInfo trade_info;
  InitTradeInfo(&trade_info, kDefaultNumColors);
  std::cout << "Num combos: " << trade_info.chip_combinations.size()
            << ", possible trades " << trade_info.possible_trades.size()
            << std::endl;

  std::cout << "Opening file: " << filename << std::endl;
  open_spiel::file::File outfile(filename, "w");
  for (int i = 0; i < num; ++i) {
    std::cout << "Generating board " << i << std::endl;
    std::string line = GenerateBoard(&rng);
    line.push_back('\n');
    std::cout << line;
    outfile.Write(line);
  }
  std::cout << "Wrote to file: " << filename << std::endl;
}

}  // namespace
}  // namespace colored_trails
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::Init("", &argc, &argv, false);
  absl::ParseCommandLine(argc, argv);
  open_spiel::colored_trails::GenerateBoards(absl::GetFlag(FLAGS_num_boards));
}
