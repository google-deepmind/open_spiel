// Copyright 2023 DeepMind Technologies Limited
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

#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

#include "open_spiel/games/abalone/abalone.h"
#include "open_spiel/games/abalone/abalone_core_ab.h"

inline constexpr int kSearchDepthAbalone = 2;

namespace open_spiel {
namespace abalone {
namespace {

namespace testing = open_spiel::testing;

void BasicAbaloneTests() {
  testing::LoadGameTest("abalone");
  testing::NoChanceOutcomesTest(*LoadGame("abalone"));
  testing::RandomSimTest(*LoadGame("abalone"), 100);
}

void RewardAbaloneTest() {
  auto game = LoadGame("abalone");

  auto state = game->NewInitialState();
  std::vector<double> rewards = state->Rewards();
  std::vector<double> returns = state->Returns();

  SPIEL_CHECK_EQ(rewards[0], 0.0);
  SPIEL_CHECK_EQ(rewards[1], 0.0);
  SPIEL_CHECK_EQ(returns[0], 0.0);
  SPIEL_CHECK_EQ(returns[1], 0.0);
}

// Force a draw (game reaches the move limit with no winner) by directly
// marking the core state as drawn, optionally with unequal marble counts.
static void ForceDraw(AbaloneState* ab, bool remove_one_p1_marble) {
  ab->core_state_.Reset(abalone_core::ABALONE_INIT_CLASSIC);
  if (remove_one_p1_marble)
    ab->core_state_.board_[8][8] = abalone_core::CellState::Empty;
  // outcome_ == Empty is the signal set by Eval on a move-limit draw.
  ab->core_state_.outcome_ = abalone_core::CellState::Empty;
}

void DrawPenaltyTest() {
  // Default draw_penalty = 0: a draw returns {0, 0} regardless of marbles.
  {
    auto game = LoadGame("abalone");
    auto state = game->NewInitialState();
    auto* ab = static_cast<AbaloneState*>(state.get());
    ForceDraw(ab, /*remove_one_p1_marble=*/true);  // 14 vs 13
    SPIEL_CHECK_TRUE(ab->IsTerminal());
    auto ret = ab->Returns();
    SPIEL_CHECK_EQ(ret[0], 0.0);
    SPIEL_CHECK_EQ(ret[1], 0.0);
  }
  // draw_penalty=0.5: the player with fewer marbles is penalized (zero-sum).
  {
    auto game = LoadGame("abalone(draw_penalty=0.5)");
    auto state = game->NewInitialState();
    auto* ab = static_cast<AbaloneState*>(state.get());
    ForceDraw(ab, /*remove_one_p1_marble=*/true);  // P0=14, P1=13
    SPIEL_CHECK_TRUE(ab->IsTerminal());
    auto ret = ab->Returns();
    SPIEL_CHECK_EQ(ret[0], 0.5);   // P0 (more marbles) rewarded
    SPIEL_CHECK_EQ(ret[1], -0.5);  // P1 (fewer marbles) penalized
  }
  // Equal marble counts on a draw: no penalty (cannot break a tie in zero-sum).
  {
    auto game = LoadGame("abalone(draw_penalty=0.5)");
    auto state = game->NewInitialState();
    auto* ab = static_cast<AbaloneState*>(state.get());
    ForceDraw(ab, /*remove_one_p1_marble=*/false);  // 14 vs 14
    SPIEL_CHECK_TRUE(ab->IsTerminal());
    auto ret = ab->Returns();
    SPIEL_CHECK_EQ(ret[0], 0.0);
    SPIEL_CHECK_EQ(ret[1], 0.0);
  }
  // A non-terminal state never incurs the draw penalty.
  {
    auto game = LoadGame("abalone(draw_penalty=0.5)");
    auto state = game->NewInitialState();
    SPIEL_CHECK_TRUE(!state->IsTerminal());
    auto ret = state->Returns();
    SPIEL_CHECK_EQ(ret[0], 0.0);
    SPIEL_CHECK_EQ(ret[1], 0.0);
  }
}

// Count marbles of a player on the core board.
static int CountMarbles(const abalone_core::core_state& stt,
                        abalone_core::CellState p) {
  return std::count(&stt.board_[0][0],
                    &stt.board_[0][0] +
                        sizeof(stt.board_) / sizeof(stt.board_[0][0]), p);
}

void RandomBoardTest() {
  // Same seed -> identical boards.
  auto g1 = LoadGame("abalone(board=random-symmetric,seed=42)");
  auto s1 = g1->NewInitialState();
  auto g2 = LoadGame("abalone(board=random-symmetric,seed=42)");
  auto s2 = g2->NewInitialState();
  auto* c1 = &static_cast<AbaloneState*>(s1.get())->core_state_;
  auto* c2 = &static_cast<AbaloneState*>(s2.get())->core_state_;
  SPIEL_CHECK_TRUE(std::equal(&c1->board_[0][0],
                              &c1->board_[0][0] +
                                  sizeof(c1->board_) / sizeof(c1->board_[0][0]),
                              &c2->board_[0][0]));

  // kMarblesPerPlayer marbles per player, all on valid (non-Invalid) cells.
  SPIEL_CHECK_EQ(CountMarbles(*c1, abalone_core::CellState::Player0),
                 abalone_core::kMarblesPerPlayer);
  SPIEL_CHECK_EQ(CountMarbles(*c1, abalone_core::CellState::Player1),
                 abalone_core::kMarblesPerPlayer);
  for (int r = 0; r < abalone_core::kNumRows; ++r) {
    for (int c = 0; c < abalone_core::kNumCols; ++c) {
      if (abalone_core::VALID_BOARD[r][c] == abalone_core::CellState::Invalid)
        SPIEL_CHECK_EQ(c1->board_[r][c], abalone_core::CellState::Invalid);
      else
        SPIEL_CHECK_TRUE(c1->board_[r][c] == abalone_core::CellState::Empty ||
                         c1->board_[r][c] == abalone_core::CellState::Player0 ||
                         c1->board_[r][c] == abalone_core::CellState::Player1);
    }
  }

  // Central symmetry: board[r][c] is the opposite player at the 180-degree
  // rotated cell (kNumRows-1-r, kNumCols-1-c), except the self-symmetric
  // center which must be empty.
  auto opposite = [](abalone_core::CellState s) {
    if (s == abalone_core::CellState::Player0) return abalone_core::CellState::Player1;
    if (s == abalone_core::CellState::Player1) return abalone_core::CellState::Player0;
    return s;
  };
  for (int r = 0; r < abalone_core::kNumRows; ++r) {
    for (int c = 0; c < abalone_core::kNumCols; ++c) {
      int sr = abalone_core::kNumRows - 1 - r;
      int sc = abalone_core::kNumCols - 1 - c;
      if (r == sr && c == sc)
        SPIEL_CHECK_EQ(c1->board_[r][c], abalone_core::CellState::Empty);
      else
        SPIEL_CHECK_EQ(c1->board_[sr][sc], opposite(c1->board_[r][c]));
    }
  }

  // Different seeds -> (almost surely) different boards.
  auto g3 = LoadGame("abalone(board=random-symmetric,seed=7)");
  auto s3 = g3->NewInitialState();
  auto* c3 = &static_cast<AbaloneState*>(s3.get())->core_state_;
  SPIEL_CHECK_FALSE(std::equal(&c1->board_[0][0],
                               &c1->board_[0][0] +
                                   sizeof(c1->board_) / sizeof(c1->board_[0][0]),
                               &c3->board_[0][0]));

  // A random board must still produce a playable game: random sims + a
  // couple of legal moves applied + Undo round-trip.
  testing::RandomSimTest(*g1, 20);
  auto st = g1->NewInitialState();
  auto before = static_cast<AbaloneState*>(st.get())->core_state_;
  auto legal = st->LegalActions();
  SPIEL_CHECK_FALSE(legal.empty());
  st->ApplyAction(legal[0]);
  st->UndoAction(0, legal[0]);
  auto* after = &static_cast<AbaloneState*>(st.get())->core_state_;
  SPIEL_CHECK_TRUE(std::equal(&before.board_[0][0],
                              &before.board_[0][0] +
                                  sizeof(before.board_) /
                                      sizeof(before.board_[0][0]),
                              &after->board_[0][0]));
}

void AlphaBetaSeedTest() {
  // A seeded search (seed >= 0) must be reproducible: the same state and
  // seed yield the same best action and the same (action, value) list,
  // even across separate game instances. The legacy mode (seed < 0) uses
  // a process-wide static RNG and is not required to be reproducible here.
  auto game = LoadGame("abalone");
  auto state = game->NewInitialState();

  auto run = [&]() {
    return AbaloneAB(*state, kSearchDepthAbalone, /*seed=*/42);
  };
  auto r1 = run();
  auto r2 = run();
  // Same best action.
  SPIEL_CHECK_EQ(r1.first, r2.first);
  // Same evaluated move list (same length, same actions, same values).
  SPIEL_CHECK_EQ(r1.second.size(), r2.second.size());
  for (std::size_t i = 0; i < r1.second.size(); ++i) {
    SPIEL_CHECK_EQ(r1.second[i].first, r2.second[i].first);
    SPIEL_CHECK_FLOAT_EQ(r1.second[i].second, r2.second[i].second);
  }

  // Reproducible across a fresh game instance.
  auto game2 = LoadGame("abalone");
  auto state2 = game2->NewInitialState();
  auto r3 = AbaloneAB(*state2, kSearchDepthAbalone, /*seed=*/42);
  SPIEL_CHECK_EQ(r1.first, r3.first);
  SPIEL_CHECK_EQ(r1.second.size(), r3.second.size());
}

std::pair<open_spiel::Action, float> _LogABSpiel(
    std::unique_ptr<State>& state) {
  auto max_move = AllAbaloneMoves_ABSpiel(state, kSearchDepthAbalone);
  return max_move;
}

std::pair<abalone_core::core_Action, float> _LogAB(
    std::unique_ptr<State>& state) {
  auto best_move = abalone_core::AlphaBeta(
      static_cast<abalone::AbaloneState*>(state.get())->core_state_,
      kSearchDepthAbalone, -1.f, 1.f);
  return best_move;
}

void DatasetTest() {
  // https://abaloneonline.wordpress.com/abalone-bank/belgian-daisy-games/
  // http://www.ist.tugraz.at/staff/aichholzer/research/rp/abalone/games.php
  typedef int game_length;
  typedef std::tuple<int, int, abalone_core::CellState> game_result;

  std::vector<
    std::tuple<std::string, game_length, std::string, game_result>
    > scenarios = {
    // Game 1: Standard
    std::make_tuple(
      std::string("classic"),
      116,
      std::string(
          "1.a1b2 i5h5 2.a5b5 i6h6 3.b5c5 i7h7 4.b6b5 i8h8 "
          "5.b1c2 h9g8 6.a3b3 h7g7 7.b4c4 h4h5 8.a4b5a3 e6e7 "
          "9.a3a2 h6g6 10.b3c3 i9i8 11.a2b3 i8h8 12.b4b3 f8e7 "
          "13.d3d4 h7g6 14.b1b2 e4f5 15.b4c4 d7d8 16.d6c5 d8e8 "
          "17.b4b3 f8g9 18.b1b2 g9f8 19.b4b3 f5g6 20.c2d3 f8f7 "
          "21.b2c2 i8i7 22.b1a1b2 i7i8 23.a2b2a1 i8h8 24.b1b2 f5g6 "
          "25.b2c2 i8h7 26.d2c2 g5h6 27.f4e3 h6g5 28.d2c2 e6f6 "
          "29.a2b2 h6g6 30.d2c2 f8e7 31.a2b2 d6e7 32.d2c2 f5g6 "
          "33.a2b2 f8f7 34.d3d4 h7g6 35.d6c5 i8h7 36.b4c4 e6f6 "
          "37.e3e4 f4g4 38.d2e3 g4f4 39.e4e5 h6g6 40.e7d6 e9e8 "
          "41.b4c4 e6f6 42.d6c5 f4g4 43.b3c4 h6g6 44.b2b3 h7h6 "
          "45.d6c5 f7f6 46.c2d3 f4g5 47.d3e4 i7h7 48.d5e5 f7f6 "
          "49.a3b3 h7g7 50.b3c4 i5h4 51.g6g5 g3h4 52.f4g4 h6h5 "
          "53.g5f4 h4h5 54.c5d5 i8h8 55.g4f3 g5h6g6 56.b4c5 h5i5h6 "
          "57.d3e4 g8g7 58.g5f4 i6i7"),
      // 1 - 1 tie
      std::make_tuple(1, 1, abalone_core::CellState::Empty)),

    // Game 2 : Belgian Daisy
    std::make_tuple(
      std::string("belgian-daisy"),
      -1,
      std::string(
          "1.a1b2 a5b5 2.i9h8 a4b4 3.i8h7 b5c5 4.h7g6 g5f4 "
          "5.c3d4d3 h4g4 6.h9g8 c5d5 7.h8g8 i6h5 8.f6f8e6 c4d5 "
          "9.g6f6 h6g6 10.a2b3 b6c6 11.e8f8 b4c5 12.b3b4 c6c5 "
          "13.b2c2 c4d5 14.c3c4 d5e6 15.h9h8 c5d5 16.d4d3 f7e6 "
          "17.h8g7 f4f5 18.g7f7 g4f3f4 19.f8e7 g8g7 20.b3c3 e3f4 "
          "21.b4b3 f6e6 22.d7e7 g5g6 23.f7f6 f4f5 24.f7g8f8 i5h5 "
          "25.g9f8 g6f6 26.c6b6b5 b4c5 27.f8g9e8 e6e7"),
      // 0 - 1 unfinished
      std::make_tuple(0, 1, abalone_core::CellState::Invalid)),

    // Game 3 : Belgian Daisy
    std::make_tuple(
      std::string("belgian-daisy"),
      148,
      std::string(
          "1.i9h8 i5h5 2.i8h7 h5g5 3.a1b2 a5b5 4.a2b3 h4g4 "
          "5.h7g6 b4c5 6.g6f5 i6h6 7.g7g8f7 b6c6 8.b3c3 a4b5 "
          "9.c4d4 d7d6 10.f4e4 b5c6 11.f7f6 c6c5 12.c1d1 g4g5 "
          "13.f8f7 c4d5 14.f4f5 g5g6 15.b1c1 g6h7 16.b2c2 h4h5 "
          "17.h9g9 h5h6 18.h9g9 h7g7 19.c1d2 d6e6 20.g6g5 h6g6 "
          "21.d1e2 g6f6 22.d4e4 g7g8 23.f9e8 c3d4 24.e4f4 h8g8 "
          "25.d8c7 h4h5 26.c2d3 g9g8 27.c7c6 h5h6 28.c6b5 h6h7 "
          "29.b5b4 d7c6 30.g4f4 h7g7 31.b4b3 g7f6 32.g5g4 g8g7 "
          "33.b3b2 c5b4 34.b2c3b1 g7f7 35.c7b6 d7c6 36.b6a5 f8e7 "
          "37.e9f9 c6c5 38.c2d2 c5d5 39.d3e4 b5c5 40.d2d3 c5d5 "
          "41.b1c2 c4c3 42.c1d2 c2c3 43.g5g4 c3c4 44.f9g9 d4e5 "
          "45.d2c2 b4c4 46.g3g4 d6d5 47.f3e3 c4c3 48.f2e2 g6f6 "
          "49.c2d3 c3b2c4 50.c1d1 b3b4 51.g9h9 e8f8 52.d1e2 d6d5 "
          "53.d2e2 b4b5 54.h9h8 e5f6 55.h5g5 d4d5 56.a5a4 h8g7 "
          "57.g4h5 g7f6 58.a4a3 f6e5 59.d1e2 d6c5 60.a3a2 e5d4 "
          "61.a1b1 f8e7 62.b1c1 e6f7e5 63.e2e3 g6f6g7 64.c1d2 b2c3 "
          "65.d2e2 g7f7 66.a2b2 e5d4 67.a1b1 b2c3 68.b1c1 c3d4 "
          "69.c1d2 b4c4b3 70.f2g3 c3b3c4 71.g7g6 d3c3 72.g6f5 d6d5 "
          "73.g3f3 c3c4 74.i9i8 b4c5"),
      // 0 - 1 tie
      std::make_tuple(0, 1, abalone_core::CellState::Empty)),

    // Game 4 : Belgian Daisy
    std::make_tuple(
      std::string("belgian-daisy"),
      -1,
      std::string(
          "1.a1b2 i5h5 2.a2b3 a5b6 3.i9h8 i6h6 4.h8g7 h6g6 "
          "5.b3c4 c5c7d6 6.b1c2 b4b6c5 7.g8g7f8 g4g5f3 8.c2d3 h4h5 "
          "9.c3d4 h6g5 10.b2c2 a4b5 11.i8h8 c5d6 12.c4c5 c7c6 "
          "13.h9h8 c5d6 14.h7h6 g7h7 15.g9g8 h4i5 16.h8g8 e7d6 "
          "17.h5h6 h8g7 18.h7h8 g7h7 19.h8g8 i5h5 20.e8e7 b5b4 "
          "21.c2c3 b4b5 22.e4d3 h5g4 23.c2c3 c7b6 24.c5d5 h7g6 "
          "25.f8e7 d8e8 26.c4d4 h4h5 27.f6e6 b6b5 28.g8f8 b4c5b3 "
          "29.f8e7 h6g5 30.f7e6 a2b2 31.c4d4 h4h5 32.c6d6 e8f8 "
          "33.h7g7 f3g4 34.g7f7 f8g8 35.f4e4 g4f4 36.e7f8 f5g6 "
          "37.f8f7 f4g4 38.d5e5 i5h4 39.d3e4 i8h8 40.e6f6 i6h5 "
          "41.c5d6d5 h4g3 42.h6g5 h5h4 43.d4e4 g3f2 44.f6f5 h8g7 "
          "45.f4f3"),
      // 6 - 0 P0
      std::make_tuple(6, 0, abalone_core::CellState::Player0)),

    // Game 5 : Belgian Daisy
    std::make_tuple(
      std::string("belgian-daisy"),
      -1,
      std::string(
          "1.i9h8 i6h6 2.i8h7 h5g5 3.h7g6 a5b5 4.h8g7 a4b4 "
          "5.a1b1 g5f4 6.g8g7 f4h4e3 7.b1c2 e3e4 8.a2b2 b6c6 "
          "9.g6f6 b6b5 10.b1c1 g3f3 11.f5g5f4 b3c4 12.c2d3 c6c5 "
          "13.d6e7 c5c4 14.d1d2 c4c3 15.g7g8 b4b5c4 16.d2d3 d6e6 "
          "17.e7f8 h6g5 18.f4e4 d7d6 19.g4f4 c5d6 20.d5d4 d6e6 "
          "21.h6h7 e5d5 22.e4e5 e7e6 23.h7g7 i5h4 24.b2b3 h4h5 "
          "25.h9g9 e5f6 26.g9f8 h5g5 27.h8g8 f3g4 28.f8e7 f5f6 "
          "29.g8f8 c1c2 30.d6c6 c2c3 31.d2d3 b4c5 32.d3d4 c3c4 "
          "33.c7b6 e3d3 34.f4e4 c6c5 35.b3c3 g7f6 36.c3d3 g6f6 "
          "37.c6b6b5 c5d6 38.a5b5 d6d7 39.g9g8 f8f7 40.e8f8 b4c4 "
          "41.f4e3 d8d7 42.b2c2 e6f7 43.h9h8 d6e7 44.g9h9 f7g8"),
      // 0 - 6 P1
      std::make_tuple(0, 6, abalone_core::CellState::Player1)
    )
  };

  for (auto scenario : scenarios) {
    auto game_board = std::get<0>(scenario);
    auto game = LoadGame(std::string("abalone(board=") + game_board
                        + std::string(")"));
    auto state = game->NewInitialState();

    auto str = std::get<2>(scenario);
    int i = 0;
    for (i = 0; ; ++i) {
      char buff[100];
      snprintf(buff, sizeof(buff), "%i.", i + 1);
      auto ply = std::string(buff);
      auto ply_pos = str.find(ply);
      if (ply_pos == std::string::npos)
        break;
      char dl = ' ';
      size_t start, end;
      start = end = ply_pos + ply.size();
      if ((start = str.find_first_not_of(dl, end)) == std::string::npos)
        break;
      end = str.find(dl, start);
      auto p1 = str.substr(start, end - start);
      auto p1_move_flag = abalone_core::Move::FromString(p1);
      auto p1_move = std::get<1>(p1_move_flag);

      // Smoke-test the alpha-beta search on real, complex late-game
      // positions (both the spiel-facing and core-only entry points).
      if (i >= 42) {
        _LogABSpiel(state);
        _LogAB(state);
      }

      auto moveId1 = abalone_core::Move::MoveToAction(p1_move);
      state->ApplyAction(moveId1);

      if ((start = str.find_first_not_of(dl, end)) == std::string::npos)
        break;
      end = str.find(dl, start);
      auto p2 = str.substr(start, end - start);
      auto p2_move_flag = abalone_core::Move::FromString(p2);
      auto p2_move = std::get<1>(p2_move_flag);

      if (i >= 42) {
        _LogABSpiel(state);
        _LogAB(state);
      }

      auto moveId2 = abalone_core::Move::MoveToAction(p2_move);
      state->ApplyAction(moveId2);

      auto p1_moveOut = abalone_core::Move::ActionToMove(moveId1);
      SPIEL_CHECK_TRUE(p1_moveOut == p1_move);

      auto p2_moveOut = abalone_core::Move::ActionToMove(moveId2);
      SPIEL_CHECK_TRUE(p2_moveOut == p2_move);
    }

    auto core_state =
        static_cast<abalone::AbaloneState*>(state.get())->core_state_;
    game_result res = std::get<3>(scenario);
    auto board_balls_P0 =
        abalone_core::kMarblesPerPlayer - std::count(&core_state.board_[0][0],
                        &core_state.board_[0][0] +
                        sizeof(core_state.board_) /
                        sizeof(core_state.board_[0][0]),
                        abalone_core::CellState::Player1);
    auto board_balls_P1 =
        abalone_core::kMarblesPerPlayer - std::count(&core_state.board_[0][0],
                        &core_state.board_[0][0] +
                        sizeof(core_state.board_) /
                        sizeof(core_state.board_[0][0]),
                        abalone_core::CellState::Player0);
    auto result_balls_P0 = std::get<0>(res);
    auto result_balls_P1 = std::get<1>(res);
    SPIEL_CHECK_TRUE(result_balls_P0 == board_balls_P0);
    SPIEL_CHECK_TRUE(result_balls_P1 == board_balls_P1);
    if (std::get<1>(scenario) < 0) {
      SPIEL_CHECK_TRUE(std::get<2>(res) == core_state.outcome_);
    } else {
      SPIEL_CHECK_TRUE(std::get<1>(scenario) == core_state.turn_);
    }
  }
}

void StringAbaloneTests() {
  abalone_core::core_state stt;
  stt.Reset();
  for (auto i = 0;
       i < abalone_core::kNumCells * abalone_core::kNumActionsPerCell; ++i) {
    auto move = abalone_core::Move::ActionToMove(i);
    auto move_str = move.ToString();
    auto maybe_move = abalone_core::Move::FromString(move_str);
    if (move.IsValid(stt)) {
      SPIEL_CHECK_TRUE(std::get<0>(maybe_move));
      auto move2 = std::get<1>(maybe_move);
      auto action = abalone_core::Move::MoveToAction(move2);
      SPIEL_CHECK_TRUE(action == i);
    }
  }
}

}  // namespace
}  // namespace abalone
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::abalone::BasicAbaloneTests();
  open_spiel::abalone::RewardAbaloneTest();
  open_spiel::abalone::DrawPenaltyTest();
  open_spiel::abalone::RandomBoardTest();
  open_spiel::abalone::AlphaBetaSeedTest();
  open_spiel::abalone::StringAbaloneTests();
  open_spiel::abalone::DatasetTest();
}
