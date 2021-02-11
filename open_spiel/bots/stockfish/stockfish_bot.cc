// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/bots/stockfish/stockfish/src/bitboard.h"
#include "open_spiel/bots/stockfish/stockfish/src/endgame.h"
#include "open_spiel/bots/stockfish/stockfish/src/position.h"
#include "open_spiel/bots/stockfish/stockfish/src/search.h"
#include "open_spiel/bots/stockfish/stockfish/src/thread.h"
#include "open_spiel/bots/stockfish/stockfish/src/tt.h"
#include "open_spiel/bots/stockfish/stockfish/src/uci.h"
#include "open_spiel/bots/stockfish/stockfish/src/syzygy/tbprobe.h"
#include "open_spiel/bots/stockfish/stockfish/src/types.h"
#include "open_spiel/bots/stockfish/stockfish_bot.h"
#include "open_spiel/spiel_utils.h"

#include "open_spiel/games/chess.h"

namespace open_spiel {
namespace stockfish {
namespace {

// FEN string of the initial position, normal chess
const char* StartFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";


}  // namespace

StockfishBot::StockfishBot(int argc, char **argv, int move_time) :
  move_time_(move_time) {


  CommandLine::init(argc, argv);
  UCI::init(Options);
  Tune::init();
  PSQT::init();
  Bitboards::init();
  Position::init();
  Bitbases::init();
  Endgames::init();
  Threads.set(size_t(Options["Threads"]));
  Search::clear(); // After threads are up
  Eval::init_NNUE();

  if (Options.count("Use NNUE")) {
    Options["Use NNUE"] = std::string("false");
  }
}

void StockfishBot::Restart() {
  Search::clear(); // ucinewgame command
}

void StockfishBot::RestartAt(const State& state) {
  Restart();
}


Action StockfishBot::Step(const State& state) {

  auto chess_state = dynamic_cast<const chess::ChessState *>(&state);

  Position pos;
  auto sf_states_ = StateListPtr(new std::deque<StateInfo>(1)); // Drop old and create a new one
  pos.set(state.ToString(), false, &sf_states_->back(), Threads.main());


  Search::LimitsType limits;
  limits.startTime = now();
  limits.movetime = move_time_;

  Threads.start_thinking(pos, sf_states_, limits, false);

  Threads.main()->wait_for_search_finished();

  Thread* bestThread = Threads.get_best_thread();

  // Send again PV info if we have a new best thread
  if (bestThread != Threads.main())
    sync_cout << UCI::pv(bestThread->rootPos, bestThread->completedDepth, -VALUE_INFINITE, VALUE_INFINITE) << sync_endl;

  sync_cout << "bestmove " << UCI::move(bestThread->rootMoves[0].pv[0], false);

  std::cout << sync_endl;

  Move best_move = bestThread->rootMoves[0].pv[0];

  chess::Move move = MoveFromStockfishMove(best_move, chess_state->Board());

  return chess::MoveToAction(move);
}

chess::Move StockfishBot::MoveFromStockfishMove(Move move, const chess::StandardChessBoard &board) {
  Square from = from_sq(move);
  Square to = to_sq(move);

  std::string from_str = UCI::square(from);
  std::string to_str = UCI::square(to);

  auto from_sq = chess::SquareFromString(from_str);
  auto to_sq = chess::SquareFromString(to_str);
  if (!from_sq || !to_sq) {
    SpielFatalError("Conversion of move from stockfish to openspiel failed");
  }

  std::optional<chess::PieceType> promotion_piece_type = chess::PieceType::kEmpty;
  if (type_of(move) == PROMOTION) {
    const char promotion_str = " pnbrqk"[promotion_type(move)];
    promotion_piece_type = chess::PieceTypeFromChar(promotion_str);
    if (!promotion_piece_type) {
      SpielFatalError("Conversion of move from stockfish to openspiel failed");
    }
  }

  return chess::Move(*from_sq, *to_sq, board.at(*from_sq), *promotion_piece_type, type_of(move) == CASTLING);
}

void StockfishBot::InformAction(const State& state, Player player_id,
                                Action action) {

}

std::unique_ptr<Bot> stockfish::MakeStockfishBot(int argc, char **argv, int move_time) {
  return std::make_unique<StockfishBot>(argc, argv, move_time);
}
}  // namespace stockfish
}  // namespace open_spiel
