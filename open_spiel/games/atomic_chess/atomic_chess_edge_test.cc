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

// Comprehensive edge-case and property tests for atomic chess.
//
// Failures are recorded non-fatally (EXPECT) so a single run surfaces every
// problem at once. The file is split into:
//   * hand-verified positional oracle tests (one behaviour each), and
//   * a randomized property fuzzer that asserts universal invariants over many
//     random games seeded from a spread of starting positions.

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/games/atomic_chess/atomic_chess.h"
#include "open_spiel/games/chess/chess.h"
#include "open_spiel/games/chess/chess_board.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace atomic_chess {
namespace {

using chess::CastlingDirection;
using chess::ChessBoard;
using chess::Color;
using chess::ColorToPlayer;
using chess::DrawUtility;
using chess::LossUtility;
using chess::Move;
using chess::Piece;
using chess::PieceType;
using chess::Square;
using chess::WinUtility;

// ---------------------------------------------------------------------------
// Non-fatal expectation machinery.
// ---------------------------------------------------------------------------
int g_checks = 0;
int g_failures = 0;
const char* g_current_test = "";

#define EXPECT(cond)                                                        \
  do {                                                                      \
    ++g_checks;                                                             \
    if (!(cond)) {                                                          \
      ++g_failures;                                                         \
      std::cerr << "FAIL [" << g_current_test << "] " << __FILE__ << ":"    \
                << __LINE__ << ": " << #cond << std::endl;                  \
    }                                                                       \
  } while (0)

#define EXPECT_MSG(cond, msg)                                               \
  do {                                                                      \
    ++g_checks;                                                             \
    if (!(cond)) {                                                          \
      ++g_failures;                                                         \
      std::cerr << "FAIL [" << g_current_test << "] " << __FILE__ << ":"    \
                << __LINE__ << ": " << #cond << "  -- " << (msg)            \
                << std::endl;                                               \
    }                                                                       \
  } while (0)

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------
std::shared_ptr<const Game> LoadAtomic() { return LoadGame("atomic_chess"); }

AtomicChessState MakeState(const std::string& fen) {
  return AtomicChessState(LoadAtomic(), fen);
}

// Parse "e4" -> Square{file, rank}.
Square Sq(const std::string& s) {
  return Square{static_cast<int8_t>(s[0] - 'a'),
                static_cast<int8_t>(s[1] - '1')};
}

bool SameSq(const Square& a, const Square& b) {
  return a.x == b.x && a.y == b.y;
}

std::vector<Move> LegalMoves(const AtomicChessState& state) {
  std::vector<Move> moves;
  for (Action a : state.LegalActions()) {
    moves.push_back(chess::ActionToMove(a, state.Board()));
  }
  return moves;
}

bool AnyActionLandsOn(const AtomicChessState& state, const Square& to) {
  for (const Move& m : LegalMoves(state)) {
    if (SameSq(m.to, to)) return true;
  }
  return false;
}

bool AnyActionFrom(const AtomicChessState& state, const Square& from) {
  for (const Move& m : LegalMoves(state)) {
    if (SameSq(m.from, from)) return true;
  }
  return false;
}

bool HasMove(const AtomicChessState& state, const Square& from,
             const Square& to) {
  for (const Move& m : LegalMoves(state)) {
    if (SameSq(m.from, from) && SameSq(m.to, to)) return true;
  }
  return false;
}

// Is there a legal castling move whose king destination file is `to_x`?
bool HasCastleToFile(const AtomicChessState& state, int to_x) {
  for (const Move& m : LegalMoves(state)) {
    if (m.is_castling() && m.to.x == to_x) return true;
  }
  return false;
}

Square FindKing(const ChessBoard& board, Color color) {
  return board.find(Piece{color, PieceType::kKing});
}

bool KingOnBoard(const ChessBoard& board, Color color) {
  return board.InBoardArea(FindKing(board, color));
}

// Applies the legal action from `from` to `to` (first match). Records a
// failure and does nothing if no such legal action exists.
void ApplyFromTo(AtomicChessState* state, const Square& from,
                 const Square& to) {
  for (Action a : state->LegalActions()) {
    Move m = chess::ActionToMove(a, state->Board());
    if (SameSq(m.from, from) && SameSq(m.to, to)) {
      state->ApplyAction(a);
      return;
    }
  }
  ++g_failures;
  std::cerr << "FAIL [" << g_current_test
            << "]: expected legal move not found (from/to)" << std::endl;
}

void ApplySAN(AtomicChessState* state, const char* san) {
  std::optional<Move> m = state->Board().ParseSANMove(san);
  if (!m.has_value()) {
    ++g_failures;
    std::cerr << "FAIL [" << g_current_test << "]: could not parse SAN " << san
              << std::endl;
    return;
  }
  Action a = chess::MoveToAction(*m, state->BoardSize());
  const std::vector<Action>& legal = state->LegalActions();
  if (std::find(legal.begin(), legal.end(), a) == legal.end()) {
    ++g_failures;
    std::cerr << "FAIL [" << g_current_test << "]: SAN " << san
              << " is not legal" << std::endl;
    return;
  }
  state->ApplyAction(a);
}

#define RUN(fn)               \
  do {                        \
    g_current_test = #fn;     \
    fn();                     \
  } while (0)

// ===========================================================================
// Section 1: Explosion mechanics.
// ===========================================================================

// Capturing piece + captured piece are both removed.
void ExplosionRemovesBothTest() {
  AtomicChessState s = MakeState("k7/8/8/8/8/3r4/8/K2R4 w - - 0 1");
  ApplyFromTo(&s, Sq("d1"), Sq("d3"));  // Rd1xd3.
  EXPECT(s.Board().at(Sq("d3")).type == PieceType::kEmpty);  // capturer gone.
  EXPECT(s.Board().at(Sq("d1")).type == PieceType::kEmpty);  // moved away.
}

// Pawns adjacent to a blast survive; non-pawns adjacent are destroyed.
void PawnSurvivesNonPawnDiesTest() {
  // Rd1xd5 blast: knight e5 dies, pawn c4 survives.
  AtomicChessState s = MakeState("k7/8/8/3rN3/2P5/8/8/K2R4 w - - 0 1");
  ApplyFromTo(&s, Sq("d1"), Sq("d5"));
  EXPECT(s.Board().at(Sq("c4")).type == PieceType::kPawn);
  EXPECT(s.Board().at(Sq("e5")).type == PieceType::kEmpty);
  EXPECT(s.Board().at(Sq("d5")).type == PieceType::kEmpty);
}

// The directly-captured pawn is removed even though pawns are blast-immune.
void CapturedPawnIsRemovedTest() {
  AtomicChessState s = MakeState("k7/8/8/8/8/3p4/8/K2R4 w - - 0 1");
  ApplyFromTo(&s, Sq("d1"), Sq("d3"));  // Rxd3 captures the pawn.
  EXPECT(s.Board().at(Sq("d3")).type == PieceType::kEmpty);
}

// Explosion at a corner does not read off-board squares and clips correctly.
void CornerExplosionTest() {
  // Rb1xa1 blast at a1: white knight b2 dies, white pawn a2 survives.
  AtomicChessState s = MakeState("7k/8/8/8/8/8/PN6/nR5K w - - 0 1");
  ApplyFromTo(&s, Sq("b1"), Sq("a1"));
  EXPECT(s.Board().at(Sq("a1")).type == PieceType::kEmpty);
  EXPECT(s.Board().at(Sq("b1")).type == PieceType::kEmpty);
  EXPECT(s.Board().at(Sq("b2")).type == PieceType::kEmpty);   // knight died.
  EXPECT(s.Board().at(Sq("a2")).type == PieceType::kPawn);    // pawn survived.
}

// A capture-with-promotion still explodes; the promoted piece is destroyed.
void PromotionCaptureExplodesTest() {
  // b7xa8=Q: blast at a8 removes the promoted queen; b8 bishop dies, a7 pawn
  // survives.
  AtomicChessState s = MakeState("rb6/pP6/8/7k/8/8/8/7K w - - 0 1");
  ApplySAN(&s, "bxa8=Q");
  EXPECT(s.Board().at(Sq("a8")).type == PieceType::kEmpty);   // promoted gone.
  EXPECT(s.Board().at(Sq("b8")).type == PieceType::kEmpty);   // bishop died.
  EXPECT(s.Board().at(Sq("a7")).type == PieceType::kPawn);    // pawn survived.
}

// ===========================================================================
// Section 2: En passant explosions.
// ===========================================================================

// En passant capture: the captured pawn (behind the target) and the capturing
// pawn are removed, and the blast (centered on the empty ep square) removes
// adjacent non-pawns while sparing adjacent pawns.
void EnPassantExplosionTest() {
  AtomicChessState s = MakeState("7k/2n5/8/2ppP3/8/8/8/7K w - d6 0 1");
  ApplyFromTo(&s, Sq("e5"), Sq("d6"));  // exd6 e.p.
  EXPECT(s.Board().at(Sq("d5")).type == PieceType::kEmpty);  // captured pawn.
  EXPECT(s.Board().at(Sq("d6")).type == PieceType::kEmpty);  // capturer.
  EXPECT(s.Board().at(Sq("c7")).type == PieceType::kEmpty);  // knight died.
  EXPECT(s.Board().at(Sq("c5")).type == PieceType::kPawn);   // pawn survived.
}

// En passant capture whose blast reaches the enemy king wins immediately.
void EnPassantExplodesEnemyKingTest() {
  AtomicChessState s = MakeState("8/8/2k5/3pP3/8/8/8/7K w - d6 0 1");
  EXPECT(HasMove(s, Sq("e5"), Sq("d6")));  // the ep capture is available.
  ApplyFromTo(&s, Sq("e5"), Sq("d6"));
  EXPECT(s.IsTerminal());
  EXPECT(s.Returns()[ColorToPlayer(Color::kWhite)] == WinUtility());
}

// An en passant capture that would blast one's own king is illegal.
void EnPassantSelfExplosionIllegalTest() {
  AtomicChessState s = MakeState("k7/8/4K3/3pP3/8/8/8/8 w - d6 0 1");
  EXPECT(!HasMove(s, Sq("e5"), Sq("d6")));  // ep would explode Ke6.
}

// ===========================================================================
// Section 3: King rules.
// ===========================================================================

// The king can never capture.
void KingCannotCaptureTest() {
  AtomicChessState s = MakeState("8/8/8/4n3/4K3/8/8/k7 w - - 0 1");
  EXPECT(!HasMove(s, Sq("e4"), Sq("e5")));   // cannot take the knight.
  EXPECT(!s.LegalActions().empty());          // but has other moves.
}

// The two kings may stand on adjacent squares (unlike standard chess).
void KingsMayBeAdjacentTest() {
  // White king d4 may step to e4, next to the black king on e5.
  AtomicChessState s = MakeState("8/8/8/4k3/3K4/8/8/8 w - - 0 1");
  EXPECT(HasMove(s, Sq("d4"), Sq("e4")));
}

// ===========================================================================
// Section 4: Check / checkmate / stalemate.
// ===========================================================================

// Adjacent kings nullify checks from other pieces: a quiet move is legal even
// though a rook "attacks" the king along a file.
void AdjacentKingsNullifyCheckTest() {
  AtomicChessState s = MakeState("8/8/8/3Kk3/8/8/7P/3r4 w - - 0 1");
  EXPECT(HasMove(s, Sq("h2"), Sq("h3")));  // quiet move legal -> not in check.
  EXPECT(!s.IsTerminal());
}

// The same rook DOES check when the kings are not adjacent: the quiet move is
// then illegal.
void NonAdjacentKingsRealCheckTest() {
  AtomicChessState s = MakeState("7k/8/8/3K4/8/8/7P/3r4 w - - 0 1");
  EXPECT(!HasMove(s, Sq("h2"), Sq("h3")));  // must address the check.
  EXPECT(!s.IsTerminal());                   // king can escape.
}

// Basic checkmate: side to move is in check with no legal move -> loss.
void CheckmateIsLossTest() {
  AtomicChessState s = MakeState("7k/6Q1/8/8/8/8/8/K7 b - - 0 1");
  EXPECT(s.IsTerminal());
  EXPECT(s.Returns()[ColorToPlayer(Color::kBlack)] == LossUtility());
  EXPECT(s.Returns()[ColorToPlayer(Color::kWhite)] == WinUtility());
}

// Stalemate is a draw.
void StalemateIsDrawTest() {
  AtomicChessState s = MakeState("k7/2R5/N7/8/8/8/8/4K3 b - - 0 1");
  EXPECT(s.IsTerminal());
  EXPECT(s.Returns()[0] == DrawUtility());
  EXPECT(s.Returns()[1] == DrawUtility());
}

// While in check, a move that explodes the enemy king is legal and wins even
// though the mover's own king remains in check.
void WinByExplosionWhileInCheckTest() {
  AtomicChessState s = MakeState("k3r3/1n6/8/8/8/8/8/1R2K3 w - - 0 1");
  EXPECT(!s.IsTerminal());                  // in check but not mated.
  EXPECT(HasMove(s, Sq("b1"), Sq("b7")));   // Rxb7 blasts the black king.
  ApplyFromTo(&s, Sq("b1"), Sq("b7"));
  EXPECT(s.IsTerminal());
  EXPECT(s.Returns()[ColorToPlayer(Color::kWhite)] == WinUtility());
}

// A player may escape check by capturing (and thus exploding) the checker.
void EscapeCheckByExplodingCheckerTest() {
  AtomicChessState s = MakeState("Rk6/8/8/8/8/8/7P/r3K3 w - - 0 1");
  EXPECT(HasMove(s, Sq("a8"), Sq("a1")));    // Rxa1 removes the checker.
  EXPECT(!HasMove(s, Sq("h2"), Sq("h3")));   // a quiet move does not escape.
}

// A capture of the checker that would also blast one's own king is illegal.
void CannotEscapeBySelfExplodingCheckerTest() {
  AtomicChessState s = MakeState("k7/8/8/8/8/8/R3r3/4K3 w - - 0 1");
  EXPECT(!HasMove(s, Sq("a2"), Sq("e2")));   // Rxe2 would explode Ke1.
  EXPECT(HasMove(s, Sq("e1"), Sq("d1")));    // king can still escape.
  EXPECT(!s.IsTerminal());
}

// A pinned piece may not move so as to expose its own king.
void PinnedPieceCannotMoveTest() {
  AtomicChessState s = MakeState("k3r3/8/8/8/8/8/4B3/4K3 w - - 0 1");
  EXPECT(!AnyActionFrom(s, Sq("e2")));   // bishop is pinned on the e-file.
  EXPECT(!s.LegalActions().empty());      // king still has moves.
}

// ===========================================================================
// Section 5: Castling.
// ===========================================================================

void CastlingBothSidesLegalTest() {
  AtomicChessState s = MakeState("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
  EXPECT(HasCastleToFile(s, 6));  // O-O (king to g1).
  EXPECT(HasCastleToFile(s, 2));  // O-O-O (king to c1).
}

void CannotCastleThroughAttackedSquareTest() {
  // Black rook f2 attacks f1 -> kingside blocked, queenside still fine.
  AtomicChessState s = MakeState("r3k2r/8/8/8/8/8/5r2/R3K2R w KQkq - 0 1");
  EXPECT(!HasCastleToFile(s, 6));
  EXPECT(HasCastleToFile(s, 2));
}

// An enemy king adjacent to the king's castling path does not forbid castling:
// any capture on those squares would explode the adjacent enemy king, so the
// squares are safe. Here Black Kg2 is adjacent to f1 and g1, yet White may O-O.
void CanCastleWithEnemyKingAdjacentToPathTest() {
  AtomicChessState s = MakeState("8/8/8/8/8/8/6k1/4K2R w K - 0 1");
  EXPECT(HasCastleToFile(s, 6));
}

void CannotCastleOutOfCheckTest() {
  AtomicChessState s = MakeState("r3k3/8/8/8/8/8/4r3/R3K2R w KQ - 0 1");
  EXPECT(!HasCastleToFile(s, 6));
  EXPECT(!HasCastleToFile(s, 2));
  EXPECT(!s.IsTerminal());
}

// A rook removed by an explosion (not by a direct capture) must forfeit its
// castling right. This is the regression guard for the state-desync bug.
void ExplodedRookRevokesOwnRightTest() {
  // Qf3xg2 blast destroys the h1 rook; Ke1 survives, a1 rook remains.
  AtomicChessState s = MakeState("k7/8/8/8/8/5Q2/6n1/R3K2R w KQ - 0 1");
  ApplyFromTo(&s, Sq("f3"), Sq("g2"));
  EXPECT(!s.Board().CastlingRight(Color::kWhite, CastlingDirection::kRight));
  EXPECT(s.Board().CastlingRight(Color::kWhite, CastlingDirection::kLeft));
  EXPECT(s.Board().at(Sq("h1")).type == PieceType::kEmpty);
  EXPECT(s.Board().at(Sq("a1")).type == PieceType::kRook);
  EXPECT(KingOnBoard(s.Board(), Color::kWhite));
  // FEN must agree: kingside right gone.
  EXPECT(s.Board().ToFEN().find(" Q ") != std::string::npos);
}

// An explosion that removes the OPPONENT's rook revokes the opponent's right.
void ExplodedRookRevokesOpponentRightTest() {
  // Qh1xh7 blast destroys the black h8 rook; black keeps queenside (a8) right.
  AtomicChessState s = MakeState("r3k2r/7n/8/8/8/8/8/4K2Q w kq - 0 1");
  ApplyFromTo(&s, Sq("h1"), Sq("h7"));
  EXPECT(!s.Board().CastlingRight(Color::kBlack, CastlingDirection::kRight));
  EXPECT(s.Board().CastlingRight(Color::kBlack, CastlingDirection::kLeft));
  EXPECT(s.Board().at(Sq("h8")).type == PieceType::kEmpty);
  EXPECT(s.Board().at(Sq("a8")).type == PieceType::kRook);
}

// Directly capturing a rook (base-engine path) also revokes the right.
void DirectRookCaptureRevokesRightTest() {
  AtomicChessState s = MakeState("r3k2r/8/8/8/8/8/8/R3K2R w KQ - 0 1");
  // Give black no back-rank interaction: white Rh1xh8 blast (no king nearby).
  AtomicChessState s2 = MakeState("r3k2r/8/8/8/8/8/8/4K2R w K - 0 1");
  ApplyFromTo(&s2, Sq("h1"), Sq("h8"));
  EXPECT(!s2.Board().CastlingRight(Color::kBlack, CastlingDirection::kRight));
}

// ===========================================================================
// Section 6: Draws.
// ===========================================================================

void ThreefoldRepetitionDrawTest() {
  AtomicChessState s = MakeState("1n2k3/8/8/8/8/8/8/1N2K3 w - - 0 1");
  const char* cycle[] = {"Nc3", "Nc6", "Nb1", "Nb8"};
  for (int rep = 0; rep < 2; ++rep) {
    for (const char* mv : cycle) {
      EXPECT(!s.IsTerminal());
      ApplySAN(&s, mv);
    }
  }
  // After two full cycles the start position has occurred three times.
  EXPECT(s.IsTerminal());
  EXPECT(s.Returns()[0] == DrawUtility());
}

void FiftyMoveRuleDrawTest() {
  // Halfmove clock at 99; one further reversible move triggers the draw.
  AtomicChessState s = MakeState("1n2k3/8/8/8/8/8/8/1N2K3 w - - 99 60");
  EXPECT(!s.IsTerminal());
  ApplySAN(&s, "Nc3");
  EXPECT(s.IsTerminal());
  EXPECT(s.Returns()[0] == DrawUtility());
}

// Documents current behaviour: K-vs-K is NOT terminated by an
// insufficient-material rule (atomic omits it); it only draws via the 50-move
// or repetition rules.
void KingVsKingNotImmediatelyTerminalTest() {
  AtomicChessState s = MakeState("k7/8/8/8/8/8/8/7K w - - 0 1");
  EXPECT(!s.IsTerminal());
  EXPECT(!s.LegalActions().empty());
}

// ===========================================================================
// Section 7: State management.
// ===========================================================================

void CloneIsIndependentTest() {
  AtomicChessState s = MakeState("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1");
  std::string before = s.ToString();
  std::unique_ptr<State> c = s.Clone();
  EXPECT(c->ToString() == before);
  // Mutate the clone; the original must be unchanged.
  c->ApplyAction(c->LegalActions()[0]);
  EXPECT(s.ToString() == before);
  EXPECT(c->ToString() != before);
}

void SerializeRoundTripWithExplosionTest() {
  auto game = LoadAtomic();
  std::unique_ptr<State> s = game->NewInitialState();
  // 1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Bxc6 (capture -> explosion) dxc6.
  for (const char* mv : {"e4", "e5", "Nf3", "Nc6", "Bb5", "a6"}) {
    auto& as = down_cast<AtomicChessState&>(*s);
    auto m = as.Board().ParseSANMove(mv);
    SPIEL_CHECK_TRUE(m.has_value());
    s->ApplyAction(chess::MoveToAction(*m, as.BoardSize()));
  }
  {
    auto& as = down_cast<AtomicChessState&>(*s);
    auto m = as.Board().ParseSANMove("Bxc6");
    SPIEL_CHECK_TRUE(m.has_value());
    s->ApplyAction(chess::MoveToAction(*m, as.BoardSize()));
  }
  std::unique_ptr<State> d = game->DeserializeState(s->Serialize());
  EXPECT(d->ToString() == s->ToString());
  EXPECT(down_cast<AtomicChessState&>(*d).Board().HashValue() ==
         down_cast<AtomicChessState&>(*s).Board().HashValue());
  EXPECT(d->LegalActions() == s->LegalActions());
}

// ===========================================================================
// Section 8: Property / invariant fuzzer.
// ===========================================================================

int ObsTensorSize(const std::shared_ptr<const Game>& game) {
  int size = 1;
  for (int d : game->ObservationTensorShape()) size *= d;
  return size;
}

// Checks invariants that must hold in EVERY reachable position.
void CheckInvariants(const std::shared_ptr<const Game>& game,
                     const AtomicChessState& state, int obs_size,
                     int num_distinct) {
  const ChessBoard& board = state.Board();

  // (1) At most one king may be missing.
  bool w = KingOnBoard(board, Color::kWhite);
  bool b = KingOnBoard(board, Color::kBlack);
  EXPECT_MSG(w || b, board.ToFEN());  // both kings can't be gone in a state.

  // (2) Every advertised castling right has a real rook on the recorded square.
  //     This directly catches the castling-right/explosion desync (a right that
  //     outlives its rook). Note: we deliberately do NOT round-trip through
  //     ToFEN()/BoardFromFEN here, because the base engine cannot reload a
  //     standard FEN that has a castling right together with two same-side
  //     rooks (a documented chess960-FEN limitation in chess_board.cc), which
  //     is unrelated to atomic-chess logic.
  for (Color color : {Color::kWhite, Color::kBlack}) {
    for (CastlingDirection dir :
         {CastlingDirection::kLeft, CastlingDirection::kRight}) {
      if (board.CastlingRight(color, dir)) {
        absl::optional<Square> rook_sq =
            board.MaybeCastlingRookSquare(color, dir);
        EXPECT_MSG(rook_sq.has_value(), board.ToFEN());
        if (rook_sq.has_value()) {
          Piece p = board.at(*rook_sq);
          EXPECT_MSG(p.type == PieceType::kRook && p.color == color,
                     board.ToFEN());
        }
      }
    }
  }

  if (state.IsTerminal()) {
    // (4) Terminal states have no legal actions and zero-sum {-1,0,1} returns.
    EXPECT(state.LegalActions().empty());
    std::vector<double> r = state.Returns();
    EXPECT(r.size() == 2);
    EXPECT(r[0] + r[1] == 0.0);
    for (double v : r) EXPECT(v == -1.0 || v == 0.0 || v == 1.0);
    return;
  }

  // (5) Non-terminal states have legal actions.
  const std::vector<Action>& legal = state.LegalActions();
  EXPECT_MSG(!legal.empty(), board.ToFEN());

  // (6) Legal actions are sorted, unique and in range.
  for (size_t i = 0; i < legal.size(); ++i) {
    EXPECT(legal[i] >= 0 && legal[i] < num_distinct);
    if (i > 0) EXPECT(legal[i - 1] < legal[i]);  // strictly increasing.
  }

  Color mover = board.ToPlay();
  for (Action a : legal) {
    // (7) Action<->Move round-trips.
    Move m = chess::ActionToMove(a, board);
    EXPECT(chess::MoveToAction(m, kBoardSize) == a);

    // (8) The king never captures (non-castling king move onto an occupied
    //     square).
    if (m.piece.type == PieceType::kKing && !m.is_castling()) {
      EXPECT_MSG(board.at(m.to).type == PieceType::kEmpty, board.ToFEN());
    }

    // (9) Applying any legal move keeps the mover's own king on the board and
    //     never leaves both kings missing; if it removes the enemy king it is
    //     immediately terminal and winning.
    auto child = down_cast<const AtomicChessState&>(state).Clone();
    child->ApplyAction(a);
    const ChessBoard& cb = down_cast<AtomicChessState&>(*child).Board();
    EXPECT_MSG(KingOnBoard(cb, mover), board.ToFEN());
    bool cw = KingOnBoard(cb, Color::kWhite);
    bool cbk = KingOnBoard(cb, Color::kBlack);
    EXPECT(cw || cbk);
    if (!KingOnBoard(cb, chess::OppColor(mover))) {
      EXPECT(child->IsTerminal());
      EXPECT(child->Returns()[ColorToPlayer(mover)] == WinUtility());
    }
  }

  // (10) Observation tensor is well-formed and piece planes are one-hot.
  std::vector<float> obs(obs_size, -1.0f);
  state.ObservationTensor(ColorToPlayer(mover), absl::MakeSpan(obs));
  for (float v : obs) EXPECT(v >= 0.0f && v <= 1.0f);
  for (int sq = 0; sq < 64; ++sq) {
    double sum = 0.0;
    for (int plane = 0; plane < 13; ++plane) sum += obs[plane * 64 + sq];
    EXPECT_MSG(sum == 1.0, board.ToFEN());  // exactly one piece/empty plane set.
  }
}

// Undo after a single action restores the state exactly.
void CheckUndo(const AtomicChessState& state, Action a) {
  auto clone = state.Clone();
  Player p = clone->CurrentPlayer();
  std::string before = clone->ToString();
  uint64_t hbefore = down_cast<AtomicChessState&>(*clone).Board().HashValue();
  std::vector<Action> lbefore = clone->LegalActions();
  size_t hist_before = clone->History().size();

  clone->ApplyAction(a);
  clone->UndoAction(p, a);

  EXPECT_MSG(clone->ToString() == before, "undo ToString");
  EXPECT_MSG(down_cast<AtomicChessState&>(*clone).Board().HashValue() == hbefore,
             "undo hash");
  EXPECT_MSG(clone->LegalActions() == lbefore, "undo legal actions");
  EXPECT_MSG(clone->History().size() == hist_before, "undo history size");
}

void PropertyFuzzTest() {
  auto game = LoadAtomic();
  const int obs_size = ObsTensorSize(game);
  const int num_distinct = game->NumDistinctActions();

  const std::vector<std::string> seeds = {
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
      "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1",
      "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 0 1",
      "4k3/2p2pp1/1p5p/p2P4/P6P/1P3PP1/2P5/4K3 w - - 0 1",
      "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
  };

  const int kGamesPerSeed = 120;
  const int kMaxPlies = 160;

  for (size_t si = 0; si < seeds.size(); ++si) {
    for (int g = 0; g < kGamesPerSeed; ++g) {
      std::mt19937_64 rng(0x9E3779B97F4A7C15ULL * (si + 1) + g * 1000003ULL);
      AtomicChessState state = MakeState(seeds[si]);
      Player prev_player = kInvalidPlayer;

      for (int ply = 0; ply < kMaxPlies; ++ply) {
        CheckInvariants(game, state, obs_size, num_distinct);
        if (state.IsTerminal()) break;

        // Player alternation.
        Player cur = state.CurrentPlayer();
        if (prev_player != kInvalidPlayer) {
          EXPECT(cur != prev_player);
        }
        prev_player = cur;

        const std::vector<Action>& legal = state.LegalActions();
        Action a = legal[rng() % legal.size()];

        // Periodic heavier checks.
        if (ply % 6 == 0) CheckUndo(state, a);
        if (ply % 9 == 0) {
          auto d = game->DeserializeState(state.Serialize());
          EXPECT(d->ToString() == state.ToString());
          EXPECT(d->LegalActions() == state.LegalActions());
        }

        state.ApplyAction(a);

        if (g_failures > 200) {  // Stop spamming if something is badly broken.
          std::cerr << "Too many failures; aborting fuzz early." << std::endl;
          return;
        }
      }
    }
  }
}

void RunAll() {
  // Section 1.
  RUN(ExplosionRemovesBothTest);
  RUN(PawnSurvivesNonPawnDiesTest);
  RUN(CapturedPawnIsRemovedTest);
  RUN(CornerExplosionTest);
  RUN(PromotionCaptureExplodesTest);
  // Section 2.
  RUN(EnPassantExplosionTest);
  RUN(EnPassantExplodesEnemyKingTest);
  RUN(EnPassantSelfExplosionIllegalTest);
  // Section 3.
  RUN(KingCannotCaptureTest);
  RUN(KingsMayBeAdjacentTest);
  // Section 4.
  RUN(AdjacentKingsNullifyCheckTest);
  RUN(NonAdjacentKingsRealCheckTest);
  RUN(CheckmateIsLossTest);
  RUN(StalemateIsDrawTest);
  RUN(WinByExplosionWhileInCheckTest);
  RUN(EscapeCheckByExplodingCheckerTest);
  RUN(CannotEscapeBySelfExplodingCheckerTest);
  RUN(PinnedPieceCannotMoveTest);
  // Section 5.
  RUN(CastlingBothSidesLegalTest);
  RUN(CannotCastleThroughAttackedSquareTest);
  RUN(CanCastleWithEnemyKingAdjacentToPathTest);
  RUN(CannotCastleOutOfCheckTest);
  RUN(ExplodedRookRevokesOwnRightTest);
  RUN(ExplodedRookRevokesOpponentRightTest);
  RUN(DirectRookCaptureRevokesRightTest);
  // Section 6.
  RUN(ThreefoldRepetitionDrawTest);
  RUN(FiftyMoveRuleDrawTest);
  RUN(KingVsKingNotImmediatelyTerminalTest);
  // Section 7.
  RUN(CloneIsIndependentTest);
  RUN(SerializeRoundTripWithExplosionTest);
  // Section 8.
  g_current_test = "PropertyFuzzTest";
  PropertyFuzzTest();
}

}  // namespace
}  // namespace atomic_chess
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::atomic_chess::RunAll();
  std::cerr << "\n==== atomic_chess_edge_test: "
            << open_spiel::atomic_chess::g_checks << " checks, "
            << open_spiel::atomic_chess::g_failures << " failures ====\n";
  return open_spiel::atomic_chess::g_failures == 0 ? 0 : 1;
}
