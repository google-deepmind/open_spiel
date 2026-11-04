// Copyright 2026 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_GAMES_SIGUO_JUNQI_H_
#define OPEN_SPIEL_GAMES_SIGUO_JUNQI_H_

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// Siguo Junqi (四国军棋, "Four-State Military Chess", a.k.a. Four-state-war):
// the four-player, two-team version of Luzhanqi (陆战棋).
// See https://zh.wikipedia.org/wiki/四国军棋 and
// https://en.wikipedia.org/wiki/Luzhanqi
//
// Players 0 (South), 1 (East), 2 (North) and 3 (West) sit around a cross-shaped
// board and play in that (counterclockwise) order. Opposite seats are partners:
// team 0 is {0, 2} and team 1 is {1, 3}.
//
// Board: each player owns a 5-column x 6-row territory of 30 points (23 posts,
// 5 camps, 2 headquarters). The four territories surround a 3x3 railway hub,
// for 4 * 30 + 9 = 129 points. Local coordinates (row, col) are used within a
// territory: row 1 is the back row (holding the headquarters at columns 2 and
// 4) and row 6 is the front row facing the hub; columns are numbered 1 to 5
// from the owner's left to right. Camps are at (3,2), (3,4), (4,3), (5,2) and
// (5,4). Railways run along rows 2 and 6 and columns 1 and 5 of every
// territory, through the hub (front-row columns 1, 3 and 5 connect to it), and
// along four curved arcs that join column 1 of each territory's front row to
// column 5 of the front row of the territory on the owner's left. Roads join
// orthogonal neighbours inside a territory, plus the diagonals of every camp.
//
// Pieces (25 per player): Field Marshal (司令, rank 9) x1, General (军长, 8)
// x1, Major General (师长, 7) x2, Brigadier (旅长, 6) x2, Colonel (团长, 5) x2,
// Major (营长, 4) x2, Captain (连长, 3) x3, Lieutenant (排长, 2) x3, Engineer
// (工兵, 1) x3, Bomb (炸弹) x2, Landmine (地雷) x3 and Flag (军旗) x1.
//
// The game has two phases.
//  1. Deployment: each player in turn places their 25 pieces on the 23 posts
//     and 2 headquarters of their territory (camps start empty), one point at a
//     time from the back row to the front row. The flag must be in a
//     headquarters, landmines in the last two rows and bombs not in the front
//     row. Each deployment action names the piece type for the next point.
//  2. Play: on their turn a player moves one piece. Every piece may move one
//     step along a road or railway. On railways a piece may slide any distance
//     along an unobstructed straight line, and may follow the curved arcs;
//     engineers may additionally turn freely on the railway network. Landmines,
//     the flag and any piece standing in a headquarters can never move. A piece
//     in a camp cannot be attacked and allies cannot attack each other.
//
// Combat: the higher rank removes the lower rank and takes its point; equal
// ranks are both removed. A bomb is removed together with whatever it meets.
// Only an engineer can remove a landmine; any other attacker is removed (and
// by default the landmine survives). Any piece may capture the flag. When a
// player's Field Marshal is removed, the location of that player's flag is
// revealed to everyone.
//
// A player is eliminated (and all their pieces are removed) when their flag
// is captured or when they have no legal move at the start of their turn.
// A team loses when both of its players are eliminated. The game is drawn
// after `no_collision_draw_plies` consecutive moves without any attack, or
// after `max_moves` moves.
//
// By default no player sees the identity of anyone else's pieces, not even
// their partner's (四暗). Only the outcome of each battle is public.
//
// Parameters:
//   "information_mode"       string  "hidden" (四暗, default),
//                                    "partner_visible" (双明: partners see each
//                                    other's pieces) or "all_visible" (全明).
//                                    Visibility of other players' pieces only
//                                    starts once deployment is over.
//   "flying_bombs"           bool    Bombs may turn on the railway like
//                                    engineers (default false).
//   "mine_survives_attack"   bool    A landmine survives killing an attacker
//                                    other than an engineer or bomb (default
//                                    true).
//   "marshal_death_reveals_flag"
//                            bool    Reveal a player's flag when their Field
//                                    Marshal is removed (default true).
//   "no_collision_draw_plies"
//                            int     Draw after this many consecutive moves
//                                    without an attack; 0 disables (default
//                                    70).
//   "max_moves"              int     Draw after this many moves in the play
//                                    phase (default 1000).

namespace open_spiel {
namespace siguo_junqi {

inline constexpr int kNumPlayers = 4;
inline constexpr int kZoneRows = 6;
inline constexpr int kZoneCols = 5;
inline constexpr int kZoneSize = kZoneRows * kZoneCols;  // 30
inline constexpr int kHubSize = 3;
inline constexpr int kNumHubPoints = kHubSize * kHubSize;  // 9
inline constexpr int kHubStart = kNumPlayers * kZoneSize;  // 120
inline constexpr int kNumNodes = kHubStart + kNumHubPoints;  // 129
inline constexpr int kPiecesPerPlayer = 25;

// Piece types. Officers are ordered by rank: rank = type - kEngineer + 1.
enum PieceType {
  kNoPiece = -1,
  kFlag = 0,
  kLandmine = 1,
  kBomb = 2,
  kEngineer = 3,
  kLieutenant = 4,
  kCaptain = 5,
  kMajor = 6,
  kColonel = 7,
  kBrigadier = 8,
  kMajorGeneral = 9,
  kGeneral = 10,
  kFieldMarshal = 11,
};
inline constexpr int kNumPieceTypes = 12;
inline constexpr std::array<int, kNumPieceTypes> kPieceCounts = {
    1, 3, 2, 3, 3, 3, 2, 2, 2, 2, 1, 1};

enum class NodeType { kPost, kCamp, kHeadquarters };

enum class InformationMode { kHidden, kPartnerVisible, kAllVisible };

enum class BattleResult { kAttackerWins, kDefenderWins, kBothRemoved };

// Action encoding:
//   Deployment: action = piece type (0..11), placed on the next point.
//   Move:       action = kNumPieceTypes + from * kNumNodes + to.
inline constexpr int kNumDeployActions = kNumPieceTypes;
inline constexpr int kNumMoveActions = kNumNodes * kNumNodes;
inline constexpr int kNumDistinctActions = kNumDeployActions + kNumMoveActions;

// Observation tensor: for each of the four seats (relative to the observer:
// self, next, partner, previous), 12 planes for pieces whose type is visible
// to the observer plus one plane for pieces whose type is hidden, over all 129
// points. Then: deployment phase flag (1), current player (4, relative),
// eliminated players (4, relative) and the no-collision counter as a fraction
// of the draw threshold (1).
inline constexpr int kPlanesPerSeat = kNumPieceTypes + 1;
inline constexpr int kNumBoardPlanes = kPlanesPerSeat * kNumPlayers;
inline constexpr int kObservationTensorSize =
    kNumBoardPlanes * kNumNodes + 1 + kNumPlayers + kNumPlayers + 1;

inline constexpr int kDefaultNoCollisionDrawPlies = 70;
inline constexpr int kDefaultMaxMoves = 1000;

struct Piece {
  int owner = kInvalidPlayer;
  PieceType type = kNoPiece;
  bool empty() const { return type == kNoPiece; }
};

// Board topology helpers.
NodeType NodeTypeOf(int node);
bool IsRailNode(int node);
// Zone (= owning seat) of a node, or kInvalidPlayer for hub points.
int ZoneOf(int node);
// Node index of point (row, col) in `zone`'s territory, 1-based coordinates.
int ZoneNode(int zone, int row, int col);
// Node index of hub point (row, col), 1-based; row 1 is next to South and
// column 1 is next to West.
int HubNode(int row, int col);
// Names such as "S12" (South, row 1, col 2) or "C22" (hub centre).
std::string NodeName(int node);
std::string PieceTypeToString(PieceType type);
// Combat outcome when `attacker` attacks `defender`.
BattleResult ResolveBattle(PieceType attacker, PieceType defender,
                           bool mine_survives_attack);

struct SiguoJunqiStructContents {
  std::string phase;  // "deployment", "play" or "terminal".
  int current_player = kInvalidPlayer;
  // Point name -> piece, e.g. {"S12": "S:Flag"}; hidden pieces are "S:?".
  std::map<std::string, std::string> board;
  std::vector<bool> eliminated;
  std::vector<bool> flag_revealed;
  int moves_since_collision = 0;
  int winning_team = -1;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(SiguoJunqiStructContents, phase,
                                 current_player, board, eliminated,
                                 flag_revealed, moves_since_collision,
                                 winning_team);
};

SPIEL_DEFINE_STRUCT(SiguoJunqiStateStruct, StateStruct,
                    SiguoJunqiStructContents);
SPIEL_DEFINE_STRUCT(SiguoJunqiObservationStruct, ObservationStruct,
                    SiguoJunqiStructContents);

// A deployment action sets `piece`; a move sets `from` and `to`.
struct SiguoJunqiActionStruct : public ActionStruct {
  std::string piece;
  std::string from;
  std::string to;
  SPIEL_STRUCT_BOILERPLATE(SiguoJunqiActionStruct, piece, from, to);
};

class SiguoJunqiGame;

class SiguoJunqiState : public State {
 public:
  explicit SiguoJunqiState(std::shared_ptr<const Game> game);
  SiguoJunqiState(const SiguoJunqiState&) = default;

  Player CurrentPlayer() const override;
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  std::unique_ptr<StateStruct> ToStruct() const override;
  std::unique_ptr<ObservationStruct> ToObservationStruct(
      Player player) const override;
  std::unique_ptr<ActionStruct> ActionToStruct(Player player,
                                               Action action_id) const override;
  std::vector<Action> StructToActions(
      const ActionStruct& action_struct) const override;

  // Accessors, mainly for tests.
  const Piece& PieceAt(int node) const { return board_[node]; }
  bool IsDeploymentPhase() const { return in_deployment_; }
  bool IsEliminated(Player player) const { return eliminated_[player]; }
  bool FlagRevealed(Player player) const { return flag_revealed_[player]; }
  int MovesSinceCollision() const { return moves_since_collision_; }
  // Replaces the board and starts the play phase with `player` to move. The
  // board must be consistent with the rules (e.g. immobile pieces); this is
  // intended for tests.
  void SetBoardForTesting(const std::array<Piece, kNumNodes>& board,
                          Player player);
  // Destinations reachable by the piece on `node`, ignoring whose turn it is.
  std::vector<int> Destinations(int node) const;

 protected:
  void DoApplyAction(Action action) override;

 private:
  const SiguoJunqiGame& ParentGame() const;
  void ApplyDeployAction(Action action);
  void ApplyMoveAction(int from, int to);
  std::vector<Action> LegalDeployActions() const;
  std::vector<Action> LegalMoveActions(Player player) const;
  bool CanMove(int node) const;
  bool CanEnter(int mover, int node) const;
  bool IsAlly(Player a, Player b) const { return a % 2 == b % 2; }
  bool SeesTypesOf(Player observer, Player owner) const;
  void Eliminate(Player player, const std::string& reason);
  // Moves to the next player that can move, eliminating players that cannot,
  // and ends the game when a team has been eliminated.
  void AdvanceTurn();
  bool CheckTeamEliminated();
  std::string PieceString(const Piece& piece, bool visible) const;
  std::string BoardString(Player observer) const;
  SiguoJunqiStructContents StructContents(Player observer) const;

  std::array<Piece, kNumNodes> board_;
  bool in_deployment_ = true;
  int deploy_slot_ = 0;
  std::array<std::array<int, kNumPieceTypes>, kNumPlayers> remaining_ = {};
  // Deployment choices, kept for information states.
  std::array<std::array<int, kPiecesPerPlayer>, kNumPlayers> deployment_ = {};
  Player current_player_ = 0;
  std::array<bool, kNumPlayers> eliminated_ = {};
  std::array<bool, kNumPlayers> flag_revealed_ = {};
  int moves_since_collision_ = 0;
  int num_moves_ = 0;
  int winning_team_ = -1;
  bool is_draw_ = false;
  // Public events of the play phase (moves, battle outcomes, revelations and
  // eliminations), shared by all information states.
  std::vector<std::string> public_log_;
};

class SiguoJunqiGame : public Game {
 public:
  explicit SiguoJunqiGame(const GameParameters& params);

  int NumDistinctActions() const override { return kNumDistinctActions; }
  std::unique_ptr<State> NewInitialState() const override;
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  double MaxUtility() const override { return 1; }
  absl::optional<double> UtilitySum() const override { return 0; }
  std::vector<int> ObservationTensorShape() const override {
    return {kObservationTensorSize};
  }
  int MaxGameLength() const override {
    return kNumPlayers * kPiecesPerPlayer + max_moves_;
  }

  InformationMode information_mode() const { return information_mode_; }
  bool flying_bombs() const { return flying_bombs_; }
  bool mine_survives_attack() const { return mine_survives_attack_; }
  bool marshal_death_reveals_flag() const {
    return marshal_death_reveals_flag_;
  }
  int no_collision_draw_plies() const { return no_collision_draw_plies_; }
  int max_moves() const { return max_moves_; }

 private:
  InformationMode information_mode_;
  bool flying_bombs_;
  bool mine_survives_attack_;
  bool marshal_death_reveals_flag_;
  int no_collision_draw_plies_;
  int max_moves_;
};

}  // namespace siguo_junqi
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_SIGUO_JUNQI_H_
