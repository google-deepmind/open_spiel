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

#include "open_spiel/games/siguo_junqi/siguo_junqi.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace siguo_junqi {
namespace {

const GameType kGameType{
    /*short_name=*/"siguo_junqi",
    /*long_name=*/"Siguo Junqi (Four-State Military Chess)",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/kNumPlayers,
    /*min_num_players=*/kNumPlayers,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {{"information_mode", GameParameter(std::string("hidden"))},
     {"flying_bombs", GameParameter(false)},
     {"mine_survives_attack", GameParameter(true)},
     {"marshal_death_reveals_flag", GameParameter(true)},
     {"no_collision_draw_plies", GameParameter(kDefaultNoCollisionDrawPlies)},
     {"max_moves", GameParameter(kDefaultMaxMoves)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new SiguoJunqiGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

constexpr char kSeatLetters[] = "SENW";
constexpr char kPieceChars[] = "FMB123456789";
constexpr std::array<const char*, kNumPieceTypes> kPieceNames = {
    "Flag",    "Landmine", "Bomb",      "Engineer",     "Lieutenant", "Captain",
    "Major",   "Colonel",  "Brigadier", "MajorGeneral", "General",
    "FieldMarshal"};

// Directions of travel: +x (east), +y (north), -x (west), -y (south).
constexpr int kNumDirections = 4;
constexpr std::array<int, kNumDirections> kDx = {1, 0, -1, 0};
constexpr std::array<int, kNumDirections> kDy = {0, 1, 0, -1};

int Sign(int v) { return (v > 0) - (v < 0); }

int DirectionOf(int dx, int dy) {
  for (int d = 0; d < kNumDirections; ++d) {
    if (kDx[d] == Sign(dx) && kDy[d] == Sign(dy)) return d;
  }
  SpielFatalError(absl::StrCat("Not an axis direction: ", dx, ",", dy));
}

// Rotates (x, y) counterclockwise by `quarter_turns` * 90 degrees.
std::pair<int, int> Rotate(int x, int y, int quarter_turns) {
  for (int i = 0; i < quarter_turns; ++i) {
    int nx = -y;
    y = x;
    x = nx;
  }
  return {x, y};
}

bool IsCampPoint(int row, int col) {
  return ((row == 3 || row == 5) && (col == 2 || col == 4)) ||
         (row == 4 && col == 3);
}
bool IsHeadquartersPoint(int row, int col) {
  return row == 1 && (col == 2 || col == 4);
}
bool IsRailPoint(int row, int col) {
  return (row == 2 || row == 6 || col == 1 || col == 5) && row >= 2;
}

// A curved railway: arriving at `from` while travelling in `dir_in`, a piece
// may continue to `to`, where it then travels in `dir_out`.
struct Arc {
  int from;
  int dir_in;
  int to;
  int dir_out;
};

// Static board topology. Points are placed on integer coordinates with the
// origin at the centre of the hub, x pointing east and y pointing north.
// South's territory spans x in [-2, 2], y in [-8, -3]; the other territories
// are rotations of it. Hub points sit at x, y in {-2, 0, 2}.
struct Topology {
  std::array<NodeType, kNumNodes> type;
  std::array<bool, kNumNodes> rail = {};
  std::array<int, kNumNodes> x;
  std::array<int, kNumNodes> y;
  // One-step neighbours along roads, railways and arcs.
  std::array<std::vector<int>, kNumNodes> step;
  // Next railway point when travelling straight in each direction, or -1.
  std::array<std::array<int, kNumDirections>, kNumNodes> rail_next;
  std::vector<Arc> arcs;
  std::array<std::string, kNumNodes> name;
  std::map<std::string, int> node_by_name;

  Topology() {
    for (auto& next : rail_next) next.fill(-1);
    for (int zone = 0; zone < kNumPlayers; ++zone) {
      for (int row = 1; row <= kZoneRows; ++row) {
        for (int col = 1; col <= kZoneCols; ++col) {
          int node = ZoneNode(zone, row, col);
          type[node] = IsCampPoint(row, col)            ? NodeType::kCamp
                       : IsHeadquartersPoint(row, col) ? NodeType::kHeadquarters
                                                       : NodeType::kPost;
          rail[node] = IsRailPoint(row, col);
          std::tie(x[node], y[node]) = Rotate(col - 3, row - 9, zone);
          name[node] = absl::StrCat(std::string(1, kSeatLetters[zone]), row,
                                    col);
        }
      }
    }
    for (int row = 1; row <= kHubSize; ++row) {
      for (int col = 1; col <= kHubSize; ++col) {
        int node = HubNode(row, col);
        type[node] = NodeType::kPost;
        rail[node] = true;
        x[node] = 2 * (col - 2);
        y[node] = 2 * (row - 2);
        name[node] = absl::StrCat("C", row, col);
      }
    }
    for (int node = 0; node < kNumNodes; ++node) {
      node_by_name[name[node]] = node;
    }

    for (int zone = 0; zone < kNumPlayers; ++zone) {
      for (int row = 1; row <= kZoneRows; ++row) {
        for (int col = 1; col <= kZoneCols; ++col) {
          int node = ZoneNode(zone, row, col);
          // Roads between orthogonal neighbours; railway where both ends are
          // on the same railway line.
          if (col < kZoneCols) {
            int right = ZoneNode(zone, row, col + 1);
            AddStep(node, right);
            if (row == 2 || row == 6) AddRail(node, right);
          }
          if (row < kZoneRows) {
            int forward = ZoneNode(zone, row + 1, col);
            AddStep(node, forward);
            if ((col == 1 || col == 5) && row >= 2) AddRail(node, forward);
          }
          // Diagonal roads out of camps (camp-to-camp diagonals added once).
          if (type[node] == NodeType::kCamp) {
            for (int dr : {-1, 1}) {
              for (int dc : {-1, 1}) {
                int other = ZoneNode(zone, row + dr, col + dc);
                if (type[other] != NodeType::kCamp || other > node) {
                  AddStep(node, other);
                }
              }
            }
          }
        }
      }
      // Front-row columns 1, 3 and 5 connect straight ahead into the hub.
      for (int col : {1, 3, 5}) {
        int front = ZoneNode(zone, kZoneRows, col);
        AddRail(front, NearestHubPoint(front));
      }
      // Arc from this zone's front-row column 1 to the front-row column 5 of
      // the zone on the owner's left, curving around the hub's corner.
      int p = ZoneNode(zone, kZoneRows, 1);
      int q = ZoneNode((zone + kNumPlayers - 1) % kNumPlayers, kZoneRows, 5);
      int cx, cy;
      std::tie(cx, cy) = Rotate(-3, -3, zone);
      AddStep(p, q);
      arcs.push_back({p, DirectionOf(x[q] - cx, y[q] - cy), q,
                      DirectionOf(cx - x[p], cy - y[p])});
      arcs.push_back({q, DirectionOf(x[p] - cx, y[p] - cy), p,
                      DirectionOf(cx - x[q], cy - y[q])});
    }
    for (int row = 1; row <= kHubSize; ++row) {
      for (int col = 1; col <= kHubSize; ++col) {
        if (col < kHubSize) {
          AddRail(HubNode(row, col), HubNode(row, col + 1));
        }
        if (row < kHubSize) {
          AddRail(HubNode(row, col), HubNode(row + 1, col));
        }
      }
    }
  }

  int NearestHubPoint(int node) const {
    for (int h = kHubStart; h < kNumNodes; ++h) {
      if (std::abs(x[h] - x[node]) + std::abs(y[h] - y[node]) == 1) return h;
    }
    SpielFatalError(absl::StrCat("No hub point next to ", name[node]));
  }

  void AddStep(int a, int b) {
    step[a].push_back(b);
    step[b].push_back(a);
  }

  void AddRail(int a, int b) {
    if (std::find(step[a].begin(), step[a].end(), b) == step[a].end()) {
      AddStep(a, b);
    }
    rail_next[a][DirectionOf(x[b] - x[a], y[b] - y[a])] = b;
    rail_next[b][DirectionOf(x[a] - x[b], y[a] - y[b])] = a;
  }
};

const Topology& GetTopology() {
  static const Topology* topology = new Topology();
  return *topology;
}

// The 25 deployment points of a territory, from the back row to the front.
std::vector<std::pair<int, int>> MakeDeploymentPoints() {
  std::vector<std::pair<int, int>> points;
  for (int row = 1; row <= kZoneRows; ++row) {
    for (int col = 1; col <= kZoneCols; ++col) {
      if (!IsCampPoint(row, col)) points.push_back({row, col});
    }
  }
  SPIEL_CHECK_EQ(points.size(), kPiecesPerPlayer);
  return points;
}

const std::vector<std::pair<int, int>>& DeploymentPoints() {
  static const auto* points =
      new std::vector<std::pair<int, int>>(MakeDeploymentPoints());
  return *points;
}

bool AllowedAt(int type, int row, int col) {
  switch (type) {
    case kFlag:
      return IsHeadquartersPoint(row, col);
    case kLandmine:
      return row <= 2;
    case kBomb:
      return row < kZoneRows;
    default:
      return true;
  }
}

std::string PlayerName(Player player) {
  return std::string(1, kSeatLetters[player]);
}

InformationMode ParseInformationMode(const std::string& mode) {
  if (mode == "hidden") return InformationMode::kHidden;
  if (mode == "partner_visible") return InformationMode::kPartnerVisible;
  if (mode == "all_visible") return InformationMode::kAllVisible;
  SpielFatalError(absl::StrCat("Unknown information_mode: ", mode));
}

}  // namespace

NodeType NodeTypeOf(int node) { return GetTopology().type[node]; }

bool IsRailNode(int node) { return GetTopology().rail[node]; }

int ZoneOf(int node) {
  return node < kHubStart ? node / kZoneSize : kInvalidPlayer;
}

int ZoneNode(int zone, int row, int col) {
  return zone * kZoneSize + (row - 1) * kZoneCols + (col - 1);
}

int HubNode(int row, int col) {
  return kHubStart + (row - 1) * kHubSize + (col - 1);
}

std::string NodeName(int node) { return GetTopology().name[node]; }

std::string PieceTypeToString(PieceType type) {
  SPIEL_CHECK_GE(type, 0);
  SPIEL_CHECK_LT(type, kNumPieceTypes);
  return kPieceNames[type];
}

BattleResult ResolveBattle(PieceType attacker, PieceType defender,
                           bool mine_survives_attack) {
  SPIEL_CHECK_GE(attacker, kBomb);
  if (attacker == kBomb || defender == kBomb) {
    return BattleResult::kBothRemoved;
  }
  if (defender == kLandmine) {
    if (attacker == kEngineer) return BattleResult::kAttackerWins;
    return mine_survives_attack ? BattleResult::kDefenderWins
                                : BattleResult::kBothRemoved;
  }
  if (defender == kFlag || attacker > defender) {
    return BattleResult::kAttackerWins;
  }
  return attacker == defender ? BattleResult::kBothRemoved
                              : BattleResult::kDefenderWins;
}

SiguoJunqiState::SiguoJunqiState(std::shared_ptr<const Game> game)
    : State(game) {
  for (auto& counts : remaining_) {
    std::copy(kPieceCounts.begin(), kPieceCounts.end(), counts.begin());
  }
  for (auto& deployment : deployment_) deployment.fill(kNoPiece);
}

const SiguoJunqiGame& SiguoJunqiState::ParentGame() const {
  return static_cast<const SiguoJunqiGame&>(*game_);
}

Player SiguoJunqiState::CurrentPlayer() const {
  return IsTerminal() ? kTerminalPlayerId : current_player_;
}

bool SiguoJunqiState::IsTerminal() const {
  return winning_team_ >= 0 || is_draw_;
}

std::vector<double> SiguoJunqiState::Returns() const {
  std::vector<double> returns(kNumPlayers, 0.0);
  if (winning_team_ >= 0) {
    for (Player p = 0; p < kNumPlayers; ++p) {
      returns[p] = p % 2 == winning_team_ ? 1.0 : -1.0;
    }
  }
  return returns;
}

std::vector<Action> SiguoJunqiState::LegalActions() const {
  if (IsTerminal()) return {};
  if (in_deployment_) return LegalDeployActions();
  return LegalMoveActions(current_player_);
}

std::vector<Action> SiguoJunqiState::LegalDeployActions() const {
  const auto& points = DeploymentPoints();
  int row = points[deploy_slot_].first;
  int col = points[deploy_slot_].second;
  // Points left after this one, by the most restrictive category they accept.
  int headquarters = 0;
  int back = 0;
  int non_front = 0;
  for (int slot = deploy_slot_ + 1; slot < kPiecesPerPlayer; ++slot) {
    const auto& [point_row, point_col] = points[slot];
    headquarters += IsHeadquartersPoint(point_row, point_col);
    back += point_row <= 2;
    non_front += point_row < kZoneRows;
  }
  std::vector<Action> actions;
  for (int type = 0; type < kNumPieceTypes; ++type) {
    std::array<int, kNumPieceTypes> counts = remaining_[current_player_];
    if (counts[type] == 0 || !AllowedAt(type, row, col)) continue;
    --counts[type];
    // The categories are nested (headquarters within the back two rows, those
    // within the non-front rows), so these conditions guarantee that the
    // remaining pieces can still be placed.
    int needs_headquarters = counts[kFlag];
    int needs_back = needs_headquarters + counts[kLandmine];
    int needs_non_front = needs_back + counts[kBomb];
    if (needs_headquarters <= headquarters && needs_back <= back &&
        needs_non_front <= non_front) {
      actions.push_back(type);
    }
  }
  return actions;
}

bool SiguoJunqiState::CanMove(int node) const {
  const Piece& piece = board_[node];
  return !piece.empty() && piece.type != kFlag && piece.type != kLandmine &&
         NodeTypeOf(node) != NodeType::kHeadquarters;
}

bool SiguoJunqiState::CanEnter(int mover, int node) const {
  if (node == mover) return false;
  const Piece& target = board_[node];
  if (target.empty()) return true;
  return !IsAlly(board_[mover].owner, target.owner) &&
         NodeTypeOf(node) != NodeType::kCamp;
}

std::vector<int> SiguoJunqiState::Destinations(int node) const {
  std::vector<int> destinations;
  if (!CanMove(node)) return destinations;
  const Topology& topology = GetTopology();
  for (int next : topology.step[node]) {
    if (CanEnter(node, next)) destinations.push_back(next);
  }
  if (topology.rail[node]) {
    PieceType type = board_[node].type;
    bool turns = type == kEngineer ||
                 (type == kBomb && ParentGame().flying_bombs());
    // Search over (point, direction of travel); turning pieces ignore the
    // direction. Every reached point ends the move if it is occupied.
    std::vector<std::pair<int, int>> frontier;
    std::vector<std::array<bool, kNumDirections>> visited(kNumNodes);
    auto push = [&](int next, int dir) {
      if (next == node) return;
      int key = turns ? 0 : dir;
      if (visited[next][key]) return;
      visited[next][key] = true;
      if (CanEnter(node, next)) destinations.push_back(next);
      if (board_[next].empty()) frontier.push_back({next, dir});
    };
    auto expand = [&](int from, int dir, bool any_direction) {
      for (int d = 0; d < kNumDirections; ++d) {
        if (!any_direction && d != dir) continue;
        if (topology.rail_next[from][d] >= 0) {
          push(topology.rail_next[from][d], d);
        }
      }
      for (const Arc& arc : topology.arcs) {
        if (arc.from == from && (any_direction || arc.dir_in == dir)) {
          push(arc.to, arc.dir_out);
        }
      }
    };
    expand(node, 0, /*any_direction=*/true);
    while (!frontier.empty()) {
      auto [from, dir] = frontier.back();
      frontier.pop_back();
      expand(from, dir, /*any_direction=*/turns);
    }
  }
  std::sort(destinations.begin(), destinations.end());
  destinations.erase(std::unique(destinations.begin(), destinations.end()),
                     destinations.end());
  return destinations;
}

std::vector<Action> SiguoJunqiState::LegalMoveActions(Player player) const {
  std::vector<Action> actions;
  for (int from = 0; from < kNumNodes; ++from) {
    if (board_[from].owner != player) continue;
    for (int to : Destinations(from)) {
      actions.push_back(kNumDeployActions + from * kNumNodes + to);
    }
  }
  return actions;
}

void SiguoJunqiState::DoApplyAction(Action action) {
  if (in_deployment_) {
    ApplyDeployAction(action);
  } else {
    SPIEL_CHECK_GE(action, kNumDeployActions);
    action -= kNumDeployActions;
    ApplyMoveAction(action / kNumNodes, action % kNumNodes);
  }
}

void SiguoJunqiState::ApplyDeployAction(Action action) {
  SPIEL_CHECK_GE(action, 0);
  SPIEL_CHECK_LT(action, kNumDeployActions);
  const auto& point = DeploymentPoints()[deploy_slot_];
  int node = ZoneNode(current_player_, point.first, point.second);
  board_[node] = {current_player_, static_cast<PieceType>(action)};
  --remaining_[current_player_][action];
  deployment_[current_player_][deploy_slot_] = action;
  if (++deploy_slot_ < kPiecesPerPlayer) return;
  deploy_slot_ = 0;
  if (++current_player_ < kNumPlayers) return;
  in_deployment_ = false;
  current_player_ = kNumPlayers - 1;
  AdvanceTurn();
}

void SiguoJunqiState::ApplyMoveAction(int from, int to) {
  const SiguoJunqiGame& game = ParentGame();
  Piece mover = board_[from];
  Piece target = board_[to];
  ++num_moves_;
  std::string event = absl::StrCat(PlayerName(mover.owner), " ",
                                   NodeName(from), "-", NodeName(to));
  board_[from] = Piece();
  std::vector<Piece> removed;
  if (target.empty()) {
    board_[to] = mover;
    ++moves_since_collision_;
  } else {
    moves_since_collision_ = 0;
    switch (ResolveBattle(mover.type, target.type,
                          game.mine_survives_attack())) {
      case BattleResult::kAttackerWins:
        board_[to] = mover;
        removed.push_back(target);
        absl::StrAppend(&event, " attacks ", PlayerName(target.owner),
                        ": attacker wins");
        break;
      case BattleResult::kDefenderWins:
        removed.push_back(mover);
        absl::StrAppend(&event, " attacks ", PlayerName(target.owner),
                        ": defender wins");
        break;
      case BattleResult::kBothRemoved:
        board_[to] = Piece();
        removed.push_back(mover);
        removed.push_back(target);
        absl::StrAppend(&event, " attacks ", PlayerName(target.owner),
                        ": both removed");
        break;
    }
  }
  public_log_.push_back(event);

  for (const Piece& piece : removed) {
    if (piece.type == kFieldMarshal && game.marshal_death_reveals_flag() &&
        !flag_revealed_[piece.owner]) {
      flag_revealed_[piece.owner] = true;
      for (int node = 0; node < kNumNodes; ++node) {
        if (board_[node].owner == piece.owner && board_[node].type == kFlag) {
          public_log_.push_back(absl::StrCat(PlayerName(piece.owner),
                                             " flag revealed at ",
                                             NodeName(node)));
        }
      }
    }
  }
  if (target.type == kFlag) {
    Eliminate(target.owner, "flag captured");
    if (CheckTeamEliminated()) return;
  }

  int draw_plies = game.no_collision_draw_plies();
  if ((draw_plies > 0 && moves_since_collision_ >= draw_plies) ||
      num_moves_ >= game.max_moves()) {
    is_draw_ = true;
    return;
  }
  AdvanceTurn();
}

void SiguoJunqiState::Eliminate(Player player, const std::string& reason) {
  for (int node = 0; node < kNumNodes; ++node) {
    if (board_[node].owner == player) board_[node] = Piece();
  }
  eliminated_[player] = true;
  public_log_.push_back(
      absl::StrCat(PlayerName(player), " eliminated (", reason, ")"));
}

bool SiguoJunqiState::CheckTeamEliminated() {
  for (int team = 0; team < 2; ++team) {
    if (eliminated_[team] && eliminated_[team + 2]) {
      winning_team_ = 1 - team;
      return true;
    }
  }
  return false;
}

void SiguoJunqiState::AdvanceTurn() {
  Player player = current_player_;
  // Each iteration either finds a player to move or eliminates one, so two
  // laps around the table always suffice.
  for (int i = 0; i < 2 * kNumPlayers; ++i) {
    player = (player + 1) % kNumPlayers;
    if (eliminated_[player]) continue;
    if (LegalMoveActions(player).empty()) {
      Eliminate(player, "no legal move");
      if (CheckTeamEliminated()) return;
      continue;
    }
    current_player_ = player;
    return;
  }
  SpielFatalError("No player can move but no team has been eliminated.");
}

void SiguoJunqiState::SetBoardForTesting(
    const std::array<Piece, kNumNodes>& board, Player player) {
  board_ = board;
  in_deployment_ = false;
  deploy_slot_ = 0;
  for (auto& counts : remaining_) counts.fill(0);
  current_player_ = player;
}

bool SiguoJunqiState::SeesTypesOf(Player observer, Player owner) const {
  if (observer == owner) return true;
  if (in_deployment_) return false;
  switch (ParentGame().information_mode()) {
    case InformationMode::kHidden:
      return false;
    case InformationMode::kPartnerVisible:
      return IsAlly(observer, owner);
    case InformationMode::kAllVisible:
      return true;
  }
  return false;
}

std::string SiguoJunqiState::ActionToString(Player player,
                                            Action action_id) const {
  if (action_id < kNumDeployActions) {
    std::string str = absl::StrCat(
        "Deploy ", PieceTypeToString(static_cast<PieceType>(action_id)));
    if (in_deployment_ && player == current_player_) {
      const auto& point = DeploymentPoints()[deploy_slot_];
      absl::StrAppend(&str, " at ",
                      NodeName(ZoneNode(player, point.first, point.second)));
    }
    return str;
  }
  action_id -= kNumDeployActions;
  return absl::StrCat(NodeName(action_id / kNumNodes), "-",
                      NodeName(action_id % kNumNodes));
}

std::string SiguoJunqiState::PieceString(const Piece& piece,
                                         bool visible) const {
  bool shown = visible || (piece.type == kFlag && flag_revealed_[piece.owner]);
  return absl::StrCat(PlayerName(piece.owner),
                      std::string(1, shown ? kPieceChars[piece.type] : '?'));
}

std::string SiguoJunqiState::BoardString(Player observer) const {
  const Topology& topology = GetTopology();
  constexpr int kExtent = 8;
  constexpr int kWidth = 2 * kExtent + 1;
  std::vector<std::string> grid(kWidth, std::string(3 * kWidth, ' '));
  for (int node = 0; node < kNumNodes; ++node) {
    std::string cell;
    const Piece& piece = board_[node];
    if (!piece.empty()) {
      cell = PieceString(piece, observer == kInvalidPlayer ||
                                    SeesTypesOf(observer, piece.owner));
    } else if (NodeTypeOf(node) == NodeType::kCamp) {
      cell = "()";
    } else if (NodeTypeOf(node) == NodeType::kHeadquarters) {
      cell = "[]";
    } else {
      cell = topology.rail[node] ? "==" : "--";
    }
    int row = kExtent - topology.y[node];
    int col = topology.x[node] + kExtent;
    grid[row].replace(3 * col, 2, cell);
  }
  for (auto& line : grid) {
    line.erase(line.find_last_not_of(' ') + 1);
  }
  grid.erase(std::remove(grid.begin(), grid.end(), ""), grid.end());
  return absl::StrJoin(grid, "\n");
}

std::string SiguoJunqiState::ToString() const {
  std::string str = BoardString(kInvalidPlayer);
  absl::StrAppend(&str, "\nPhase: ", in_deployment_ ? "deployment" : "play");
  if (IsTerminal()) {
    absl::StrAppend(&str, "\nTerminal: ",
                    is_draw_ ? "draw"
                             : absl::StrCat("team ", winning_team_, " wins"));
  } else {
    absl::StrAppend(&str, "\nCurrent player: ", PlayerName(current_player_));
  }
  absl::StrAppend(&str, "\nMoves since collision: ", moves_since_collision_);
  return str;
}

std::string SiguoJunqiState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, kNumPlayers);
  std::string str = BoardString(player);
  absl::StrAppend(&str, "\nObserver: ", PlayerName(player));
  absl::StrAppend(&str, "\nPhase: ", in_deployment_ ? "deployment" : "play");
  if (!IsTerminal()) {
    absl::StrAppend(&str, "\nCurrent player: ", PlayerName(current_player_));
  }
  absl::StrAppend(&str, "\nMoves since collision: ", moves_since_collision_);
  return str;
}

std::string SiguoJunqiState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, kNumPlayers);
  std::string str = absl::StrCat("Observer: ", PlayerName(player));
  for (Player owner = 0; owner < kNumPlayers; ++owner) {
    if (!SeesTypesOf(player, owner)) continue;
    int placed = !in_deployment_ || owner < current_player_ ? kPiecesPerPlayer
                 : owner == current_player_                ? deploy_slot_
                                                           : 0;
    std::string pieces;
    for (int slot = 0; slot < placed; ++slot) {
      pieces.push_back(kPieceChars[deployment_[owner][slot]]);
    }
    absl::StrAppend(&str, "\n", PlayerName(owner), " deployment: ", pieces);
  }
  for (const std::string& event : public_log_) {
    absl::StrAppend(&str, "\n", event);
  }
  return str;
}

void SiguoJunqiState::ObservationTensor(Player player,
                                        absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, kNumPlayers);
  SPIEL_CHECK_EQ(values.size(), kObservationTensorSize);
  std::fill(values.begin(), values.end(), 0.0f);
  for (int node = 0; node < kNumNodes; ++node) {
    const Piece& piece = board_[node];
    if (piece.empty()) continue;
    int seat = (piece.owner - player + kNumPlayers) % kNumPlayers;
    bool visible = SeesTypesOf(player, piece.owner) ||
                   (piece.type == kFlag && flag_revealed_[piece.owner]);
    int plane = seat * kPlanesPerSeat + (visible ? piece.type : kNumPieceTypes);
    values[plane * kNumNodes + node] = 1.0f;
  }
  int offset = kNumBoardPlanes * kNumNodes;
  values[offset++] = in_deployment_ ? 1.0f : 0.0f;
  if (!IsTerminal()) {
    values[offset + (current_player_ - player + kNumPlayers) % kNumPlayers] =
        1.0f;
  }
  offset += kNumPlayers;
  for (Player p = 0; p < kNumPlayers; ++p) {
    if (eliminated_[p]) {
      values[offset + (p - player + kNumPlayers) % kNumPlayers] = 1.0f;
    }
  }
  offset += kNumPlayers;
  int draw_plies = ParentGame().no_collision_draw_plies();
  if (draw_plies > 0) {
    values[offset] =
        std::min(1.0f, static_cast<float>(moves_since_collision_) / draw_plies);
  }
}

std::unique_ptr<State> SiguoJunqiState::Clone() const {
  return std::unique_ptr<State>(new SiguoJunqiState(*this));
}

SiguoJunqiStructContents SiguoJunqiState::StructContents(
    Player observer) const {
  SiguoJunqiStructContents contents;
  contents.phase = IsTerminal()      ? "terminal"
                   : in_deployment_ ? "deployment"
                                    : "play";
  contents.current_player = CurrentPlayer();
  for (int node = 0; node < kNumNodes; ++node) {
    const Piece& piece = board_[node];
    if (piece.empty()) continue;
    bool visible = observer == kInvalidPlayer ||
                   SeesTypesOf(observer, piece.owner) ||
                   (piece.type == kFlag && flag_revealed_[piece.owner]);
    contents.board[NodeName(node)] =
        absl::StrCat(PlayerName(piece.owner), ":",
                     visible ? PieceTypeToString(piece.type) : "?");
  }
  contents.eliminated.assign(eliminated_.begin(), eliminated_.end());
  contents.flag_revealed.assign(flag_revealed_.begin(), flag_revealed_.end());
  contents.moves_since_collision = moves_since_collision_;
  contents.winning_team = winning_team_;
  return contents;
}

std::unique_ptr<StateStruct> SiguoJunqiState::ToStruct() const {
  auto state_struct = std::make_unique<SiguoJunqiStateStruct>();
  static_cast<SiguoJunqiStructContents&>(*state_struct) =
      StructContents(kInvalidPlayer);
  return state_struct;
}

std::unique_ptr<ObservationStruct> SiguoJunqiState::ToObservationStruct(
    Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, kNumPlayers);
  auto observation_struct = std::make_unique<SiguoJunqiObservationStruct>();
  static_cast<SiguoJunqiStructContents&>(*observation_struct) =
      StructContents(player);
  return observation_struct;
}

std::unique_ptr<ActionStruct> SiguoJunqiState::ActionToStruct(
    Player player, Action action_id) const {
  auto action_struct = std::make_unique<SiguoJunqiActionStruct>();
  if (action_id < kNumDeployActions) {
    action_struct->piece =
        PieceTypeToString(static_cast<PieceType>(action_id));
  } else {
    action_id -= kNumDeployActions;
    action_struct->from = NodeName(action_id / kNumNodes);
    action_struct->to = NodeName(action_id % kNumNodes);
  }
  return action_struct;
}

std::vector<Action> SiguoJunqiState::StructToActions(
    const ActionStruct& action_struct) const {
  const auto* action = SafeActionCast<SiguoJunqiActionStruct>(action_struct);
  if (!action->piece.empty()) {
    for (int type = 0; type < kNumPieceTypes; ++type) {
      if (action->piece == kPieceNames[type]) return {type};
    }
    SpielFatalError(absl::StrCat("Unknown piece: ", action->piece));
  }
  const auto& node_by_name = GetTopology().node_by_name;
  auto from = node_by_name.find(action->from);
  auto to = node_by_name.find(action->to);
  SPIEL_CHECK_TRUE(from != node_by_name.end());
  SPIEL_CHECK_TRUE(to != node_by_name.end());
  return {kNumDeployActions + from->second * kNumNodes + to->second};
}

SiguoJunqiGame::SiguoJunqiGame(const GameParameters& params)
    : Game(kGameType, params),
      information_mode_(ParseInformationMode(
          ParameterValue<std::string>("information_mode"))),
      flying_bombs_(ParameterValue<bool>("flying_bombs")),
      mine_survives_attack_(ParameterValue<bool>("mine_survives_attack")),
      marshal_death_reveals_flag_(
          ParameterValue<bool>("marshal_death_reveals_flag")),
      no_collision_draw_plies_(ParameterValue<int>("no_collision_draw_plies")),
      max_moves_(ParameterValue<int>("max_moves")) {
  SPIEL_CHECK_GE(no_collision_draw_plies_, 0);
  SPIEL_CHECK_GT(max_moves_, 0);
}

std::unique_ptr<State> SiguoJunqiGame::NewInitialState() const {
  return std::unique_ptr<State>(new SiguoJunqiState(shared_from_this()));
}

}  // namespace siguo_junqi
}  // namespace open_spiel
