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

#include "open_spiel/games/catan/catan.h"

#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace catan {
namespace {

constexpr int kDefaultPlayers = 4;
constexpr int kDefaultVictoryPoints = 10;
constexpr int kDefaultMaxTradeOffers = 5;
constexpr int kDefaultMaxCounterSteps = 3;
constexpr int kDefaultMaxTurns = 1000;

const GameType kGameType{
    /*short_name=*/"catan",
    /*long_name=*/"Catan",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/kMaxPlayers,
    /*min_num_players=*/kMinPlayers,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {{"players", GameParameter(kDefaultPlayers)},
     {"victory_points", GameParameter(kDefaultVictoryPoints)},
     {"board", GameParameter("random")},
     {"max_trade_offers", GameParameter(kDefaultMaxTradeOffers)},
     {"max_counter_steps", GameParameter(kDefaultMaxCounterSteps)},
     {"max_turns", GameParameter(kDefaultMaxTurns)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new CatanGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
RegisterSingleTensorObserver single_tensor(kGameType.short_name);

using Cards = std::array<int, kNumResources>;
constexpr Cards kRoadCost = {1, 1, 0, 0, 0};
constexpr Cards kSettlementCost = {1, 1, 1, 1, 0};
constexpr Cards kCityCost = {0, 0, 0, 2, 3};
constexpr Cards kDevCardCost = {0, 0, 1, 1, 1};

constexpr std::array<int, kNumTerrains> kTerrainCounts = {4, 3, 4, 4, 3, 1};
constexpr std::array<int, kNumPortTypes> kPortCounts = {1, 1, 1, 1, 1, 4};
constexpr std::array<int, kNumDevCardTypes> kDevCardCounts = {14, 5, 2, 2, 2};
constexpr std::array<int, kNumDiceSums> kDiceWays = {1, 2, 3, 4, 5, 6,
                                                     5, 4, 3, 2, 1};

// Number tokens in spiral order.
constexpr std::array<int, kNumHexes - 1> kNumberTokens = {
    5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11};

// Ports sit on these coastal edges, counted counterclockwise from the top,
// 3 or 4 edges apart around the 30-edge coast: as on the frame, three take the
// outer edges of alternate corner hexes and six sit on the side hexes.
constexpr std::array<int, kNumPorts> kPortCoastalEdges = {2,  5,  9,  12, 15,
                                                          19, 22, 25, 29};

// The layout used with board=fixed, in hex order.
constexpr std::array<int, kNumHexes> kFixedTerrain = {
    kMountains, kPasture, kForest, kFields, kHills,     kPasture, kHills,
    kFields,    kForest,  kDesert, kForest, kMountains, kForest,  kMountains,
    kFields,    kPasture, kHills,  kFields, kPasture};
constexpr std::array<int, kNumPorts> kFixedPorts = {
    kGenericPort, kWool,        kGenericPort, kGenericPort, kBrick,
    kLumber,      kGenericPort, kGrain,       kOre};

constexpr std::array<const char*, kNumResources> kResourceNames = {
    "Lumber", "Brick", "Wool", "Grain", "Ore"};
constexpr std::array<const char*, kNumTerrains> kTerrainNames = {
    "Forest", "Hills", "Pasture", "Fields", "Mountains", "Desert"};
constexpr std::array<const char*, kNumDevCardTypes> kDevCardNames = {
    "Knight", "Victory Point", "Road Building", "Year of Plenty", "Monopoly"};
constexpr std::array<const char*, kNumPhases> kPhaseNames = {
    // The board and setup.
    "BoardTerrain", "BoardPorts", "BoardSpiral", "FirstPlayer",
    "SetupSettlement", "SetupRoad",
    // A turn.
    "PreRoll", "Roll", "Discard", "MoveRobber", "ChooseVictim", "Steal", "Main",
    "RoadBuilding", "ComposeTrade", "TradeResponse", "TradeDecide",
    "DrawDevCard", "GameOver"};
constexpr std::array<const char*, kNumTradeStatuses> kTradeStatusNames = {
    "none", "pending", "accepted", "declined", "countered"};

constexpr int kMaxCardsInGame = kNumResources * kBankResourceCount;
// Upper bound on the player actions in a turn, trading aside: every action
// that spends cards (at most the cards in the game, twice with a monopoly),
// discarding every card, and a few more for the roll, the robber, a
// development card and its follow-ups, and ending the turn.
constexpr int kMaxActionsPerTurn = 3 * kMaxCardsInGame + 25;
// Building an offer or counter: at most every card in the game, then propose
// or withdraw.
constexpr int kMaxComposeActions = kMaxCardsInGame + 1;

// Observation tensor layout. Seats are numbered from the observer, seat 0.
constexpr int kNumSeatFeatures = 11;
constexpr int kNumTradeSeatFeatures = kNumTradeStatuses + 2 * kNumResources + 1;
constexpr int kObservationTensorSize =
    kNumHexes * (kNumTerrains + kNumDiceSums + 1) +     // Hexes and robber.
    kNumVertices * (kNumPortTypes + 2 * kMaxPlayers) +  // Ports, buildings.
    kNumEdges * kMaxPlayers +                           // Roads.
    kNumResources + 2 * kNumDevCardTypes +              // Own cards.
    kMaxPlayers * kNumSeatFeatures +                    // Every player.
    kNumResources + 1 +                                 // Bank.
    kNumPhases + 4 * kMaxPlayers + kNumDiceSums + 3 +   // Turn.
    2 * kMaxPlayers + 4 * kNumResources + 4 +           // Bundle being built.
    kMaxPlayers * kNumTradeSeatFeatures +               // Trades.
    kNumVertices;                                       // Setup settlement.

// Angle in degrees, counterclockwise from east, of a point in the integer
// layout coordinates of BuildTopology.
double Angle(double x, double y) {
  constexpr double kPi = 3.14159265358979323846;
  return std::atan2(-y / 2, x * std::sqrt(3.0) / 2) * 180 / kPi;
}

// Position of an angle counterclockwise from `start`, in [0, 360). The
// epsilon keeps a point exactly at `start` at 0 rather than just under 360.
double AngleFrom(double angle, double start) {
  return std::fmod(angle - start + 720 + 1e-6, 360);
}

// Hex (q, r) in axial coordinates has its center at x = 2q + r, y = 3r, in
// units of half a hex width and half a side, so corners are integer points.
Topology BuildTopology() {
  Topology t;
  struct Hex {
    int q, r, x, y;
  };
  std::vector<Hex> hexes;
  for (int r = -2; r <= 2; ++r) {
    for (int q = -2; q <= 2; ++q) {
      if (std::abs(q + r) <= 2) hexes.push_back({q, r, 2 * q + r, 3 * r});
    }
  }
  SPIEL_CHECK_EQ(hexes.size(), kNumHexes);

  constexpr std::array<int, kNumHexCorners> kCornerDx = {0, 1, 1, 0, -1, -1};
  constexpr std::array<int, kNumHexCorners> kCornerDy = {-2, -1, 1, 2, 1, -1};
  std::vector<std::pair<int, int>> corners;  // (y, x), sorted row by row.
  for (const Hex& hex : hexes) {
    for (int c = 0; c < kNumHexCorners; ++c) {
      corners.push_back({hex.y + kCornerDy[c], hex.x + kCornerDx[c]});
    }
  }
  std::sort(corners.begin(), corners.end());
  corners.erase(std::unique(corners.begin(), corners.end()), corners.end());
  SPIEL_CHECK_EQ(corners.size(), kNumVertices);
  auto vertex_at = [&corners](int y, int x) {
    return static_cast<int>(
        std::lower_bound(corners.begin(), corners.end(), std::make_pair(y, x)) -
        corners.begin());
  };

  std::vector<std::pair<int, int>> edges;
  for (int h = 0; h < kNumHexes; ++h) {
    for (int c = 0; c < kNumHexCorners; ++c) {
      t.hex_vertices[h][c] =
          vertex_at(hexes[h].y + kCornerDy[c], hexes[h].x + kCornerDx[c]);
    }
    for (int c = 0; c < kNumHexCorners; ++c) {
      int a = t.hex_vertices[h][c];
      int b = t.hex_vertices[h][(c + 1) % kNumHexCorners];
      edges.push_back({std::min(a, b), std::max(a, b)});
    }
  }
  std::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
  SPIEL_CHECK_EQ(edges.size(), kNumEdges);
  // Number edges row by row by their midpoints.
  auto midpoint = [&corners](const std::pair<int, int>& e) {
    return std::make_pair(corners[e.first].first + corners[e.second].first,
                          corners[e.first].second + corners[e.second].second);
  };
  std::sort(edges.begin(), edges.end(),
            [&midpoint](const auto& a, const auto& b) {
              return midpoint(a) < midpoint(b);
            });

  for (int h = 0; h < kNumHexes; ++h) {
    for (int v : t.hex_vertices[h]) t.vertex_hexes[v].push_back(h);
  }
  for (int e = 0; e < kNumEdges; ++e) {
    auto [a, b] = edges[e];
    t.edge_vertices[e] = {a, b};
    t.vertex_edges[a].push_back(e);
    t.vertex_edges[b].push_back(e);
    t.vertex_neighbors[a].push_back(b);
    t.vertex_neighbors[b].push_back(a);
  }
  for (auto& neighbors : t.vertex_neighbors) {
    std::sort(neighbors.begin(), neighbors.end());
  }

  // Coastal edges border a single hex. Ports go on them, counterclockwise from
  // the top.
  std::vector<int> coast;
  for (int e = 0; e < kNumEdges; ++e) {
    auto [a, b] = edges[e];
    int hexes_on_edge = 0;
    for (int h : t.vertex_hexes[a]) {
      for (int g : t.vertex_hexes[b]) hexes_on_edge += h == g;
    }
    if (hexes_on_edge == 1) coast.push_back(e);
  }
  SPIEL_CHECK_EQ(coast.size(), 30);
  auto edge_angle = [&](int e) {
    auto [y, x] = midpoint(edges[e]);
    return AngleFrom(Angle(x / 2.0, y / 2.0), 90);
  };
  std::sort(coast.begin(), coast.end(), [&edge_angle](int a, int b) {
    return edge_angle(a) < edge_angle(b);
  });
  t.vertex_port.fill(-1);
  for (int i = 0; i < kNumPorts; ++i) {
    t.port_vertices[i] = t.edge_vertices[coast[kPortCoastalEdges[i]]];
    for (int v : t.port_vertices[i]) t.vertex_port[v] = i;
  }

  // Number tokens go counterclockwise ring by ring toward the center. Spiral s
  // starts on the outer hex at 120 + 60s degrees, the top-left one for s = 0.
  auto ring = [&hexes](int h) {
    return std::max({std::abs(hexes[h].q), std::abs(hexes[h].r),
                     std::abs(hexes[h].q + hexes[h].r)});
  };
  for (int s = 0; s < kNumSpiralStarts; ++s) {
    auto hex_angle = [&hexes, s](int h) {
      return AngleFrom(Angle(hexes[h].x, hexes[h].y), 120 + 60 * s);
    };
    std::vector<int> order(kNumHexes);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      if (ring(a) != ring(b)) return ring(a) > ring(b);
      return hex_angle(a) < hex_angle(b);
    });
    std::copy(order.begin(), order.end(), t.spirals[s].begin());
  }
  return t;
}

// Length of the longest trail of `player`'s roads leaving `vertex` without
// reusing an edge or passing through another player's building.
int Trail(const Topology& topo, const std::array<Player, kNumEdges>& roads,
          const std::array<Player, kNumVertices>& buildings, Player player,
          int vertex, std::bitset<kNumEdges>* used) {
  int best = 0;
  for (int e : topo.vertex_edges[vertex]) {
    if (roads[e] != player || (*used)[e]) continue;
    int next = topo.edge_vertices[e][0] == vertex ? topo.edge_vertices[e][1]
                                                  : topo.edge_vertices[e][0];
    used->set(e);
    int length = 1;
    if (buildings[next] == kInvalidPlayer || buildings[next] == player) {
      length += Trail(topo, roads, buildings, player, next, used);
    }
    used->reset(e);
    best = std::max(best, length);
  }
  return best;
}

int Total(absl::Span<const int> counts) {
  return std::accumulate(counts.begin(), counts.end(), 0);
}

std::string PortTypeString(int type) {
  return type == kGenericPort ? "3:1"
                              : absl::StrCat(kResourceNames[type], " 2:1");
}

// "2 Lumber + 1 Wool for 1 Ore", "1 Wool for 1 any".
std::string BundleString(const Bundle& bundle) {
  auto cards = [](const Cards& counts, int any) {
    std::string out;
    for (int r = 0; r < kNumResources; ++r) {
      if (counts[r] == 0) continue;
      absl::StrAppend(&out, out.empty() ? "" : " + ", counts[r], " ",
                      kResourceNames[r]);
    }
    if (any > 0) absl::StrAppend(&out, out.empty() ? "" : " + ", any, " any");
    return out.empty() ? std::string("nothing") : out;
  };
  return absl::StrCat(cards(bundle.give, bundle.give_any), " for ",
                      cards(bundle.ask, bundle.ask_any));
}

}  // namespace

std::ostream& operator<<(std::ostream& os, Phase phase) {
  return os << kPhaseNames[static_cast<int>(phase)];
}

std::ostream& operator<<(std::ostream& os, TradeStatus status) {
  return os << kTradeStatusNames[static_cast<int>(status)];
}

const Topology& GetTopology() {
  static const Topology* topology = new Topology(BuildTopology());
  return *topology;
}

std::pair<int, int> ResourcePairFromIndex(int index) {
  int give = index / (kNumResources - 1);
  int get = index % (kNumResources - 1);
  return {give, get < give ? get : get + 1};
}

std::pair<int, int> YearOfPlentyPair(int index) {
  for (int a = 0; a < kNumResources; ++a) {
    for (int b = a; b < kNumResources; ++b) {
      if (index-- == 0) return {a, b};
    }
  }
  SpielFatalError(absl::StrCat("Invalid year of plenty index ", index));
}

CatanState::CatanState(std::shared_ptr<const Game> game) : State(game) {
  const auto& catan = static_cast<const CatanGame&>(*game);
  num_players_ = catan.NumPlayers();
  victory_points_ = catan.victory_points();
  max_trade_offers_ = catan.max_trade_offers();
  max_counter_steps_ = catan.max_counter_steps();
  max_turns_ = catan.max_turns();

  terrain_.fill(-1);
  port_type_.fill(-1);
  terrains_left_ = kTerrainCounts;
  ports_left_ = kPortCounts;
  vertex_owner_.fill(kInvalidPlayer);
  edge_owner_.fill(kInvalidPlayer);
  roads_left_.fill(kNumRoads);
  settlements_left_.fill(kNumSettlements);
  cities_left_.fill(kNumCities);
  bank_.fill(kBankResourceCount);
  dev_deck_ = kDevCardCounts;

  if (catan.fixed_board()) {
    terrain_ = kFixedTerrain;
    port_type_ = kFixedPorts;
    PlaceNumbers(/*spiral_start=*/0);
  }
}

Player CatanState::CurrentPlayer() const {
  switch (phase_) {
    case Phase::kBoardTerrain:
    case Phase::kBoardPorts:
    case Phase::kBoardSpiral:
    case Phase::kFirstPlayer:
    case Phase::kRoll:
    case Phase::kSteal:
    case Phase::kDrawDevCard:
      return kChancePlayerId;
    case Phase::kGameOver:
      return kTerminalPlayerId;
    case Phase::kDiscard:
      return DiscardingPlayer();
    case Phase::kComposeTrade:
      return composer_;
    case Phase::kTradeResponse:
      return responder_;
    default:
      return turn_player_;
  }
}

// Players discard in seat order from the player on turn.
Player CatanState::DiscardingPlayer() const {
  for (int i = 0; i < num_players_; ++i) {
    Player p = (turn_player_ + i) % num_players_;
    if (discard_left_[p] > 0) return p;
  }
  return kInvalidPlayer;
}

void CatanState::PlaceNumbers(int spiral_start) {
  int token = 0;
  for (int h : GetTopology().spirals[spiral_start]) {
    if (terrain_[h] == kDesert) {
      robber_hex_ = h;
    } else {
      number_[h] = kNumberTokens[token++];
    }
  }
  phase_ = Phase::kFirstPlayer;
}

std::vector<std::pair<Action, double>> CatanState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  std::vector<std::pair<Action, double>> outcomes;
  auto add_weighted = [&outcomes](absl::Span<const int> weights) {
    double total = Total(weights);
    for (int i = 0; i < weights.size(); ++i) {
      if (weights[i] > 0) outcomes.push_back({i, weights[i] / total});
    }
  };
  switch (phase_) {
    case Phase::kBoardTerrain:
      add_weighted(terrains_left_);
      break;
    case Phase::kBoardPorts:
      add_weighted(ports_left_);
      break;
    case Phase::kBoardSpiral:
      add_weighted(std::vector<int>(kNumSpiralStarts, 1));
      break;
    case Phase::kFirstPlayer:
      add_weighted(std::vector<int>(num_players_, 1));
      break;
    case Phase::kRoll:
      add_weighted(kDiceWays);
      break;
    case Phase::kSteal:
      add_weighted(resources_[victim_]);
      break;
    case Phase::kDrawDevCard:
      add_weighted(dev_deck_);
      break;
    default:
      SpielFatalError("Not a chance phase");
  }
  return outcomes;
}

bool CatanState::CanAfford(Player player, const Cards& cost) const {
  for (int r = 0; r < kNumResources; ++r) {
    if (resources_[player][r] < cost[r]) return false;
  }
  return true;
}

void CatanState::Pay(Player player, const Cards& cost) {
  for (int r = 0; r < kNumResources; ++r) {
    resources_[player][r] -= cost[r];
    bank_[r] += cost[r];
  }
}

bool CatanState::CanPlaceSettlement(Player player, int vertex,
                                    bool setup) const {
  const Topology& topo = GetTopology();
  if (vertex_owner_[vertex] != kInvalidPlayer) return false;
  for (int v : topo.vertex_neighbors[vertex]) {
    if (vertex_owner_[v] != kInvalidPlayer) return false;
  }
  if (setup) return true;
  for (int e : topo.vertex_edges[vertex]) {
    if (edge_owner_[e] == player) return true;
  }
  return false;
}

bool CatanState::CanPlaceRoad(Player player, int edge) const {
  const Topology& topo = GetTopology();
  if (edge_owner_[edge] != kInvalidPlayer) return false;
  for (int v : topo.edge_vertices[edge]) {
    if (vertex_owner_[v] == player) return true;
    // Roads cannot connect through another player's building.
    if (vertex_owner_[v] != kInvalidPlayer) continue;
    for (int e : topo.vertex_edges[v]) {
      if (edge_owner_[e] == player) return true;
    }
  }
  return false;
}

bool CatanState::AnyRoadPlaceable(Player player) const {
  if (roads_left_[player] == 0) return false;
  for (int e = 0; e < kNumEdges; ++e) {
    if (CanPlaceRoad(player, e)) return true;
  }
  return false;
}

bool CatanState::CanPlayDevCard(Player player, int card) const {
  return !dev_card_played_ &&
         dev_cards_[player][card] - dev_bought_[player][card] > 0;
}

Cards CatanState::TradeRates(Player player) const {
  const Topology& topo = GetTopology();
  Cards rates;
  rates.fill(4);
  for (int i = 0; i < kNumPorts; ++i) {
    if (vertex_owner_[topo.port_vertices[i][0]] != player &&
        vertex_owner_[topo.port_vertices[i][1]] != player) {
      continue;
    }
    if (port_type_[i] == kGenericPort) {
      for (int& rate : rates) rate = std::min(rate, 3);
    } else {
      rates[port_type_[i]] = 2;
    }
  }
  return rates;
}

int CatanState::NumResourceCards(Player player) const {
  return Total(resources_[player]);
}

int CatanState::PublicVictoryPoints(Player player) const {
  int points = 0;
  for (int v = 0; v < kNumVertices; ++v) {
    if (vertex_owner_[v] == player) points += is_city_[v] ? 2 : 1;
  }
  if (longest_road_holder_ == player) points += 2;
  if (largest_army_holder_ == player) points += 2;
  return points;
}

int CatanState::VictoryPoints(Player player) const {
  return PublicVictoryPoints(player) + dev_cards_[player][kVictoryPoint];
}

int CatanState::LongestRoad(Player player) const {
  const Topology& topo = GetTopology();
  std::bitset<kNumEdges> used;
  int best = 0;
  for (int e = 0; e < kNumEdges; ++e) {
    if (edge_owner_[e] != player) continue;
    for (int v : topo.edge_vertices[e]) {
      best = std::max(
          best, Trail(topo, edge_owner_, vertex_owner_, player, v, &used));
    }
  }
  return best;
}

// A holder tied for the longest road keeps it. Otherwise it goes to the single
// player with the longest road of at least kMinLongestRoad, or to nobody.
void CatanState::UpdateLongestRoadHolder() {
  int longest = *std::max_element(road_length_.begin(), road_length_.end());
  if (longest_road_holder_ != kInvalidPlayer &&
      road_length_[longest_road_holder_] >= kMinLongestRoad &&
      road_length_[longest_road_holder_] == longest) {
    return;
  }
  longest_road_holder_ = kInvalidPlayer;
  if (longest < kMinLongestRoad) return;
  if (std::count(road_length_.begin(), road_length_.end(), longest) == 1) {
    longest_road_holder_ =
        std::find(road_length_.begin(), road_length_.end(), longest) -
        road_length_.begin();
  }
}

void CatanState::UpdateLargestArmy() {
  int knights = knights_played_[turn_player_];
  if (knights < kMinLargestArmy) return;
  if (largest_army_holder_ == kInvalidPlayer ||
      knights > knights_played_[largest_army_holder_]) {
    largest_army_holder_ = turn_player_;
  }
}

void CatanState::AddDevCardActions(std::vector<Action>* actions) const {
  Player p = turn_player_;
  if (CanPlayDevCard(p, kKnight)) actions->push_back(kPlayKnight);
  if (CanPlayDevCard(p, kRoadBuilding) && AnyRoadPlaceable(p)) {
    actions->push_back(kPlayRoadBuilding);
  }
  if (CanPlayDevCard(p, kYearOfPlenty)) {
    for (int i = 0; i < kNumYearOfPlentyActions; ++i) {
      auto [a, b] = YearOfPlentyPair(i);
      if (a == b ? bank_[a] >= 2 : (bank_[a] >= 1 && bank_[b] >= 1)) {
        actions->push_back(kYearOfPlentyActionBase + i);
      }
    }
  }
  if (CanPlayDevCard(p, kMonopoly)) {
    for (int r = 0; r < kNumResources; ++r) {
      actions->push_back(kMonopolyActionBase + r);
    }
  }
}

// The cards `composer` may add to `built`: one it holds more of than it gives,
// one it does not hold all of to ask for, never a resource on both sides, and
// for an offer an open card on one side. Neither side can exceed the cards
// there are to give or ask for.
void CatanState::AddComposeActions(Player composer, const Bundle& built,
                                   bool offer,
                                   std::vector<Action>* actions) const {
  int held = NumResourceCards(composer);
  if (held == 0) return;  // Nothing to give, so nothing to propose.
  bool room_to_give = Total(built.give) + built.give_any < held;
  bool room_to_ask = Total(built.ask) + built.ask_any < kMaxCardsInGame - held;
  for (int r = 0; r < kNumResources; ++r) {
    if (room_to_give && built.ask[r] == 0 &&
        built.give[r] < resources_[composer][r]) {
      actions->push_back(kTradeGiveActionBase + r);
    }
  }
  for (int r = 0; r < kNumResources; ++r) {
    if (room_to_ask && built.give[r] == 0 &&
        built.ask[r] < kBankResourceCount - resources_[composer][r]) {
      actions->push_back(kTradeAskActionBase + r);
    }
  }
  if (!offer) return;
  if (room_to_give && built.ask_any == 0) actions->push_back(kTradeGiveAny);
  if (room_to_ask && built.give_any == 0) actions->push_back(kTradeAskAny);
}

std::vector<Player> CatanState::RobberVictims() const {
  std::vector<Player> victims;
  for (Player p = 0; p < num_players_; ++p) {
    if (p == turn_player_ || NumResourceCards(p) == 0) continue;
    for (int v : GetTopology().hex_vertices[robber_hex_]) {
      if (vertex_owner_[v] == p) {
        victims.push_back(p);
        break;
      }
    }
  }
  return victims;
}

std::vector<Action> CatanState::LegalActions() const {
  if (IsChanceNode()) return LegalChanceOutcomes();
  if (IsTerminal()) return {};
  Player p = CurrentPlayer();
  std::vector<Action> actions;
  switch (phase_) {
    case Phase::kSetupSettlement:
      for (int v = 0; v < kNumVertices; ++v) {
        if (CanPlaceSettlement(p, v, /*setup=*/true)) {
          actions.push_back(kSettlementActionBase + v);
        }
      }
      break;
    case Phase::kSetupRoad:
      for (int e : GetTopology().vertex_edges[setup_vertex_]) {
        if (edge_owner_[e] == kInvalidPlayer) {
          actions.push_back(kRoadActionBase + e);
        }
      }
      break;
    case Phase::kPreRoll:
      AddDevCardActions(&actions);
      actions.push_back(kRollDice);
      break;
    case Phase::kDiscard:
      for (int r = 0; r < kNumResources; ++r) {
        if (resources_[p][r] > 0) actions.push_back(kDiscardActionBase + r);
      }
      break;
    case Phase::kMoveRobber:
      for (int h = 0; h < kNumHexes; ++h) {
        if (h != robber_hex_) actions.push_back(kMoveRobberActionBase + h);
      }
      break;
    case Phase::kChooseVictim:
      for (Player victim : RobberVictims()) {
        actions.push_back(kStealActionBase + victim);
      }
      break;
    case Phase::kRoadBuilding:
      for (int e = 0; e < kNumEdges; ++e) {
        if (CanPlaceRoad(p, e)) actions.push_back(kRoadActionBase + e);
      }
      break;
    case Phase::kComposeTrade:
      AddComposeActions(
          p, compose_,
          /*offer=*/p == turn_player_ && counterparty_ == kInvalidPlayer,
          &actions);
      if (Total(compose_.give) + compose_.give_any > 0 &&
          Total(compose_.ask) + compose_.ask_any > 0) {
        actions.push_back(kProposeTrade);
      }
      actions.push_back(kWithdrawTrade);
      break;
    case Phase::kTradeResponse:
      // A counter starts with its first card.
      if (counters_[p] < max_counter_steps_) {
        AddComposeActions(p, Bundle(), /*offer=*/false, &actions);
      }
      // An open offer can only be countered.
      if (CanAfford(p, trades_[p].ask) &&
          !(offer_round_ && offer_.give_any + offer_.ask_any > 0)) {
        actions.push_back(kAcceptTrade);
      }
      actions.push_back(kDeclineTrade);
      break;
    case Phase::kTradeDecide:
      for (Player q = 0; q < num_players_; ++q) {
        if ((trade_status_[q] == TradeStatus::kAccepted ||
             trade_status_[q] == TradeStatus::kCountered) &&
            CanAfford(p, trades_[q].give) && CanAfford(q, trades_[q].ask)) {
          actions.push_back(kExecuteTradeActionBase + q);
        }
      }
      for (Player q = 0; q < num_players_; ++q) {
        if (trade_status_[q] == TradeStatus::kCountered &&
            counters_[q] < max_counter_steps_) {
          actions.push_back(kCounterTradeActionBase + q);
        }
      }
      actions.push_back(kEndNegotiation);
      break;
    case Phase::kMain: {
      if (settlements_left_[p] > 0 && CanAfford(p, kSettlementCost)) {
        for (int v = 0; v < kNumVertices; ++v) {
          if (CanPlaceSettlement(p, v, /*setup=*/false)) {
            actions.push_back(kSettlementActionBase + v);
          }
        }
      }
      if (cities_left_[p] > 0 && CanAfford(p, kCityCost)) {
        for (int v = 0; v < kNumVertices; ++v) {
          if (vertex_owner_[v] == p && !is_city_[v]) {
            actions.push_back(kCityActionBase + v);
          }
        }
      }
      if (roads_left_[p] > 0 && CanAfford(p, kRoadCost)) {
        for (int e = 0; e < kNumEdges; ++e) {
          if (CanPlaceRoad(p, e)) actions.push_back(kRoadActionBase + e);
        }
      }
      if (Total(dev_deck_) > 0 && CanAfford(p, kDevCardCost)) {
        actions.push_back(kBuyDevCard);
      }
      AddDevCardActions(&actions);
      Cards rates = TradeRates(p);
      for (int i = 0; i < kNumResourcePairs; ++i) {
        auto [give, get] = ResourcePairFromIndex(i);
        if (resources_[p][give] >= rates[give] && bank_[get] > 0) {
          actions.push_back(kBankTradeActionBase + i);
        }
      }
      // An offer starts with its first card.
      if (trade_offers_made_ < max_trade_offers_) {
        AddComposeActions(p, Bundle(), /*offer=*/true, &actions);
      }
      actions.push_back(kEndTurn);
      break;
    }
    default:
      SpielFatalError("Unexpected phase");
  }
  return actions;
}

void CatanState::DoApplyAction(Action action) {
  uint8_t everyone = (1 << num_players_) - 1;
  if (IsChanceNode()) {
    uint8_t visible = everyone;
    if (phase_ == Phase::kSteal) {
      visible = (1 << turn_player_) | (1 << victim_);
    } else if (phase_ == Phase::kDrawDevCard) {
      visible = 1 << turn_player_;
    }
    visibility_.push_back(visible);
    ApplyChanceOutcome(action);
  } else {
    visibility_.push_back(everyone);
    ApplyPlayerAction(action);
  }
  CheckWin();
}

void CatanState::ApplyChanceOutcome(Action outcome) {
  switch (phase_) {
    case Phase::kBoardTerrain:
      terrain_[board_step_++] = outcome;
      --terrains_left_[outcome];
      if (board_step_ == kNumHexes) {
        board_step_ = 0;
        phase_ = Phase::kBoardPorts;
      }
      break;
    case Phase::kBoardPorts:
      port_type_[board_step_++] = outcome;
      --ports_left_[outcome];
      if (board_step_ == kNumPorts) phase_ = Phase::kBoardSpiral;
      break;
    case Phase::kBoardSpiral:
      PlaceNumbers(outcome);
      break;
    case Phase::kFirstPlayer:
      first_player_ = turn_player_ = outcome;
      phase_ = Phase::kSetupSettlement;
      break;
    case Phase::kRoll:
      last_roll_ = outcome + 2;
      if (last_roll_ != 7) {
        Produce(last_roll_);
        phase_ = Phase::kMain;
        break;
      }
      for (Player p = 0; p < num_players_; ++p) {
        int cards = NumResourceCards(p);
        discard_left_[p] = cards > kDiscardThreshold ? cards / 2 : 0;
      }
      if (DiscardingPlayer() == kInvalidPlayer) {
        StartRobber(Phase::kMain);
      } else {
        phase_ = Phase::kDiscard;
      }
      break;
    case Phase::kSteal:
      --resources_[victim_][outcome];
      ++resources_[turn_player_][outcome];
      victim_ = kInvalidPlayer;
      phase_ = return_phase_;
      break;
    case Phase::kDrawDevCard:
      --dev_deck_[outcome];
      ++dev_cards_[turn_player_][outcome];
      ++dev_bought_[turn_player_][outcome];
      phase_ = Phase::kMain;
      break;
    default:
      SpielFatalError("Not a chance phase");
  }
}

void CatanState::ApplyPlayerAction(Action action) {
  Player p = CurrentPlayer();
  switch (phase_) {
    case Phase::kSetupSettlement: {
      int v = action - kSettlementActionBase;
      PlaceSettlement(p, v);
      setup_vertex_ = v;
      if (setup_step_ >= num_players_) {
        for (int h : GetTopology().vertex_hexes[v]) {
          int r = terrain_[h];
          if (r != kDesert && bank_[r] > 0) {
            --bank_[r];
            ++resources_[p][r];
          }
        }
      }
      phase_ = Phase::kSetupRoad;
      break;
    }
    case Phase::kSetupRoad:
      PlaceRoad(p, action - kRoadActionBase);
      FinishSetupRoad();
      break;
    case Phase::kDiscard: {
      int r = action - kDiscardActionBase;
      --resources_[p][r];
      ++bank_[r];
      --discard_left_[p];
      if (DiscardingPlayer() == kInvalidPlayer) StartRobber(Phase::kMain);
      break;
    }
    case Phase::kMoveRobber:
      robber_hex_ = action - kMoveRobberActionBase;
      ResolveRobberMove();
      break;
    case Phase::kChooseVictim:
      victim_ = action - kStealActionBase;
      phase_ = Phase::kSteal;
      break;
    case Phase::kRoadBuilding:
      PlaceRoad(p, action - kRoadActionBase);
      if (--free_roads_left_ == 0 || !AnyRoadPlaceable(p)) {
        free_roads_left_ = 0;
        phase_ = return_phase_;
      }
      break;
    case Phase::kComposeTrade:
      if (action == kProposeTrade) {
        ProposeTrade();
      } else if (action == kWithdrawTrade) {
        WithdrawTrade();
      } else {
        AddComposeCard(action);
      }
      break;
    case Phase::kTradeResponse:
      if (action == kAcceptTrade && !offer_round_) {
        // A counter from the player on turn, who agreed to it by proposing it.
        ExecuteTrade(p);
      } else if (action == kAcceptTrade || action == kDeclineTrade) {
        trade_status_[p] = action == kAcceptTrade ? TradeStatus::kAccepted
                                                  : TradeStatus::kDeclined;
        Answered(p);
      } else {
        StartComposing(p, turn_player_);
        AddComposeCard(action);
      }
      break;
    case Phase::kTradeDecide:
      if (action == kEndNegotiation) {
        EndNegotiation();
      } else if (action < kCounterTradeActionBase) {
        ExecuteTrade(action - kExecuteTradeActionBase);
      } else {
        StartComposing(p, action - kCounterTradeActionBase);
      }
      break;
    case Phase::kPreRoll:
    case Phase::kMain:
      ApplyTurnAction(p, action);
      break;
    default:
      SpielFatalError("Unexpected phase");
  }
}

// An action of the player on turn, before or after the roll.
void CatanState::ApplyTurnAction(Player p, Action action) {
  if (action == kRollDice) {
    phase_ = Phase::kRoll;
  } else if (action == kEndTurn) {
    EndTurn();
  } else if (action < kCityActionBase) {
    Pay(p, kSettlementCost);
    PlaceSettlement(p, action - kSettlementActionBase);
  } else if (action < kRoadActionBase) {
    Pay(p, kCityCost);
    is_city_[action - kCityActionBase] = true;
    ++settlements_left_[p];
    --cities_left_[p];
  } else if (action < kBuyDevCard) {
    Pay(p, kRoadCost);
    PlaceRoad(p, action - kRoadActionBase);
  } else if (action == kBuyDevCard) {
    Pay(p, kDevCardCost);
    phase_ = Phase::kDrawDevCard;
  } else if (action == kPlayKnight) {
    PlayDevCard(p, kKnight);
    ++knights_played_[p];
    UpdateLargestArmy();
    StartRobber(phase_);
  } else if (action == kPlayRoadBuilding) {
    PlayDevCard(p, kRoadBuilding);
    free_roads_left_ = std::min(2, roads_left_[p]);
    return_phase_ = phase_;
    phase_ = Phase::kRoadBuilding;
  } else if (action < kMonopolyActionBase) {
    PlayDevCard(p, kYearOfPlenty);
    auto [a, b] = YearOfPlentyPair(action - kYearOfPlentyActionBase);
    for (int r : {a, b}) {
      --bank_[r];
      ++resources_[p][r];
    }
  } else if (action < kMoveRobberActionBase) {
    PlayDevCard(p, kMonopoly);
    int r = action - kMonopolyActionBase;
    for (Player q = 0; q < num_players_; ++q) {
      if (q == p) continue;
      resources_[p][r] += resources_[q][r];
      resources_[q][r] = 0;
    }
  } else if (action >= kBankTradeActionBase && action < kTradeGiveActionBase) {
    auto [give, get] = ResourcePairFromIndex(action - kBankTradeActionBase);
    int rate = TradeRates(p)[give];
    resources_[p][give] -= rate;
    bank_[give] += rate;
    --bank_[get];
    ++resources_[p][get];
  } else if (action >= kTradeGiveActionBase && action < kProposeTrade) {
    StartComposing(p, kInvalidPlayer);
    AddComposeCard(action);
  } else {
    SpielFatalError(absl::StrCat("Invalid action ", action));
  }
}

void CatanState::PlaceSettlement(Player player, int vertex) {
  vertex_owner_[vertex] = player;
  --settlements_left_[player];
  // The settlement may break other players' roads through this vertex.
  for (int e : GetTopology().vertex_edges[vertex]) {
    Player owner = edge_owner_[e];
    if (owner != kInvalidPlayer && owner != player) {
      road_length_[owner] = LongestRoad(owner);
    }
  }
  UpdateLongestRoadHolder();
}

void CatanState::PlaceRoad(Player player, int edge) {
  edge_owner_[edge] = player;
  --roads_left_[player];
  road_length_[player] = LongestRoad(player);
  UpdateLongestRoadHolder();
}

void CatanState::PlayDevCard(Player player, int card) {
  --dev_cards_[player][card];
  dev_card_played_ = true;
}

void CatanState::FinishSetupRoad() {
  ++setup_step_;
  setup_vertex_ = -1;
  if (setup_step_ == 2 * num_players_) {
    turn_player_ = first_player_;
    phase_ = Phase::kPreRoll;
    return;
  }
  // Players place in order from the first player, then again in reverse.
  int place = setup_step_ < num_players_ ? setup_step_
                                         : 2 * num_players_ - 1 - setup_step_;
  turn_player_ = (first_player_ + place) % num_players_;
  phase_ = Phase::kSetupSettlement;
}

void CatanState::Produce(int dice_sum) {
  const Topology& topo = GetTopology();
  std::array<Cards, kMaxPlayers> owed{};
  for (int h = 0; h < kNumHexes; ++h) {
    if (number_[h] != dice_sum || h == robber_hex_) continue;
    for (int v : topo.hex_vertices[h]) {
      Player owner = vertex_owner_[v];
      if (owner == kInvalidPlayer) continue;
      owed[owner][terrain_[h]] += is_city_[v] ? 2 : 1;
    }
  }
  for (int r = 0; r < kNumResources; ++r) {
    int total = 0;
    int num_owed = 0;
    for (Player p = 0; p < num_players_; ++p) {
      total += owed[p][r];
      num_owed += owed[p][r] > 0;
    }
    // A shortage is judged against the bank before anyone is paid.
    const int available = bank_[r];
    for (Player p = 0; p < num_players_; ++p) {
      int amount = 0;
      if (total <= available) {
        amount = owed[p][r];
      } else if (num_owed == 1 && owed[p][r] > 0) {
        amount = available;
      }
      bank_[r] -= amount;
      resources_[p][r] += amount;
    }
  }
}

void CatanState::StartRobber(Phase return_phase) {
  return_phase_ = return_phase;
  phase_ = Phase::kMoveRobber;
}

void CatanState::ResolveRobberMove() {
  std::vector<Player> victims = RobberVictims();
  if (victims.empty()) {
    phase_ = return_phase_;
  } else if (victims.size() == 1) {
    victim_ = victims[0];
    phase_ = Phase::kSteal;
  } else {
    phase_ = Phase::kChooseVictim;
  }
}

void CatanState::StartComposing(Player composer, Player counterparty) {
  if (counterparty == kInvalidPlayer) {
    ++trade_offers_made_;
  } else {
    ++counters_[composer == turn_player_ ? counterparty : composer];
  }
  composer_ = composer;
  counterparty_ = counterparty;
  compose_ = Bundle();
  phase_ = Phase::kComposeTrade;
}

void CatanState::AddComposeCard(Action action) {
  if (action < kTradeAskActionBase) {
    ++compose_.give[action - kTradeGiveActionBase];
  } else if (action < kTradeGiveAny) {
    ++compose_.ask[action - kTradeAskActionBase];
  } else if (action == kTradeGiveAny) {
    ++compose_.give_any;
  } else {
    ++compose_.ask_any;
  }
}

void CatanState::ProposeTrade() {
  Player composer = composer_;
  Player counterparty = counterparty_;
  Bundle built = compose_;
  composer_ = counterparty_ = kInvalidPlayer;
  compose_ = Bundle();
  if (composer != turn_player_) {
    // A counter to the player on turn, kept from that player's side.
    trades_[composer] = Bundle{built.ask, built.give};
    trade_status_[composer] = TradeStatus::kCountered;
    Answered(composer);
  } else if (counterparty == kInvalidPlayer) {
    offer_ = built;
    for (Player q = 0; q < num_players_; ++q) {
      if (q == turn_player_) continue;
      trades_[q] = built;
      trade_status_[q] = TradeStatus::kPending;
    }
    offer_round_ = true;
    NextTradeResponder(turn_player_);
  } else {
    trades_[counterparty] = built;  // A counter has no open slots.
    trade_status_[counterparty] = TradeStatus::kPending;
    responder_ = counterparty;
    phase_ = Phase::kTradeResponse;
  }
}

void CatanState::WithdrawTrade() {
  Player composer = composer_;
  Player counterparty = counterparty_;
  composer_ = counterparty_ = kInvalidPlayer;
  compose_ = Bundle();
  if (composer != turn_player_) {
    // A counter given up is a refusal.
    trade_status_[composer] = TradeStatus::kDeclined;
    Answered(composer);
  } else if (counterparty == kInvalidPlayer) {
    EndNegotiation();
  } else {
    phase_ = Phase::kTradeDecide;
  }
}

void CatanState::Answered(Player player) {
  if (offer_round_) {
    NextTradeResponder(player);
  } else {
    AwaitDecision();
  }
}

// Asks the next player after `after`, in seat order, who has yet to answer the
// offer.
void CatanState::NextTradeResponder(Player after) {
  for (Player q = NextPlayerRoundRobin(after, num_players_); q != turn_player_;
       q = NextPlayerRoundRobin(q, num_players_)) {
    if (trade_status_[q] == TradeStatus::kPending) {
      responder_ = q;
      phase_ = Phase::kTradeResponse;
      return;
    }
  }
  offer_round_ = false;
  AwaitDecision();
}

// Hands the negotiation to the player on turn while an acceptance or counter
// is on the table, and ends it otherwise.
void CatanState::AwaitDecision() {
  responder_ = kInvalidPlayer;
  for (TradeStatus status : trade_status_) {
    if (status == TradeStatus::kAccepted || status == TradeStatus::kCountered) {
      phase_ = Phase::kTradeDecide;
      return;
    }
  }
  EndNegotiation();
}

void CatanState::ExecuteTrade(Player other) {
  for (int r = 0; r < kNumResources; ++r) {
    int net = trades_[other].ask[r] - trades_[other].give[r];
    resources_[turn_player_][r] += net;
    resources_[other][r] -= net;
  }
  EndNegotiation();
}

void CatanState::EndNegotiation() {
  composer_ = counterparty_ = responder_ = kInvalidPlayer;
  offer_round_ = false;
  offer_ = Bundle();
  trades_.fill(Bundle());
  trade_status_.fill(TradeStatus::kNone);
  counters_.fill(0);
  phase_ = Phase::kMain;
}

void CatanState::EndTurn() {
  dev_bought_[turn_player_].fill(0);
  dev_card_played_ = false;
  trade_offers_made_ = 0;
  last_roll_ = 0;
  if (++turn_number_ >= max_turns_) {
    phase_ = Phase::kGameOver;
    return;
  }
  turn_player_ = NextPlayerRoundRobin(turn_player_, num_players_);
  phase_ = Phase::kPreRoll;
}

void CatanState::CheckWin() {
  if (phase_ == Phase::kGameOver || phase_ <= Phase::kSetupRoad) return;
  if (VictoryPoints(turn_player_) >= victory_points_) {
    winner_ = turn_player_;
    phase_ = Phase::kGameOver;
  }
}

std::vector<double> CatanState::Returns() const {
  std::vector<double> returns(num_players_, 0.0);
  if (winner_ == kInvalidPlayer) return returns;
  for (Player p = 0; p < num_players_; ++p) {
    returns[p] = p == winner_ ? 1.0 : -1.0 / (num_players_ - 1);
  }
  return returns;
}

std::string CatanState::ActionToString(Player player, Action action) const {
  if (player == kChancePlayerId) {
    // Labels follow the current phase, so a past outcome gets a generic one.
    switch (phase_) {
      case Phase::kBoardTerrain:
        if (action >= kNumTerrains) break;
        return absl::StrCat("Terrain ", kTerrainNames[action]);
      case Phase::kBoardPorts:
        if (action >= kNumPortTypes) break;
        return absl::StrCat("Port ", PortTypeString(action));
      case Phase::kBoardSpiral:
        return absl::StrCat("Numbers from corner ", action);
      case Phase::kFirstPlayer:
        return absl::StrCat("First player ", action);
      case Phase::kRoll:
        return absl::StrCat("Roll ", action + 2);
      case Phase::kSteal:
        if (action >= kNumResources) break;
        return absl::StrCat("Steal ", kResourceNames[action]);
      case Phase::kDrawDevCard:
        if (action >= kNumDevCardTypes) break;
        return absl::StrCat("Draw ", kDevCardNames[action]);
      default:
        break;
    }
    return absl::StrCat("Chance outcome ", action);
  }
  if (action < kCityActionBase) {
    return absl::StrCat("Settlement v", action - kSettlementActionBase);
  } else if (action < kRoadActionBase) {
    return absl::StrCat("City v", action - kCityActionBase);
  } else if (action < kBuyDevCard) {
    int e = action - kRoadActionBase;
    const auto& ends = GetTopology().edge_vertices[e];
    return absl::StrCat("Road e", e, " (v", ends[0], "-v", ends[1], ")");
  } else if (action == kBuyDevCard) {
    return "Buy development card";
  } else if (action == kPlayKnight) {
    return "Play Knight";
  } else if (action == kPlayRoadBuilding) {
    return "Play Road Building";
  } else if (action < kMonopolyActionBase) {
    auto [a, b] = YearOfPlentyPair(action - kYearOfPlentyActionBase);
    return absl::StrCat("Play Year of Plenty ", kResourceNames[a], " ",
                        kResourceNames[b]);
  } else if (action < kMoveRobberActionBase) {
    return absl::StrCat("Play Monopoly ",
                        kResourceNames[action - kMonopolyActionBase]);
  } else if (action < kStealActionBase) {
    return absl::StrCat("Robber to h", action - kMoveRobberActionBase);
  } else if (action < kDiscardActionBase) {
    return absl::StrCat("Rob player ", action - kStealActionBase);
  } else if (action < kBankTradeActionBase) {
    return absl::StrCat("Discard ",
                        kResourceNames[action - kDiscardActionBase]);
  } else if (action < kTradeGiveActionBase) {
    auto [give, get] = ResourcePairFromIndex(action - kBankTradeActionBase);
    return absl::StrCat("Bank trade ", kResourceNames[give], " for ",
                        kResourceNames[get]);
  } else if (action < kTradeAskActionBase) {
    return absl::StrCat("Trade: give ",
                        kResourceNames[action - kTradeGiveActionBase]);
  } else if (action < kTradeGiveAny) {
    return absl::StrCat("Trade: ask for ",
                        kResourceNames[action - kTradeAskActionBase]);
  } else if (action == kTradeGiveAny) {
    return "Trade: give any card";
  } else if (action == kTradeAskAny) {
    return "Trade: ask for any card";
  } else if (action == kProposeTrade) {
    return "Propose trade";
  } else if (action == kWithdrawTrade) {
    return "Withdraw trade";
  } else if (action == kAcceptTrade) {
    return "Accept trade";
  } else if (action == kDeclineTrade) {
    return "Decline trade";
  } else if (action < kCounterTradeActionBase) {
    return absl::StrCat("Trade with player ", action - kExecuteTradeActionBase);
  } else if (action < kEndNegotiation) {
    return absl::StrCat("Counter player ", action - kCounterTradeActionBase);
  } else if (action == kEndNegotiation) {
    return "End negotiation";
  } else if (action == kRollDice) {
    return "Roll dice";
  } else if (action == kEndTurn) {
    return "End turn";
  }
  return absl::StrCat("Invalid action ", action);
}

// The negotiation in progress, if any.
std::string CatanState::TradeString() const {
  std::string s;
  if (composer_ != kInvalidPlayer) {
    absl::StrAppend(&s, "\nBuilding: player ", composer_, " gives ",
                    BundleString(compose_), ", to ",
                    counterparty_ == kInvalidPlayer
                        ? std::string("the table")
                        : absl::StrCat("player ", counterparty_));
  }
  if (Total(offer_.give) + offer_.give_any > 0) {
    absl::StrAppend(&s, "\nOffer: ", BundleString(offer_),
                    offer_round_ ? ", being answered" : "");
  }
  for (Player q = 0; q < num_players_; ++q) {
    if (trade_status_[q] == TradeStatus::kNone) continue;
    absl::StrAppend(&s, "\nWith player ", q, ": ", BundleString(trades_[q]),
                    ", ", kTradeStatusNames[static_cast<int>(trade_status_[q])],
                    ", counters ", counters_[q]);
  }
  return s;
}

std::string CatanState::BoardString() const {
  const Topology& topo = GetTopology();
  std::string s = "Hexes:";
  for (int h = 0; h < kNumHexes; ++h) {
    if (terrain_[h] < 0) {
      absl::StrAppend(&s, " ", h, ":?");
    } else {
      absl::StrAppend(&s, " ", h, ":", kTerrainNames[terrain_[h]]);
      if (number_[h] > 0) absl::StrAppend(&s, "/", number_[h]);
    }
  }
  absl::StrAppend(&s, "\nRobber: h", robber_hex_, "\nPorts:");
  for (int i = 0; i < kNumPorts; ++i) {
    absl::StrAppend(&s, " v", topo.port_vertices[i][0], "-v",
                    topo.port_vertices[i][1], ":",
                    port_type_[i] < 0 ? "?" : PortTypeString(port_type_[i]));
  }
  absl::StrAppend(&s, "\nBuildings:");
  for (int v = 0; v < kNumVertices; ++v) {
    if (vertex_owner_[v] != kInvalidPlayer) {
      absl::StrAppend(&s, " v", v, ":", is_city_[v] ? "C" : "S",
                      vertex_owner_[v]);
    }
  }
  absl::StrAppend(&s, "\nRoads:");
  for (int e = 0; e < kNumEdges; ++e) {
    if (edge_owner_[e] != kInvalidPlayer) {
      absl::StrAppend(&s, " e", e, ":", edge_owner_[e]);
    }
  }
  absl::StrAppend(&s, "\nPhase: ", kPhaseNames[static_cast<int>(phase_)],
                  ", turn ", turn_number_, ", player ", turn_player_,
                  ", first player ", first_player_, ", roll ", last_roll_);
  if (dev_card_played_) absl::StrAppend(&s, ", development card played");
  if (free_roads_left_ > 0) {
    absl::StrAppend(&s, ", free roads ", free_roads_left_);
  }
  if (trade_offers_made_ > 0) {
    absl::StrAppend(&s, ", trade offers ", trade_offers_made_);
  }
  if (victim_ != kInvalidPlayer) {
    absl::StrAppend(&s, ", robbing player ", victim_);
  }
  absl::StrAppend(&s, TradeString(), "\nBank:");
  for (int r = 0; r < kNumResources; ++r) {
    absl::StrAppend(&s, " ", kResourceNames[r], " ", bank_[r]);
  }
  absl::StrAppend(&s, ", development cards ", Total(dev_deck_), "\n");
  return s;
}

// Shows `player`'s hidden cards if `observer` is that player, or is
// kInvalidPlayer for the full state.
std::string CatanState::PlayerString(Player player, Player observer) const {
  bool visible = observer == kInvalidPlayer || observer == player;
  std::string s = absl::StrCat(
      "Player ", player, ": VP ",
      visible ? VictoryPoints(player) : PublicVictoryPoints(player));
  if (visible) {
    for (int r = 0; r < kNumResources; ++r) {
      absl::StrAppend(&s, ", ", kResourceNames[r], " ", resources_[player][r]);
    }
    for (int c = 0; c < kNumDevCardTypes; ++c) {
      if (dev_cards_[player][c] == 0) continue;
      absl::StrAppend(&s, ", ", kDevCardNames[c], " ", dev_cards_[player][c]);
      if (dev_bought_[player][c] > 0) {
        absl::StrAppend(&s, " (", dev_bought_[player][c], " new)");
      }
    }
  } else {
    absl::StrAppend(&s, ", resource cards ", NumResourceCards(player),
                    ", development cards ", Total(dev_cards_[player]));
  }
  absl::StrAppend(&s, ", knights played ", knights_played_[player],
                  ", road length ", road_length_[player]);
  if (longest_road_holder_ == player) absl::StrAppend(&s, ", longest road");
  if (largest_army_holder_ == player) absl::StrAppend(&s, ", largest army");
  if (discard_left_[player] > 0) {
    absl::StrAppend(&s, ", to discard ", discard_left_[player]);
  }
  return s + "\n";
}

std::string CatanState::ToString() const {
  std::string s = BoardString();
  for (Player p = 0; p < num_players_; ++p) {
    absl::StrAppend(&s, PlayerString(p, kInvalidPlayer));
  }
  return s;
}

std::string CatanState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  std::string s = BoardString();
  for (Player p = 0; p < num_players_; ++p) {
    absl::StrAppend(&s, PlayerString(p, player));
  }
  return s;
}

std::string CatanState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  const std::vector<PlayerAction>& history = FullHistory();
  std::string s = absl::StrCat("Player ", player, ":");
  s.reserve(s.size() + 5 * history.size());
  for (int i = 0; i < history.size(); ++i) {
    if ((visibility_[i] >> player) & 1) {
      absl::StrAppend(&s, " ", history[i].action);
    } else {
      absl::StrAppend(&s, " ?");
    }
  }
  return s;
}

int CatanState::Seat(Player player, Player observer) const {
  return (player - observer + num_players_) % num_players_;
}

void CatanState::ObservationTensor(Player player,
                                   absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  SPIEL_CHECK_EQ(values.size(), kObservationTensorSize);
  const Topology& topo = GetTopology();
  std::fill(values.begin(), values.end(), 0);
  int offset = 0;
  auto one_hot = [&values, &offset](int index, int size) {
    if (index >= 0) values[offset + index] = 1;
    offset += size;
  };
  auto value = [&values, &offset](float x) { values[offset++] = x; };
  auto seat = [&](Player p) { return p >= 0 ? Seat(p, player) : -1; };

  for (int h = 0; h < kNumHexes; ++h) one_hot(terrain_[h], kNumTerrains);
  for (int h = 0; h < kNumHexes; ++h) {
    one_hot(number_[h] > 0 ? number_[h] - 2 : -1, kNumDiceSums);
  }
  one_hot(robber_hex_, kNumHexes);
  for (int v = 0; v < kNumVertices; ++v) {
    int port = topo.vertex_port[v];
    one_hot(port >= 0 ? port_type_[port] : -1, kNumPortTypes);
  }
  for (int v = 0; v < kNumVertices; ++v) {
    Player owner = vertex_owner_[v];
    one_hot(owner < 0 ? -1 : 2 * seat(owner) + is_city_[v], 2 * kMaxPlayers);
  }
  for (int e = 0; e < kNumEdges; ++e) {
    one_hot(seat(edge_owner_[e]), kMaxPlayers);
  }

  for (int r = 0; r < kNumResources; ++r) value(resources_[player][r]);
  for (int c = 0; c < kNumDevCardTypes; ++c) value(dev_cards_[player][c]);
  for (int c = 0; c < kNumDevCardTypes; ++c) {
    value(dev_cards_[player][c] - dev_bought_[player][c]);
  }
  for (int s = 0; s < kMaxPlayers; ++s) {
    if (s >= num_players_) {
      offset += kNumSeatFeatures;
      continue;
    }
    Player p = (player + s) % num_players_;
    value(NumResourceCards(p));
    value(Total(dev_cards_[p]));
    value(knights_played_[p]);
    value(PublicVictoryPoints(p));
    value(longest_road_holder_ == p);
    value(largest_army_holder_ == p);
    value(road_length_[p]);
    value(roads_left_[p]);
    value(settlements_left_[p]);
    value(cities_left_[p]);
    value(discard_left_[p]);
  }
  for (int r = 0; r < kNumResources; ++r) value(bank_[r]);
  value(Total(dev_deck_));

  one_hot(static_cast<int>(phase_), kNumPhases);
  one_hot(seat(CurrentPlayer()), kMaxPlayers);
  one_hot(seat(turn_player_), kMaxPlayers);
  one_hot(seat(first_player_), kMaxPlayers);
  one_hot(seat(victim_), kMaxPlayers);
  one_hot(last_roll_ > 0 ? last_roll_ - 2 : -1, kNumDiceSums);
  value(dev_card_played_);
  value(free_roads_left_);
  value(trade_offers_made_);

  one_hot(seat(composer_), kMaxPlayers);
  one_hot(seat(counterparty_), kMaxPlayers);
  for (const Bundle* bundle : {&compose_, &offer_}) {
    for (int r = 0; r < kNumResources; ++r) value(bundle->give[r]);
    for (int r = 0; r < kNumResources; ++r) value(bundle->ask[r]);
    value(bundle->give_any);
    value(bundle->ask_any);
  }
  for (int s = 0; s < kMaxPlayers; ++s) {
    if (s >= num_players_) {
      offset += kNumTradeSeatFeatures;
      continue;
    }
    Player p = (player + s) % num_players_;
    one_hot(static_cast<int>(trade_status_[p]), kNumTradeStatuses);
    for (int r = 0; r < kNumResources; ++r) value(trades_[p].give[r]);
    for (int r = 0; r < kNumResources; ++r) value(trades_[p].ask[r]);
    value(counters_[p]);
  }
  one_hot(setup_vertex_, kNumVertices);
  SPIEL_CHECK_EQ(offset, kObservationTensorSize);
}

namespace {

std::map<std::string, int> CardMap(absl::Span<const int> counts,
                                   absl::Span<const char* const> names) {
  std::map<std::string, int> cards;
  for (int i = 0; i < counts.size(); ++i) cards[names[i]] = counts[i];
  return cards;
}

CatanTradeStruct TradeStruct(const Bundle& bundle) {
  CatanTradeStruct trade;
  for (int r = 0; r < kNumResources; ++r) {
    if (bundle.give[r] > 0) trade.give[kResourceNames[r]] = bundle.give[r];
    if (bundle.ask[r] > 0) trade.ask[kResourceNames[r]] = bundle.ask[r];
  }
  trade.give_any = bundle.give_any;
  trade.ask_any = bundle.ask_any;
  return trade;
}

int ResourceIndex(const std::string& name) {
  auto it = std::find(kResourceNames.begin(), kResourceNames.end(), name);
  if (it == kResourceNames.end()) {
    SpielFatalError(absl::StrCat("Unknown resource: ", name));
  }
  return it - kResourceNames.begin();
}

}  // namespace

// Hides other players' cards from `observer`, unless it is kInvalidPlayer.
void CatanState::FillStruct(Player observer,
                            CatanStructContents* contents) const {
  contents->phase = kPhaseNames[static_cast<int>(phase_)];
  contents->current_player = CurrentPlayer();
  contents->turn_player = turn_player_;
  contents->turn = turn_number_;
  contents->roll = last_roll_;
  contents->development_card_played = dev_card_played_;
  contents->free_roads = free_roads_left_;
  contents->trade_offers_made = trade_offers_made_;
  contents->victim = victim_;
  for (int h = 0; h < kNumHexes; ++h) {
    std::string hex = terrain_[h] < 0 ? "?" : kTerrainNames[terrain_[h]];
    if (number_[h] > 0) absl::StrAppend(&hex, " ", number_[h]);
    contents->hexes.push_back(hex);
  }
  for (int type : port_type_) {
    contents->ports.push_back(type < 0 ? "?" : PortTypeString(type));
  }
  contents->robber_hex = robber_hex_;
  contents->bank = CardMap(bank_, kResourceNames);
  contents->development_deck = Total(dev_deck_);
  contents->longest_road = longest_road_holder_;
  contents->largest_army = largest_army_holder_;
  for (Player p = 0; p < num_players_; ++p) {
    bool visible = observer == kInvalidPlayer || observer == p;
    CatanPlayerStruct player;
    player.victory_points = visible ? VictoryPoints(p) : PublicVictoryPoints(p);
    if (visible) {
      player.resources = CardMap(resources_[p], kResourceNames);
      player.development_cards = CardMap(dev_cards_[p], kDevCardNames);
      player.new_development_cards = CardMap(dev_bought_[p], kDevCardNames);
    }
    player.resource_count = NumResourceCards(p);
    player.development_card_count = Total(dev_cards_[p]);
    player.knights_played = knights_played_[p];
    player.road_length = road_length_[p];
    player.discard = discard_left_[p];
    for (int v = 0; v < kNumVertices; ++v) {
      if (vertex_owner_[v] == p) {
        (is_city_[v] ? player.cities : player.settlements).push_back(v);
      }
    }
    for (int e = 0; e < kNumEdges; ++e) {
      if (edge_owner_[e] == p) player.roads.push_back(e);
    }
    contents->players.push_back(player);
    contents->trades.push_back(TradeStruct(trades_[p]));
    contents->trade_status.push_back(
        kTradeStatusNames[static_cast<int>(trade_status_[p])]);
    contents->counters.push_back(counters_[p]);
  }
  contents->offer_round = offer_round_;
  contents->building = TradeStruct(compose_);
  contents->offer = TradeStruct(offer_);
}

std::unique_ptr<StateStruct> CatanState::ToStruct() const {
  auto state_struct = std::make_unique<CatanStateStruct>();
  FillStruct(kInvalidPlayer, state_struct.get());
  return state_struct;
}

std::unique_ptr<ObservationStruct> CatanState::ToObservationStruct(
    Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  auto observation = std::make_unique<CatanObservationStruct>();
  FillStruct(player, observation.get());
  observation->observing_player = player;
  return observation;
}

std::unique_ptr<ActionStruct> CatanState::ActionToStruct(Player player,
                                                         Action action) const {
  SPIEL_CHECK_GE(player, 0);
  auto a = std::make_unique<CatanActionStruct>();
  auto add_resource = [&a](int r) {
    a->resources.push_back(kResourceNames[r]);
  };
  if (action < kCityActionBase) {
    a->kind = "settlement";
    a->vertex = action - kSettlementActionBase;
  } else if (action < kRoadActionBase) {
    a->kind = "city";
    a->vertex = action - kCityActionBase;
  } else if (action < kBuyDevCard) {
    a->kind = "road";
    a->edge = action - kRoadActionBase;
  } else if (action == kBuyDevCard) {
    a->kind = "buy_development_card";
  } else if (action == kPlayKnight) {
    a->kind = "play_knight";
  } else if (action == kPlayRoadBuilding) {
    a->kind = "play_road_building";
  } else if (action < kMonopolyActionBase) {
    a->kind = "play_year_of_plenty";
    auto [first, second] = YearOfPlentyPair(action - kYearOfPlentyActionBase);
    add_resource(first);
    add_resource(second);
  } else if (action < kMoveRobberActionBase) {
    a->kind = "play_monopoly";
    add_resource(action - kMonopolyActionBase);
  } else if (action < kStealActionBase) {
    a->kind = "move_robber";
    a->hex = action - kMoveRobberActionBase;
  } else if (action < kDiscardActionBase) {
    a->kind = "rob";
    a->player = action - kStealActionBase;
  } else if (action < kBankTradeActionBase) {
    a->kind = "discard";
    add_resource(action - kDiscardActionBase);
  } else if (action < kTradeGiveActionBase) {
    a->kind = "bank_trade";
    auto [give, get] = ResourcePairFromIndex(action - kBankTradeActionBase);
    add_resource(give);
    add_resource(get);
  } else if (action < kTradeAskActionBase) {
    a->kind = "trade_give";
    add_resource(action - kTradeGiveActionBase);
  } else if (action < kTradeGiveAny) {
    a->kind = "trade_ask";
    add_resource(action - kTradeAskActionBase);
  } else if (action == kTradeGiveAny) {
    a->kind = "trade_give_any";
  } else if (action == kTradeAskAny) {
    a->kind = "trade_ask_any";
  } else if (action == kProposeTrade) {
    a->kind = "propose";
  } else if (action == kWithdrawTrade) {
    a->kind = "withdraw_trade";
  } else if (action == kAcceptTrade) {
    a->kind = "accept_trade";
  } else if (action == kDeclineTrade) {
    a->kind = "decline_trade";
  } else if (action < kCounterTradeActionBase) {
    a->kind = "execute_trade";
    a->player = action - kExecuteTradeActionBase;
  } else if (action < kEndNegotiation) {
    a->kind = "counter_trade";
    a->player = action - kCounterTradeActionBase;
  } else if (action == kEndNegotiation) {
    a->kind = "end_negotiation";
  } else if (action == kRollDice) {
    a->kind = "roll";
  } else if (action == kEndTurn) {
    a->kind = "end_turn";
  } else {
    SpielFatalError(absl::StrCat("Invalid action ", action));
  }
  return a;
}

std::vector<Action> CatanState::StructToActions(
    const ActionStruct& action_struct) const {
  const auto* a = SafeActionCast<CatanActionStruct>(action_struct);
  if (a->kind == "propose_trade") {
    std::vector<Action> actions;
    if (phase_ == Phase::kTradeDecide) {
      if (a->player < 0 || a->player >= num_players_) {
        SpielFatalError("A counter to a counter must name the player.");
      }
      actions.push_back(kCounterTradeActionBase + a->player);
    }
    auto add = [&actions](int count, Action action) {
      if (count < 0) SpielFatalError("A trade cannot have negative counts.");
      actions.insert(actions.end(), count, action);
    };
    for (const auto& [name, count] : a->trade.give) {
      add(count, kTradeGiveActionBase + ResourceIndex(name));
    }
    for (const auto& [name, count] : a->trade.ask) {
      add(count, kTradeAskActionBase + ResourceIndex(name));
    }
    add(a->trade.give_any, kTradeGiveAny);
    add(a->trade.ask_any, kTradeAskAny);
    actions.push_back(kProposeTrade);
    return actions;
  }
  // Any other kind names a single legal action.
  nlohmann::json wanted = a->to_json_base();
  for (Action action : LegalActions()) {
    if (ActionToStruct(CurrentPlayer(), action)->to_json_base() == wanted) {
      return {action};
    }
  }
  SpielFatalError(absl::StrCat("Not a legal action: ", a->ToJson()));
}

std::unique_ptr<ActionStruct> CatanState::ActionsToStruct(
    Player player, absl::Span<const Action> actions) const {
  if (actions.size() == 1) return ActionToStruct(player, actions[0]);
  auto a = std::make_unique<CatanActionStruct>();
  a->kind = "propose_trade";
  for (Action action : actions) {
    if (action >= kTradeGiveActionBase && action < kTradeAskActionBase) {
      ++a->trade.give[kResourceNames[action - kTradeGiveActionBase]];
    } else if (action >= kTradeAskActionBase && action < kTradeGiveAny) {
      ++a->trade.ask[kResourceNames[action - kTradeAskActionBase]];
    } else if (action == kTradeGiveAny) {
      ++a->trade.give_any;
    } else if (action == kTradeAskAny) {
      ++a->trade.ask_any;
    } else if (action >= kCounterTradeActionBase && action < kEndNegotiation) {
      a->player = action - kCounterTradeActionBase;
    } else {
      SPIEL_CHECK_EQ(action, kProposeTrade);
    }
  }
  return a;
}

std::unique_ptr<State> CatanState::Clone() const {
  return std::unique_ptr<State>(new CatanState(*this));
}

CatanGame::CatanGame(const GameParameters& params)
    : Game(kGameType, params),
      num_players_(ParameterValue<int>("players")),
      victory_points_(ParameterValue<int>("victory_points")),
      max_trade_offers_(ParameterValue<int>("max_trade_offers")),
      max_counter_steps_(ParameterValue<int>("max_counter_steps")),
      max_turns_(ParameterValue<int>("max_turns")) {
  SPIEL_CHECK_GE(num_players_, kMinPlayers);
  SPIEL_CHECK_LE(num_players_, kMaxPlayers);
  SPIEL_CHECK_GE(victory_points_, 1);
  SPIEL_CHECK_GE(max_trade_offers_, 0);
  SPIEL_CHECK_GE(max_counter_steps_, 0);
  SPIEL_CHECK_GE(max_turns_, 1);
  std::string board = ParameterValue<std::string>("board");
  if (board != "random" && board != "fixed") {
    SpielFatalError(absl::StrCat("Unknown board: ", board));
  }
  fixed_board_ = board == "fixed";
  MaxGameLength();  // Fails now if the parameters make it overflow.
}

std::vector<int> CatanGame::ObservationTensorShape() const {
  return {kObservationTensorSize};
}

int CatanGame::MaxGameLength() const {
  // Per offer: building it and each counter, an answer to each, and the
  // decisions between them.
  int64_t counters = int64_t{kMaxPlayers - 1} * max_counter_steps_;
  int64_t per_offer = (kMaxComposeActions + 2) * (1 + counters) + kMaxPlayers;
  int64_t per_turn = kMaxActionsPerTurn + max_trade_offers_ * per_offer;
  int64_t length = 4 * num_players_ + max_turns_ * per_turn;
  SPIEL_CHECK_LE(length, std::numeric_limits<int>::max());
  return length;
}

std::pair<std::string, std::string> CatanGame::ActionStructSpec() const {
  return {R"({"kind": string, "vertex": int, "edge": int, "hex": int, )"
          R"("player": int, "resources": [string], "trade": {"give": )"
          R"({string: int}, "ask": {string: int}, "give_any": int, )"
          R"("ask_any": int}})",
          R"({"kind": "propose_trade", "trade": {"give": {"Wool": 2}, )"
          R"("ask": {"Ore": 1}}})"};
}

int CatanGame::MaxChanceNodesInHistory() const {
  // The board, then per turn a roll and up to two robberies, and every
  // development card.
  return kNumHexes + kNumPorts + 2 + 3 * max_turns_ + Total(kDevCardCounts);
}

}  // namespace catan
}  // namespace open_spiel
