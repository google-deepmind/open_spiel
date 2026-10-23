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
#include <memory>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace catan {
namespace {

namespace testing = open_spiel::testing;

std::shared_ptr<const Game> FixedGame(int players, GameParameters params = {}) {
  params["players"] = GameParameter(players);
  params["board"] = GameParameter("fixed");
  return LoadGame("catan", params);
}

const CatanState& AsCatan(const State& state) {
  return static_cast<const CatanState&>(state);
}

// Plays the setup with player 0 first, each player taking the first legal
// settlement and road.
std::unique_ptr<State> PlayFirstLegalSetup(const Game& game) {
  std::unique_ptr<State> state = game.NewInitialState();
  while (AsCatan(*state).phase() <= Phase::kSetupRoad) {
    state->ApplyAction(state->LegalActions().front());
  }
  return state;
}

// The setup, then player 0 rolls a 12.
std::unique_ptr<State> StartFirstTurn(const Game& game) {
  std::unique_ptr<State> state = PlayFirstLegalSetup(game);
  state->ApplyAction(kRollDice);
  state->ApplyAction(12 - 2);
  return state;
}

// A resource player 0 holds and a different one player 1 holds.
std::pair<int, int> TradablePair(const CatanState& state) {
  for (int give = 0; give < kNumResources; ++give) {
    for (int get = 0; get < kNumResources; ++get) {
      if (get != give && state.resources(0, give) > 0 &&
          state.resources(1, get) > 0) {
        return {give, get};
      }
    }
  }
  SpielFatalError("No tradable pair");
}

std::vector<int> HexNeighbors(int hex) {
  const Topology& topo = GetTopology();
  std::vector<int> neighbors;
  for (int g = 0; g < kNumHexes; ++g) {
    int shared = 0;
    for (int v : topo.hex_vertices[hex]) {
      shared += absl::c_linear_search(topo.hex_vertices[g], v);
    }
    if (g != hex && shared == 2) neighbors.push_back(g);
  }
  return neighbors;
}

void TopologyTest() {
  const Topology& topo = GetTopology();
  int coastal_vertices = 0;
  for (int v = 0; v < kNumVertices; ++v) {
    SPIEL_CHECK_TRUE(topo.vertex_edges[v].size() == 2 ||
                     topo.vertex_edges[v].size() == 3);
    SPIEL_CHECK_GE(topo.vertex_hexes[v].size(), 1);
    SPIEL_CHECK_LE(topo.vertex_hexes[v].size(), 3);
    coastal_vertices += topo.vertex_hexes[v].size() < 3;
  }
  SPIEL_CHECK_EQ(coastal_vertices, 30);

  // Each port is a coastal edge, and no two ports share a vertex.
  std::set<int> port_vertices;
  for (const auto& port : topo.port_vertices) {
    SPIEL_CHECK_TRUE(
        absl::c_linear_search(topo.vertex_neighbors[port[0]], port[1]));
    SPIEL_CHECK_LT(topo.vertex_hexes[port[0]].size(), 3);
    SPIEL_CHECK_LT(topo.vertex_hexes[port[1]].size(), 3);
    port_vertices.insert(port.begin(), port.end());
  }
  SPIEL_CHECK_EQ(port_vertices.size(), 2 * kNumPorts);
  // As on the frame, three ports take the outer edges of corner hexes, touching
  // no other hex, and the other six sit on side hexes (four neighbors).
  int corner_ports = 0;
  int side_ports = 0;
  for (const auto& port : topo.port_vertices) {
    std::vector<int> hexes;
    for (int h : topo.vertex_hexes[port[0]]) {
      if (absl::c_linear_search(topo.vertex_hexes[port[1]], h)) {
        hexes.push_back(h);
      }
    }
    SPIEL_CHECK_EQ(hexes.size(), 1);
    corner_ports += topo.vertex_hexes[port[0]].size() == 1 &&
                    topo.vertex_hexes[port[1]].size() == 1;
    side_ports += HexNeighbors(hexes[0]).size() == 4;
  }
  SPIEL_CHECK_EQ(corner_ports, 3);
  SPIEL_CHECK_EQ(side_ports, 6);

  // Each spiral starts on its own corner hex (three neighbors), ends at the
  // center (six), covers every hex and steps between adjacent hexes.
  std::set<int> starts;
  for (const auto& spiral : topo.spirals) {
    SPIEL_CHECK_EQ(std::set<int>(spiral.begin(), spiral.end()).size(),
                   kNumHexes);
    SPIEL_CHECK_EQ(HexNeighbors(spiral.front()).size(), 3);
    SPIEL_CHECK_EQ(HexNeighbors(spiral.back()).size(), 6);
    starts.insert(spiral.front());
    for (int i = 1; i < kNumHexes; ++i) {
      SPIEL_CHECK_TRUE(
          absl::c_linear_search(HexNeighbors(spiral[i - 1]), spiral[i]));
    }
  }
  SPIEL_CHECK_EQ(starts.size(), kNumSpiralStarts);
}

void ActionEncodingTest() {
  SPIEL_CHECK_EQ(kNumDistinctActions, 278);
  std::set<std::pair<int, int>> resource_pairs;
  for (int i = 0; i < kNumResourcePairs; ++i) {
    auto [give, get] = ResourcePairFromIndex(i);
    SPIEL_CHECK_NE(give, get);
    resource_pairs.insert({give, get});
  }
  SPIEL_CHECK_EQ(resource_pairs.size(), kNumResourcePairs);
  std::set<std::pair<int, int>> plenty_pairs;
  for (int i = 0; i < kNumYearOfPlentyActions; ++i) {
    auto [a, b] = YearOfPlentyPair(i);
    SPIEL_CHECK_LE(a, b);
    plenty_pairs.insert({a, b});
  }
  SPIEL_CHECK_EQ(plenty_pairs.size(), kNumYearOfPlentyActions);
}

// No two hexes with a 6 or an 8 may be adjacent.
void CheckRedNumbersApart(const CatanState& state) {
  for (int h = 0; h < kNumHexes; ++h) {
    if (state.number(h) != 6 && state.number(h) != 8) continue;
    for (int g : HexNeighbors(h)) {
      SPIEL_CHECK_NE(state.number(g), 6);
      SPIEL_CHECK_NE(state.number(g), 8);
    }
  }
}

void CheckInvariants(const State& generic_state) {
  const auto& state = AsCatan(generic_state);
  if (state.phase() < Phase::kFirstPlayer) return;
  const Topology& topo = GetTopology();
  int num_players = state.NumPlayers();

  for (int r = 0; r < kNumResources; ++r) {
    int total = state.bank(r);
    for (Player p = 0; p < num_players; ++p) {
      SPIEL_CHECK_GE(state.resources(p, r), 0);
      total += state.resources(p, r);
    }
    SPIEL_CHECK_EQ(total, kBankResourceCount);
  }

  std::vector<int> roads(num_players, 0);
  std::vector<int> settlements(num_players, 0);
  std::vector<int> cities(num_players, 0);
  for (int v = 0; v < kNumVertices; ++v) {
    Player owner = state.vertex_owner(v);
    if (owner == kInvalidPlayer) continue;
    ++(state.is_city(v) ? cities : settlements)[owner];
    // The distance rule.
    for (int w : topo.vertex_neighbors[v]) {
      SPIEL_CHECK_EQ(state.vertex_owner(w), kInvalidPlayer);
    }
  }
  for (int e = 0; e < kNumEdges; ++e) {
    if (state.edge_owner(e) != kInvalidPlayer) ++roads[state.edge_owner(e)];
  }
  std::vector<int> road_lengths(num_players);
  for (Player p = 0; p < num_players; ++p) {
    SPIEL_CHECK_LE(roads[p], kNumRoads);
    SPIEL_CHECK_LE(settlements[p], kNumSettlements);
    SPIEL_CHECK_LE(cities[p], kNumCities);
    SPIEL_CHECK_EQ(state.PublicVictoryPoints(p),
                   settlements[p] + 2 * cities[p] +
                       2 * (state.longest_road_holder() == p) +
                       2 * (state.largest_army_holder() == p));
    road_lengths[p] = state.LongestRoad(p);
  }
  // Outside setup, the game ends once the player on turn reaches the target.
  if (!state.IsTerminal() && state.phase() > Phase::kSetupRoad) {
    const auto& game = static_cast<const CatanGame&>(*state.GetGame());
    SPIEL_CHECK_LT(state.VictoryPoints(state.turn_player()),
                   game.victory_points());
    CheckRedNumbersApart(state);
  }

  Player road_holder = state.longest_road_holder();
  if (road_holder != kInvalidPlayer) {
    SPIEL_CHECK_GE(road_lengths[road_holder], kMinLongestRoad);
    SPIEL_CHECK_EQ(road_lengths[road_holder],
                   *absl::c_max_element(road_lengths));
  }
  // A player who only ties the holder does not take it.
  Player army_holder = state.largest_army_holder();
  if (army_holder != kInvalidPlayer) {
    SPIEL_CHECK_GE(state.knights_played(army_holder), kMinLargestArmy);
    for (Player p = 0; p < num_players; ++p) {
      SPIEL_CHECK_LE(state.knights_played(p),
                     state.knights_played(army_holder));
    }
  }

  // A player with no cards cannot start a counter it could never propose.
  if (state.phase() == Phase::kTradeResponse &&
      state.NumResourceCards(state.CurrentPlayer()) == 0) {
    for (Action a : state.LegalActions()) {
      SPIEL_CHECK_TRUE(a == kAcceptTrade || a == kDeclineTrade);
    }
  }
}

void RandomSimTests() {
  // The test keeps every state of a game, so games with trading are capped.
  std::shared_ptr<const Game> game =
      LoadGame("catan", {{"max_turns", GameParameter(100)}});
  testing::RandomSimTest(*game, 2, /*serialize=*/false, /*verbose=*/false,
                         /*mask_test=*/true, &CheckInvariants);
  for (int players = 2; players <= 4; ++players) {
    for (const char* board : {"random", "fixed"}) {
      game = LoadGame("catan", {{"players", GameParameter(players)},
                                {"board", GameParameter(board)},
                                {"max_trade_offers", GameParameter(0)}});
      testing::RandomSimTest(*game, 1, /*serialize=*/false, /*verbose=*/false,
                             /*mask_test=*/true, &CheckInvariants);
    }
  }
  // Short games with serialization checks.
  game = LoadGame("catan", {{"players", GameParameter(3)},
                            {"max_turns", GameParameter(30)}});
  testing::RandomSimTest(*game, 5, /*serialize=*/true, /*verbose=*/false,
                         /*mask_test=*/true, &CheckInvariants);
}

void SetupTest() {
  std::shared_ptr<const Game> game = FixedGame(3);
  std::unique_ptr<State> state = game->NewInitialState();
  const Topology& topo = GetTopology();
  // Player 2 goes first, so the order is 2, 0, 1, 1, 0, 2.
  SPIEL_CHECK_EQ(AsCatan(*state).phase(), Phase::kFirstPlayer);
  state->ApplyAction(2);
  std::vector<Player> order;
  std::vector<int> second_settlements(3, -1);
  while (AsCatan(*state).phase() <= Phase::kSetupRoad) {
    Player p = state->CurrentPlayer();
    Action settlement = state->LegalActions().front();
    order.push_back(p);
    state->ApplyAction(settlement);
    if (order.size() > 3) second_settlements[p] = settlement;
    // The road must touch the new settlement.
    for (Action road : state->LegalActions()) {
      SPIEL_CHECK_TRUE(absl::c_linear_search(
          topo.edge_vertices[road - kRoadActionBase], settlement));
    }
    state->ApplyAction(state->LegalActions().front());
  }
  SPIEL_CHECK_EQ(order, (std::vector<Player>{2, 0, 1, 1, 0, 2}));
  const CatanState& catan = AsCatan(*state);
  SPIEL_CHECK_EQ(catan.phase(), Phase::kPreRoll);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  state->ApplyAction(kRollDice);
  state->ApplyAction(12 - 2);
  state->ApplyAction(kEndTurn);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  // Starting resources come from the hexes around the second settlement.
  for (Player p = 0; p < 3; ++p) {
    std::vector<int> expected(kNumResources, 0);
    for (int h : topo.vertex_hexes[second_settlements[p]]) {
      if (catan.terrain(h) != kDesert) ++expected[catan.terrain(h)];
    }
    for (int r = 0; r < kNumResources; ++r) {
      SPIEL_CHECK_EQ(catan.resources(p, r), expected[r]);
    }
    SPIEL_CHECK_EQ(catan.VictoryPoints(p), 2);
  }
}

// Cards of `resource` that `player` is owed when `number` is rolled.
int ExpectedProduction(const CatanState& state, Player player, int number,
                       int resource) {
  int amount = 0;
  for (int h = 0; h < kNumHexes; ++h) {
    if (state.number(h) != number || state.terrain(h) != resource ||
        h == state.robber_hex()) {
      continue;
    }
    for (int v : GetTopology().hex_vertices[h]) {
      if (state.vertex_owner(v) == player) amount += state.is_city(v) ? 2 : 1;
    }
  }
  return amount;
}

void ProductionAndRobberTest() {
  std::shared_ptr<const Game> game = FixedGame(2);
  std::unique_ptr<State> state = PlayFirstLegalSetup(*game);
  const auto& catan = AsCatan(*state);

  // A hex next to one of player 0's settlements.
  int hex = -1;
  for (int v = 0; v < kNumVertices && hex < 0; ++v) {
    if (catan.vertex_owner(v) != 0) continue;
    for (int h : GetTopology().vertex_hexes[v]) {
      if (catan.terrain(h) != kDesert) hex = h;
    }
  }
  SPIEL_CHECK_GE(hex, 0);
  int resource = catan.terrain(hex);
  int number = catan.number(hex);
  int expected = ExpectedProduction(catan, 0, number, resource);
  SPIEL_CHECK_GT(expected, 0);
  int before = catan.resources(0, resource);
  state->ApplyAction(kRollDice);
  state->ApplyAction(number - 2);
  SPIEL_CHECK_EQ(catan.phase(), Phase::kMain);
  SPIEL_CHECK_EQ(catan.resources(0, resource), before + expected);
  state->ApplyAction(kEndTurn);

  // Player 1 rolls a 7: nobody holds more than 7 cards, so the robber moves,
  // and steals from player 0 if it holds cards next to the hex.
  state->ApplyAction(kRollDice);
  state->ApplyAction(7 - 2);
  SPIEL_CHECK_EQ(catan.phase(), Phase::kMoveRobber);
  SPIEL_CHECK_FALSE(absl::c_linear_search(
      state->LegalActions(), kMoveRobberActionBase + catan.robber_hex()));
  state->ApplyAction(kMoveRobberActionBase + hex);
  if (catan.phase() == Phase::kSteal) {
    int total_before = catan.NumResourceCards(1);
    state->ApplyAction(state->LegalActions().front());
    SPIEL_CHECK_EQ(catan.NumResourceCards(1), total_before + 1);
  }
  SPIEL_CHECK_EQ(catan.phase(), Phase::kMain);
  state->ApplyAction(kEndTurn);

  // The robber blocks production.
  SPIEL_CHECK_EQ(catan.robber_hex(), hex);
  expected = ExpectedProduction(catan, 0, number, resource);
  before = catan.resources(0, resource);
  state->ApplyAction(kRollDice);
  state->ApplyAction(number - 2);
  SPIEL_CHECK_EQ(catan.resources(0, resource), before + expected);
}

// At each roll, checks every player's payout for every possible sum: what each
// is owed, unless the bank cannot pay everyone owed a resource, when only a
// lone claimant gets it, taking what is left.
void CheckPayouts(const State& state, int* shared, int* shortages) {
  const auto& before = AsCatan(state);
  if (before.phase() != Phase::kRoll) return;
  for (Action outcome = 0; outcome < kNumDiceSums; ++outcome) {
    int sum = outcome + 2;
    if (sum == 7) continue;
    std::unique_ptr<State> child = state.Child(outcome);
    const auto& after = AsCatan(*child);
    for (int r = 0; r < kNumResources; ++r) {
      std::vector<int> owed(state.NumPlayers());
      int total = 0;
      int num_owed = 0;
      for (Player p = 0; p < state.NumPlayers(); ++p) {
        owed[p] = ExpectedProduction(before, p, sum, r);
        total += owed[p];
        num_owed += owed[p] > 0;
      }
      *shared += num_owed > 1;
      *shortages += total > before.bank(r);
      for (Player p = 0; p < state.NumPlayers(); ++p) {
        int expected = 0;
        if (total <= before.bank(r)) {
          expected = owed[p];
        } else if (num_owed == 1 && owed[p] > 0) {
          expected = before.bank(r);
        }
        SPIEL_CHECK_EQ(after.resources(p, r) - before.resources(p, r),
                       expected);
      }
    }
  }
}

void ProductionRuleTest() {
  int shared = 0;
  int shortages = 0;
  std::shared_ptr<const Game> game =
      LoadGame("catan", {{"max_trade_offers", GameParameter(0)}});
  testing::RandomSimTest(
      *game, 5, /*serialize=*/false, /*verbose=*/false, /*mask_test=*/false,
      [&](const State& state) { CheckPayouts(state, &shared, &shortages); });
  SPIEL_CHECK_GT(shared, 0);
  SPIEL_CHECK_GT(shortages, 0);
}

// At each steal and development card draw, only the players who see the
// outcome can tell two outcomes apart.
void CheckSecrets(const State& state, int* steals, int* draws) {
  const auto& catan = AsCatan(state);
  bool steal = catan.phase() == Phase::kSteal;
  if (!steal && catan.phase() != Phase::kDrawDevCard) return;
  std::vector<std::pair<Action, double>> outcomes = state.ChanceOutcomes();
  if (outcomes.size() < 2) return;
  std::unique_ptr<State> a = state.Child(outcomes[0].first);
  std::unique_ptr<State> b = state.Child(outcomes[1].first);
  // A victory point card can end the game, which everyone sees.
  if (a->IsTerminal() || b->IsTerminal()) return;
  for (Player p = 0; p < state.NumPlayers(); ++p) {
    // The victim of a steal is whoever loses the card.
    bool sees =
        p == catan.turn_player() ||
        (steal && AsCatan(*a).NumResourceCards(p) < catan.NumResourceCards(p));
    SPIEL_CHECK_EQ(a->InformationStateString(p) == b->InformationStateString(p),
                   !sees);
    SPIEL_CHECK_EQ(a->ObservationString(p) == b->ObservationString(p), !sees);
    SPIEL_CHECK_EQ(a->ObservationTensor(p) == b->ObservationTensor(p), !sees);
  }
  ++*(steal ? steals : draws);
}

void HiddenInformationTest() {
  int steals = 0;
  int draws = 0;
  std::shared_ptr<const Game> game = LoadGame(
      "catan",
      {{"players", GameParameter(3)}, {"max_trade_offers", GameParameter(0)}});
  testing::RandomSimTest(
      *game, 3, /*serialize=*/false, /*verbose=*/false, /*mask_test=*/false,
      [&](const State& state) { CheckSecrets(state, &steals, &draws); });
  SPIEL_CHECK_GT(steals, 0);
  SPIEL_CHECK_GT(draws, 0);
}

void DiscardTest() {
  // Roll the same producing number until player 0 holds more than 7 cards,
  // then roll a 7.
  std::shared_ptr<const Game> game = FixedGame(2);
  std::unique_ptr<State> state = PlayFirstLegalSetup(*game);
  const auto& catan = AsCatan(*state);
  int number = 0;
  for (int v = 0; v < kNumVertices && number == 0; ++v) {
    if (catan.vertex_owner(v) != 0) continue;
    for (int h : GetTopology().vertex_hexes[v]) {
      number = std::max(number, catan.number(h));
    }
  }
  while (catan.NumResourceCards(0) <= kDiscardThreshold) {
    state->ApplyAction(kRollDice);
    state->ApplyAction(number - 2);
    state->ApplyAction(kEndTurn);
  }
  int cards = catan.NumResourceCards(0);
  Player roller = catan.turn_player();
  state->ApplyAction(kRollDice);
  state->ApplyAction(7 - 2);
  SPIEL_CHECK_EQ(catan.phase(), Phase::kDiscard);
  SPIEL_CHECK_EQ(
      state->CurrentPlayer(),
      catan.NumResourceCards(roller) > kDiscardThreshold ? roller : 0);
  while (catan.phase() == Phase::kDiscard) {
    state->ApplyAction(state->LegalActions().front());
  }
  SPIEL_CHECK_EQ(catan.NumResourceCards(0), cards - cards / 2);
  SPIEL_CHECK_EQ(catan.phase(), Phase::kMoveRobber);
}

void TradeTest() {
  std::shared_ptr<const Game> game =
      FixedGame(3, {{"max_trade_offers", GameParameter(2)},
                    {"max_counter_steps", GameParameter(2)}});
  std::unique_ptr<State> state = StartFirstTurn(*game);
  const auto& catan = AsCatan(*state);
  auto [give, ask] = TradablePair(catan);
  std::vector<int> before = {catan.resources(0, give), catan.resources(0, ask),
                             catan.resources(1, give), catan.resources(1, ask)};

  // An offer has no resource on both sides and cards going both ways. Any
  // other resource may be asked for, whoever holds it.
  state->ApplyAction(kTradeGiveActionBase + give);
  SPIEL_CHECK_EQ(catan.phase(), Phase::kComposeTrade);
  SPIEL_CHECK_FALSE(
      absl::c_linear_search(state->LegalActions(), kProposeTrade));
  for (int r = 0; r < kNumResources; ++r) {
    SPIEL_CHECK_EQ(
        absl::c_linear_search(state->LegalActions(), kTradeAskActionBase + r),
        r != give);
  }
  state->ApplyAction(kTradeAskActionBase + ask);
  SPIEL_CHECK_FALSE(
      absl::c_linear_search(state->LegalActions(), kTradeGiveActionBase + ask));
  state->ApplyAction(kProposeTrade);

  // Player 1 counters with the same trade, built from its side; player 2
  // declines.
  SPIEL_CHECK_EQ(catan.phase(), Phase::kTradeResponse);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_TRUE(absl::c_linear_search(state->LegalActions(), kAcceptTrade));
  state->ApplyAction(kTradeGiveActionBase + ask);
  state->ApplyAction(kTradeAskActionBase + give);
  state->ApplyAction(kProposeTrade);
  SPIEL_CHECK_EQ(catan.trade_status(1), TradeStatus::kCountered);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 2);
  state->ApplyAction(kDeclineTrade);

  // Player 0 counters back. Player 1, its counters used up, can only accept
  // or decline, and its acceptance executes the trade.
  SPIEL_CHECK_EQ(catan.phase(), Phase::kTradeDecide);
  SPIEL_CHECK_EQ(
      state->LegalActions(),
      (std::vector<Action>{kExecuteTradeActionBase + 1,
                           kCounterTradeActionBase + 1, kEndNegotiation}));
  state->ApplyAction(kCounterTradeActionBase + 1);
  state->ApplyAction(kTradeGiveActionBase + give);
  state->ApplyAction(kTradeAskActionBase + ask);
  state->ApplyAction(kProposeTrade);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_EQ(state->LegalActions(),
                 (std::vector<Action>{kAcceptTrade, kDeclineTrade}));
  state->ApplyAction(kAcceptTrade);
  SPIEL_CHECK_EQ(catan.phase(), Phase::kMain);
  SPIEL_CHECK_EQ(catan.resources(0, give), before[0] - 1);
  SPIEL_CHECK_EQ(catan.resources(0, ask), before[1] + 1);
  SPIEL_CHECK_EQ(catan.resources(1, give), before[2] + 1);
  SPIEL_CHECK_EQ(catan.resources(1, ask), before[3] - 1);

  // Starting an offer uses one up, even if it is withdrawn.
  state->ApplyAction(kTradeGiveActionBase + ask);
  state->ApplyAction(kWithdrawTrade);
  SPIEL_CHECK_EQ(catan.phase(), Phase::kMain);
  for (Action a : state->LegalActions()) {
    SPIEL_CHECK_FALSE(a >= kTradeGiveActionBase && a < kProposeTrade);
  }
}

void OpenOfferTest() {
  std::shared_ptr<const Game> game = FixedGame(3);
  std::unique_ptr<State> state = StartFirstTurn(*game);
  const auto& catan = AsCatan(*state);
  auto [give, counter] = TradablePair(catan);

  // "My card for any card": open on one side only.
  state->ApplyAction(kTradeGiveActionBase + give);
  state->ApplyAction(kTradeAskAny);
  SPIEL_CHECK_FALSE(
      absl::c_linear_search(state->LegalActions(), kTradeGiveAny));
  state->ApplyAction(kProposeTrade);

  // It cannot be accepted, only countered with named cards.
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  for (Action a : {kAcceptTrade, kTradeGiveAny, kTradeAskAny}) {
    SPIEL_CHECK_FALSE(absl::c_linear_search(state->LegalActions(), a));
  }
  state->ApplyAction(kTradeGiveActionBase + counter);
  state->ApplyAction(kTradeAskActionBase + give);
  state->ApplyAction(kProposeTrade);
  state->ApplyAction(kDeclineTrade);
  SPIEL_CHECK_EQ(catan.phase(), Phase::kTradeDecide);
  int before = catan.resources(0, counter);
  state->ApplyAction(kExecuteTradeActionBase + 1);
  SPIEL_CHECK_EQ(catan.phase(), Phase::kMain);
  SPIEL_CHECK_EQ(catan.resources(0, counter), before + 1);
}

void EndOfGameTest() {
  // With 2 victory points, player 0 wins as soon as its first turn starts.
  std::shared_ptr<const Game> game =
      FixedGame(3, {{"victory_points", GameParameter(2)}});
  std::unique_ptr<State> state = PlayFirstLegalSetup(*game);
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->Returns(), (std::vector<double>{1.0, -0.5, -0.5}));

  // A game that hits the turn limit is a draw.
  game = FixedGame(2, {{"max_turns", GameParameter(1)}});
  state = StartFirstTurn(*game);
  state->ApplyAction(kEndTurn);
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->Returns(), (std::vector<double>{0.0, 0.0}));
}

void RandomBoardTest() {
  std::shared_ptr<const Game> game = LoadGame("catan");
  std::mt19937 rng(3);
  std::set<Action> spiral_starts;
  std::set<Player> first_players;
  for (int sim = 0; sim < 200; ++sim) {
    std::unique_ptr<State> state = game->NewInitialState();
    while (state->IsChanceNode()) {
      std::vector<Action> outcomes = state->LegalActions();
      Action outcome = outcomes[rng() % outcomes.size()];
      if (AsCatan(*state).phase() == Phase::kBoardSpiral) {
        spiral_starts.insert(outcome);
      }
      state->ApplyAction(outcome);
    }
    const auto& catan = AsCatan(*state);
    SPIEL_CHECK_EQ(state->CurrentPlayer(), catan.first_player());
    first_players.insert(catan.first_player());
    std::vector<int> terrains(kNumTerrains, 0);
    std::vector<int> ports(kNumPortTypes, 0);
    for (int h = 0; h < kNumHexes; ++h) {
      ++terrains[catan.terrain(h)];
      SPIEL_CHECK_EQ(catan.number(h) == 0, catan.terrain(h) == kDesert);
    }
    for (int i = 0; i < kNumPorts; ++i) ++ports[catan.port_type(i)];
    SPIEL_CHECK_EQ(terrains, (std::vector<int>{4, 3, 4, 4, 3, 1}));
    SPIEL_CHECK_EQ(ports, (std::vector<int>{1, 1, 1, 1, 1, 4}));
    SPIEL_CHECK_EQ(catan.terrain(catan.robber_hex()), kDesert);
    CheckRedNumbersApart(catan);
  }
  SPIEL_CHECK_EQ(spiral_starts.size(), kNumSpiralStarts);
  SPIEL_CHECK_EQ(first_players.size(), 4);
}

// Every legal action converts to a struct and back, also through JSON, and an
// observation shows only the observer's cards.
void CheckStructs(const State& state) {
  if (state.IsChanceNode() || state.IsTerminal()) return;
  for (Action action : state.LegalActions()) {
    std::unique_ptr<ActionStruct> action_struct = state.ActionToStruct(action);
    SPIEL_CHECK_EQ(state.StructToActions(*action_struct),
                   std::vector<Action>{action});
    SPIEL_CHECK_EQ(
        state.StructToActions(CatanActionStruct(action_struct->ToJson())),
        std::vector<Action>{action});
  }
  for (Player p = 0; p < state.NumPlayers(); ++p) {
    auto observation =
        CatanObservationStruct(state.ToObservationStruct(p)->ToJson());
    for (Player q = 0; q < state.NumPlayers(); ++q) {
      SPIEL_CHECK_EQ(observation.players[q].resources.empty(), q != p);
    }
    // Whether the turn's dice have been rolled is part of the observation.
    if (AsCatan(state).phase() == Phase::kPreRoll) {
      SPIEL_CHECK_EQ(observation.roll, 0);
    }
  }
}

void StructTest() {
  std::shared_ptr<const Game> game =
      LoadGame("catan", {{"max_turns", GameParameter(5)}});
  testing::RandomSimTest(*game, 1, /*serialize=*/false, /*verbose=*/false,
                         /*mask_test=*/false, &CheckStructs);

  // A whole offer is one struct, standing for the actions that build it.
  const std::vector<std::string> names = {"Lumber", "Brick", "Wool", "Grain",
                                          "Ore"};
  game = FixedGame(3);
  std::unique_ptr<State> state = StartFirstTurn(*game);
  auto [give, ask] = TradablePair(AsCatan(*state));
  CatanActionStruct offer;
  offer.kind = "propose_trade";
  offer.trade.give[names[give]] = 1;
  offer.trade.ask[names[ask]] = 1;
  std::vector<Action> actions = state->StructToActions(offer);
  SPIEL_CHECK_EQ(
      actions, (std::vector<Action>{kTradeGiveActionBase + give,
                                    kTradeAskActionBase + ask, kProposeTrade}));
  SPIEL_CHECK_EQ(state->ActionsToStruct(actions)->ToJson(), offer.ToJson());
  for (Action action : actions) state->ApplyAction(action);
  SPIEL_CHECK_EQ(AsCatan(*state).phase(), Phase::kTradeResponse);
}

void BasicCatanTests() {
  testing::LoadGameTest("catan");
  testing::ChanceOutcomesTest(*LoadGame("catan"));
  TopologyTest();
  ActionEncodingTest();
  SetupTest();
  ProductionAndRobberTest();
  ProductionRuleTest();
  DiscardTest();
  TradeTest();
  OpenOfferTest();
  EndOfGameTest();
  RandomBoardTest();
  HiddenInformationTest();
  StructTest();
  RandomSimTests();
}

}  // namespace
}  // namespace catan
}  // namespace open_spiel

int main(int argc, char** argv) { open_spiel::catan::BasicCatanTests(); }
