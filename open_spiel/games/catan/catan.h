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

#ifndef OPEN_SPIEL_GAMES_CATAN_H_
#define OPEN_SPIEL_GAMES_CATAN_H_

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// The base game of Catan (Settlers of Catan) for 2 to 4 players, see
// https://en.wikipedia.org/wiki/Catan
//
// Rules follow the base game with the variable board setup:
// - Terrain and port types are shuffled by chance. Number tokens follow the
//   standard spiral from a corner drawn by chance, skipping the desert. With
//   board=fixed, a fixed layout is used instead.
// - The first player is drawn by chance. Setup places a settlement and a road
//   per player in seat order from them, then again in reverse; the second
//   settlement collects one resource from each adjacent hex.
// - Trading and building happen in one phase after the roll. One development
//   card may be played per turn, before or after rolling, but not on the turn
//   it was bought. Victory point cards count as soon as they are drawn.
// - On a 7, players holding more than 7 resource cards discard half, one card
//   per action, then the robber moves and steals.
// - If the bank cannot pay a resource to everyone owed it, nobody gets it,
//   unless only one player is owed it, who takes what is left.
// - The first player to reach `victory_points` on their own turn wins.
//
// Trading between players is only with the player on turn, after the roll,
// with cards going both ways and no resource on both sides. The player on turn
// builds an offer a card at a time and proposes it. Each other player in seat
// order accepts, declines or builds a counter-offer. The player on turn then
// executes an acceptance or counter, counters a counter (whose acceptance
// executes it at once), or ends the negotiation. An offer may leave one side
// open ("any card"); it can then only be countered, and counters always name
// their cards.
//
// To keep games finite, `max_trade_offers` caps the offers started per turn
// and `max_counter_steps` the counters per player per offer. Starting to build
// one uses it up, even if it is withdrawn.
//
// Returns are +1 for the winner and -1/(n-1) for the others, or 0 for all if
// the game reaches `max_turns` turns after setup.
//
// Resource and development cards are private. A stolen card is seen only by
// the two players involved, and a drawn development card only by its buyer.
// The information state string is the history with those outcomes hidden.
//
// Parameters:
//   "players"           int     number of players, 2 to 4 (default = 4)
//   "victory_points"    int     points needed to win (default = 10)
//   "board"             string  "random" or "fixed" (default = "random")
//   "max_trade_offers"  int     trade offers per turn, 0 for no trading
//                               between players (default = 5)
//   "max_counter_steps" int     counters per player per offer (default = 3)
//   "max_turns"         int     turns after setup before a draw
//                               (default = 1000)

namespace open_spiel {
namespace catan {

inline constexpr int kMinPlayers = 2;
inline constexpr int kMaxPlayers = 4;

inline constexpr int kNumHexes = 19;
inline constexpr int kNumHexCorners = 6;
inline constexpr int kNumVertices = 54;
inline constexpr int kNumEdges = 72;
inline constexpr int kNumPorts = 9;
inline constexpr int kNumResources = 5;
inline constexpr int kNumTerrains = 6;
inline constexpr int kNumPortTypes = 6;
inline constexpr int kNumDevCardTypes = 5;
inline constexpr int kNumDiceSums = 11;  // 2 to 12.
inline constexpr int kNumSpiralStarts = 6;

inline constexpr int kBankResourceCount = 19;
inline constexpr int kNumRoads = 15;
inline constexpr int kNumSettlements = 5;
inline constexpr int kNumCities = 4;
inline constexpr int kMinLongestRoad = 5;
inline constexpr int kMinLargestArmy = 3;
inline constexpr int kDiscardThreshold = 7;

enum Resource { kLumber = 0, kBrick, kWool, kGrain, kOre };

// Terrain t < kDesert produces resource t.
enum Terrain { kForest = 0, kHills, kPasture, kFields, kMountains, kDesert };

// Port types 0-4 trade that resource 2:1; kGenericPort trades anything 3:1.
inline constexpr int kGenericPort = 5;

enum DevCard {
  kKnight = 0,
  kVictoryPoint,
  kRoadBuilding,
  kYearOfPlenty,
  kMonopoly
};

// Unordered pairs of resources, with repetition.
inline constexpr int kNumYearOfPlentyActions =
    kNumResources * (kNumResources + 1) / 2;
// A resource to give and a different one to get.
inline constexpr int kNumResourcePairs = kNumResources * (kNumResources - 1);

// Player actions.
inline constexpr Action kSettlementActionBase = 0;
inline constexpr Action kCityActionBase = kSettlementActionBase + kNumVertices;
inline constexpr Action kRoadActionBase = kCityActionBase + kNumVertices;
inline constexpr Action kBuyDevCard = kRoadActionBase + kNumEdges;
inline constexpr Action kPlayKnight = kBuyDevCard + 1;
inline constexpr Action kPlayRoadBuilding = kPlayKnight + 1;
inline constexpr Action kYearOfPlentyActionBase = kPlayRoadBuilding + 1;
inline constexpr Action kMonopolyActionBase =
    kYearOfPlentyActionBase + kNumYearOfPlentyActions;
inline constexpr Action kMoveRobberActionBase =
    kMonopolyActionBase + kNumResources;
inline constexpr Action kStealActionBase = kMoveRobberActionBase + kNumHexes;
inline constexpr Action kDiscardActionBase = kStealActionBase + kMaxPlayers;
inline constexpr Action kBankTradeActionBase =
    kDiscardActionBase + kNumResources;
// An offer or counter is built a card at a time, from the side of the player
// building it.
inline constexpr Action kTradeGiveActionBase =
    kBankTradeActionBase + kNumResourcePairs;
inline constexpr Action kTradeAskActionBase =
    kTradeGiveActionBase + kNumResources;
inline constexpr Action kTradeGiveAny = kTradeAskActionBase + kNumResources;
inline constexpr Action kTradeAskAny = kTradeGiveAny + 1;
inline constexpr Action kProposeTrade = kTradeAskAny + 1;
inline constexpr Action kWithdrawTrade = kProposeTrade + 1;
inline constexpr Action kAcceptTrade = kWithdrawTrade + 1;
inline constexpr Action kDeclineTrade = kAcceptTrade + 1;
// The player on turn executes the trade on the table with a player, or
// counters that player's counter.
inline constexpr Action kExecuteTradeActionBase = kDeclineTrade + 1;
inline constexpr Action kCounterTradeActionBase =
    kExecuteTradeActionBase + kMaxPlayers;
inline constexpr Action kEndNegotiation = kCounterTradeActionBase + kMaxPlayers;
inline constexpr Action kRollDice = kEndNegotiation + 1;
inline constexpr Action kEndTurn = kRollDice + 1;
inline constexpr int kNumDistinctActions = kEndTurn + 1;

// Chance outcomes index the current phase's outcomes: terrain, port type,
// spiral start, first player, dice sum minus 2, stolen resource or drawn card.
inline constexpr int kMaxChanceOutcomes = kNumDiceSums;

enum class Phase {
  kBoardTerrain = 0,  // Chance: terrain of the next hex.
  kBoardPorts,        // Chance: type of the next port.
  kBoardSpiral,       // Chance: the corner the number tokens start from.
  kFirstPlayer,       // Chance: who plays first.
  kSetupSettlement,
  kSetupRoad,
  kPreRoll,  // Roll, or play a development card first.
  kRoll,     // Chance: dice sum.
  kDiscard,
  kMoveRobber,
  kChooseVictim,
  kSteal,  // Chance: stolen resource.
  kMain,
  kRoadBuilding,
  kComposeTrade,   // Building an offer or counter.
  kTradeResponse,  // Answering an offer or counter.
  kTradeDecide,    // The player on turn, once the answers are in.
  kDrawDevCard,    // Chance: drawn development card.
  kGameOver,
};
inline constexpr int kNumPhases = static_cast<int>(Phase::kGameOver) + 1;

std::ostream& operator<<(std::ostream& os, Phase phase);

// Where the trade between the player on turn and another player stands.
enum class TradeStatus {
  kNone = 0,
  kPending,  // Waiting for that player's answer.
  kAccepted,
  kDeclined,
  kCountered,  // Waiting for the player on turn.
};
inline constexpr int kNumTradeStatuses =
    static_cast<int>(TradeStatus::kCountered) + 1;

std::ostream& operator<<(std::ostream& os, TradeStatus status);

// Fixed board topology. Hexes, vertices and edges are numbered row by row from
// the top, left to right; hexes are pointy-topped.
struct Topology {
  std::array<std::array<int, kNumHexCorners>, kNumHexes> hex_vertices;
  std::array<std::vector<int>, kNumVertices> vertex_hexes;
  std::array<std::vector<int>, kNumVertices> vertex_edges;
  std::array<std::vector<int>, kNumVertices> vertex_neighbors;
  std::array<std::array<int, 2>, kNumEdges> edge_vertices;
  std::array<std::array<int, 2>, kNumPorts> port_vertices;
  std::array<int, kNumVertices> vertex_port;  // Port index, or -1.
  // Number token order from each spiral start, counterclockwise.
  std::array<std::array<int, kNumHexes>, kNumSpiralStarts> spirals;
};
const Topology& GetTopology();

std::pair<int, int> ResourcePairFromIndex(int index);
std::pair<int, int> YearOfPlentyPair(int index);

// Cards to give and to ask for, with "any card" slots.
struct Bundle {
  std::array<int, kNumResources> give{};
  std::array<int, kNumResources> ask{};
  int give_any = 0;
  int ask_any = 0;
};

// OpenSpiel's struct API. Cards are counted by name.
struct CatanTradeStruct {
  std::map<std::string, int> give;
  std::map<std::string, int> ask;
  int give_any = 0;
  int ask_any = 0;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(CatanTradeStruct, give, ask,
                                              give_any, ask_any);
};

struct CatanPlayerStruct {
  int victory_points = 0;  // Public points only, when observed by another.
  // Empty when hidden. New development cards were bought this turn.
  std::map<std::string, int> resources;
  std::map<std::string, int> development_cards;
  std::map<std::string, int> new_development_cards;
  int resource_count = 0;
  int development_card_count = 0;
  int knights_played = 0;
  int road_length = 0;
  int discard = 0;  // Cards still to discard.
  std::vector<int> settlements;
  std::vector<int> cities;
  std::vector<int> roads;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(
      CatanPlayerStruct, victory_points, resources, development_cards,
      new_development_cards, resource_count, development_card_count,
      knights_played, road_length, discard, settlements, cities, roads);
};

struct CatanStructContents {
  std::string phase;
  int current_player = kInvalidPlayer;
  int turn_player = kInvalidPlayer;
  int turn = 0;
  int roll = 0;  // This turn's dice, 0 before the roll.
  bool development_card_played = false;
  int free_roads = 0;
  int trade_offers_made = 0;
  int victim = kInvalidPlayer;     // Of the robber, while a card is stolen.
  std::vector<std::string> hexes;  // "Forest 11", "Desert", or "?".
  std::vector<std::string> ports;  // "3:1", "Wool 2:1", or "?".
  int robber_hex = 0;
  std::map<std::string, int> bank;
  int development_deck = 0;
  int longest_road = kInvalidPlayer;
  int largest_army = kInvalidPlayer;
  std::vector<CatanPlayerStruct> players;
  // The offer or counter being built, the offer on the table, and each
  // player's trade with the player on turn, from the player on turn's side.
  CatanTradeStruct building;
  CatanTradeStruct offer;
  bool offer_round = false;  // The other players are answering the offer.
  std::vector<CatanTradeStruct> trades;
  std::vector<std::string> trade_status;
  std::vector<int> counters;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(
      CatanStructContents, phase, current_player, turn_player, turn, roll,
      development_card_played, free_roads, trade_offers_made, victim, hexes,
      ports, robber_hex, bank, development_deck, longest_road, largest_army,
      players, building, offer, offer_round, trades, trade_status, counters);
};

SPIEL_DEFINE_STRUCT(CatanStateStruct, StateStruct, CatanStructContents);

struct CatanObservationExtras {
  int observing_player = kInvalidPlayer;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(CatanObservationExtras,
                                              observing_player);
};

SPIEL_DEFINE_STRUCT_WITH_EXTRAS(CatanObservationStruct, ObservationStruct,
                                CatanStructContents, CatanObservationExtras);

// One action, by kind: settlement and city (vertex); road (edge);
// buy_development_card; play_knight; play_road_building; play_year_of_plenty
// (two resources); play_monopoly and discard (a resource); move_robber (hex);
// rob (player); bank_trade (resources to give and get); trade_give and
// trade_ask (a resource); trade_give_any; trade_ask_any; propose;
// withdraw_trade; accept_trade; decline_trade; execute_trade and
// counter_trade (player); end_negotiation; roll; end_turn.
// Or "propose_trade": a whole offer or counter (trade), which stands for the
// actions that build and propose it; when the player on turn counters a
// counter, `player` names whom.
struct CatanActionStruct : public ActionStruct {
  std::string kind;
  int vertex = -1;
  int edge = -1;
  int hex = -1;
  int player = -1;
  std::vector<std::string> resources;
  CatanTradeStruct trade;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(CatanActionStruct, kind, vertex,
                                              edge, hex, player, resources,
                                              trade);
  CatanActionStruct() = default;
  explicit CatanActionStruct(const std::string& json_str) {
    nlohmann::json::parse(json_str).get_to(*this);
  }
  explicit CatanActionStruct(const nlohmann::json& json) { json.get_to(*this); }
  nlohmann::json to_json_base() const override { return *this; }
};

class CatanState : public State {
 public:
  explicit CatanState(std::shared_ptr<const Game> game);
  CatanState(const CatanState&) = default;

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override { return phase_ == Phase::kGameOver; }
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::vector<Action> LegalActions() const override;
  std::unique_ptr<StateStruct> ToStruct() const override;
  std::unique_ptr<ObservationStruct> ToObservationStruct(
      Player player) const override;
  std::unique_ptr<ActionStruct> ActionToStruct(Player player,
                                               Action action) const override;
  std::vector<Action> StructToActions(
      const ActionStruct& action_struct) const override;
  std::unique_ptr<ActionStruct> ActionsToStruct(
      Player player, absl::Span<const Action> actions) const override;

  // Accessors, mainly for tests.
  Phase phase() const { return phase_; }
  Player turn_player() const { return turn_player_; }
  Player first_player() const { return first_player_; }
  int terrain(int hex) const { return terrain_[hex]; }
  int number(int hex) const { return number_[hex]; }
  int port_type(int port) const { return port_type_[port]; }
  int robber_hex() const { return robber_hex_; }
  Player vertex_owner(int vertex) const { return vertex_owner_[vertex]; }
  bool is_city(int vertex) const { return is_city_[vertex]; }
  Player edge_owner(int edge) const { return edge_owner_[edge]; }
  int resources(Player player, int resource) const {
    return resources_[player][resource];
  }
  int bank(int resource) const { return bank_[resource]; }
  int knights_played(Player player) const { return knights_played_[player]; }
  Player longest_road_holder() const { return longest_road_holder_; }
  Player largest_army_holder() const { return largest_army_holder_; }
  TradeStatus trade_status(Player player) const {
    return trade_status_[player];
  }
  int LongestRoad(Player player) const;
  int VictoryPoints(Player player) const;
  int PublicVictoryPoints(Player player) const;
  int NumResourceCards(Player player) const;

 protected:
  void DoApplyAction(Action action_id) override;

 private:
  void ApplyChanceOutcome(Action outcome);
  void ApplyPlayerAction(Action action);
  void ApplyTurnAction(Player player, Action action);
  void PlaceNumbers(int spiral_start);
  void PlaceSettlement(Player player, int vertex);
  void PlaceRoad(Player player, int edge);
  void PlayDevCard(Player player, int card);
  void FinishSetupRoad();
  void Produce(int dice_sum);
  void StartRobber(Phase return_phase);
  void ResolveRobberMove();
  void EndTurn();
  void UpdateLongestRoadHolder();
  void UpdateLargestArmy();
  void CheckWin();

  // Trading between players.
  void StartComposing(Player composer, Player counterparty);
  void AddComposeCard(Action action);
  void ProposeTrade();
  void WithdrawTrade();
  void Answered(Player player);
  void NextTradeResponder(Player after);
  void AwaitDecision();
  void ExecuteTrade(Player other);
  void EndNegotiation();

  Player DiscardingPlayer() const;
  std::vector<Player> RobberVictims() const;
  std::array<int, kNumResources> TradeRates(Player player) const;
  bool CanAfford(Player player,
                 const std::array<int, kNumResources>& cost) const;
  void Pay(Player player, const std::array<int, kNumResources>& cost);
  bool CanPlaceSettlement(Player player, int vertex, bool setup) const;
  bool CanPlaceRoad(Player player, int edge) const;
  bool AnyRoadPlaceable(Player player) const;
  bool CanPlayDevCard(Player player, int card) const;
  void AddDevCardActions(std::vector<Action>* actions) const;
  void AddComposeActions(Player composer, const Bundle& built, bool offer,
                         std::vector<Action>* actions) const;
  std::string BoardString() const;
  std::string TradeString() const;
  std::string PlayerString(Player player, Player observer) const;
  void FillStruct(Player observer, CatanStructContents* contents) const;
  int Seat(Player player, Player observer) const;

  int num_players_;
  int victory_points_;
  int max_trade_offers_;
  int max_counter_steps_;
  int max_turns_;

  Phase phase_ = Phase::kBoardTerrain;
  Player turn_player_ = 0;
  Player first_player_ = 0;
  int turn_number_ = 0;
  Player winner_ = kInvalidPlayer;

  // Board.
  std::array<int, kNumHexes> terrain_;
  std::array<int, kNumHexes> number_{};  // 0 for the desert.
  std::array<int, kNumPorts> port_type_;
  std::array<int, kNumTerrains> terrains_left_;
  std::array<int, kNumPortTypes> ports_left_;
  int board_step_ = 0;
  int robber_hex_ = 0;

  // Pieces.
  std::array<Player, kNumVertices> vertex_owner_;
  std::array<bool, kNumVertices> is_city_{};
  std::array<Player, kNumEdges> edge_owner_;
  std::array<int, kMaxPlayers> roads_left_;
  std::array<int, kMaxPlayers> settlements_left_;
  std::array<int, kMaxPlayers> cities_left_;

  // Cards.
  std::array<std::array<int, kNumResources>, kMaxPlayers> resources_{};
  std::array<int, kNumResources> bank_;
  std::array<std::array<int, kNumDevCardTypes>, kMaxPlayers> dev_cards_{};
  std::array<std::array<int, kNumDevCardTypes>, kMaxPlayers> dev_bought_{};
  std::array<int, kNumDevCardTypes> dev_deck_;
  std::array<int, kMaxPlayers> knights_played_{};
  std::array<int, kMaxPlayers> road_length_{};
  Player longest_road_holder_ = kInvalidPlayer;
  Player largest_army_holder_ = kInvalidPlayer;

  // Turn.
  int setup_step_ = 0;
  int setup_vertex_ = -1;
  int last_roll_ = 0;
  bool dev_card_played_ = false;
  Phase return_phase_ = Phase::kMain;
  std::array<int, kMaxPlayers> discard_left_{};
  Player victim_ = kInvalidPlayer;
  int free_roads_left_ = 0;

  // Trading. The offer and each player's trade are from the side of the
  // player on turn; the bundle being built is from its composer's side.
  int trade_offers_made_ = 0;
  Player composer_ = kInvalidPlayer;
  Player counterparty_ = kInvalidPlayer;  // Of a counter; none for an offer.
  Bundle compose_;
  Bundle offer_;
  bool offer_round_ = false;  // The other players are answering the offer.
  Player responder_ = kInvalidPlayer;
  std::array<Bundle, kMaxPlayers> trades_;
  std::array<TradeStatus, kMaxPlayers> trade_status_{};
  std::array<int, kMaxPlayers> counters_{};

  // For each history entry, a bit mask of the players who see it.
  std::vector<uint8_t> visibility_;
};

class CatanGame : public Game {
 public:
  explicit CatanGame(const GameParameters& params);

  int NumDistinctActions() const override { return kNumDistinctActions; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::make_unique<CatanState>(shared_from_this());
  }
  int MaxChanceOutcomes() const override { return kMaxChanceOutcomes; }
  int NumPlayers() const override { return num_players_; }
  double MinUtility() const override { return -1; }
  double MaxUtility() const override { return 1; }
  absl::optional<double> UtilitySum() const override { return 0; }
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override;
  int MaxChanceNodesInHistory() const override;
  std::pair<std::string, std::string> ActionStructSpec() const override;

  int victory_points() const { return victory_points_; }
  bool fixed_board() const { return fixed_board_; }
  int max_trade_offers() const { return max_trade_offers_; }
  int max_counter_steps() const { return max_counter_steps_; }
  int max_turns() const { return max_turns_; }

 private:
  int num_players_;
  int victory_points_;
  bool fixed_board_;
  int max_trade_offers_;
  int max_counter_steps_;
  int max_turns_;
};

}  // namespace catan
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_CATAN_H_
