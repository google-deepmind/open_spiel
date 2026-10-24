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

#ifndef OPEN_SPIEL_GAMES_CATCH_THE_HIDDEN_EVADER_H_
#define OPEN_SPIEL_GAMES_CATCH_THE_HIDDEN_EVADER_H_

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

// Implemented by Jacopo888 (Jacopo D'Abundo). For other map files, see
// https://github.com/Jacopo888.
//
// Catch the Hidden Evader: a two-player, zero-sum pursuit-evasion game with
// imperfect information on a graph whose links have transport types.
//
// Player 0 is the evader, player 1 a team of pursuers that it moves one at a
// time. A chance event draws distinct start stations, first for the evader
// (privately), then for each pursuer. In every turn the evader moves, then the
// pursuers move in order. Each move follows one link and pays a ticket of the
// link's type (taxi, bus or underground). The evader also holds secret tickets,
// valid on any link and the only way to cross a ferry link, and double tickets:
// a double move is declared with the first of two evader moves, and the
// pursuers skip that turn. Every ticket a pursuer pays goes to the evader.
// The evader's station is shown to the pursuers only after the evader moves
// listed in `reveal_moves`; otherwise they see the ticket it used.
//
// The pursuers win by a capture: a pursuer moving onto the evader's station,
// or the evader moving onto a pursuer's station. They also win if the evader
// has no usable ticket at the start of one of its moves. The evader wins once
// the pursuers have answered its last move (`max_evader_moves`), or at once if
// that move starts a double move; it also wins as soon as no pursuer holds a
// ticket for a transport of the station it stands on. A pursuer with no legal
// move passes. A capture always takes precedence.
//
// Parameters:
//   "map"               string  built-in map, "city150" or "tiny"
//                                                         (def. "city150")
//   "map_file"          string  map file, overrides "map" when set (def. "")
//   "num_pursuers"      int     number of pursuers                  (def. 5)
//   "max_evader_moves"  int     evader moves in a game              (def. 24)
//   "reveal_moves"      string  evader moves after which its station is
//                               shown                 (def. "[3;8;13;18;24]")
//   "pursuer_tickets"   string  taxi, bus and underground tickets of each
//                               pursuer                     (def. "[10;8;4]")
//   "evader_tickets"    string  taxi, bus, underground, secret and double
//                               tickets of the evader     (def. "[4;3;3;5;2]")
// Lists are written in brackets with ';' between values, as in battleship.
//
// Map file format: plain text, '#' starts a comment. The first line is
// "stations <n>", the second "starts <s1> <s2> ...", then one link per line,
// "<a> <b> <taxi|bus|underground|ferry>". Stations are numbered from 1 to n
// and links are undirected; a pair of stations may have several transports,
// one line each. Replaying a serialized game needs the same map.
//
// Observation tensor, of the same size for both players, in this order: the
// one-hot station of each pursuer; the one-hot station last shown; the one-hot
// station of the evader (in the evader's own observation only); the evader's
// tickets, then each pursuer's, each divided by the most it can ever hold; the
// evader moves made and the moves since the last reveal, both divided by
// max_evader_moves; the one-hot phase (setup, evader, pursuers, terminal);
// double move pending; the one-hot pursuer to move; the one-hot winner. A
// block with nothing to show is zero.
//
// Information-state tensor: the observation tensor, then the whole history in
// a fixed layout: the one-hot start of each pursuer and of the evader (the
// evader's own only); for each evader move, whether it was played, its one-hot
// ticket and whether it was a double; the one-hot station shown at each
// reveal; the one-hot station after each evader move (the evader's own only);
// for each evader move and each pursuer, the pursuer's one-hot status (not
// played, moved, passed, skipped by a double), destination and ticket. Moves
// not played yet are zero, apart from the status "not played".

namespace open_spiel {
namespace catch_the_hidden_evader {

// Constants.
inline constexpr int kNumPlayers = 2;
inline constexpr Player kEvader = 0;
inline constexpr Player kPursuers = 1;

// Tickets. The first three are transports, which pursuers also use.
enum Ticket {
  kTaxi = 0,
  kBus = 1,
  kUnderground = 2,
  kSecret = 3,
  kDouble = 4,
};
inline constexpr int kNumTransports = 3;
inline constexpr int kNumMoveTickets = 4;  // Tickets that pay for one move.
inline constexpr int kNumEvaderTicketTypes = 5;

// The transports of a link, as a bit mask: one bit per transport, plus one
// for ferries. Masks are not ticket types: a ferry link takes only secret
// tickets, and a secret ticket fits any link.
inline constexpr int kTaxiLink = 1 << kTaxi;
inline constexpr int kBusLink = 1 << kBus;
inline constexpr int kUndergroundLink = 1 << kUnderground;
inline constexpr int kFerryLink = 1 << 3;

// Status of a pursuer in one pursuers' turn, recorded for perfect recall.
enum class PursuerStatus {
  kNotPlayed = 0,
  kMoved = 1,
  kPassed = 2,
  kSkipped = 3,  // The turn was skipped by a double move.
};
inline constexpr int kNumPursuerStatuses = 4;

// A graph of stations. Stations are indexed from 0; station s has the number
// s + 1 in map files, strings and structs.
struct Map {
  int num_stations = 0;
  std::vector<int> starts;
  // For each station, its neighbours and the link mask, by neighbour.
  std::vector<std::vector<std::pair<int, int>>> links;
};

// Parses and validates a map in the format above.
Map ParseMap(const std::string& text);

// One evader move, as recorded in the history.
struct EvaderMove {
  int to;
  int ticket;
  bool is_double;
  bool revealed;
};

// A move action, decoded.
struct MoveAction {
  int ticket;
  int target;
  bool is_double;  // Evader moves only.
};

// One pursuer's part of a pursuers' turn.
struct PursuerRecord {
  PursuerStatus status = PursuerStatus::kNotPlayed;
  int to = -1;
  int ticket = -1;
};

// Struct layer (OpenSpiel 2.0). Stations are numbered from 1, and -1 means
// none or hidden.
struct EvaderLogEntry {
  std::string ticket;
  bool double_move = false;
  int revealed_station = -1;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(EvaderLogEntry, ticket, double_move,
                                 revealed_station);
};

struct CatchTheHiddenEvaderStructContents {
  std::string phase;
  std::string current_player;
  int evader_station = -1;
  std::vector<int> pursuer_stations;
  std::vector<int> evader_tickets;
  std::vector<std::vector<int>> pursuer_tickets;
  int evader_moves = 0;
  bool double_pending = false;
  int pursuer_to_move = -1;
  int last_revealed_station = -1;
  std::vector<EvaderLogEntry> evader_log;
  std::string winner;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(CatchTheHiddenEvaderStructContents, phase,
                                 current_player, evader_station,
                                 pursuer_stations, evader_tickets,
                                 pursuer_tickets, evader_moves, double_pending,
                                 pursuer_to_move, last_revealed_station,
                                 evader_log, winner);
};

SPIEL_DEFINE_STRUCT(CatchTheHiddenEvaderStateStruct, StateStruct,
                    CatchTheHiddenEvaderStructContents);

struct CatchTheHiddenEvaderObservationExtras {
  int observing_player = -1;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(
      CatchTheHiddenEvaderObservationExtras, observing_player);
};

SPIEL_DEFINE_STRUCT_WITH_EXTRAS(CatchTheHiddenEvaderObservationStruct,
                                ObservationStruct,
                                CatchTheHiddenEvaderStructContents,
                                CatchTheHiddenEvaderObservationExtras);

// An action. `actor` is "chance", "evader" or "pursuer". A chance action draws
// the start station `to`; `pursuer` is the index of the pursuer acting.
struct CatchTheHiddenEvaderActionStruct : public ActionStruct {
  std::string actor;
  int pursuer = -1;
  int from = -1;
  int to = -1;
  std::string ticket;
  bool double_move = false;
  bool pass = false;
  SPIEL_STRUCT_BOILERPLATE(CatchTheHiddenEvaderActionStruct, actor, pursuer,
                           from, to, ticket, double_move, pass);
};

class CatchTheHiddenEvaderGame;

// State of an in-play game.
class CatchTheHiddenEvaderState : public State {
 public:
  explicit CatchTheHiddenEvaderState(std::shared_ptr<const Game> game);
  CatchTheHiddenEvaderState(const CatchTheHiddenEvaderState&) = default;

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override { return winner_ != kInvalidPlayer; }
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action action) override;
  std::vector<Action> LegalActions() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::unique_ptr<State> ResampleFromInfostate(
      int player_id, std::function<double()> rng) const override;

  std::unique_ptr<StateStruct> ToStruct() const override;
  std::unique_ptr<ObservationStruct> ToObservationStruct(
      Player player) const override;
  std::unique_ptr<ActionStruct> ActionToStruct(Player player,
                                               Action action_id) const override;
  std::vector<Action> StructToActions(
      const ActionStruct& action_struct) const override;

  // Accessors. Stations are indexed from 0; -1 means not placed yet.
  int EvaderStation() const { return evader_; }
  const std::vector<int>& PursuerStations() const { return pursuers_; }
  int EvaderTickets(int ticket) const { return evader_tickets_[ticket]; }
  int PursuerTickets(int pursuer, int ticket) const {
    return pursuer_tickets_[pursuer][ticket];
  }
  int NumEvaderMoves() const { return static_cast<int>(evader_log_.size()); }
  bool DoublePending() const { return double_pending_; }
  Player Winner() const { return winner_; }

 protected:
  void DoApplyAction(Action action) override;

 private:
  const CatchTheHiddenEvaderGame& GetGame() const;
  void Reset();  // Back to the game's initial state, before the setup.
  bool InSetup() const { return drawn_.size() < pursuers_.size() + 1; }
  bool IsReveal(int move_number) const;
  int LastRevealedStation() const;
  std::vector<Action> EvaderActions() const;
  std::vector<Action> PursuerActions(int pursuer) const;
  bool EvaderCanMove(int station) const;  // With the evader's tickets.
  bool PursuersImmobile() const;
  void ApplyChance(Action action);
  void ApplyEvaderMove(Action action);
  void ApplyPursuerAction(Action action);
  void StartEvaderMove();
  std::string PhaseString() const;
  std::string EvaderTicketsString() const;
  std::string PursuersString() const;
  CatchTheHiddenEvaderStructContents StructContents() const;
  // Steps of ResampleFromInfostate.
  std::unique_ptr<State> ResampleSetup(std::function<double()> rng) const;
  std::vector<std::vector<double>> EvaderPathCounts() const;
  std::vector<double> EndingWeights(const std::vector<double>& arrivals,
                                    const std::vector<int>& pursuers) const;
  std::unique_ptr<State> ReplayWithEvaderStations(
      const std::vector<int>& stations) const;

  // Start-list indices drawn so far: the evader's first, then the pursuers'.
  std::vector<int> drawn_;
  int evader_ = -1;
  std::vector<int> pursuers_;
  std::array<int, kNumEvaderTicketTypes> evader_tickets_;
  std::vector<std::array<int, kNumTransports>> pursuer_tickets_;
  bool double_pending_ = false;
  int pursuer_to_move_ = 0;
  Player current_player_ = kChancePlayerId;
  Player winner_ = kInvalidPlayer;
  std::vector<EvaderMove> evader_log_;
  // The pursuers' turn answering each evader move, one record per pursuer.
  std::vector<std::vector<PursuerRecord>> pursuer_log_;
};

// Game object.
class CatchTheHiddenEvaderGame : public Game {
 public:
  explicit CatchTheHiddenEvaderGame(const GameParameters& params);

  int NumDistinctActions() const override { return PassAction() + 1; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::make_unique<CatchTheHiddenEvaderState>(shared_from_this());
  }
  int MaxChanceOutcomes() const override {
    return static_cast<int>(map_.starts.size());
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  double MaxUtility() const override { return 1; }
  absl::optional<double> UtilitySum() const override { return 0; }
  std::vector<int> ObservationTensorShape() const override;
  std::vector<int> InformationStateTensorShape() const override;
  int MaxGameLength() const override {
    return max_evader_moves_ * (num_pursuers_ + 1);
  }
  int MaxChanceNodesInHistory() const override { return num_pursuers_ + 1; }

  const Map& GetMap() const { return map_; }
  int NumStations() const { return map_.num_stations; }
  int NumPursuers() const { return num_pursuers_; }
  int MaxEvaderMoves() const { return max_evader_moves_; }
  const std::vector<int>& RevealMoves() const { return reveal_moves_; }
  int StartingEvaderTickets(int ticket) const {
    return evader_tickets_[ticket];
  }
  int StartingPursuerTickets(int ticket) const {
    return pursuer_tickets_[ticket];
  }

  // Action ids, with stations indexed from 0: first the evader's moves, a
  // single and a double one for each ticket and target, then the pursuers'
  // moves, one for each transport and target, then the pass.
  Action EvaderMoveAction(int ticket, int target, bool is_double) const {
    return 2 * (ticket * NumStations() + target) + (is_double ? 1 : 0);
  }
  Action PursuerMoveAction(int ticket, int target) const {
    return 2 * kNumMoveTickets * NumStations() + ticket * NumStations() +
           target;
  }
  Action PassAction() const { return PursuerMoveAction(kNumTransports, 0); }
  MoveAction DecodeEvaderMove(Action action) const {
    return {static_cast<int>(action / 2 / NumStations()),
            static_cast<int>(action / 2 % NumStations()), action % 2 == 1};
  }
  MoveAction DecodePursuerMove(Action action) const {  // Not the pass.
    const int offset = action - PursuerMoveAction(0, 0);
    return {offset / NumStations(), offset % NumStations(), false};
  }

  // Divisors that bring ticket counts within [0, 1] in tensors.
  double EvaderTicketScale(int ticket) const;
  double PursuerTicketScale(int ticket) const;

 private:
  Map map_;
  int num_pursuers_;
  int max_evader_moves_;
  std::vector<int> reveal_moves_;
  std::array<int, kNumEvaderTicketTypes> evader_tickets_;
  std::array<int, kNumTransports> pursuer_tickets_;
};

}  // namespace catch_the_hidden_evader
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_CATCH_THE_HIDDEN_EVADER_H_
