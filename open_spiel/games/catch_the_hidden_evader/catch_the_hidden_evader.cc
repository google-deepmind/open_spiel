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

#include "open_spiel/games/catch_the_hidden_evader/catch_the_hidden_evader.h"

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/strings/ascii.h"
#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/abseil-cpp/absl/strings/strip.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/catch_the_hidden_evader/catch_the_hidden_evader_maps.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/file.h"

namespace open_spiel {
namespace catch_the_hidden_evader {
namespace {

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"catch_the_hidden_evader",
    /*long_name=*/"Catch the Hidden Evader",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/kNumPlayers,
    /*min_num_players=*/kNumPlayers,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {{"map", GameParameter(std::string("city150"))},
     {"map_file", GameParameter(std::string(""))},
     {"num_pursuers", GameParameter(5)},
     {"max_evader_moves", GameParameter(24)},
     {"reveal_moves", GameParameter(std::string("[3;8;13;18;24]"))},
     {"pursuer_tickets", GameParameter(std::string("[10;8;4]"))},
     {"evader_tickets", GameParameter(std::string("[4;3;3;5;2]"))}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new CatchTheHiddenEvaderGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

constexpr std::array<const char*, kNumEvaderTicketTypes> kTicketNames = {
    "taxi", "bus", "underground", "secret", "double"};

// Station numbers as shown to users: stations are indexed from 0 internally.
std::string StationString(int station) {
  return station < 0 ? "-" : absl::StrCat(station + 1);
}

int StationNumber(int station) { return station < 0 ? -1 : station + 1; }

std::string PlayerString(Player player) {
  switch (player) {
    case kEvader:
      return "evader";
    case kPursuers:
      return "pursuers";
    case kChancePlayerId:
      return "chance";
    case kTerminalPlayerId:
      return "terminal";
    default:
      return DefaultPlayerString(player);
  }
}

int TicketFromString(const std::string& name, int num_tickets) {
  for (int ticket = 0; ticket < num_tickets; ++ticket) {
    if (name == kTicketNames[ticket]) return ticket;
  }
  SpielFatalError(absl::StrCat("Unknown ticket: ", name));
}

// Whether `ticket` pays for a move along a link with this mask.
bool TicketFitsLink(int ticket, int link_mask) {
  return ticket == kSecret || (link_mask & (1 << ticket)) != 0;
}

// Parses a list of integers in brackets, separated by ';', e.g. "[10;8;4]",
// as battleship does: commas would clash with the separator of parameters in
// game strings, and the brackets keep a single value a string.
std::vector<int> ParseIntList(const std::string& name,
                              const std::string& text) {
  absl::string_view list = absl::StripAsciiWhitespace(text);
  if (!absl::ConsumePrefix(&list, "[") || !absl::ConsumeSuffix(&list, "]")) {
    SpielFatalError(absl::StrCat(
        "Parameter ", name, ": expected a list such as [1;2], got ", text));
  }
  std::vector<int> values;
  for (absl::string_view piece :
       absl::StrSplit(list, ';', absl::SkipWhitespace())) {
    int value;
    if (!absl::SimpleAtoi(piece, &value)) {
      SpielFatalError(
          absl::StrCat("Parameter ", name, ": not an integer: ", piece));
    }
    values.push_back(value);
  }
  return values;
}

// Writes consecutive blocks of a tensor.
class TensorWriter {
 public:
  explicit TensorWriter(absl::Span<float> values) : values_(values) {
    absl::c_fill(values_, 0.0f);
  }
  // A one-hot block of `size` values; nothing is set when index < 0.
  void OneHot(int size, int index) {
    if (index >= 0) {
      SPIEL_CHECK_LT(index, size);
      values_[position_ + index] = 1.0f;
    }
    position_ += size;
  }
  void Value(double value) { values_[position_++] = value; }
  void Skip(int size) { position_ += size; }
  int position() const { return position_; }

 private:
  absl::Span<float> values_;
  int position_ = 0;
};

// Draws an index with probability proportional to its weight.
int DrawIndex(const std::vector<double>& weights,
              const std::function<double()>& rng) {
  const double total = absl::c_accumulate(weights, 0.0);
  SPIEL_CHECK_GT(total, 0.0);
  double r = rng() * total;
  const int size = weights.size();
  for (int i = 0; i < size; ++i) {
    if (weights[i] == 0.0) continue;
    if (r < weights[i]) return i;
    r -= weights[i];
  }
  for (int i = size - 1; i >= 0; --i) {
    if (weights[i] > 0.0) return i;  // Rounding left r at the very end.
  }
  SpielFatalError("No consistent evader history.");
}

}  // namespace

Map ParseMap(const std::string& text) {
  Map map;
  std::set<std::tuple<int, int, int>> seen;
  bool has_stations = false;
  bool has_starts = false;
  int line_number = 0;
  for (absl::string_view raw_line : absl::StrSplit(text, '\n')) {
    ++line_number;
    absl::string_view line = raw_line.substr(0, raw_line.find('#'));
    std::vector<std::string> tokens =
        absl::StrSplit(line, absl::ByAnyChar(" \t\r"), absl::SkipEmpty());
    if (tokens.empty()) continue;
    auto fail = [line_number](const std::string& message) {
      SpielFatalError(absl::StrCat("Map line ", line_number, ": ", message));
    };
    auto station = [&](const std::string& token) {
      int number = 0;
      if (!absl::SimpleAtoi(token, &number) || number < 1 ||
          number > map.num_stations) {
        fail(absl::StrCat("invalid station ", token));
      }
      return number - 1;
    };
    if (!has_stations) {
      if (tokens.size() != 2 || tokens[0] != "stations" ||
          !absl::SimpleAtoi(tokens[1], &map.num_stations) ||
          map.num_stations < 2) {
        fail("expected \"stations <n>\" with n >= 2");
      }
      map.links.resize(map.num_stations);
      has_stations = true;
    } else if (!has_starts) {
      if (tokens.size() < 2 || tokens[0] != "starts") {
        fail("expected \"starts <s1> <s2> ...\"");
      }
      for (const std::string& token : absl::MakeConstSpan(tokens).subspan(1)) {
        int s = station(token);
        if (absl::c_linear_search(map.starts, s)) {
          fail(absl::StrCat("repeated start station ", token));
        }
        map.starts.push_back(s);
      }
      has_starts = true;
    } else {
      if (tokens.size() != 3) fail("expected \"<a> <b> <transport>\"");
      int a = station(tokens[0]);
      int b = station(tokens[1]);
      if (a == b) fail("a link must join two different stations");
      int mask = 0;
      if (tokens[2] == "taxi") {
        mask = kTaxiLink;
      } else if (tokens[2] == "bus") {
        mask = kBusLink;
      } else if (tokens[2] == "underground") {
        mask = kUndergroundLink;
      } else if (tokens[2] == "ferry") {
        mask = kFerryLink;
      } else {
        fail(absl::StrCat("unknown transport ", tokens[2]));
      }
      if (!seen.insert({std::min(a, b), std::max(a, b), mask}).second) {
        fail("repeated link");
      }
      for (auto [from, to] : {std::make_pair(a, b), std::make_pair(b, a)}) {
        auto it = absl::c_find_if(map.links[from],
                                  [to = to](const std::pair<int, int>& link) {
                                    return link.first == to;
                                  });
        if (it == map.links[from].end()) {
          map.links[from].push_back({to, mask});
        } else {
          it->second |= mask;
        }
      }
    }
  }
  if (!has_starts) SpielFatalError("Map: missing \"stations\" or \"starts\"");
  for (int s = 0; s < map.num_stations; ++s) {
    if (map.links[s].empty()) {
      SpielFatalError(absl::StrCat("Map: station ", s + 1, " has no link"));
    }
    absl::c_sort(map.links[s]);
  }
  return map;
}

// ---------------------------------------------------------------------------
// Game.

CatchTheHiddenEvaderGame::CatchTheHiddenEvaderGame(const GameParameters& params)
    : Game(kGameType, params),
      num_pursuers_(ParameterValue<int>("num_pursuers")),
      max_evader_moves_(ParameterValue<int>("max_evader_moves")) {
  const std::string map_file = ParameterValue<std::string>("map_file");
  const std::string map_name = ParameterValue<std::string>("map");
  if (!map_file.empty()) {
    map_ = ParseMap(file::ReadContentsFromFile(map_file, "r"));
  } else if (map_name == "city150") {
    map_ = ParseMap(kCity150Map);
  } else if (map_name == "tiny") {
    map_ = ParseMap(kTinyMap);
  } else {
    SpielFatalError(absl::StrCat("Unknown map: ", map_name));
  }
  if (num_pursuers_ < 1 ||
      static_cast<int>(map_.starts.size()) < num_pursuers_ + 1) {
    SpielFatalError(
        absl::StrCat("num_pursuers must be at least 1 and the map "
                     "needs num_pursuers + 1 start stations; got ",
                     num_pursuers_, " pursuers and ", map_.starts.size(),
                     " start stations"));
  }
  if (max_evader_moves_ < 1) SpielFatalError("max_evader_moves must be >= 1");

  reveal_moves_ =
      ParseIntList("reveal_moves", ParameterValue<std::string>("reveal_moves"));
  int previous = 0;
  for (int move : reveal_moves_) {
    if (move <= previous || move > max_evader_moves_) {
      SpielFatalError(
          "reveal_moves must be strictly increasing, within "
          "[1, max_evader_moves]");
    }
    previous = move;
  }
  const std::vector<int> pursuer = ParseIntList(
      "pursuer_tickets", ParameterValue<std::string>("pursuer_tickets"));
  const std::vector<int> evader = ParseIntList(
      "evader_tickets", ParameterValue<std::string>("evader_tickets"));
  if (pursuer.size() != kNumTransports ||
      evader.size() != kNumEvaderTicketTypes ||
      absl::c_any_of(pursuer, [](int n) { return n < 0; }) ||
      absl::c_any_of(evader, [](int n) { return n < 0; })) {
    SpielFatalError(
        "pursuer_tickets needs 3 and evader_tickets 5 non-negative counts");
  }
  absl::c_copy(pursuer, pursuer_tickets_.begin());
  absl::c_copy(evader, evader_tickets_.begin());
}

double CatchTheHiddenEvaderGame::EvaderTicketScale(int ticket) const {
  // Pursuers hand their transport tickets over to the evader.
  int most = evader_tickets_[ticket];
  if (ticket < kNumTransports) most += num_pursuers_ * pursuer_tickets_[ticket];
  return std::max(1, most);
}

double CatchTheHiddenEvaderGame::PursuerTicketScale(int ticket) const {
  return std::max(1, pursuer_tickets_[ticket]);
}

std::vector<int> CatchTheHiddenEvaderGame::ObservationTensorShape() const {
  const int s = NumStations();
  const int n = num_pursuers_;
  // Pursuers, last revealed station, evader (private), tickets, two counters,
  // phase, double pending, the pursuer to move and the winner.
  return {n * s + s + s + kNumEvaderTicketTypes + kNumTransports * n + 2 + 4 +
          1 + n + kNumPlayers};
}

std::vector<int> CatchTheHiddenEvaderGame::InformationStateTensorShape() const {
  const int s = NumStations();
  const int n = num_pursuers_;
  const int m = max_evader_moves_;
  // The observation, then the history in the order the tensor is written.
  const int start_stations = (n + 1) * s;
  const int evader_moves = m * (1 + kNumMoveTickets + 1);  // Public part.
  const int shown_stations = static_cast<int>(reveal_moves_.size()) * s;
  const int evader_stations = m * s;  // Private.
  const int pursuer_actions =
      m * n * (kNumPursuerStatuses + s + kNumTransports);
  return {ObservationTensorShape()[0] + start_stations + evader_moves +
          shown_stations + evader_stations + pursuer_actions};
}

// ---------------------------------------------------------------------------
// State.

CatchTheHiddenEvaderState::CatchTheHiddenEvaderState(
    std::shared_ptr<const Game> game)
    : State(game) {
  Reset();
}

void CatchTheHiddenEvaderState::Reset() {
  const CatchTheHiddenEvaderGame& game = GetGame();
  drawn_.clear();
  evader_ = -1;
  pursuers_.assign(game.NumPursuers(), -1);
  for (int ticket = 0; ticket < kNumEvaderTicketTypes; ++ticket) {
    evader_tickets_[ticket] = game.StartingEvaderTickets(ticket);
  }
  std::array<int, kNumTransports> pursuer;
  for (int ticket = 0; ticket < kNumTransports; ++ticket) {
    pursuer[ticket] = game.StartingPursuerTickets(ticket);
  }
  pursuer_tickets_.assign(game.NumPursuers(), pursuer);
  double_pending_ = false;
  pursuer_to_move_ = 0;
  current_player_ = kChancePlayerId;
  winner_ = kInvalidPlayer;
  evader_log_.clear();
  pursuer_log_.clear();
}

const CatchTheHiddenEvaderGame& CatchTheHiddenEvaderState::GetGame() const {
  return open_spiel::down_cast<const CatchTheHiddenEvaderGame&>(*game_);
}

Player CatchTheHiddenEvaderState::CurrentPlayer() const {
  return IsTerminal() ? kTerminalPlayerId : current_player_;
}

bool CatchTheHiddenEvaderState::IsReveal(int move_number) const {
  return absl::c_binary_search(GetGame().RevealMoves(), move_number);
}

int CatchTheHiddenEvaderState::LastRevealedStation() const {
  for (auto it = evader_log_.rbegin(); it != evader_log_.rend(); ++it) {
    if (it->revealed) return it->to;
  }
  return -1;
}

std::vector<std::pair<Action, double>>
CatchTheHiddenEvaderState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  const int num_starts = GetGame().GetMap().starts.size();
  const double probability = 1.0 / (num_starts - drawn_.size());
  std::vector<std::pair<Action, double>> outcomes;
  for (int index = 0; index < num_starts; ++index) {
    if (!absl::c_linear_search(drawn_, index)) {
      outcomes.push_back({index, probability});
    }
  }
  return outcomes;
}

std::vector<Action> CatchTheHiddenEvaderState::LegalActions() const {
  if (IsTerminal()) return {};
  if (IsChanceNode()) return LegalChanceOutcomes();
  if (current_player_ == kEvader) return EvaderActions();
  return PursuerActions(pursuer_to_move_);
}

std::vector<Action> CatchTheHiddenEvaderState::EvaderActions() const {
  const CatchTheHiddenEvaderGame& game = GetGame();
  // The evader may move onto a pursuer (a capture), and one double move per
  // turn may be declared whatever the second move can be.
  const bool can_double = !double_pending_ && evader_tickets_[kDouble] > 0;
  std::vector<Action> actions;
  for (const auto& [target, mask] : game.GetMap().links[evader_]) {
    for (int ticket = 0; ticket < kNumMoveTickets; ++ticket) {
      if (evader_tickets_[ticket] == 0 || !TicketFitsLink(ticket, mask)) {
        continue;
      }
      actions.push_back(game.EvaderMoveAction(ticket, target, false));
      if (can_double) {
        actions.push_back(game.EvaderMoveAction(ticket, target, true));
      }
    }
  }
  absl::c_sort(actions);
  return actions;
}

std::vector<Action> CatchTheHiddenEvaderState::PursuerActions(
    int pursuer) const {
  const CatchTheHiddenEvaderGame& game = GetGame();
  std::vector<Action> actions;
  for (const auto& [target, mask] : game.GetMap().links[pursuers_[pursuer]]) {
    if (absl::c_linear_search(pursuers_, target)) continue;
    for (int ticket = 0; ticket < kNumTransports; ++ticket) {
      if (pursuer_tickets_[pursuer][ticket] > 0 && (mask & (1 << ticket))) {
        actions.push_back(game.PursuerMoveAction(ticket, target));
      }
    }
  }
  if (actions.empty()) return {game.PassAction()};
  absl::c_sort(actions);
  return actions;
}

// No pursuer holds a ticket for a transport of its own station. Destinations
// taken by other pursuers do not count: pursuers blocked only by each other
// pass, and the game goes on.
bool CatchTheHiddenEvaderState::PursuersImmobile() const {
  const CatchTheHiddenEvaderGame& game = GetGame();
  for (int pursuer = 0; pursuer < game.NumPursuers(); ++pursuer) {
    int transports = 0;
    for (const auto& [target, mask] : game.GetMap().links[pursuers_[pursuer]]) {
      transports |= mask;
    }
    for (int ticket = 0; ticket < kNumTransports; ++ticket) {
      if (pursuer_tickets_[pursuer][ticket] > 0 && (transports & (1 << ticket)))
        return false;
    }
  }
  return true;
}

void CatchTheHiddenEvaderState::DoApplyAction(Action action) {
  if (IsChanceNode()) {
    ApplyChance(action);
  } else if (current_player_ == kEvader) {
    ApplyEvaderMove(action);
  } else {
    ApplyPursuerAction(action);
  }
}

void CatchTheHiddenEvaderState::ApplyChance(Action action) {
  const int station = GetGame().GetMap().starts[action];
  if (drawn_.empty()) {
    evader_ = station;
  } else {
    pursuers_[drawn_.size() - 1] = station;
  }
  drawn_.push_back(action);
  if (!InSetup()) {
    current_player_ = kEvader;
    StartEvaderMove();
  }
}

bool CatchTheHiddenEvaderState::EvaderCanMove(int station) const {
  // Moving onto a pursuer counts: it is a legal move, and a capture.
  for (const auto& [target, mask] : GetGame().GetMap().links[station]) {
    for (int ticket = 0; ticket < kNumMoveTickets; ++ticket) {
      if (evader_tickets_[ticket] > 0 && TicketFitsLink(ticket, mask)) {
        return true;
      }
    }
  }
  return false;
}

void CatchTheHiddenEvaderState::StartEvaderMove() {
  if (!EvaderCanMove(evader_)) winner_ = kPursuers;  // No usable ticket.
}

void CatchTheHiddenEvaderState::ApplyEvaderMove(Action action) {
  const CatchTheHiddenEvaderGame& game = GetGame();
  const auto [ticket, target, is_double] = game.DecodeEvaderMove(action);
  evader_tickets_[ticket]--;
  if (is_double) evader_tickets_[kDouble]--;
  const int move_number = NumEvaderMoves() + 1;
  evader_log_.push_back({target, ticket, is_double, IsReveal(move_number)});
  pursuer_log_.push_back(std::vector<PursuerRecord>(game.NumPursuers()));
  evader_ = target;
  if (absl::c_linear_search(pursuers_, target)) {
    winner_ = kPursuers;  // The evader moved onto a pursuer.
    return;
  }
  if (is_double) {
    // Immobile pursuers are checked after every move, the first move of a
    // double included.
    if (PursuersImmobile()) {
      winner_ = kEvader;
      return;
    }
    for (PursuerRecord& record : pursuer_log_.back()) {
      record.status = PursuerStatus::kSkipped;
    }
    if (move_number == game.MaxEvaderMoves()) {
      winner_ = kEvader;  // The pursuers skip their last turn.
      return;
    }
    double_pending_ = true;
    StartEvaderMove();
    return;
  }
  double_pending_ = false;
  if (PursuersImmobile()) {
    winner_ = kEvader;
    return;
  }
  current_player_ = kPursuers;
  pursuer_to_move_ = 0;
}

void CatchTheHiddenEvaderState::ApplyPursuerAction(Action action) {
  const CatchTheHiddenEvaderGame& game = GetGame();
  const int pursuer = pursuer_to_move_;
  PursuerRecord& record = pursuer_log_.back()[pursuer];
  if (action == game.PassAction()) {
    record.status = PursuerStatus::kPassed;
  } else {
    const MoveAction move = game.DecodePursuerMove(action);
    pursuer_tickets_[pursuer][move.ticket]--;
    evader_tickets_[move.ticket]++;
    pursuers_[pursuer] = move.target;
    record = {PursuerStatus::kMoved, move.target, move.ticket};
    if (move.target == evader_) {
      winner_ = kPursuers;
      return;
    }
  }
  if (PursuersImmobile()) {
    winner_ = kEvader;
    return;
  }
  if (++pursuer_to_move_ < game.NumPursuers()) return;
  if (NumEvaderMoves() >= game.MaxEvaderMoves()) {
    winner_ = kEvader;  // The pursuers have answered the evader's last move.
    return;
  }
  current_player_ = kEvader;
  StartEvaderMove();
}

void CatchTheHiddenEvaderState::UndoAction(Player player, Action action) {
  SPIEL_CHECK_FALSE(history_.empty());
  SPIEL_CHECK_EQ(history_.back().player, player);
  SPIEL_CHECK_EQ(history_.back().action, action);
  // Replaying the shorter history keeps the rules in one place.
  std::vector<PlayerAction> history = history_;
  history.pop_back();
  Reset();
  history_.clear();
  move_number_ = 0;
  for (const PlayerAction& player_action : history) {
    ApplyAction(player_action.action);
  }
}

std::vector<double> CatchTheHiddenEvaderState::Returns() const {
  if (winner_ == kEvader) return {1.0, -1.0};
  if (winner_ == kPursuers) return {-1.0, 1.0};
  return {0.0, 0.0};
}

std::string CatchTheHiddenEvaderState::ActionToString(Player player,
                                                      Action action) const {
  const CatchTheHiddenEvaderGame& game = GetGame();
  if (player == kChancePlayerId) {
    const std::string who =
        drawn_.empty() ? "Evader" : absl::StrCat("Pursuer ", drawn_.size() - 1);
    return absl::StrCat(who, " starts at ",
                        StationString(game.GetMap().starts[action]));
  }
  if (player == kEvader) {
    const MoveAction move = game.DecodeEvaderMove(action);
    return absl::StrCat(
        "Evader: ", StationString(evader_), "->", StationString(move.target),
        " by ", kTicketNames[move.ticket], move.is_double ? " (double)" : "");
  }
  const std::string pursuer = absl::StrCat("Pursuer ", pursuer_to_move_, ": ");
  if (action == game.PassAction()) return absl::StrCat(pursuer, "pass");
  const MoveAction move = game.DecodePursuerMove(action);
  return absl::StrCat(pursuer, StationString(pursuers_[pursuer_to_move_]), "->",
                      StationString(move.target), " by ",
                      kTicketNames[move.ticket]);
}

std::string CatchTheHiddenEvaderState::PhaseString() const {
  if (IsTerminal()) return "terminal";
  if (IsChanceNode()) return "setup";
  return current_player_ == kEvader ? "evader" : "pursuers";
}

std::string CatchTheHiddenEvaderState::EvaderTicketsString() const {
  std::string str;
  for (int ticket = 0; ticket < kNumEvaderTicketTypes; ++ticket) {
    absl::StrAppend(&str, " ", kTicketNames[ticket], "=",
                    evader_tickets_[ticket]);
  }
  return str;
}

std::string CatchTheHiddenEvaderState::PursuersString() const {
  std::string str;
  for (int pursuer = 0; pursuer < GetGame().NumPursuers(); ++pursuer) {
    absl::StrAppend(&str, " ", StationString(pursuers_[pursuer]), " (",
                    absl::StrJoin(pursuer_tickets_[pursuer], "/"), ")");
  }
  return str;
}

std::string CatchTheHiddenEvaderState::ToString() const {
  std::string str = absl::StrCat(
      "Phase: ", PhaseString(), "\nEvader: ", StationString(evader_),
      ", tickets", EvaderTicketsString(), "\nPursuers:", PursuersString(),
      "\nEvader moves: ", evader_log_.size(), "/", GetGame().MaxEvaderMoves());
  for (const EvaderMove& move : evader_log_) {
    absl::StrAppend(&str, " ", kTicketNames[move.ticket],
                    move.is_double ? "+double" : "", "@",
                    StationString(move.to), move.revealed ? "(shown)" : "");
  }
  if (double_pending_) absl::StrAppend(&str, "\nDouble move pending");
  if (IsTerminal()) absl::StrAppend(&str, "\nWinner: ", PlayerString(winner_));
  return str;
}

std::string CatchTheHiddenEvaderState::InformationStateString(
    Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  // Everything this player has seen, in order: the setup, then every evader
  // move (station only if shown, or to the evader itself) and pursuer action.
  const CatchTheHiddenEvaderGame& game = GetGame();
  std::string str = absl::StrCat("Player ", player, "\nStarts:");
  for (auto it = drawn_.begin(); it != drawn_.end(); ++it) {
    if (it == drawn_.begin() && player != kEvader) {
      absl::StrAppend(&str, " ?");
    } else {
      absl::StrAppend(&str, " ", StationString(game.GetMap().starts[*it]));
    }
  }
  for (int k = 0; k < NumEvaderMoves(); ++k) {
    const EvaderMove& move = evader_log_[k];
    absl::StrAppend(&str, "\nE", k + 1, ":", kTicketNames[move.ticket],
                    move.is_double ? "+double" : "");
    if (move.revealed || player == kEvader) {
      absl::StrAppend(&str, "@", StationString(move.to));
    }
    for (int pursuer = 0; pursuer < game.NumPursuers(); ++pursuer) {
      const PursuerRecord& record = pursuer_log_[k][pursuer];
      switch (record.status) {
        case PursuerStatus::kNotPlayed:
          break;
        case PursuerStatus::kMoved:
          absl::StrAppend(&str, " P", pursuer, ":", kTicketNames[record.ticket],
                          "@", StationString(record.to));
          break;
        case PursuerStatus::kPassed:
          absl::StrAppend(&str, " P", pursuer, ":pass");
          break;
        case PursuerStatus::kSkipped:
          absl::StrAppend(&str, " P", pursuer, ":skip");
          break;
      }
    }
  }
  // The end of the game is public, and so is who won.
  if (IsTerminal()) absl::StrAppend(&str, "\nWinner: ", PlayerString(winner_));
  return str;
}

std::string CatchTheHiddenEvaderState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  std::string str = absl::StrCat("Phase: ", PhaseString());
  if (current_player_ == kPursuers && !IsTerminal()) {
    absl::StrAppend(&str, " (pursuer ", pursuer_to_move_, ")");
  }
  if (player == kEvader) {
    absl::StrAppend(&str, "\nEvader: ", StationString(evader_));
  }
  absl::StrAppend(&str, "\nLast shown: ", StationString(LastRevealedStation()),
                  "\nEvader tickets:", EvaderTicketsString(),
                  "\nPursuers:", PursuersString());
  absl::StrAppend(&str, "\nEvader moves: ", evader_log_.size(),
                  double_pending_ ? " (double pending)" : "");
  if (IsTerminal()) absl::StrAppend(&str, "\nWinner: ", PlayerString(winner_));
  return str;
}

void CatchTheHiddenEvaderState::ObservationTensor(
    Player player, absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  const CatchTheHiddenEvaderGame& game = GetGame();
  SPIEL_CHECK_EQ(static_cast<int>(values.size()), game.ObservationTensorSize());
  const int num_stations = game.NumStations();
  const int num_pursuers = game.NumPursuers();
  const double max_moves = game.MaxEvaderMoves();
  TensorWriter writer(values);
  for (int pursuer = 0; pursuer < num_pursuers; ++pursuer) {
    writer.OneHot(num_stations, pursuers_[pursuer]);
  }
  writer.OneHot(num_stations, LastRevealedStation());
  writer.OneHot(num_stations, player == kEvader ? evader_ : -1);
  for (int ticket = 0; ticket < kNumEvaderTicketTypes; ++ticket) {
    writer.Value(evader_tickets_[ticket] / game.EvaderTicketScale(ticket));
  }
  for (int pursuer = 0; pursuer < num_pursuers; ++pursuer) {
    for (int ticket = 0; ticket < kNumTransports; ++ticket) {
      writer.Value(pursuer_tickets_[pursuer][ticket] /
                   game.PursuerTicketScale(ticket));
    }
  }
  int last_reveal = 0;
  for (int k = 0; k < NumEvaderMoves(); ++k) {
    if (evader_log_[k].revealed) last_reveal = k + 1;
  }
  writer.Value(evader_log_.size() / max_moves);
  writer.Value((evader_log_.size() - last_reveal) / max_moves);
  const int phase = IsTerminal()                 ? 3
                    : IsChanceNode()             ? 0
                    : current_player_ == kEvader ? 1
                                                 : 2;
  writer.OneHot(4, phase);
  writer.Value(double_pending_ ? 1.0 : 0.0);
  writer.OneHot(num_pursuers, phase == 2 ? pursuer_to_move_ : -1);
  writer.OneHot(kNumPlayers, winner_);  // Nothing set before the end.
  SPIEL_CHECK_EQ(writer.position(), game.ObservationTensorSize());
}

void CatchTheHiddenEvaderState::InformationStateTensor(
    Player player, absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  const CatchTheHiddenEvaderGame& game = GetGame();
  SPIEL_CHECK_EQ(static_cast<int>(values.size()),
                 game.InformationStateTensorSize());
  const int num_stations = game.NumStations();
  const int num_pursuers = game.NumPursuers();
  const int max_moves = game.MaxEvaderMoves();
  const int num_moves = NumEvaderMoves();
  const int num_drawn = drawn_.size();
  const int observation_size = game.ObservationTensorSize();
  ObservationTensor(player, values.subspan(0, observation_size));
  TensorWriter writer(values.subspan(observation_size));
  const std::vector<int>& starts = game.GetMap().starts;
  for (int pursuer = 0; pursuer < num_pursuers; ++pursuer) {
    writer.OneHot(num_stations,
                  pursuer + 1 < num_drawn ? starts[drawn_[pursuer + 1]] : -1);
  }
  writer.OneHot(num_stations,
                player == kEvader && !drawn_.empty() ? starts[drawn_[0]] : -1);
  for (int k = 0; k < max_moves; ++k) {
    if (k < num_moves) {
      writer.Value(1.0);
      writer.OneHot(kNumMoveTickets, evader_log_[k].ticket);
      writer.Value(evader_log_[k].is_double ? 1.0 : 0.0);
    } else {
      writer.Skip(1 + kNumMoveTickets + 1);
    }
  }
  int reveal = 0;
  for (const EvaderMove& move : evader_log_) {
    if (move.revealed) {
      writer.OneHot(num_stations, move.to);
      ++reveal;
    }
  }
  writer.Skip((game.RevealMoves().size() - reveal) * num_stations);
  for (int k = 0; k < max_moves; ++k) {
    writer.OneHot(num_stations,
                  player == kEvader && k < num_moves ? evader_log_[k].to : -1);
  }
  for (int k = 0; k < max_moves; ++k) {
    for (int pursuer = 0; pursuer < num_pursuers; ++pursuer) {
      PursuerRecord record;
      if (k < num_moves) record = pursuer_log_[k][pursuer];
      writer.OneHot(kNumPursuerStatuses, static_cast<int>(record.status));
      writer.OneHot(num_stations, record.to);
      writer.OneHot(kNumTransports, record.ticket);
    }
  }
  SPIEL_CHECK_EQ(observation_size + writer.position(),
                 game.InformationStateTensorSize());
}

std::unique_ptr<State> CatchTheHiddenEvaderState::Clone() const {
  return std::unique_ptr<State>(new CatchTheHiddenEvaderState(*this));
}

std::unique_ptr<State> CatchTheHiddenEvaderState::ResampleFromInfostate(
    int player_id, std::function<double()> rng) const {
  // The evader sees the whole history: only the pursuers miss its stations.
  if (player_id == kEvader || drawn_.empty()) return Clone();
  if (InSetup()) return ResampleSetup(rng);
  // The sample is uniform over the evader histories (its start and its station
  // after each move) consistent with the pursuers' information. Paths are
  // counted forwards, then drawn backwards in proportion to the counts.
  const std::vector<std::vector<double>> count = EvaderPathCounts();
  const Map& map = GetGame().GetMap();
  const int num_stations = GetGame().NumStations();
  const int num_moves = NumEvaderMoves();
  std::vector<int> stations(num_moves + 1);
  stations[num_moves] = DrawIndex(count[num_moves], rng);
  for (int k = num_moves; k >= 1; --k) {
    std::vector<double> weights(num_stations, 0.0);
    for (int from = 0; from < num_stations; ++from) {
      for (const auto& [to, mask] : map.links[from]) {
        if (to == stations[k] &&
            TicketFitsLink(evader_log_[k - 1].ticket, mask)) {
          weights[from] = count[k - 1][from];
        }
      }
    }
    stations[k - 1] = DrawIndex(weights, rng);
  }
  std::unique_ptr<State> state = ReplayWithEvaderStations(stations);
  SPIEL_CHECK_EQ(InformationStateString(player_id),
                 state->InformationStateString(player_id));
  return state;
}

std::unique_ptr<State> CatchTheHiddenEvaderState::ResampleSetup(
    std::function<double()> rng) const {
  // During the setup the pursuers know only that the evader's start differs
  // from theirs.
  const CatchTheHiddenEvaderGame& game = GetGame();
  const std::vector<int> pursuer_draws(drawn_.begin() + 1, drawn_.end());
  std::vector<double> weights(game.GetMap().starts.size(), 1.0);
  for (int index : pursuer_draws) weights[index] = 0.0;
  std::unique_ptr<State> state = game.NewInitialState();
  state->ApplyAction(DrawIndex(weights, rng));
  for (int index : pursuer_draws) state->ApplyAction(index);
  return state;
}

// count[k][s] is proportional to the number of evader paths consistent with
// the pursuers' information that are at station s after k moves. Each row is
// rescaled before the next one is computed, so that long games cannot
// overflow: only proportions within a row matter.
std::vector<std::vector<double>> CatchTheHiddenEvaderState::EvaderPathCounts()
    const {
  const CatchTheHiddenEvaderGame& game = GetGame();
  const Map& map = game.GetMap();
  const int num_stations = game.NumStations();
  const int num_moves = NumEvaderMoves();
  std::vector<std::vector<double>> count(
      num_moves + 1, std::vector<double>(num_stations, 0.0));
  std::vector<int> pursuers(
      game.NumPursuers());  // Their stations before a turn.
  for (int pursuer = 0; pursuer < game.NumPursuers(); ++pursuer) {
    pursuers[pursuer] = map.starts[drawn_[pursuer + 1]];
  }
  for (int start : map.starts) {
    if (!absl::c_linear_search(pursuers, start)) count[0][start] = 1.0;
  }
  for (int k = 1; k <= num_moves; ++k) {
    const double largest = *absl::c_max_element(count[k - 1]);
    SPIEL_CHECK_GT(largest, 0.0);
    for (double& paths : count[k - 1]) paths /= largest;
    const EvaderMove& move = evader_log_[k - 1];
    std::vector<double>& row = count[k];
    for (int from = 0; from < num_stations; ++from) {
      if (count[k - 1][from] == 0.0) continue;
      for (const auto& [to, mask] : map.links[from]) {
        if (TicketFitsLink(move.ticket, mask)) row[to] += count[k - 1][from];
      }
    }
    if (move.revealed) {
      for (int s = 0; s < num_stations; ++s) {
        if (s != move.to) row[s] = 0.0;
      }
    }
    // In a game the pursuers won, the last step may be a capture.
    if (k == num_moves && winner_ == kPursuers) break;
    // Otherwise neither side moved onto the other: that is a capture.
    for (int station : pursuers) row[station] = 0.0;
    for (int pursuer = 0; pursuer < game.NumPursuers(); ++pursuer) {
      const PursuerRecord& record = pursuer_log_[k - 1][pursuer];
      if (record.status != PursuerStatus::kMoved) continue;
      row[record.to] = 0.0;
      pursuers[pursuer] = record.to;
    }
  }
  std::vector<double>& last = count[num_moves];
  if (winner_ == kPursuers) {
    last = EndingWeights(last, pursuers);
  } else if (!IsTerminal() && current_player_ == kEvader) {
    // An evader to move has a usable ticket: otherwise the game would be over.
    for (int s = 0; s < num_stations; ++s) {
      if (!EvaderCanMove(s)) last[s] = 0.0;
    }
  }
  return count;
}

// In a game the pursuers won, the weights of the evader's last station, from
// `arrivals`, the paths reaching each station with the last evader move (or
// the starts, before any move), and the pursuers' stations at that move. The
// endings follow the order of ApplyEvaderMove and ApplyPursuerAction.
std::vector<double> CatchTheHiddenEvaderState::EndingWeights(
    const std::vector<double>& arrivals,
    const std::vector<int>& pursuers) const {
  const int num_stations = GetGame().NumStations();
  std::vector<double> weights(num_stations, 0.0);
  // Paths after which the evader had no usable ticket for its next move.
  auto add_no_ticket = [&](const std::vector<double>& paths) {
    for (int s = 0; s < num_stations; ++s) {
      if (!EvaderCanMove(s)) weights[s] += paths[s];
    }
  };
  const Player last_player = history_.back().player;
  if (last_player == kChancePlayerId) {  // Before the first move.
    add_no_ticket(arrivals);
    return weights;
  }
  std::vector<double> uncaught = arrivals;
  for (int station : pursuers) uncaught[station] = 0.0;
  if (last_player == kEvader) {
    // Pursuers skipping their turn mark the first move of a double that left
    // no usable ticket; otherwise the evader moved onto a pursuer.
    if (pursuer_log_.back()[0].status == PursuerStatus::kSkipped) {
      add_no_ticket(uncaught);
    } else {
      for (int station : pursuers) weights[station] = arrivals[station];
    }
    return weights;
  }
  // The last pursuer to act moved onto the evader. Or it was the last of the
  // turn, nobody reached the evader, and the evader then had no usable ticket,
  // unless the game had already ended: by time, or with pursuers left unable
  // to move.
  const std::vector<PursuerRecord>& turn = pursuer_log_.back();
  const int num_pursuers = turn.size();
  int actor = num_pursuers - 1;
  while (turn[actor].status == PursuerStatus::kNotPlayed) --actor;
  for (int pursuer = 0; pursuer < actor; ++pursuer) {
    if (turn[pursuer].status == PursuerStatus::kMoved) {
      uncaught[turn[pursuer].to] = 0.0;
    }
  }
  if (turn[actor].status == PursuerStatus::kMoved) {
    weights[turn[actor].to] = uncaught[turn[actor].to];
    uncaught[turn[actor].to] = 0.0;
  }
  if (actor == num_pursuers - 1 &&
      NumEvaderMoves() < GetGame().MaxEvaderMoves() && !PursuersImmobile()) {
    add_no_ticket(uncaught);
  }
  return weights;
}

// This game's history, with the evader starting at stations[0] and at
// stations[k] after its k-th move.
std::unique_ptr<State> CatchTheHiddenEvaderState::ReplayWithEvaderStations(
    const std::vector<int>& stations) const {
  const CatchTheHiddenEvaderGame& game = GetGame();
  const std::vector<int>& starts = game.GetMap().starts;
  std::unique_ptr<State> state = game.NewInitialState();
  int evader_moves = 0;
  for (const PlayerAction& player_action : history_) {
    if (player_action.player == kChancePlayerId && state->MoveNumber() == 0) {
      state->ApplyAction(
          std::distance(starts.begin(), absl::c_find(starts, stations[0])));
    } else if (player_action.player == kEvader) {
      const EvaderMove& move = evader_log_[evader_moves];
      ++evader_moves;
      state->ApplyAction(game.EvaderMoveAction(
          move.ticket, stations[evader_moves], move.is_double));
    } else {
      state->ApplyAction(player_action.action);
    }
  }
  return state;
}

CatchTheHiddenEvaderStructContents CatchTheHiddenEvaderState::StructContents()
    const {
  CatchTheHiddenEvaderStructContents contents;
  contents.phase = PhaseString();
  contents.current_player = PlayerString(CurrentPlayer());
  contents.evader_station = StationNumber(evader_);
  for (int station : pursuers_) {
    contents.pursuer_stations.push_back(StationNumber(station));
  }
  contents.evader_tickets.assign(evader_tickets_.begin(),
                                 evader_tickets_.end());
  for (const auto& tickets : pursuer_tickets_) {
    contents.pursuer_tickets.push_back({tickets.begin(), tickets.end()});
  }
  contents.evader_moves = evader_log_.size();
  contents.double_pending = double_pending_;
  contents.pursuer_to_move =
      !IsTerminal() && current_player_ == kPursuers ? pursuer_to_move_ : -1;
  contents.last_revealed_station = StationNumber(LastRevealedStation());
  for (const EvaderMove& move : evader_log_) {
    EvaderLogEntry entry;
    entry.ticket = kTicketNames[move.ticket];
    entry.double_move = move.is_double;
    entry.revealed_station = move.revealed ? StationNumber(move.to) : -1;
    contents.evader_log.push_back(entry);
  }
  contents.winner = IsTerminal() ? PlayerString(winner_) : "";
  return contents;
}

std::unique_ptr<StateStruct> CatchTheHiddenEvaderState::ToStruct() const {
  auto state_struct = std::make_unique<CatchTheHiddenEvaderStateStruct>();
  static_cast<CatchTheHiddenEvaderStructContents&>(*state_struct) =
      StructContents();
  return state_struct;
}

std::unique_ptr<ObservationStruct>
CatchTheHiddenEvaderState::ToObservationStruct(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  auto obs_struct = std::make_unique<CatchTheHiddenEvaderObservationStruct>();
  static_cast<CatchTheHiddenEvaderStructContents&>(*obs_struct) =
      StructContents();
  obs_struct->observing_player = player;
  if (player != kEvader) obs_struct->evader_station = -1;  // Hidden.
  return obs_struct;
}

std::unique_ptr<ActionStruct> CatchTheHiddenEvaderState::ActionToStruct(
    Player player, Action action) const {
  const CatchTheHiddenEvaderGame& game = GetGame();
  auto action_struct = std::make_unique<CatchTheHiddenEvaderActionStruct>();
  if (player == kChancePlayerId) {
    action_struct->actor = "chance";
    action_struct->pursuer = static_cast<int>(drawn_.size()) - 1;
    action_struct->to = StationNumber(game.GetMap().starts[action]);
  } else if (player == kEvader) {
    action_struct->actor = "evader";
    const MoveAction move = game.DecodeEvaderMove(action);
    action_struct->from = StationNumber(evader_);
    action_struct->to = StationNumber(move.target);
    action_struct->ticket = kTicketNames[move.ticket];
    action_struct->double_move = move.is_double;
  } else {
    SPIEL_CHECK_EQ(player, kPursuers);
    action_struct->actor = "pursuer";
    action_struct->pursuer = pursuer_to_move_;
    action_struct->from = StationNumber(pursuers_[pursuer_to_move_]);
    if (action == game.PassAction()) {
      action_struct->pass = true;
    } else {
      const MoveAction move = game.DecodePursuerMove(action);
      action_struct->to = StationNumber(move.target);
      action_struct->ticket = kTicketNames[move.ticket];
    }
  }
  return action_struct;
}

std::vector<Action> CatchTheHiddenEvaderState::StructToActions(
    const ActionStruct& action_struct) const {
  const CatchTheHiddenEvaderGame& game = GetGame();
  const auto* a =
      SafeActionCast<CatchTheHiddenEvaderActionStruct>(action_struct);
  if (!a->pass) {
    SPIEL_CHECK_GE(a->to, 1);
    SPIEL_CHECK_LE(a->to, game.NumStations());
  }
  if (a->actor == "chance") {
    SPIEL_CHECK_TRUE(IsChanceNode());
    auto it = absl::c_find(game.GetMap().starts, a->to - 1);
    SPIEL_CHECK_TRUE(it != game.GetMap().starts.end());
    return {std::distance(game.GetMap().starts.begin(), it)};
  }
  if (a->actor == "evader") {
    SPIEL_CHECK_EQ(current_player_, kEvader);
    return {game.EvaderMoveAction(TicketFromString(a->ticket, kNumMoveTickets),
                                  a->to - 1, a->double_move)};
  }
  SPIEL_CHECK_EQ(a->actor, "pursuer");
  SPIEL_CHECK_EQ(current_player_, kPursuers);
  SPIEL_CHECK_EQ(a->pursuer, pursuer_to_move_);
  if (a->pass) return {game.PassAction()};
  return {game.PursuerMoveAction(TicketFromString(a->ticket, kNumTransports),
                                 a->to - 1)};
}

}  // namespace catch_the_hidden_evader
}  // namespace open_spiel
