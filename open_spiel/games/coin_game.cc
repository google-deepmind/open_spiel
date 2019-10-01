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

#include "open_spiel/games/coin_game.h"

#include <array>

namespace open_spiel {
namespace coin_game {

namespace {

// Facts about the game
const GameType kGameType{
    /*short_name=*/"coin_game",
    /*long_name=*/"The Coin Game",
    GameType::Dynamics::kSequential,
    // Getting a NewInitialState randomly initializes player and coin positions
    // and player preferences, but from that point on no chance nodes are
    // involved.
    GameType::ChanceMode::kExplicitStochastic,
    // Imperfect information game because players only know their own preferred
    // coin.
    GameType::Information::kImperfectInformation,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/10,
    /*min_num_players=*/1,
    /*provides_information_state=*/false,
    /*provides_information_state_as_normalized_vector=*/false,
    /*provides_observation=*/true,
    /*provides_observation_as_normalized_vector=*/false,
    /*parameter_specification=*/
    {
        {"players", {GameParameter::Type::kInt, false}},
        {"rows", {GameParameter::Type::kInt, false}},
        {"columns", {GameParameter::Type::kInt, false}},
        {"episode_length", {GameParameter::Type::kInt, false}},
        // Number of extra coin colors to use apart from the
        // players' preferred color.
        {"num_extra_coin_colors", {GameParameter::Type::kInt, false}},
        {"num_coins_per_color", {GameParameter::Type::kInt, false}},
    }};

std::unique_ptr<Game> Factory(const GameParameters& params) {
  return std::unique_ptr<Game>(new CoinGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

// Defaults match the paper https://arxiv.org/pdf/1802.09640.pdf
constexpr int kDefaultPlayers = 2;
constexpr int kDefaultRows = 8;
constexpr int kDefaultColumns = 8;
constexpr int kDefaultExtraCoinColors = 1;
constexpr int kDefaultCoinsPerColor = 4;
constexpr int kDefaultEpisodeLength = 20;

std::string GamePhaseToString(GamePhase phase) {
  switch (phase) {
    case GamePhase::kAssignPreferences:
      return "AssignPreferences";
    case GamePhase::kDeployPlayers:
      return "DeployPlayers";
    case GamePhase::kDeployCoins:
      return "DeployCoins";
    case GamePhase::kPlay:
      return "Play";
    default:
      SpielFatalError("Unknown phase.");
      return "This will never return.";
  }
}

enum struct SymbolType { kEmpty = 0, kCoin = 1, kPlayer = 2 };
constexpr char kEmptySymbol = ' ';

SymbolType GetSymbolType(char symbol) {
  if (symbol == kEmptySymbol) {
    return SymbolType::kEmpty;
  } else if ('a' <= symbol && symbol <= 'z') {
    return SymbolType::kCoin;
  } else if ('0' <= symbol && symbol <= '9') {
    return SymbolType::kPlayer;
  }
  SpielFatalError(absl::StrCat("Unexpected symbol: ", std::string(1, symbol)));
}

inline char PlayerSymbol(Player player) {
  return '0' + static_cast<char>(player);
}
inline char CoinSymbol(int coin) { return 'a' + static_cast<char>(coin); }
int CoinId(char symbol) { return symbol - 'a'; }

// Movement.
enum MovementType { kUp = 0, kDown = 1, kLeft = 2, kRight = 3, kStand = 4 };

constexpr std::array<Location, 5> offsets = {
    {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {0, 0}}};

Location operator+(const Location& a, const Location& b) {
  return {a.first + b.first, a.second + b.second};
}

std::set<int> RangeAsSet(int n) {
  std::set<int> result;
  for (int i = 0; i < n; i++) {
    result.insert(i);
  }
  return result;
}

std::vector<Action> Range(int n) {
  std::vector<Action> result(n);
  for (int i = 0; i < n; i++) {
    result[i] = i;
  }
  return result;
}

ActionsAndProbs ActionProbRange(const std::set<int> set) {
  ActionsAndProbs result;
  result.reserve(set.size());
  const double prob = 1.0 / set.size();
  for (int elem : set) {
    result.push_back({elem, prob});
  }
  return result;
}

std::vector<Action> ActionRange(const std::set<int> set) {
  std::vector<Action> result;
  result.reserve(set.size());
  for (int elem : set) {
    result.push_back(elem);
  }
  return result;
}
}  // namespace

Setup::Setup(int num_rows, int num_columns, int num_coin_colors)
    : available_coin_colors_(RangeAsSet(num_coin_colors)),
      available_positions_(RangeAsSet(num_rows * num_columns)) {}

CoinState::CoinState(const CoinGame& parent_game)
    : State(/*num_distinct_actions=*/parent_game.NumDistinctActions(),
            /*num_players=*/parent_game.NumPlayers()),
      game_(parent_game),
      setup_(parent_game.NumRows(), parent_game.NumColumns(),
             parent_game.NumCoinColors()),
      player_preferences_(parent_game.NumPlayers()),
      player_location_(parent_game.NumPlayers()),
      field_(parent_game.NumRows() * parent_game.NumColumns(), kEmptySymbol),
      player_coins_(parent_game.NumPlayers() * parent_game.NumCoinColors(), 0) {
}

GamePhase CoinState::GetPhase() const {
  if (cur_player_ != kChancePlayerId) {
    return GamePhase::kPlay;
  } else if (setup_.num_players_assigned_preference < num_players_) {
    return GamePhase::kAssignPreferences;
  } else if (setup_.num_players_on_field < num_players_) {
    return GamePhase::kDeployPlayers;
  } else if (setup_.num_coins_on_field < game_.TotalCoins()) {
    return GamePhase::kDeployCoins;
  } else {
    SpielFatalError("Inconsistent setup versus current_player state");
  }
}

std::vector<Action> CoinState::LegalActions() const {
  if (IsTerminal()) return {};
  switch (GetPhase()) {
    case GamePhase::kAssignPreferences:
      return ActionRange(setup_.available_coin_colors_);
    case GamePhase::kDeployPlayers:
      return ActionRange(setup_.available_positions_);
    case GamePhase::kDeployCoins:
      return ActionRange(setup_.available_positions_);
    case GamePhase::kPlay:
      return Range(offsets.size());
    default:
      SpielFatalError("Unknown phase.");
  }
}

ActionsAndProbs CoinState::ChanceOutcomes() const {
  switch (GetPhase()) {
    case GamePhase::kAssignPreferences:
      return ActionProbRange(setup_.available_coin_colors_);
    case GamePhase::kDeployPlayers:
      return ActionProbRange(setup_.available_positions_);
    case GamePhase::kDeployCoins:
      return ActionProbRange(setup_.available_positions_);
    case GamePhase::kPlay:
      SpielFatalError("ChanceOutcomes invoked in play phase");
    default:
      SpielFatalError("Unknown phase.");
      return {};
  }
}

std::string CoinState::Observation(Player player) const {
  std::ostringstream out;
  // A player only learns its own preference.
  SpielStrOut(out, player_preferences_[player], "\n");
  // Table of how many coins of each type were collected by each player.
  PrintCoinsCollected(out);
  // Current positions of all coins and players on the board.
  PrintBoard(out);
  return out.str();
}

bool CoinState::InBounds(Location loc) const {
  return (loc.first >= 0 && loc.second >= 0 && loc.first < game_.NumRows() &&
          loc.second < game_.NumColumns());
}

void CoinState::SetField(Location loc, char symbol) {
  field_[loc.first * game_.NumColumns() + loc.second] = symbol;
}

char CoinState::GetField(Location loc) const {
  return field_[loc.first * game_.NumColumns() + loc.second];
}

Location CoinState::LocationFromIndex(int index) const {
  return {index / game_.NumColumns(), index % game_.NumColumns()};
}

void CoinState::ApplyAssignPreferenceAction(Action coin_color) {
  SPIEL_CHECK_LT(coin_color, game_.NumCoinColors());
  player_preferences_[setup_.num_players_assigned_preference] = coin_color;
  ++setup_.num_players_assigned_preference;
  setup_.available_coin_colors_.erase(coin_color);
}

void CoinState::ApplyDeployPlayersAction(Action index) {
  SPIEL_CHECK_LT(index, field_.size());
  SPIEL_CHECK_TRUE(GetSymbolType(field_[index]) == SymbolType::kEmpty);
  field_[index] = PlayerSymbol(setup_.num_players_on_field);
  player_location_[setup_.num_players_on_field] = LocationFromIndex(index);
  ++setup_.num_players_on_field;
  setup_.available_positions_.erase(index);
}

void CoinState::ApplyDeployCoinsAction(Action index) {
  SPIEL_CHECK_LT(index, field_.size());
  SPIEL_CHECK_TRUE(GetSymbolType(field_[index]) == SymbolType::kEmpty);

  int coin_color = setup_.num_coins_on_field / game_.NumCoinsPerColor();
  field_[index] = CoinSymbol(coin_color);
  ++setup_.num_coins_on_field;
  setup_.available_positions_.erase(index);

  if (setup_.num_coins_on_field == game_.TotalCoins()) {
    // Switch to play phase.
    setup_.available_positions_.clear();    // Release memory.
    setup_.available_coin_colors_.clear();  // Release memory.
    cur_player_ = 0;
  }
}

void CoinState::ApplyPlayAction(Action move) {
  ++total_moves_;

  Location old_loc = player_location_[cur_player_];
  SPIEL_CHECK_EQ(GetField(old_loc), PlayerSymbol(cur_player_));

  Location new_loc = old_loc + offsets[move];
  if (InBounds(new_loc)) {
    char target = GetField(new_loc);
    SymbolType target_type = GetSymbolType(target);
    if (target_type == SymbolType::kCoin) {
      IncPlayerCoinCount(cur_player_, CoinId(target));
    }
    if (target_type == SymbolType::kCoin || target_type == SymbolType::kEmpty) {
      player_location_[cur_player_] = new_loc;
      SetField(old_loc, kEmptySymbol);
      SetField(new_loc, PlayerSymbol(cur_player_));
    }
  }
  cur_player_ = (cur_player_ + 1) % num_players_;
}

void CoinState::DoApplyAction(Action action) {
  switch (GetPhase()) {
    case GamePhase::kAssignPreferences:
      ApplyAssignPreferenceAction(action);
      break;
    case GamePhase::kDeployPlayers:
      ApplyDeployPlayersAction(action);
      break;
    case GamePhase::kDeployCoins:
      ApplyDeployCoinsAction(action);
      break;
    case GamePhase::kPlay:
      ApplyPlayAction(action);
      break;
  }
}

void CoinState::IncPlayerCoinCount(Player player, int coin_color) {
  player_coins_[player * game_.NumCoinColors() + coin_color]++;
}

int CoinState::GetPlayerCoinCount(Player player, int coin_color) const {
  return player_coins_[player * game_.NumCoinColors() + coin_color];
}

std::string CoinState::ActionToString(Player player, Action action_id) const {
  if (player == kChancePlayerId) {
    return absl::StrCat(action_id);
  } else {
    if (action_id == kUp) {
      return "up";
    } else if (action_id == kDown) {
      return "down";
    } else if (action_id == kLeft) {
      return "left";
    } else if (action_id == kRight) {
      return "right";
    } else if (action_id == kStand) {
      return "stand";
    } else {
      SpielFatalError(absl::StrCat("Unexpected action ", action_id));
    }
  }
}

void CoinState::PrintCoinsCollected(std::ostream& out) const {
  // Prints table with players as rows and coin_colors as columns.
  SpielStrOut(out, "        ");
  for (int coint_color = 0; coint_color < game_.NumCoinColors();
       coint_color++) {
    SpielStrOut(out, CoinSymbol(coint_color), " ");
  }
  SpielStrOut(out, "\n");
  for (auto player = Player{0}; player < num_players_; player++) {
    SpielStrOut(out, "player", player, " ");
    for (int coint_color = 0; coint_color < game_.NumCoinColors();
         coint_color++) {
      SpielStrOut(out, GetPlayerCoinCount(player, coint_color), " ");
    }
    SpielStrOut(out, "\n");
  }
}

void CoinState::PrintPreferences(std::ostream& out) const {
  SpielStrOut(out, "preferences=");
  for (Player player = 0; player < setup_.num_players_assigned_preference;
       player++) {
    SpielStrOut(out, player, ":", CoinSymbol(player_preferences_[player]), " ");
  }
  SpielStrOut(out, "\n");
}

void CoinState::PrintBoardDelimiterRow(std::ostream& out) const {
  SpielStrOut(out, "+");
  for (int c = 0; c < game_.NumColumns(); c++) {
    SpielStrOut(out, "-");
  }
  SpielStrOut(out, "+\n");
}

void CoinState::PrintBoard(std::ostream& out) const {
  PrintBoardDelimiterRow(out);
  for (int r = 0; r < game_.NumRows(); r++) {
    SpielStrOut(out, "|");
    for (int c = 0; c < game_.NumColumns(); c++) {
      SpielStrOut(out, GetField({r, c}));
    }
    SpielStrOut(out, "|\n");
  }
  PrintBoardDelimiterRow(out);
}

std::string CoinState::ToString() const {
  std::ostringstream out;
  SpielStrOut(out, "phase=", GamePhaseToString(GetPhase()), "\n");
  PrintPreferences(out);
  SpielStrOut(out, "moves=", total_moves_, "\n");
  PrintCoinsCollected(out);
  PrintBoard(out);
  return out.str();
}

bool CoinState::IsTerminal() const {
  return total_moves_ >= game_.EpisodeLength();
}

std::vector<double> CoinState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>(num_players_, 0.0);
  }

  int collected_coins = 0;
  std::vector<int> coin_count(game_.NumCoinColors());
  for (int coin_color = 0; coin_color < game_.NumCoinColors(); coin_color++) {
    for (auto player = Player{0}; player < num_players_; player++) {
      Player player_coins = GetPlayerCoinCount(player, coin_color);
      coin_count[coin_color] += player_coins;
      collected_coins += player_coins;
    }
  }
  int good_coins = 0;
  for (int preference : player_preferences_) {
    good_coins += coin_count[preference];
  }
  const int bad_coins = collected_coins - good_coins;
  std::vector<double> rewards(num_players_);
  for (auto player = Player{0}; player < num_players_; player++) {
    int self_coins = coin_count[player_preferences_[player]];
    int other_coins = good_coins - self_coins;
    rewards[player] = (std::pow(self_coins, 2) + std::pow(other_coins, 2) -
                       std::pow(bad_coins, 2));
  }
  return rewards;
}

std::unique_ptr<State> CoinState::Clone() const {
  return std::unique_ptr<State>(new CoinState(*this));
}

int CoinState::CurrentPlayer() const {
  return IsTerminal() ? kTerminalPlayerId : cur_player_;
}

double CoinGame::MaxUtility() const { return std::pow(TotalCoins(), 2); }

double CoinGame::MinUtility() const { return -MaxUtility(); }

CoinGame::CoinGame(const GameParameters& params)
    : Game(kGameType, params),
      num_players_(ParameterValue<int>("players", kDefaultPlayers)),
      num_rows_(ParameterValue<int>("rows", kDefaultRows)),
      num_columns_(ParameterValue<int>("columns", kDefaultColumns)),
      episode_length_(
          ParameterValue<int>("episode_length", kDefaultEpisodeLength)),
      num_coin_colors_(num_players_ +
                       ParameterValue<int>("num_extra_coin_colors",
                                           kDefaultExtraCoinColors)),
      num_coins_per_color_(
          ParameterValue<int>("num_coins_per_color", kDefaultCoinsPerColor)) {
  int total_items = num_players_ + num_coin_colors_ * num_coins_per_color_;
  SPIEL_CHECK_LE(total_items, num_rows_ * num_columns_);
}

int CoinGame::MaxGameLength() const { return (episode_length_); }

// Chance nodes must not be considered in NumDistinctActions.
int CoinGame::NumDistinctActions() const { return offsets.size(); }

std::unique_ptr<Game> CoinGame::Clone() const {
  return std::unique_ptr<Game>(new CoinGame(*this));
}

std::unique_ptr<State> CoinGame::NewInitialState() const {
  return std::unique_ptr<State>(new CoinState(*this));
}

}  // namespace coin_game
}  // namespace open_spiel
