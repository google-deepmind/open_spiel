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

#include "open_spiel/games/coop_box_pushing.h"

#include <cstddef>
#include <memory>
#include <utility>

#include "open_spiel/spiel.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace coop_box_pushing {
namespace {

// Valid characters: <>^v .bB
// However first 4 characters each have a player (0 or 1) attached to them
// So, 4 + 4 + 3 = 11
constexpr int kCellStates = 11;
constexpr char kLeft = '<';
constexpr char kRight = '>';
constexpr char kUp = '^';
constexpr char kDown = 'v';
constexpr char kField = '.';
constexpr char kSmallBox = 'b';
constexpr char kBigBox = 'B';

// Some constants for this game.
constexpr int kRows = 8;
constexpr int kCols = 8;
constexpr int kNumPlayers = 2;
constexpr int kNumDistinctActions = 4;

// Chance outcomes.
enum ChanceOutcome {
  kChanceSuccess = 0,
  kChanceFail = 1,
  kChanceInit1 = 2,  // determines order of moves
  kChanceInit2 = 3
};

// Rewards.
constexpr double kBumpPenalty = -5;
constexpr double kDelayPenalty = -0.1;
constexpr double kSmallBoxReward = 10;
constexpr double kBigBoxReward = 100;

// Default parameters.
constexpr int kDefaultHorizon = 100;
constexpr bool kDefaultFullyObservable = false;

constexpr std::array<int, 4> row_offsets = {{-1, 0, 1, 0}};
constexpr std::array<int, 4> col_offsets = {{0, 1, 0, -1}};

// Facts about the game
const GameType kGameType{
    /*short_name=*/"coop_box_pushing",
    /*long_name=*/"Cooperative Box Pushing",
    GameType::Dynamics::kSimultaneous,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kIdentical,
    GameType::RewardModel::kRewards,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {{"fully_observable", GameParameter(kDefaultFullyObservable)},
     {"horizon", GameParameter(kDefaultHorizon)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new CoopBoxPushingGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

ActionType ToAction(Action action) {
  switch (action) {
    case 0:
      return ActionType::kTurnLeft;
    case 1:
      return ActionType::kTurnRight;
    case 2:
      return ActionType::kMoveForward;
    case 3:
      return ActionType::kStay;
  }

  SpielFatalError(absl::StrCat("Invalid action: ", action));
}

std::string ActionToString(Action action) {
  switch (action) {
    case 0:
      return "turn left";
    case 1:
      return "turn right";
    case 2:
      return "move forward";
    case 3:
      return "stay";
  }

  SpielFatalError(absl::StrCat("Invalid action: ", action));
}

char ToCharacter(int orientation) {
  switch (orientation) {
    case OrientationType::kNorth:
      return '^';
    case OrientationType::kEast:
      return '>';
    case OrientationType::kSouth:
      return 'v';
    case OrientationType::kWest:
      return '<';
  }

  SpielFatalError(absl::StrCat("invalid orientation ", orientation));
}

OrientationType Rotate(OrientationType orientation, ActionType move) {
  if (move == ActionType::kTurnLeft) {
    return (orientation == 0 ? static_cast<OrientationType>(3)
                             : static_cast<OrientationType>(orientation - 1));
  } else {
    return (orientation == 3 ? static_cast<OrientationType>(0)
                             : static_cast<OrientationType>(orientation + 1));
  }
}

std::pair<int, int> NextCoord(std::pair<int, int> coord, int direction) {
  return {coord.first + row_offsets[direction],
          coord.second + col_offsets[direction]};
}
}  // namespace

CoopBoxPushingState::CoopBoxPushingState(std::shared_ptr<const Game> game,
                                         int horizon, bool fully_observable)
    : SimMoveState(game),
      total_rewards_(0),
      horizon_(horizon),
      cur_player_(kSimultaneousPlayerId),
      total_moves_(0),
      initiative_(0),
      win_(false),
      fully_observable_(fully_observable),
      reward_(0),
      action_status_(
          {ActionStatusType::kUnresolved, ActionStatusType::kUnresolved}) {
  field_.resize(kRows * kCols, '.');

  // Small boxes.
  SetField({3, 1}, 'b');
  SetField({3, 6}, 'b');

  // Big box.
  SetField({3, 3}, 'B');
  SetField({3, 4}, 'B');

  // Agents.
  SetPlayer({6, 1}, 0, OrientationType::kEast);
  SetPlayer({6, 6}, 1, OrientationType::kWest);
}

std::string CoopBoxPushingState::ActionToString(Player player,
                                                Action action) const {
  return ::open_spiel::coop_box_pushing::ActionToString(action);
}

void CoopBoxPushingState::SetField(std::pair<int, int> coord, char v) {
  field_[coord.first * kCols + coord.second] = v;
}

void CoopBoxPushingState::SetPlayer(std::pair<int, int> coord, Player player,
                                    OrientationType orientation) {
  SetField(coord, ToCharacter(orientation));
  player_coords_[player] = coord;
  player_orient_[player] = orientation;
}

void CoopBoxPushingState::SetPlayer(std::pair<int, int> coord, Player player) {
  SetPlayer(coord, player, player_orient_[player]);
}

char CoopBoxPushingState::field(std::pair<int, int> coord) const {
  return field_[coord.first * kCols + coord.second];
}

void CoopBoxPushingState::DoApplyActions(const std::vector<Action>& actions) {
  SPIEL_CHECK_EQ(actions.size(), 2);
  SPIEL_CHECK_EQ(cur_player_, kSimultaneousPlayerId);
  moves_[0] = ToAction(actions[0]);
  moves_[1] = ToAction(actions[1]);
  cur_player_ = kChancePlayerId;
}

bool CoopBoxPushingState::InBounds(std::pair<int, int> coord) const {
  return (coord.first >= 0 && coord.second >= 0 && coord.first < kRows &&
          coord.second < kCols);
}

void CoopBoxPushingState::MoveForward(Player player) {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LE(player, 1);

  OrientationType dir = player_orient_[player];
  auto next = NextCoord(player_coords_[player], dir);

  if (!InBounds(next)) {
    // Bump.. out of bounds!
    AddReward(kBumpPenalty);
  } else if (field(next) == '.') {
    // Uninterrupted move.
    SetField(player_coords_[player], '.');
    SetPlayer(next, player);
  } else if (field(next) == 'b') {
    auto next_next = NextCoord(next, dir);
    if (!InBounds(next_next)) {
      // Bump, can't push box out of bounds!
      AddReward(kBumpPenalty);
    } else if (field(next_next) == '.') {
      // Move the small box.
      SetField(next_next, 'b');
      SetField(player_coords_[player], '.');
      SetPlayer(next, player);

      // Check for reward.
      if (next_next.first == 0 && next.first != 0) {
        AddReward(kSmallBoxReward);
      }
    } else {
      // Trying to move box into something else.. bump!
      AddReward(kBumpPenalty);
    }
  } else {
    // Also bump!
    AddReward(kBumpPenalty);
  }
}

void CoopBoxPushingState::AddReward(double reward) {
  reward_ += reward;
  total_rewards_ += reward;
}

void CoopBoxPushingState::ResolveMoves() {
  // Check for successful move of the big box.
  if (moves_[0] == ActionType::kMoveForward &&
      moves_[1] == ActionType::kMoveForward &&
      action_status_[0] == ActionStatusType::kSuccess &&
      action_status_[1] == ActionStatusType::kSuccess) {
    std::array<std::pair<int, int>, 2> next_coords;
    std::array<std::pair<int, int>, 2> next_next_coords;

    next_coords[0] = NextCoord(player_coords_[0], player_orient_[0]);
    next_coords[1] = NextCoord(player_coords_[1], player_orient_[1]);
    next_next_coords[0] = NextCoord(next_coords[0], player_orient_[0]);
    next_next_coords[1] = NextCoord(next_coords[1], player_orient_[1]);

    if (InBounds(next_coords[0]) && InBounds(next_coords[1]) &&
        InBounds(next_next_coords[0]) && InBounds(next_next_coords[1]) &&
        field(next_coords[0]) == 'B' && field(next_coords[1]) == 'B' &&
        field(next_next_coords[0]) == '.' &&
        field(next_next_coords[1]) == '.') {
      SetField(next_next_coords[0], 'B');
      SetField(next_next_coords[1], 'B');
      SetField(player_coords_[0], '.');
      SetField(player_coords_[1], '.');
      SetPlayer(next_coords[0], 0);
      SetPlayer(next_coords[1], 1);

      if (next_next_coords[0].first == 0 && next_coords[0].first != 0) {
        AddReward(kBigBoxReward);
        win_ = true;
        return;
      }
    }
  }

  // Otherwise, just resolve them independently.
  for (int i = 0; i < 2; i++) {
    // Player order depends on initiative.
    int p = (i + initiative_) % 2;

    SPIEL_CHECK_GE(p, 0);
    SPIEL_CHECK_LT(p, 2);
    SPIEL_CHECK_TRUE(action_status_[p] != ActionStatusType::kUnresolved);

    ActionType move = moves_[p];

    // Action failed or deliberate stay => nothing happens to this agent.
    if (action_status_[p] == ActionStatusType::kFail ||
        move == ActionType::kStay) {
      continue;
    }

    if (move == ActionType::kTurnLeft || move == ActionType::kTurnRight) {
      SetPlayer(player_coords_[p], p, Rotate(player_orient_[p], move));
    } else if (move == ActionType::kMoveForward) {
      MoveForward(p);
    }
  }

  // Reset the action statuses and current player.
  cur_player_ = kSimultaneousPlayerId;
  action_status_[0] = ActionStatusType::kUnresolved;
  action_status_[1] = ActionStatusType::kUnresolved;

  AddReward(kDelayPenalty);
  total_moves_++;
}

void CoopBoxPushingState::DoApplyAction(Action action) {
  reward_ = 0;
  if (IsSimultaneousNode()) return ApplyFlatJointAction(action);

  if (action == kChanceSuccess) {
    // Success.
    if (action_status_[0] == ActionStatusType::kUnresolved) {
      action_status_[0] = ActionStatusType::kSuccess;
    } else if (action_status_[1] == ActionStatusType::kUnresolved) {
      action_status_[1] = ActionStatusType::kSuccess;
    } else {
      SpielFatalError(absl::StrCat("Invalid chance move case: ", action));
    }
  } else if (action == kChanceFail) {
    // Fail!
    if (action_status_[0] == ActionStatusType::kUnresolved) {
      action_status_[0] = ActionStatusType::kFail;
    } else if (action_status_[1] == ActionStatusType::kUnresolved) {
      action_status_[1] = ActionStatusType::kFail;
    } else {
      SpielFatalError(absl::StrCat("Invalid chance move case: ", action));
    }
  } else if (action == kChanceInit1) {
    // Player 1 moves first.
    initiative_ = 0;
    ResolveMoves();
  } else {
    // Player 2 moves first.
    initiative_ = 1;
    ResolveMoves();
  }
}

std::vector<Action> CoopBoxPushingState::LegalActions(Player player) const {
  if (player == kSimultaneousPlayerId) {
    return LegalFlatJointActions();
  } else if (IsTerminal()) {
    return {};
  } else if (IsChanceNode()) {
    if (action_status_[0] == ActionStatusType::kUnresolved ||
        action_status_[1] == ActionStatusType::kUnresolved) {
      return {0, 1};
    } else {
      return {2, 3};
    }
  }
  // All the actions are legal at every state.
  return {0, 1, 2, 3};
}

ActionsAndProbs CoopBoxPushingState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());

  if (action_status_[0] == ActionStatusType::kUnresolved ||
      action_status_[1] == ActionStatusType::kUnresolved) {
    // Determine success (0) or failure (1) of a player's action.
    return {std::pair<Action, double>(0, 0.9),
            std::pair<Action, double>(1, 0.1)};
  } else {
    // Determine initiative outcomes (2 and 3)
    return {std::pair<Action, double>(2, 0.5),
            std::pair<Action, double>(3, 0.5)};
  }
}

std::string CoopBoxPushingState::ToString() const {
  std::string result = "";
  absl::StrAppend(&result, "Total moves: ", total_moves_, "\n");
  absl::StrAppend(&result, "Most recent reward: ", reward_, "\n");
  absl::StrAppend(&result, "Total rewards: ", total_rewards_, "\n");

  for (int r = 0; r < kRows; r++) {
    for (int c = 0; c < kCols; c++) {
      result += field({r, c});
    }

    absl::StrAppend(&result, "\n");
  }

  return result;
}

ObservationType CoopBoxPushingState::PartialObservation(Player player) const {
  std::pair<int, int> adj_coord = {
      player_coords_[player].first + row_offsets[player_orient_[player]],
      player_coords_[player].second + col_offsets[player_orient_[player]]};

  if (!InBounds(adj_coord)) {
    return kWallObs;
  } else {
    switch (field(adj_coord)) {
      case kField:
        return kEmptyFieldObs;
      case kLeft:
      case kRight:
      case kUp:
      case kDown:
        return kOtherAgentObs;
      case kSmallBox:
        return kSmallBoxObs;
      case kBigBox:
        return kBigBoxObs;
      default:
        SpielFatalError("Unrecognized field char: " +
                        std::to_string(field(adj_coord)));
    }
  }
}

std::string CoopBoxPushingState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  if (fully_observable_) {
    return ToString();
  } else {
    ObservationType obs = PartialObservation(player);
    switch (obs) {
      case kEmptyFieldObs:
        return "field";
      case kWallObs:
        return "wall";
      case kOtherAgentObs:
        return "other agent";
      case kSmallBoxObs:
        return "small box";
      case kBigBoxObs:
        return "big box";
      default:
        SpielFatalError("Unrecognized observation!");
    }
  }
}

bool CoopBoxPushingState::IsTerminal() const {
  return (total_moves_ >= horizon_ || win_);
}

std::vector<double> CoopBoxPushingState::Returns() const {
  // Cooperative game: all players get same reward.
  return {total_rewards_, total_rewards_};
}

std::vector<double> CoopBoxPushingState::Rewards() const {
  // Cooperative game: all players get same reward.
  return {reward_, reward_};
}

bool CoopBoxPushingState::SameAsPlayer(std::pair<int, int> coord,
                                       Player player) const {
  return coord == player_coords_[player];
}

int CoopBoxPushingState::ObservationPlane(std::pair<int, int> coord,
                                          Player player) const {
  int plane = 0;
  switch (field(coord)) {
    case kField:
      plane = 0;
      break;
    case kSmallBox:
      plane = 1;
      break;
    case kBigBox:
      plane = 2;
      break;
    case kLeft:
      plane = (SameAsPlayer(coord, player)) ? 3 : 4;
      break;
    case kRight:
      plane = (SameAsPlayer(coord, player)) ? 5 : 6;
      break;
    case kUp:
      plane = (SameAsPlayer(coord, player)) ? 7 : 8;
      break;
    case kDown:
      plane = (SameAsPlayer(coord, player)) ? 9 : 10;
      break;
    default:
      std::cerr << "Invalid character on field: " << field(coord) << std::endl;
      std::cerr << ToString() << std::endl;
      plane = -1;
      break;
  }

  return plane;
}

void CoopBoxPushingState::ObservationTensor(Player player,
                                            absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  if (fully_observable_) {
    TensorView<3> view(values, {kCellStates, kRows, kCols}, true);

    for (int r = 0; r < kRows; r++) {
      for (int c = 0; c < kCols; c++) {
        int plane = ObservationPlane({r, c}, player);
        SPIEL_CHECK_TRUE(plane >= 0 && plane < kCellStates);
        view[{plane, r, c}] = 1.0;
      }
    }
  } else {
    SPIEL_CHECK_EQ(values.size(), kNumObservations);
    std::fill(values.begin(), values.end(), 0);
    ObservationType obs = PartialObservation(player);
    values[obs] = 1;
  }
}

std::unique_ptr<State> CoopBoxPushingState::Clone() const {
  return std::unique_ptr<State>(new CoopBoxPushingState(*this));
}

CoopBoxPushingGame::CoopBoxPushingGame(const GameParameters& params)
    : SimMoveGame(kGameType, params),
      horizon_(ParameterValue<int>("horizon")),
      fully_observable_(ParameterValue<bool>("fully_observable")) {}

std::vector<int> CoopBoxPushingGame::ObservationTensorShape() const {
  if (fully_observable_) {
    return {kCellStates, kRows, kCols};
  } else {
    return {kNumObservations};
  }
}

int CoopBoxPushingGame::NumDistinctActions() const {
  return kNumDistinctActions;
}

int CoopBoxPushingGame::NumPlayers() const { return kNumPlayers; }

std::unique_ptr<State> CoopBoxPushingGame::NewInitialState() const {
  std::unique_ptr<State> state(
      new CoopBoxPushingState(shared_from_this(), horizon_, fully_observable_));
  return state;
}

// This is a cooperative game where rewards are summed over players.
// So multiply the lower/upper bound by number of players. Also, utility is
// handed out at the end of the episode, so multiply this lower bound by the
// episode length.
double CoopBoxPushingGame::MaxUtility() const {
  return MaxGameLength() * NumPlayers() * (kBigBoxReward + kDelayPenalty);
}

double CoopBoxPushingGame::MinUtility() const {
  return MaxGameLength() * NumPlayers() * (kBumpPenalty + kDelayPenalty);
}

}  // namespace coop_box_pushing
}  // namespace open_spiel
