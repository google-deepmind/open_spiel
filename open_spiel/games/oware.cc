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

#include "open_spiel/games/oware.h"

#include <iomanip>

#include "open_spiel/game_parameters.h"

namespace open_spiel {
namespace oware {

namespace {

// Facts about the game
const GameType kGameType{
    /*short_name=*/"oware",
    /*long_name=*/"Oware",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {{"num_houses_per_player", GameParameter(kDefaultHousesPerPlayer)},
     {"num_seeds_per_house", GameParameter(kDdefaultSeedsPerHouse)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new OwareGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

OwareState::OwareState(std::shared_ptr<const Game> game,
                       int num_houses_per_player, int num_seeds_per_house)
    : State(game),
      num_houses_per_player_(num_houses_per_player),
      total_seeds_(kNumPlayers * num_seeds_per_house * num_houses_per_player),
      board_(/*num_houses_per_player=*/num_houses_per_player,
             /*num_seeds_per_house=*/num_seeds_per_house) {
  boards_since_last_capture_.insert(board_);
}

OwareState::OwareState(std::shared_ptr<const Game> game,
                       const OwareBoard& board)
    : State(game),
      num_houses_per_player_(board.seeds.size() / kNumPlayers),
      total_seeds_(board.TotalSeeds()),
      board_(board) {
  SPIEL_CHECK_EQ(0, board.seeds.size() % kNumPlayers);
  SPIEL_CHECK_TRUE(IsTerminal() || !LegalActions().empty());
  boards_since_last_capture_.insert(board_);
}

std::vector<Action> OwareState::LegalActions() const {
  std::vector<Action> actions;
  if (IsTerminal()) return actions;
  const Player lower = PlayerLowerHouse(board_.current_player);
  const Player upper = PlayerUpperHouse(board_.current_player);
  if (OpponentSeeds() == 0) {
    // In case the opponent does not have any seeds, a player must make
    // a move which gives the opponent seeds.
    for (int house = lower; house <= upper; house++) {
      const int first_seeds_in_own_row = upper - house;
      if (board_.seeds[house] - first_seeds_in_own_row > 0) {
        actions.push_back(HouseToAction(house));
      }
    }
  } else {
    for (int house = lower; house <= upper; house++) {
      if (board_.seeds[house] > 0) {
        actions.push_back(HouseToAction(house));
      }
    }
  }
  return actions;
}

std::string OwareState::ActionToString(Player player, Action action) const {
  return std::string(1, (player == Player{0} ? 'A' : 'a') + action);
}

void OwareState::WritePlayerScore(std::ostringstream& out,
                                  Player player) const {
  out << "Player " << player << " score = " << board_.score[player];
  if (CurrentPlayer() == player) {
    out << " [PLAYING]" << std::endl;
  } else {
    out << std::endl;
  }
}

std::string OwareState::ToString() const {
  std::ostringstream out;
  if (IsTerminal()) {
    out << "[FINISHED]" << std::endl;
  }
  WritePlayerScore(out, 1);

  // Add player 1 labels.
  for (int action = num_houses_per_player_ - 1; action >= 0; action--) {
    out << std::setw(3) << std::right << ActionToString(1, action);
  }
  out << std::endl;

  // Add player 1 house seeds.
  for (int house = kNumPlayers * num_houses_per_player_ - 1;
       house >= num_houses_per_player_; house--) {
    out << std::setw(3) << std::right << board_.seeds[house];
  }
  out << std::endl;

  // Add player 0 house seeds.
  for (int house = 0; house < num_houses_per_player_; house++) {
    out << std::setw(3) << std::right << board_.seeds[house];
  }
  out << std::endl;

  // Add player 0 labels.
  for (int action = 0; action < num_houses_per_player_; action++) {
    out << std::setw(3) << std::right << ActionToString(0, action);
  }
  out << std::endl;

  WritePlayerScore(out, 0);
  return out.str();
}

bool OwareState::IsTerminal() const {
  // Terminate when one player has more than half of the seeds
  // (works both for even and odd number of seeds), or when all seeds
  // are equally shared.
  const int limit = total_seeds_ / 2;
  return board_.score[0] > limit || board_.score[1] > limit ||
         (board_.score[0] == limit && board_.score[1] == limit);
}

std::vector<double> OwareState::Returns() const {
  if (IsTerminal()) {
    if (board_.score[0] > board_.score[1]) {
      return {1, -1};
    } else if (board_.score[0] < board_.score[1]) {
      return {-1, 1};
    } else {
      return {0, 0};
    }
  } else {
    return {0, 0};
  }
}

std::unique_ptr<State> OwareState::Clone() const {
  return std::unique_ptr<State>(new OwareState(*this));
}

int OwareState::DistributeSeeds(int house) {
  int to_distribute = board_.seeds[house];
  SPIEL_CHECK_NE(to_distribute, 0);
  board_.seeds[house] = 0;
  int index = house;
  while (to_distribute > 0) {
    index = (index + 1) % NumHouses();
    // Seeds are never sown into the house they were drawn from.
    if (index != house) {
      board_.seeds[index]++;
      to_distribute--;
    }
  }
  return index;
}

bool OwareState::InOpponentRow(int house) const {
  return (house / num_houses_per_player_) != board_.current_player;
}

bool OwareState::IsGrandSlam(int house) const {
  // If there are seeds beyond the house in which the last seed was dropped,
  // it is not a Grand Slam.
  for (int index = UpperHouse(house); index > house; index--) {
    if (board_.seeds[index] > 0) {
      return false;
    }
  }
  // If not all houses are captured starting from the house in which the last
  // seed was dropped, it is not a Grand Slam. It means the opponent will still
  // have some seeds left because none of these houses can be empty due to
  // the way seeds are sown.
  const int lower = LowerHouse(house);
  for (int index = house; index >= lower; index--) {
    SPIEL_CHECK_GT(board_.seeds[index], 0);
    if (!ShouldCapture(board_.seeds[index])) {
      return false;
    }
  }
  return true;
}

int OwareState::OpponentSeeds() const {
  int count = 0;
  const Player opponent = 1 - board_.current_player;
  const int lower = PlayerLowerHouse(opponent);
  const int upper = PlayerUpperHouse(opponent);
  for (int house = lower; house <= upper; house++) {
    count += board_.seeds[house];
  }
  return count;
}

int OwareState::DoCaptureFrom(int house) {
  const int lower = LowerHouse(house);
  int captured = 0;
  for (int index = house; index >= lower; index--) {
    if (ShouldCapture(board_.seeds[index])) {
      captured += board_.seeds[index];
      board_.seeds[index] = 0;
    } else {
      break;
    }
  }
  board_.score[board_.current_player] += captured;
  return captured;
}

void OwareState::DoApplyAction(Action action) {
  SPIEL_CHECK_LT(history_.size(), kMaxGameLength);

  int last_house = DistributeSeeds(ActionToHouse(CurrentPlayer(), action));

  if (InOpponentRow(last_house) && !IsGrandSlam(last_house)) {
    const int captured = DoCaptureFrom(last_house);
    if (captured > 0) {
      // No need to keep previous boards for checking game repetition because
      // captured seeds do not re-enter the game.
      boards_since_last_capture_.clear();
    }
  }
  board_.current_player = 1 - board_.current_player;

  if (!boards_since_last_capture_.insert(board_).second) {
    // We have game repetition, the game is ended.
    CollectAndTerminate();
  }

  if (LegalActions().empty()) {
    CollectAndTerminate();
  }
}

void OwareState::CollectAndTerminate() {
  for (int house = 0; house < NumHouses(); house++) {
    const Player player = house / num_houses_per_player_;
    board_.score[player] += board_.seeds[house];
    board_.seeds[house] = 0;
  }
}

std::string OwareState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return board_.ToString();
}

void OwareState::ObservationTensor(Player player,
                                   absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  SPIEL_CHECK_EQ(values.size(), /*seeds*/ NumHouses() + /*scores*/ kNumPlayers);
  for (int house = 0; house < NumHouses(); ++house) {
    values[house] = ((double)board_.seeds[house]) / total_seeds_;
  }
  for (Player player = 0; player < kNumPlayers; ++player) {
    values[NumHouses() + player] =
        ((double)board_.score[player]) / total_seeds_;
  }
}

OwareGame::OwareGame(const GameParameters& params)
    : Game(kGameType, params),
      num_houses_per_player_(ParameterValue<int>("num_houses_per_player")),
      num_seeds_per_house_(ParameterValue<int>("num_seeds_per_house")) {}

std::vector<int> OwareGame::ObservationTensorShape() const {
  return {/*seeds*/ num_houses_per_player_ * kNumPlayers +
          /*scores*/ kNumPlayers};
}

}  // namespace oware
}  // namespace open_spiel
