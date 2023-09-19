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

#include "open_spiel/games/phantom_go/phantom_go.h"

#include <random>
#include <sstream>

#include "open_spiel/game_parameters.h"
#include "open_spiel/games/phantom_go/phantom_go_board.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace phantom_go {
namespace {

// Facts about the game
const GameType kGameType{
    /*short_name=*/"phantom_go",
    /*long_name=*/"Phantom Go",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {{"komi", GameParameter(7.5)},
     {"board_size", GameParameter(9)},
     {"handicap", GameParameter(0)},
     // After the maximum game length, the game will end arbitrarily and the
     // score is computed as usual (i.e. number of stones + komi).
     // It's advised to only use shorter games to compute win-rates.
     // When not provided, it defaults to DefaultMaxGameLength(board_size)
     {"max_game_length",
      GameParameter(GameParameter::Type::kInt, /*is_mandatory=*/false)}},
};

std::shared_ptr<const Game> Factory(const GameParameters &params) {
  return std::shared_ptr<const Game>(new PhantomGoGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

std::vector<VirtualPoint> HandicapStones(int num_handicap) {
  if (num_handicap < 2 || num_handicap > 9) return {};

  static std::array<VirtualPoint, 9> placement = {
      {MakePoint("d4"), MakePoint("q16"), MakePoint("d16"), MakePoint("q4"),
       MakePoint("d10"), MakePoint("q10"), MakePoint("k4"), MakePoint("k16"),
       MakePoint("k10")}};
  static VirtualPoint center = MakePoint("k10");

  std::vector points(placement.begin(), placement.begin() + num_handicap);

  if (num_handicap >= 5 && num_handicap % 2 == 1) {
    points[num_handicap - 1] = center;
  }

  return points;
}

}  // namespace

class PhantomGoObserver : public Observer {
 public:
  PhantomGoObserver(IIGObservationType iig_obs_type)
      : Observer(/*has_string=*/true, /*has_tensor=*/true),
        iig_obs_type_(iig_obs_type) {}

  void WriteTensor(const State &observed_state, int player,
                   Allocator *allocator) const override {
    const PhantomGoState &state =
        open_spiel::down_cast<const PhantomGoState &>(observed_state);

    const int totalBoardPoints =
        state.board().board_size() * state.board().board_size();

    {
      auto out = allocator->Get("stone-counts", {2});
      auto stoneCount = state.GetStoneCount();
      out.at(0) = stoneCount[0];
      out.at(1) = stoneCount[1];
    }

    if (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer) {
      {
        auto observation = state.board().GetObservationByID(player);

        auto out_empty =
            allocator->Get("player_observation_empty", {totalBoardPoints});
        auto out_white =
            allocator->Get("player_observation_white", {totalBoardPoints});
        auto out_black =
            allocator->Get("player_observation_black", {totalBoardPoints});
        auto out_komi = allocator->Get("komi", {totalBoardPoints});

        for (int i = 0; i < totalBoardPoints; i++) {
          switch (observation[i]) {
            case GoColor::kBlack:
              out_black.at(i) = true;
              out_white.at(i) = false;
              out_empty.at(i) = false;
              break;

            case GoColor::kWhite:
              out_black.at(i) = false;
              out_white.at(i) = true;
              out_empty.at(i) = false;
              break;

            case GoColor::kEmpty:
              out_black.at(i) = false;
              out_white.at(i) = false;
              out_empty.at(i) = true;
              break;

            default:
              SpielFatalError(absl::StrCat("Unhandled case: ", observation[i]));
          }
          if (state.CurrentPlayer() == (uint8_t)GoColor::kWhite) {
            out_komi.at(i) = 1;
          } else {
            out_komi.at(i) = 0;
          }
        }
      }
    }
  }

  std::string StringFrom(const State &observed_state,
                         int player) const override {
    const PhantomGoState &state =
        open_spiel::down_cast<const PhantomGoState &>(observed_state);

    return state.ObservationString(player);
  }

 private:
  IIGObservationType iig_obs_type_;
};

PhantomGoState::PhantomGoState(std::shared_ptr<const Game> game, int board_size,
                               float komi, int handicap)
    : State(std::move(game)),
      board_(board_size),
      komi_(komi),
      handicap_(handicap),
      max_game_length_(game_->MaxGameLength()),
      to_play_(GoColor::kBlack) {
  ResetBoard();
}

std::string PhantomGoState::ObservationString(int player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return absl::StrCat(board_.ObservationToString(player),
                      board_.LastMoveInformationToString());
}

void PhantomGoState::ObservationTensor(Player player,
                                       absl::Span<float> values) const {
  ContiguousAllocator allocator(values);
  const PhantomGoGame &game =
      open_spiel::down_cast<const PhantomGoGame &>(*game_);
  game.default_observer_->WriteTensor(*this, player, &allocator);
}

std::vector<Action> PhantomGoState::LegalActions() const {
  std::vector<Action> actions{};
  if (IsTerminal()) return actions;
  for (VirtualPoint p : BoardPoints(board_.board_size())) {
    if (board_.IsLegalMove(p, to_play_)) {
      actions.push_back(board_.VirtualActionToAction(p));
    }
  }
  actions.push_back(board_.pass_action());
  return actions;
}

std::string PhantomGoState::ActionToString(Player player, Action action) const {
  return absl::StrCat(
      GoColorToString(static_cast<GoColor>(player)), " ",
      VirtualPointToString(board_.ActionToVirtualAction(action)));
}

char GoColorToChar(GoColor c) {
  switch (c) {
    case GoColor::kBlack:
      return 'X';
    case GoColor::kWhite:
      return 'O';
    case GoColor::kEmpty:
      return '+';
    case GoColor::kGuard:
      return '#';
    default:
      SpielFatalError(absl::StrCat("Unknown color ", c, " in GoColorToChar."));
      return '!';
  }
}

std::string PhantomGoState::ToString() const {
  std::array<int, 2> stoneCount = board_.GetStoneCount();

  return absl::StrCat("GoState(komi=", komi_,
                      ", to_play=", GoColorToString(to_play_),
                      ", history.size()=", history_.size(), ", ",
                      "stones_count: w", stoneCount[1], " b", stoneCount[0],
                      ")\n", board_.ToString(), board_.ObservationsToString());
}

bool PhantomGoState::IsTerminal() const {
  if (history_.size() < 2) return false;
  return (history_.size() >= max_game_length_) || superko_ ||
         (history_[history_.size() - 1].action == board_.pass_action() &&
          history_[history_.size() - 2].action == board_.pass_action());
}

std::vector<double> PhantomGoState::Returns() const {
  if (!IsTerminal()) return {0.0, 0.0};

  if (superko_) {
    // Superko rules (https://senseis.xmp.net/?Superko) are complex and vary
    // between rulesets.
    // For simplicity and because superkos are very rare, we just treat them as
    // a draw.
    return {DrawUtility(), DrawUtility()};
  }

  // Score with Tromp-Taylor.
  float black_score = TrompTaylorScore(board_, komi_, handicap_);

  std::vector<double> returns(phantom_go::NumPlayers());
  if (black_score > 0) {
    returns[ColorToPlayer(GoColor::kBlack)] = WinUtility();
    returns[ColorToPlayer(GoColor::kWhite)] = LossUtility();
  } else if (black_score < 0) {
    returns[ColorToPlayer(GoColor::kBlack)] = LossUtility();
    returns[ColorToPlayer(GoColor::kWhite)] = WinUtility();
  } else {
    returns[ColorToPlayer(GoColor::kBlack)] = DrawUtility();
    returns[ColorToPlayer(GoColor::kWhite)] = DrawUtility();
  }
  return returns;
}

std::unique_ptr<State> PhantomGoState::Clone() const {
  return std::unique_ptr<State>(new PhantomGoState(*this));
}

void PhantomGoState::UndoAction(Player player, Action action) {
  // We don't have direct undo functionality, but copying the board and
  // replaying all actions is still pretty fast (> 1 million undos/second).
  history_.pop_back();
  --move_number_;
  ResetBoard();
  for (auto [_, action] : history_) {
    DoApplyAction(action);
  }
}

void PhantomGoState::DoApplyAction(Action action) {
  if (board_.PlayMove(board_.ActionToVirtualAction(action), to_play_)) {
    to_play_ = OppColor(to_play_);
    bool was_inserted = repetitions_.insert(board_.HashValue()).second;
    if (!was_inserted && action != board_.pass_action()) {
      // We have encountered this position before.
      superko_ = true;
    }
  }
}

void PhantomGoState::ResetBoard() {
  board_.Clear();
  if (handicap_ < 2) {
    to_play_ = GoColor::kBlack;
  } else {
    for (VirtualPoint p : HandicapStones(handicap_)) {
      board_.PlayMove(p, GoColor::kBlack);
    }
    to_play_ = GoColor::kWhite;
  }

  repetitions_.clear();
  repetitions_.insert(board_.HashValue());
  superko_ = false;
}
std::array<int, 2> PhantomGoState::GetStoneCount() const {
  return board_.GetStoneCount();
}
bool PhantomGoState::equalMetaposition(const PhantomGoState &state1,
                                       const PhantomGoState &state2,
                                       int playerID) {
  if (state1.board_.board_size() != state2.board_.board_size()) {
    return false;
  }

  std::array<int, 2> stoneCount1 = state1.board_.GetStoneCount();
  std::array<int, 2> stoneCount2 = state2.board_.GetStoneCount();

  if (stoneCount1[0] != stoneCount2[0] || stoneCount1[1] != stoneCount2[1]) {
    return false;
  }

  int boardSize = state1.board_.board_size();

  auto observation1 = state1.board_.GetObservationByID(playerID);
  auto observation2 = state2.board_.GetObservationByID(playerID);

  for (int i = 0; i < boardSize * boardSize; i++) {
    if (observation1[i] != observation2[i]) {
      return false;
    }
  }

  if (state1.to_play_ != state2.to_play_) {
    return false;
  }

  return true;
}
int PhantomGoState::GetMaxGameLenght() const { return max_game_length_; }

PhantomGoGame::PhantomGoGame(const GameParameters &params)
    : Game(kGameType, params),
      komi_(ParameterValue<double>("komi")),
      board_size_(ParameterValue<int>("board_size")),
      handicap_(ParameterValue<int>("handicap")),
      max_game_length_(ParameterValue<int>("max_game_length",
                                           DefaultMaxGameLength(board_size_))) {
  default_observer_ = std::make_shared<PhantomGoObserver>(kDefaultObsType);
}

}  // namespace phantom_go
}  // namespace open_spiel
