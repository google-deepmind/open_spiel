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

#include "open_spiel/games/phantom_go.h"

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
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {{"komi", GameParameter(7.5)},
     {"board_size", GameParameter(19)},
     {"handicap", GameParameter(0)},
     // After the maximum game length, the game will end arbitrarily and the
     // score is computed as usual (i.e. number of stones + komi).
     // It's advised to only use shorter games to compute win-rates.
     // When not provided, it defaults to DefaultMaxGameLength(board_size)
     {"max_game_length",
      GameParameter(GameParameter::Type::kInt, /*is_mandatory=*/false)}},
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new PhantomGoGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

std::vector<VirtualPoint> HandicapStones(int num_handicap) {
  if (num_handicap < 2 || num_handicap > 9) return {};

  static std::array<VirtualPoint, 9> placement = {
      {MakePoint("d4"), MakePoint("q16"), MakePoint("d16"), MakePoint("q4"),
       MakePoint("d10"), MakePoint("q10"), MakePoint("k4"), MakePoint("k16"),
       MakePoint("k10")}};
  static VirtualPoint center = MakePoint("k10");

  std::vector<VirtualPoint> points;
  points.reserve(num_handicap);
  for (int i = 0; i < num_handicap; ++i) {
    points.push_back(placement[i]);
  }

  if (num_handicap >= 5 && num_handicap % 2 == 1) {
    points[num_handicap - 1] = center;
  }

  return points;
}

}  // namespace

PhantomGoState::PhantomGoState(std::shared_ptr<const Game> game, int board_size, float komi,
                 int handicap)
    //help 
    : State(std::move(game)),
      board_(board_size),
      komi_(komi),
      handicap_(handicap),
      max_game_length_(game_->MaxGameLength()),
      to_play_(GoColor::kBlack) {
  ResetBoard();
  
}


// this method is in progress of making, the implementation is not correct
std::unique_ptr<State> PhantomGoState::ResampleFromInfostate(
    int player_id, std::function<double()> rng) const {
    int boardSize = board_.board_size();

    /*std::shared_ptr<const Game> newGame = GetGame();
    std::unique_ptr<State> newState = newGame->NewInitialState();*/

    std::shared_ptr<const Game> newGame = LoadGame("phantom_go");
    std::unique_ptr<PhantomGoState> newState = std::make_unique<PhantomGoState>(PhantomGoState(newGame, boardSize, komi_, handicap_));

    std::array<GoColor, kMaxBoardSize* kMaxBoardSize> infoState = board_.GetObservationByID(player_id);
    std::array<int, 2> stoneCount = board_.getStoneCount();

    std::array<std::vector<int>, 2> stones;


    //Find and store all stones
    for (int i = 0; i < boardSize * boardSize; i++)
    {
        if (infoState[i] != GoColor::kEmpty)
        {
            stones[(uint8_t)infoState[i]].push_back(i);
        }
    }


    int i = 0;
    int max;
    if(stoneCount[(uint8_t)GoColor::kBlack] > stoneCount[(uint8_t)GoColor::kWhite])
    {
        max = stoneCount[(uint8_t)GoColor::kBlack];
    }
    else
    {
        max = stoneCount[(uint8_t)GoColor::kWhite];
    }

    while (i < max)
    {
        for (int c = 0; c <= 1; c++)
        {
            if (i >= stones[c].size())
            {
                if(i < stoneCount[c])
                {
                    std::vector<Action> actions = newState->LegalActions();
                    std::shuffle(actions.begin(), actions.end(), std::mt19937(std::random_device()()));
                    std::array<int, 2> currStoneCount = newState->board_.getStoneCount();
                    currStoneCount[c]++;

                    for(long action : actions)
                    {
                        if(action == VirtualActionToAction(kVirtualPass, boardSize))
                            continue;

                        newState->ApplyAction(action);
                        if(newState->board_.getStoneCount()[0] == currStoneCount[0] &&
                            newState->board_.getStoneCount()[1] == currStoneCount[1])
                        { //random move was applied correctly, no captures were made
                            if(player_id != c) {
                                newState->ApplyAction(action);
                            }
                            break;
                        }
                        else
                        {
                            newState->UndoAction(-1, -1);
                        }
                    }

                }
                else {
                    newState->ApplyAction(VirtualActionToAction(kVirtualPass, boardSize));
                    //printf("pass\n");
                }
            }

            else{
                newState->ApplyAction(stones[c][i]);
                if(player_id != c) {
                    newState->ApplyAction(stones[c][i]);
                }
                //printf("%i\n", stones[c][i]);
            }

        }
        i++;
    }

    return newState;
}

std::string PhantomGoState::InformationStateString(int player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string PhantomGoState::ObservationString(int player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void PhantomGoState::ObservationTensor(int player, absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  int num_cells = board_.board_size() * board_.board_size();
  SPIEL_CHECK_EQ(values.size(), num_cells * (CellStates() + 1));
  std::fill(values.begin(), values.end(), 0.);

  // Add planes: black, white, empty.
  int cell = 0;
  for (VirtualPoint p : BoardPoints(board_.board_size())) {
    int color_val = static_cast<int>(board_.PointColor(p));
    values[num_cells * color_val + cell] = 1.0;
    ++cell;
  }
  SPIEL_CHECK_EQ(cell, num_cells);

  // Add a fourth binary plane for komi (whether white is to play).
  std::fill(values.begin() + (CellStates() * num_cells), values.end(),
            (to_play_ == GoColor::kWhite ? 1.0 : 0.0));
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
  std::stringstream ss;
  std::array<int, 2> stoneCount = board_.getStoneCount();
  ss << "GoState(komi=" << komi_ << ", to_play=" << GoColorToString(to_play_)
     << ", history.size()=" << history_.size() << ", "
     << "stones_count: w" << stoneCount[1] << " b" << stoneCount[0] << ")\n";

  ss << board_;

  ss << board_.observationToString();

  return ss.str();
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
    if (board_.PlayMove(board_.ActionToVirtualAction(action), to_play_))
    {
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

PhantomGoGame::PhantomGoGame(const GameParameters& params)
    : Game(kGameType, params),
      komi_(ParameterValue<double>("komi")),
      board_size_(ParameterValue<int>("board_size")),
      handicap_(ParameterValue<int>("handicap")),
      max_game_length_(ParameterValue<int>(
          "max_game_length", DefaultMaxGameLength(board_size_))) {}



}  // namespace phantom_go
}  // namespace open_spiel
