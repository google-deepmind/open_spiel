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
    : State(std::move(game)),
      board_(board_size),
      komi_(komi),
      handicap_(handicap),
      max_game_length_(game_->MaxGameLength()),
      to_play_(GoColor::kBlack) {
    ResetBoard();

}

//This method is used, when the Metapositon Resampling fails
//It resamples the state into a Metaposition, that corresponds to the actual state on the game board
std::unique_ptr<State> PhantomGoState::ResampleFromMetapositionHard(
    int player_id, std::function<double()> rng) const {

    int boardSize = board_.board_size();
    Action pass_action = VirtualActionToAction(kVirtualPass, boardSize);
    auto opp_player_id = (uint8_t) OppColor((GoColor) player_id);

    std::shared_ptr<const Game> game = GetGame();
    std::unique_ptr<PhantomGoState>
        state = std::make_unique<PhantomGoState>(down_cast<PhantomGoState>(*game->NewInitialState()));

    std::array<std::vector<int>, 2> stones;
    std::array<int, 2> stoneCount = board_.GetStoneCount();
    std::vector<int> enemyVisibleStones;
    std::array<GoColor, kMaxBoardSize * kMaxBoardSize> infoState = board_.GetObservationByID(player_id);

    //Find and store all enemy visible stones
    for (int i = 0; i < boardSize * boardSize; i++) {
        if (infoState[i] == (GoColor) opp_player_id) {
            enemyVisibleStones.push_back(i);
        }
    }

    for (int i = 0; i < boardSize * boardSize; i++) {
        if (board_.PointColor(ActionToVirtualAction(i, boardSize)) != GoColor::kEmpty) {
            stones[(uint8_t) board_.PointColor(ActionToVirtualAction(i, boardSize))].push_back(i);
        }
    }

    if (player_id == (uint8_t) GoColor::kWhite) {
        state->ApplyAction(pass_action);
    }

    for (long action: stones[player_id]) // Fill the board with stones of player we want to resample for
    {
        state->ApplyAction(action);
        state->ApplyAction(pass_action);
    }

    if (!state->history_.empty()) {
        state->UndoAction(opp_player_id, pass_action);
    }

    if (state->history_.empty() && (GoColor) player_id == GoColor::kBlack) {
        state->ApplyAction(pass_action);
    }

    for (long action: stones[opp_player_id]) // Fill the board with stones of player we want to resample for
    {
        state->ApplyAction(action);
        if (std::find(enemyVisibleStones.begin(), enemyVisibleStones.end(), action) != enemyVisibleStones.end()) {
            state->ApplyAction(action);
        }
        state->ApplyAction(pass_action);
    }

    if (!state->history_.empty() && !stones[opp_player_id].empty()) {
        state->UndoAction(player_id, pass_action);
    }

    if (!(state->board_.GetStoneCount()[0] == stoneCount[0] &&
        state->board_.GetStoneCount()[1] == stoneCount[1])) {
        std::cout << "hard resample\nstone count" << ToString() << state->ToString();
        SpielFatalError("after resampling, the count of stones doesn't match\n");
    }

    return state;
}

std::unique_ptr<State> PhantomGoState::ResampleFromMetaposition(
    int player_id, std::function<double()> rng) const {

    int boardSize = board_.board_size();
    Action pass_action = VirtualActionToAction(kVirtualPass, boardSize);

    std::shared_ptr<const Game> game = GetGame();
    std::unique_ptr<PhantomGoState>
        state = std::make_unique<PhantomGoState>(down_cast<PhantomGoState>(*game->NewInitialState()));

    std::array<GoColor, kMaxBoardSize * kMaxBoardSize> infoState = board_.GetObservationByID(player_id);
    std::array<int, 2> stoneCount = board_.GetStoneCount();

    std::array<std::vector<int>, 2> stones;
    std::vector<Action> enemyActions;
    std::vector<bool> enemyActionVisibility;
    std::vector<int> enemyActionNumber;

    auto opp_player_id = (uint8_t) OppColor((GoColor) player_id);

    //Find and store all stones which are in the last move on board
    for (int i = 0; i < boardSize * boardSize; i++) {
        if (infoState[i] != GoColor::kEmpty) {
            stones[(uint8_t) infoState[i]].push_back(i);
        }
    }

    if (player_id == (uint8_t) GoColor::kWhite) {
        state->ApplyAction(pass_action);
    }

    for (long action: stones[player_id]) // Fill the board with stones of player we want to resample for
    {
        state->ApplyAction(action);
        state->ApplyAction(pass_action);
    }

    if (!state->history_.empty()) {
        state->UndoAction(opp_player_id, pass_action);
    }

    if (state->history_.empty() && !history_.empty() && (GoColor) player_id == GoColor::kBlack) {
        state->ApplyAction(pass_action);
    }

    for (long action: stones[opp_player_id]) {
        state->ApplyAction(action);
        state->ApplyAction(action);
        state->ApplyAction(pass_action);
    }

    for (int i = 0; i < stoneCount[opp_player_id] - stones[opp_player_id].size(); i++) {
        std::vector<Action> actions = state->LegalActions();
        std::shuffle(actions.begin(), actions.end(), std::mt19937(std::random_device()()));
        std::array<int, 2> currStoneCount = state->board_.GetStoneCount();
        currStoneCount[opp_player_id]++;
        std::vector<int> vec = stones[opp_player_id];
        bool actionChosen = false;
        for (long action: actions) {
            // pass can't be chosen, also an action that will be played by opposing player can't be chosen
            if (action == pass_action ||
                std::find(vec.begin(), vec.end(), action) != vec.end())
                continue;

            state->ApplyAction(action);
            if (state->board_.GetStoneCount()[0] == currStoneCount[0] &&
                state->board_.GetStoneCount()[1]
                    == currStoneCount[1]) { //random move was applied correctly, no captures were made
                state->ApplyAction(pass_action);
                actionChosen = true;
                break;
            } else {
                state->UndoAction(opp_player_id, action);
            }
        }
    }

    if (!state->history_.empty() && stoneCount[opp_player_id] != 0) {
        state->UndoAction(player_id, pass_action);
    }

    if (!history_.empty() && stoneCount[opp_player_id] == 0) {
        state->ApplyAction(pass_action);
    }

    if (!(state->board_.GetStoneCount()[0] == stoneCount[0] &&
        state->board_.GetStoneCount()[1] == stoneCount[1])) {
        return PhantomGoState::ResampleFromMetapositionHard(player_id, rng);
    }

    if (CurrentPlayer() != state->CurrentPlayer()) {
        std::cout << "resampling for " << player_id << "\nwrong player" << ToString() << state->ToString();

        for (int i = 0; i < state->history_.size(); i++) {
            std::cout << state->history_[i] << "\n";
        }
        SpielFatalError("after resampling, wrong current player\n");
    }

    return state;
}

//This method is unfinished, will be later replaced by or-tools CSP solver implementation
std::unique_ptr<State> PhantomGoState::ResampleFromInfostate(
    int player_id, std::function<double()> rng) const {
    /*
    int boardSize = board_.board_size();

    std::shared_ptr<const Game> game = GetGame();
    std::unique_ptr<PhantomGoState>
        state = std::make_unique<PhantomGoState>(down_cast<PhantomGoState>(*game->NewInitialState()));

    std::array<GoColor, kMaxBoardSize * kMaxBoardSize> infoState = board_.GetObservationByID(player_id);
    std::array<int, 2> stoneCount = board_.GetStoneCount();

    std::array<std::vector<int>, 2> stones;
    std::vector<Action> enemyActions;
    std::vector<bool> enemyActionVisibility;
    std::vector<int> enemyActionNumber;

    auto opp_payer_id = (uint8_t) OppColor((GoColor) player_id);

    //Find and store all stones which are in the last move on board
    for (int i = 0; i < boardSize * boardSize; i++) {
        if (infoState[i] != GoColor::kEmpty) {
            stones[(uint8_t) infoState[i]].push_back(i);
        }
    }

    std::vector<int> captureMoves;
    std::vector<std::vector<Action>> capturedActions;
    capturedActions.emplace_back();

    { //deciding which actions are important because of captures
        std::shared_ptr<const Game> historyGame = GetGame();
        std::unique_ptr<PhantomGoState>
            historyState = std::make_unique<PhantomGoState>(down_cast<PhantomGoState>(*game->NewInitialState()));
        //this state will be used as a state to replicate the whole history to be able to observe board in each step


        for (int i = 0; i < history_.size(); i++) {
            //continiously filling in a vector of enemy moves, for which their importance will be decided
            if (history_[i].player == opp_payer_id) {
                enemyActions.push_back(history_[i].action);
                enemyActionVisibility.push_back(false);
                enemyActionNumber.push_back(i);
                //pass must be played, the count of the stones wont match up
                if (history_[i].action == VirtualActionToAction(kVirtualPass, boardSize)) {
                    enemyActionVisibility[enemyActionVisibility.size() - 1] = true;
                }
            }

            std::array<int, 2> prevStoneCount = historyState->board_.GetStoneCount();
            historyState->ApplyAction(history_[i].action);
            std::array<int, 2> currStoneCount = historyState->board_.GetStoneCount();

            if (currStoneCount[0] < prevStoneCount[0] || currStoneCount[1]
                < prevStoneCount[1]) //if one of the counts of stones is lower than in the previous move
            {
                captureMoves.push_back(i); //in this move, a capture took place

                historyState->UndoAction(-1, -1);
                bool playerCaptured;
                if (historyState->CurrentPlayer() == player_id) {
                    playerCaptured = true;
                } else {
                    playerCaptured = false;
                }
                std::unique_ptr<PhantomGoState>
                    cloneState = std::make_unique<PhantomGoState>(down_cast<PhantomGoState>(*historyState->Clone()));
                GoColor capturedStonesColor = OppColor((GoColor) historyState->CurrentPlayer());
                std::cout << historyState->ToString();
                historyState->ApplyAction(history_[i].action);
                std::cout << historyState->ToString() << "captures: ";

                for (int x = 0; x < boardSize * boardSize;
                     x++) { //there was an enemy stone on board on that box, but now it isn't
                    if (historyState->board_.PointColor(ActionToVirtualAction(x, boardSize)) == GoColor::kEmpty &&
                        cloneState->board_.PointColor(ActionToVirtualAction(x, boardSize)) == capturedStonesColor) {
                        capturedActions[capturedActions.size() - 1].push_back(x);
                        std::cout << ActionToString((uint8_t) capturedStonesColor, x) << " ";
                        if (playerCaptured) { //if the capture was made by player we are resampling for, change the importance of the move that placed captured stone
                            for (int y = enemyActions.size() - 1; y >= 0; y--) {
                                if (enemyActions[y] == x && enemyActionNumber[y] <= i) {
                                    enemyActionVisibility[y] = true;
                                    break;
                                }
                            }
                        }
                    }
                }

                if (!playerCaptured) //we must add every adjacent stone to every captured stone to the "important" stones
                {
                    std::vector<Action> importantActions;
                    for (int x = 0; x < capturedActions[capturedActions.size() - 1].size(); x++) {
                        if (historyState->board_.PointColor(ActionToVirtualAction(
                            capturedActions[capturedActions.size() - 1][x] - 1, boardSize)) ==
                            (GoColor) opp_payer_id) {
                            importantActions.push_back(capturedActions[capturedActions.size() - 1][x] - 1);
                        }
                        if (historyState->board_.PointColor(ActionToVirtualAction(
                            capturedActions[capturedActions.size() - 1][x] + 1, boardSize)) ==
                            (GoColor) opp_payer_id) {
                            importantActions.push_back(capturedActions[capturedActions.size() - 1][x] + 1);
                        }

                        if (historyState->board_.PointColor(ActionToVirtualAction(
                            capturedActions[capturedActions.size() - 1][x] + boardSize, boardSize)) ==
                            (GoColor) opp_payer_id) {
                            importantActions.push_back(capturedActions[capturedActions.size() - 1][x] + boardSize);
                        }
                        if (historyState->board_.PointColor(ActionToVirtualAction(
                            capturedActions[capturedActions.size() - 1][x] - boardSize, boardSize)) ==
                            (GoColor) opp_payer_id) {
                            importantActions.push_back(capturedActions[capturedActions.size() - 1][x] - boardSize);
                        }
                    }

                    std::cout << "important actions: ";
                    for (int x = 0; x < importantActions.size(); x++) {
                        std::cout << ActionToString((uint8_t) OppColor(capturedStonesColor), importantActions[x]) + " ";
                        for (int y = enemyActions.size() - 1; y >= 0; y--) {
                            if (enemyActions[y] == importantActions[x] && enemyActionNumber[y] <= i) {
                                enemyActionVisibility[y] = true;
                                break;
                            }
                        }
                    }
                }

                std::cout << "\n";
                capturedActions.emplace_back();

            }
        }
    }

    { //deciding if enemy moves are important, because they will be observed
        std::shared_ptr<const Game> historyGame = GetGame();
        std::unique_ptr<PhantomGoState>
            historyState = std::make_unique<PhantomGoState>(down_cast<PhantomGoState>(*game->NewInitialState()));
        //this state will be used as a state to replicate the whole history to be able to observe board in each step

        for (int i = 0; i < history_.size(); i++) {

            // if the move on i-1 was observational
            if (history_[i].player == opp_payer_id
                && historyState->board_.PointColor(ActionToVirtualAction(history_[i].action, boardSize))
                    == (GoColor) player_id) {
                for (int x = enemyActions.size() - 1; x >= 0;
                     x--) { //second part of this if is important to mark a correct action, which happened before the observation move
                    if (enemyActions[x] == history_[i].action && enemyActionNumber[x] <= i) {
                        enemyActionVisibility[x] = true;
                        break;
                    }
                }
            }

            if (history_[i].player == player_id &&
                historyState->board_.PointColor(ActionToVirtualAction(history_[i].action, boardSize))
                    == (GoColor) opp_payer_id) {
                for (int x = enemyActions.size() - 1; x >= 0;
                     x--) { //second part of this if is important to mark a correct action, which happened before the observation move
                    if (enemyActions[x] == history_[i].action && enemyActionNumber[x] <= i) {
                        enemyActionVisibility[x] = true;
                        break;
                    }
                }
            }

            historyState->ApplyAction(history_[i].action);
        }
    }

    for (int i = 0; i < history_.size(); i++) {
        std::cout << i << " " << ActionToString(history_[i].player, history_[i].action) << "\n";
    }
    std::cout << "\n";
    for (int i = 0; i < enemyActions.size(); i++) {
        std::cout << ActionToString(opp_payer_id, enemyActions[i]) << " " << enemyActionVisibility[i]
                  << " " << enemyActionNumber[i] << "\n";
    }

    int captureSection = 0;
    int enemyMove = 0;
    captureMoves.push_back(history_.size() + 1);
    capturedActions.emplace_back(); //last section has no actions that are "illegal"
    for (int i = 0; i < history_.size(); i++) {
        // moving of separator of board "phases", separated by captures
        if (captureMoves[captureSection] == i) {
            captureSection++;
        }

        if (history_[i].player == player_id) {
            state->ApplyAction(history_[i].action);
        } else {
            if (enemyActionVisibility[enemyMove]) {
                SPIEL_CHECK_EQ(enemyActions[enemyMove], history_[i].action);
                state->ApplyAction(history_[i].action);
            } else {
                std::vector<Action> actions = state->LegalActions();
                std::shuffle(actions.begin(), actions.end(), std::mt19937(std::random_device()()));
                for (long &action: actions) {
                    if (action == VirtualActionToAction(kVirtualPass, boardSize)) {
                        continue;
                    }
                    // if is an action that will be made by any player in the future
                    if (std::find(stones[0].begin(), stones[0].end(), action) != stones[0].end()
                        || std::find(stones[1].begin(), stones[1].end(), action) != stones[1].end()) {
                        continue;
                    }
                    //if the move would be observational
                    if (state->board_.PointColor(ActionToVirtualAction(action, boardSize)) == (GoColor) player_id) {
                        continue;
                    }

                    bool legal = true;
                    for (int p = captureSection; p < captureMoves.size();
                         p++) { //if the action is part of any group of actions that will be played and then captured
                        if (std::find(capturedActions[p].begin(), capturedActions[p].end(), action) !=
                            capturedActions[p].end()) {
                            legal = false;
                            break;
                        }
                    }
                    if (legal) {
                        std::array<int, 2> prevStoneCount = state->board_.GetStoneCount();
                        state->ApplyAction(action);
                        std::array<int, 2> currStoneCount = state->board_.GetStoneCount();
                        if (currStoneCount[0] < prevStoneCount[0] || currStoneCount[1]
                            < prevStoneCount[1]) //if one of the counts of stones is lower than in the previous move
                        {
                            state->UndoAction(-1, -1);
                            legal = false;
                            continue;
                        }
                        break;
                    }
                }
            }
            enemyMove++;
        }
    }*/
    SpielFatalError("Method ResampleFromInfostate is unfinished and shouldn't be used\n");
    //return state;
}

std::string PhantomGoState::InformationStateString(int player) const {
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, num_players_);
    return HistoryString();
}

std::string PhantomGoState::ObservationString(int player) const {
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, num_players_);
    return board_.ObservationToString(player);
}

void PhantomGoState::ObservationTensor(int player, absl::Span<float> values) const {
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, num_players_);

    int num_cells = board_.board_size() * board_.board_size();
    SPIEL_CHECK_EQ(values.size(), num_cells * (CellStates() + 1));
    std::fill(values.begin(), values.end(), 0.);

    // Add planes: black, white, empty.
    int cell = 0;
    for (VirtualPoint p: BoardPoints(board_.board_size())) {
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
    for (VirtualPoint p: BoardPoints(board_.board_size())) {
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
        case GoColor::kBlack:return 'X';
        case GoColor::kWhite:return 'O';
        case GoColor::kEmpty:return '+';
        case GoColor::kGuard:return '#';
        default:SpielFatalError(absl::StrCat("Unknown color ", c, " in GoColorToChar."));
            return '!';
    }
}

std::string PhantomGoState::ToString() const {
    std::stringstream ss;
    std::array<int, 2> stoneCount = board_.GetStoneCount();
    ss << "GoState(komi=" << komi_ << ", to_play=" << GoColorToString(to_play_)
       << ", history.size()=" << history_.size() << ", "
       << "stones_count: w" << stoneCount[1] << " b" << stoneCount[0] << ")\n";

    ss << board_;

    ss << board_.ObservationsToString();

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
    for (auto[_, action]: history_) {
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
        for (VirtualPoint p: HandicapStones(handicap_)) {
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
bool PhantomGoState::equalMetaposition(const PhantomGoState &state1, const PhantomGoState &state2, int playerID) {

    if(state1.board_.board_size() != state2.board_.board_size())
    {
        return false;
    }

    std::array<int, 2> stoneCount1 = state1.board_.GetStoneCount();
    std::array<int, 2> stoneCount2 = state2.board_.GetStoneCount();

    if(stoneCount1[0] != stoneCount2[0] || stoneCount1[1] != stoneCount2[1])
    {
        return false;
    }

    int boardSize = state1.board_.board_size();

    auto observation1 = state1.board_.GetObservationByID(playerID);
    auto observation2 = state2.board_.GetObservationByID(playerID);

    for(int i = 0; i < boardSize * boardSize; i++)
    {
        if(observation1[i] != observation2[i])
        {
            return false;
        }
    }

    if(state1.to_play_ != state2.to_play_)
    {
        return false;
    }

    return true;
}

PhantomGoGame::PhantomGoGame(const GameParameters &params)
    : Game(kGameType, params),
      komi_(ParameterValue<double>("komi")),
      board_size_(ParameterValue<int>("board_size")),
      handicap_(ParameterValue<int>("handicap")),
      max_game_length_(ParameterValue<int>(
          "max_game_length", DefaultMaxGameLength(board_size_))) {}

class PhantomGoObserver : public Observer {
 public:
  PhantomGoObserver(IIGObservationType iig_obs_type)
      : Observer(/*has_string=*/true, /*has_tensor=*/true),
        iig_obs_type_(iig_obs_type) {}

  void WriteTensor(const State &observed_state, int player,
                   Allocator *allocator) const override {
      const PhantomGoState &state =
          open_spiel::down_cast<const PhantomGoState &>(observed_state);

      const int totalBoardPoints = state.board().board_size() * state.board().board_size();

      {
          auto out = allocator->Get("stone-counts", {2});
          auto stoneCount = state.GetStoneCount();
          out.at(0) = stoneCount[0];
          out.at(1) = stoneCount[1];
      }

      if (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer) {
          {
              auto out = allocator->Get("player_observation", {totalBoardPoints});
              auto observation = state.board().GetObservationByID(player);
              for (int i = 0; i < totalBoardPoints; i++) {
                  out.at(i) = (uint8_t) observation[i];
              }
          }
      }

      if (iig_obs_type_.public_info) {

          {
              auto out = allocator->Get("history-turns", {state.History().size()});
              auto history = state.FullHistory();
              for (int i = 0; i < history.size(); i++) {
                  out.at(i) = history[i].player;
              }
          }

          {
              std::shared_ptr<const Game> game = state.GetGame();
              std::unique_ptr<PhantomGoState>
                  currState = std::make_unique<PhantomGoState>(down_cast<PhantomGoState>(*game->NewInitialState()));
              auto out = allocator->Get("history-turns", {state.History().size()});
              auto history = state.History();
              std::array<int, 2> prevStoneCount = currState->GetStoneCount();
              for (int i = 0; i < history.size(); i++) {
                  currState->ApplyAction(history[i]);
                  std::array<int, 2> currStoneCount = currState->GetStoneCount();
                  if (prevStoneCount[0] - currStoneCount[0] > 0) {
                      out.at(i) = prevStoneCount[0] - currStoneCount[0];
                  } else if (prevStoneCount[1] - currStoneCount[1] > 0) {
                      out.at(i) = prevStoneCount[1] - currStoneCount[1];
                  } else {
                      out.at(i) = 0;
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

}  // namespace phantom_go
}  // namespace open_spiel
