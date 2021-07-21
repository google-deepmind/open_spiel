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

// Todo:
// Implement UndoAction

#include "open_spiel/games/parcheesi.h"

#include <algorithm>
#include <cstdlib>
#include <set>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace parcheesi {
namespace {


const std::vector<std::pair<Action, double>> kChanceOutcomes = {
    std::pair<Action, double>(0, 1.0 / 18),
    std::pair<Action, double>(1, 1.0 / 18),
    std::pair<Action, double>(2, 1.0 / 18),
    std::pair<Action, double>(3, 1.0 / 18),
    std::pair<Action, double>(4, 1.0 / 18),
    std::pair<Action, double>(5, 1.0 / 18),
    std::pair<Action, double>(6, 1.0 / 18),
    std::pair<Action, double>(7, 1.0 / 18),
    std::pair<Action, double>(8, 1.0 / 18),
    std::pair<Action, double>(9, 1.0 / 18),
    std::pair<Action, double>(10, 1.0 / 18),
    std::pair<Action, double>(11, 1.0 / 18),
    std::pair<Action, double>(12, 1.0 / 18),
    std::pair<Action, double>(13, 1.0 / 18),
    std::pair<Action, double>(14, 1.0 / 18),
    std::pair<Action, double>(15, 1.0 / 36),
    std::pair<Action, double>(16, 1.0 / 36),
    std::pair<Action, double>(17, 1.0 / 36),
    std::pair<Action, double>(18, 1.0 / 36),
    std::pair<Action, double>(19, 1.0 / 36),
    std::pair<Action, double>(20, 1.0 / 36),
};

const std::vector<std::vector<int>> kChanceOutcomeValues = {
    {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}, {2, 3}, {2, 4},
    {2, 5}, {2, 6}, {3, 4}, {3, 5}, {3, 6}, {4, 5}, {4, 6},
    {5, 6}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}};


// Facts about the game
const GameType kGameType{
    /*short_name=*/"parcheesi",
    /*long_name=*/"Parcheesi",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*min_num_players=*/4,
    /*max_num_players=*/4,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {}};

static std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new ParcheesiGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace


std::string ParcheesiState::ActionToString(Player player,
                                            Action move_id) const {
  if(player == kChancePlayerId){
    return absl::StrCat("chance outcome ", move_id,
                          " (roll: ", kChanceOutcomeValues[move_id][0],
                          kChanceOutcomeValues[move_id][1], ")");
  }else if(move_id == 0){
    return absl::StrCat("Player ", kTokens[player], " does not have any valid moves");
  }else if(move_id == 1){
    return absl::StrCat("Player ", kTokens[player], " does not have any valid bonus moves");
  }else{
    ParcheesiMove parcheesiMove = SpielMoveToParcheesiMove(move_id);
    return absl::StrCat("Player ", kTokens[player], " moved token ", parcheesiMove.token_index, " from ", 
      GetHumanReadablePosForPlayer(parcheesiMove.old_pos, player), " to ", 
      GetHumanReadablePosForPlayer(parcheesiMove.new_pos, player));    
  }
}

std::string ParcheesiState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void ParcheesiState::ObservationTensor(Player player,
                                        absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  SPIEL_CHECK_EQ(values.size(), kStateEncodingSize);
  auto value_it = values.begin();
}

ParcheesiState::ParcheesiState(std::shared_ptr<const Game> game)
    : State(game),
      cur_player_(kChancePlayerId),
      prev_player_(kChancePlayerId),
      turns_(0),
      dice_({}),
      extra_turn_(0),
      bonus_move_(0),
      illegal_move_((ParcheesiMove(-1, -1, -1, -1, false))),
      home_(std::vector<std::vector<std::string>>(kNumPlayers, std::vector<std::string>())),
      base_(std::vector<std::vector<std::string>>(kNumPlayers, std::vector<std::string>())),
      token_pos_(std::vector<std::vector<int>>(kNumPlayers, std::vector<int>(4, -1))),
      board_(std::vector<std::vector<std::string>>(kNumBoardTiles, std::vector<std::string>())){
  SetupInitialBoard();
}

void ParcheesiState::SetupInitialBoard() {
  
  // Start with all tokens in base
  // The 4 players in order are denoted as: r(red), g(green), y(yellow), b(blue)
  // Tokens for each players are
  //  r(red)    : r0, r1, r2, r3
  //  g(green)  : g0, g1, g2, g3
  //  y(yellow) : y0, y1, y2, y3
  //  b(blue)   : b0, b1, b2, b3

  for(int i = 0; i < kNumPlayers; i++){
    for(int j = 0; j < kNumTokens; j++){
      base_[i].push_back(kTokens[i] + std::to_string(j));
    }
  }
}

Player ParcheesiState::CurrentPlayer() const {
  return IsTerminal() ? kTerminalPlayerId : Player{cur_player_};
}

void ParcheesiState::RollDice(int outcome) {
  dice_.push_back(kChanceOutcomeValues[outcome][0]);
  dice_.push_back(kChanceOutcomeValues[outcome][1]);
}

int ParcheesiState::GetPlayerFromToken(std::string token) const {
  for(int i = 0; i < kTokens.size(); i++){
    if(kTokens[i].at(0) == token.at(0)){
      return i;
    }
  }
  return -1;
}

ParcheesiMove ParcheesiState::SpielMoveToParcheesiMove(Action move) const {
  bool breaking_block = move % kParcheesiMoveBreakingBlockMax;
  move /= kParcheesiMoveBreakingBlockMax;
  int token_index = (move % kParcheesiMoveTokenIndexMax) - 1;
  move /= kParcheesiMoveTokenIndexMax;
  int new_pos = (move % kParcheesiMovePosMax) - 1;
  move /= kParcheesiMovePosMax;
  int old_pos = (move % kParcheesiMovePosMax) - 1;
  move /= kParcheesiMovePosMax;
  int die_index = move;
  ParcheesiMove parcheesiMove = ParcheesiMove(die_index, old_pos, new_pos, token_index, breaking_block);
  return parcheesiMove;
}

std::vector<Action> ParcheesiState::MultipleParcheesiMoveToSpielMove(std::vector<ParcheesiMove> parcheesiMoves) const {
  std::vector<Action> moves = {};
  for(int i = 0; i < parcheesiMoves.size(); i++){
    moves.push_back(ParcheesiMoveToSpielMove(parcheesiMoves[i]));
  }  
  return moves;
}

Action ParcheesiState::ParcheesiMoveToSpielMove(ParcheesiMove parcheesiMove) const {
  Action move = parcheesiMove.die_index;
  move *= kParcheesiMovePosMax;
  move += (parcheesiMove.old_pos + 1);
  move *= kParcheesiMovePosMax;
  move += (parcheesiMove.new_pos + 1);
  move *= kParcheesiMoveTokenIndexMax;
  move += (parcheesiMove.token_index + 1);
  move *= kParcheesiMoveBreakingBlockMax;
  move += parcheesiMove.breaking_block;
  return move;
}

// If a player rolls 3 consecutive doubles, their turn ends and the farthest token they have that
// is not inside ladder/home is removed from the board
void ParcheesiState::PenalisePlayer(Player player){
  std::vector<int> player_pos = token_pos_[player];
  sort(player_pos.begin(), player_pos.end(), std::greater<int>());
  for(int i = 0; i < player_pos.size(); i++){
    int pos = player_pos[i];
    if(pos < kNumBoardPos && pos >= 0){
      int grid_pos = GetGridPosForPlayer(pos, player);
      std::string token = board_[grid_pos].back();
      board_[grid_pos].pop_back();
      token_pos_[player][token.at(1) - '0'] = -1;      
      base_[player].push_back(token);      
      break;
    }      
  }
}


void ParcheesiState::DoApplyAction(Action move) {
  if (IsChanceNode()) {
    SPIEL_CHECK_TRUE(dice_.empty());
    RollDice(move);

    // Move on to the next player unless extra_turn granted due to doubles
    if(extra_turn_ == 0)
      cur_player_ = NextPlayerRoundRobin(prev_player_, num_players_);
    else
      cur_player_ = prev_player_;

    // Doubles rolled! player gets an extra turn
    if(dice_[0] == dice_[1]){
      extra_turn_++;
      if(extra_turn_ == 3){
        extra_turn_ = 0;
        PenalisePlayer(cur_player_);
        NextTurn();
      }
    }else
      extra_turn_ = 0;
    return;
  }

  // Reset bonus move here. Bonus moves are granted if a player hits an opponents token (20 moves)
  // or gets a token into home (10 moves). When bonus moves are granted, it's mandatory to play
  // them first before any die moves that the player may have left
  bonus_move_ = 0;

  // Execute a playable move
  if(move > kActionNoBonusMoves){
    // It is illegal to move a block (blocks are formed when two tokens of the same player occupies
    // the same position on the board) using a double. So if the player used one die value in a double
    // dice roll to move one token of a block, then he cannot also move the other one unless it's the
    // only move possible. We track that move with illegal_move
    illegal_move_ = ParcheesiMove(-1, -1, -1, -1, false);
    ParcheesiMove parcheesiMove = SpielMoveToParcheesiMove(move);    
    if(parcheesiMove.breaking_block)
      illegal_move_ = parcheesiMove;

    std::string token = kTokens[cur_player_] + std::to_string(parcheesiMove.token_index);       
    
    // Remove the token from it's old location. Either in base_ or board_
    if(parcheesiMove.old_pos == -1){
      token = base_[cur_player_].back();
      base_[cur_player_].pop_back();       
    }else if(parcheesiMove.old_pos < kNumBoardPos){      
      int grid_pos = GetGridPosForPlayer(parcheesiMove.old_pos, cur_player_);
      for(int i = 0; i < board_[grid_pos].size(); i++){
        if(board_[grid_pos][i] == token){
          board_[grid_pos].erase(board_[grid_pos].begin() + i);
          break;
        }
      }
    }

    // Place the token to it's new location. Remove any opponent tokens that were hit.
    // Award bonus moves as applicable.
    if(parcheesiMove.new_pos < kNumBoardPos){
      int grid_pos = GetGridPosForPlayer(parcheesiMove.new_pos, cur_player_);
      if(board_[grid_pos].size() == 1 && board_[grid_pos][0].at(0) != kTokens[cur_player_].at(0)){
        std::string killed_token = board_[grid_pos].back();
        board_[grid_pos].pop_back();
        int snubbed_player = GetPlayerFromToken(killed_token);
        base_[snubbed_player].push_back(killed_token);
        token_pos_[snubbed_player][killed_token.at(1) - '0'] = -1;
        bonus_move_ = kBonusMovesForHittingOpponentToken;
      }      
      board_[grid_pos].push_back(token);
    }else if(parcheesiMove.new_pos == kHomePos){
      home_[cur_player_].push_back(token);
      bonus_move_ = kBonusMovesForGettingTokenHome;
    } 

    // Update token_pos_   
    token_pos_[cur_player_][token.at(1) - '0'] = parcheesiMove.new_pos;

    // If the move is not a bonus move, removed used die from dice_
    if(parcheesiMove.new_pos - parcheesiMove.old_pos < kBonusMovesForGettingTokenHome){
      if(parcheesiMove.die_index == 2)
        dice_.clear();
      else
        dice_.erase(dice_.begin() + parcheesiMove.die_index);
    }    
  }  

  // End player's turn when:
  // 1. No valid moves available for player
  // 2. No bonus moves available and dice are used up
  // 3. Bonus moves are not playable and dice are used up
  if(move == kActionNoValidMoves || (bonus_move_ == 0 && dice_.size() == 0) || (move == kActionNoBonusMoves && dice_.size() == 0)){
    NextTurn();
  }  
}

void ParcheesiState::NextTurn(){
  prev_player_ = cur_player_;
  cur_player_ = kChancePlayerId;
  dice_.clear();
  turns_++;
}

std::string ParcheesiState::GetHumanReadablePosForPlayer(int pos, int player) const {
  if(pos == -1)
    return "base";
  else if(pos < kNumBoardPos)
    return std::to_string(GetGridPosForPlayer(pos, player) + 1);
  else if(pos < kHomePos)
    return "ladder pos " + std::to_string(pos - kNumBoardPos + 1);
  else
    return "home";
}
    

void ParcheesiState::UndoAction(int player, Action action) {
  history_.pop_back();
  --move_number_;
}


std::vector<Action> ParcheesiState::LegalActions() const {
  if (IsChanceNode()) return LegalChanceOutcomes();
  if (IsTerminal()) return {};

  std::vector<ParcheesiMove> moves;
  // If bonus moves are granted, mandatory to use them first.
  if(bonus_move_ != 0)
    moves = GetParcheesiMoves(cur_player_, {bonus_move_});
  else{
    moves = GetParcheesiMoves(cur_player_, dice_);
    // Remove the move that'll allow to move block if other moves are available.
    if(illegal_move_.die_index != -1 && moves.size() > 1){
      for(int i = 0; i < moves.size(); i++){
        if(moves[i].old_pos == illegal_move_.old_pos && moves[i].new_pos == illegal_move_.new_pos){
          moves.erase(moves.begin() + i);
          break;
        }
      }
    }
  }

  std::vector<Action> spielmoves = MultipleParcheesiMoveToSpielMove(moves);

  // kActionNoValidMoves: No valid move is possible now for the player and turn should end now
  // kActionNoBonusMoves: No moves possible with bonus_moves, althought turn might not be over if unused die is left
  if(spielmoves.size() == 0){
    if(bonus_move_ != 0)
      spielmoves.push_back(kActionNoBonusMoves);
    else
      spielmoves.push_back(kActionNoValidMoves);
  }

  return spielmoves;
}

std::vector<ParcheesiMove> ParcheesiState::GetParcheesiMoves(int player, std::vector<int> dice) const {
  std::vector<ParcheesiMove> moves;

  // If rolled doubles, moving blocks is mandatory
  if(dice.size() == 2 && dice[0] == dice[1]){
    std::vector<int> block_pos;
    for(int i = 0; i < token_pos_[player].size(); i++){
      int pos = token_pos_[player][i];
      if(pos >= 0 && pos < kNumBoardPos && count(token_pos_[player].begin(), token_pos_[player].end(), pos) == 2)
        block_pos.push_back(pos);
    }
    if(block_pos.size() > 0){
      int block_to_move = *max_element(block_pos.begin(), block_pos.end());
      //modified so that only block tokens are considered for moves
      std::vector<int> modified_token_pos = token_pos_[player];
      for(int i = 0; i < modified_token_pos.size(); i++){
        if(modified_token_pos[i] != block_to_move)
          replace(modified_token_pos.begin(), modified_token_pos.end(), modified_token_pos[i], -1);
      }
      moves = GetGridMoves(modified_token_pos, player, dice, true);
      if(moves.size() > 0){
        return moves;                
      }
    }        
  }

  // If player gets a 5, mandatory to move token from base to board if applicable
  if(board_[kStartPos[player]].size() < 2){
    if(base_[player].size() > 0){
      if(dice[0] == 5){
        moves.push_back(ParcheesiMove(0, -1, 0, -1, false));
        return moves;
      }
      if(dice.size() > 1){
        if(dice[1] == 5){
          moves.push_back(ParcheesiMove(1, -1, 0, -1, false));
          return moves;
        }
        if(dice[0] + dice[1] == 5){
          moves.push_back(ParcheesiMove(2, -1, 0, -1, false));
          return moves;
        }
      }
    }
  }

  if(base_[player].size() + home_[player].size() < kNumTokens)
    return GetGridMoves(token_pos_[player], player, dice, false);

  return {};
}

std::vector<ParcheesiMove> ParcheesiState::GetGridMoves(std::vector<int> player_token_pos, int player, std::vector<int> dice, bool breaking_block) const {
  std::vector<ParcheesiMove> moves;
  for(int i = 0; i < player_token_pos.size(); i++){
    int old_pos = player_token_pos[i];
    if(old_pos >= 0){
      for(int j = 0; j < dice.size(); j++){
        int new_pos = old_pos + dice[j];
        if(new_pos <= kHomePos && !DestinationOccupiedBySafeToken(new_pos, player) && !BlocksInRoute(old_pos, new_pos, player)){
          moves.push_back(ParcheesiMove(j, old_pos, new_pos, i, breaking_block));
        }
      }
    }
  }
  return moves;
}

int ParcheesiState::GetGridPosForPlayer(int pos, int player) const {
  return (pos + kStartPos[player]) % kNumBoardTiles;
}    

bool ParcheesiState::BlocksInRoute(int start, int end, int player) const {
  if(start >= kNumBoardPos)
      return false;
  bool block_found = false;
  for(int i = start + 1; i < std::min(end + 1, kNumBoardPos); i++){
    if(board_[GetGridPosForPlayer(i, player)].size() == 2){
      block_found = true;
      break;
    }      
  } 
  return block_found;
}    

bool ParcheesiState::DestinationOccupiedBySafeToken(int destination,int player) const {
  int grid_pos = GetGridPosForPlayer(destination, player);
  if(board_[grid_pos].size() == 1 && std::count(kSafePos.begin(), kSafePos.end(), grid_pos)){
    std::string token = board_[grid_pos][0];
    if(token.at(0) != kTokens[player].at(0))
      return true;
  }    
  return false;
}

std::vector<std::pair<Action, double>> ParcheesiState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  return kChanceOutcomes;
}

std::string ParcheesiState::ToString() const {
  std::string board_str = "";
  for(int i = 0; i < kNumPlayers; i++){
    absl::StrAppend(&board_str, "\nPlayer ");
    absl::StrAppend(&board_str, kTokens[i]);
    absl::StrAppend(&board_str, ": ");
    for(int j = 0; j < kNumTokens; j++){
      absl::StrAppend(&board_str, token_pos_[i][j]);
      absl::StrAppend(&board_str, " ");
    }
  }

  absl::StrAppend(&board_str, "\nTurn ");
  absl::StrAppend(&board_str, turns_);
  absl::StrAppend(&board_str, "\n");
  
  return board_str;
}

bool ParcheesiState::IsTerminal() const {
  for(int i = 0; i < 4; i++)
    if(home_[i].size() == 4)
      return true;
  return false;
}

std::vector<double> ParcheesiState::Returns() const {
  std::vector<double> returns(kNumPlayers);
  for(int i = 0; i < 4; i++)
    if(home_[i].size() == 4)
      returns[i] = 1.0;
  return returns;
}

std::unique_ptr<State> ParcheesiState::Clone() const {
  return std::unique_ptr<State>(new ParcheesiState(*this));
}

ParcheesiGame::ParcheesiGame(const GameParameters& params)
    : Game(kGameType, params) {}

double ParcheesiGame::MaxUtility() const {
  return 1;
}


}  // namespace parcheesi
}  // namespace open_spiel
