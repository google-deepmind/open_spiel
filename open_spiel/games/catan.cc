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

#include "open_spiel/games/catan.h"

#include <utility>
#include <random>

#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace catan {
namespace {

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"catan",
    /*long_name=*/"The Settlers of Catan",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kRewards,
    /*max_num_players=*/4,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {
      {"players", GameParameter(4, false)},
      {"max_turns", GameParameter(72, false)},
      {"random_start_player", GameParameter(false, false)},
      {"example_board", GameParameter(true, false)},
      {"example_starting_positions", GameParameter(true, false)},
      {"harbor_support", GameParameter(true, false)},
      {"robber_support", GameParameter(true, false)},
      {"development_card_support", GameParameter(true, false)},
    }
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new CatanGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

CatanGame::CatanGame(const GameParameters& params) : Game(kGameType, params){
  this->num_players_ = ParameterValue<int>("players");
}

std::unique_ptr<State> CatanGame::NewInitialState() const {
   return std::unique_ptr<State>(new CatanState(shared_from_this(),
      ParameterValue<int>("players"),
      ParameterValue<int>("max_turns"),
      ParameterValue<bool>("random_start_player"),
      ParameterValue<bool>("example_board"),
      ParameterValue<bool>("example_starting_positions"),
      ParameterValue<bool>("harbor_support"),
      ParameterValue<bool>("robber_support"),
      ParameterValue<bool>("development_card_support")));
}

std::shared_ptr<const Game> CatanGame::Clone() const {
  return std::shared_ptr<Game>(new CatanGame(GetParameters()));
}

CatanState::CatanState(std::shared_ptr<const Game> game,
  int players,
  int max_turns,
  bool random_start_player,
  bool example_board,
  bool example_starting_positions,
  bool harbor_support,
  bool robber_support,
  bool development_card_support):
  State(game), catan_env_(players, example_board, example_starting_positions, harbor_support, robber_support, development_card_support),
  prev_state_score_(players, 0), enc_(&catan_env_) {

  this->max_turns = max_turns;
  if (random_start_player) {
    if (example_starting_positions) {
      this->previous_player_ = this->RandomPlayer();
    }
    else {
      this->current_player_ = this->RandomPlayer();
    }
  }

  if (example_starting_positions) {
    for (int i = 0; i < players; i++) {
      this->prev_state_score_[i] = 2.0; // the first two settlements are already placed
    }
    this->current_player_ = kChancePlayerId; // we start with a dice roll
  }
}

Player CatanState::RandomPlayer() const {
  std::random_device rd;
  std::mt19937 eng(rd());
  std::uniform_int_distribution<> distr(0, this->catan_env_.players.size()-1);

  return distr(eng);
}

std::vector<Action> CatanState::LegalActions() const {
  if (this->IsTerminal()){
    return {};
  } else if (IsChanceNode()) {
    ActionsAndProbs outcomes_and_probs = this->ChanceOutcomes();
    const int n = outcomes_and_probs.size();
    std::vector<Action> chance_outcomes;
    chance_outcomes.reserve(n);
    for (int i = 0; i < n; i++) {
      chance_outcomes.emplace_back(outcomes_and_probs.at(i).first);
    }
    return chance_outcomes;
  } else {
    return this->catan_env_.get_legal_actions(this->CurrentPlayer());
  }
}

std::string CatanState::ActionToString(Player player,Action action_id) const {
  return this->catan_env_.action_to_string(player, action_id);
}

std::vector<double> CatanState::Rewards() const {
  std::vector<double> rewards;
  for (int i = 0; i < this->catan_env_.players.size(); i++){
    rewards.emplace_back(this->catan_env_.players[i].get_VP(true) - this->prev_state_score_[i]);
  }
  return rewards;
}

std::vector<double> CatanState::Returns() const {
  std::vector<double> returns;
  for (int i = 0; i < this->catan_env_.players.size(); i++){
    if (this->setup_phase_counter_ == 0) { //if there we use the example_starting_positions, so there was no setup phase
      returns.emplace_back(this->catan_env_.players[i].get_VP(true) - 2); // the players start with two settlements and therefore two victory points
    }
    else {
      returns.emplace_back(this->catan_env_.players[i].get_VP(true));
    }
  }
  return returns;
}

void CatanState::DoApplyAction(Action move) {
  if (this->turns_ >= this->max_turns) {
    this->catan_env_.has_ended = true; // end the game if the max_turns are reached
  }
  if (move == 0) {
    this->turns_++; // a player ends his turn
  }
  if (!IsChanceNode()){
    for (int i = 0; i < this->catan_env_.players.size(); i++){
      this->prev_state_score_[i] = this->catan_env_.players[i].get_VP(true);
    }
  }

  // if we are resolving a road building card decrease the roads to build value
  if (this->catan_env_.resolve_road_building) {
    this->catan_env_.roads_to_build--;
  }

  // apply the action
  this->catan_env_.apply_action(this->current_player_, move);

  // if we are in the setup phase
  if (this->catan_env_.is_starting) {
    // check if a settlement was placed and if yes switch to road placement
    if (this->catan_env_.is_starting_settlement_placement) {
      this->catan_env_.is_starting_settlement_placement = false;
      this->catan_env_.is_starting_road_placement = true;
    }
    // check if a road was placed
    else if (this->catan_env_.is_starting_road_placement) {
      // switch to settlement placement
      this->catan_env_.is_starting_road_placement = false;
      this->catan_env_.is_starting_settlement_placement = true;
      // check if players are placing their second road
      if (this->catan_env_.is_starting_round_two) {
        //check if everyone placed their second road and settlement if yes end the setup phase
        if (this->setup_phase_counter_ == 4 * this->catan_env_.players.size() - 1) {
          this->catan_env_.is_starting = false;
          this->catan_env_.is_starting_round_two = false;
          this->catan_env_.is_starting_road_placement = false;
          this->catan_env_.is_starting_settlement_placement = false;
          this->previous_player_ = PreviousPlayerRoundRobin(this->current_player_, this->catan_env_.players.size());
          this->current_player_ = kChancePlayerId;
          return;
        }
        // we are in the second round and go select the next player backwards
        this->current_player_ = PreviousPlayerRoundRobin(this->current_player_, this->catan_env_.players.size());
      }
      // check if the last player placed his first road if yes switch to second round
      if (this->setup_phase_counter_ == 2 * this->catan_env_.players.size() - 1) {
        this->catan_env_.is_starting_round_two = true;
      }
      // if we are not in the second round switch to the next player
      else if (!this->catan_env_.is_starting_round_two){
        this->current_player_ = NextPlayerRoundRobin(this->current_player_, this->catan_env_.players.size());
      }
    }
    // increment the counter so we can determine when the setup phase should end
    this->setup_phase_counter_++;
    return;
  }

  // if we are resolving road building unset the boolean if all roads were build otherwise return, to start next road building action
  if (this->catan_env_.resolve_road_building) {
    if (this->catan_env_.roads_to_build == 0) {
      this->catan_env_.resolve_road_building = false;
    }
    return;
  }

  // if we are currently resolving a seven discard we count down the discards
  if(this->to_discard_counter_ > 0) {
    this->to_discard_counter_--;
    return;
  }


  // if we are resolving a seven everyone with more than 7 cards needs to discard half his cards (rounded down)
  if(this->catan_env_.resolving_seven) {
    this->catan_env_.resolving_seven = false;
    for (unsigned int i = 0; i < this->catan_env_.players.size(); i++){
      int handsize = this->catan_env_.players.at(i).cards.size();
      if (handsize >= 8) {
        this->catan_env_.resolving_seven = true;
        this->to_discard_counter_ = (int)std::floor(handsize/2) -1;
        this->current_player_ = i;
        return;
      }
    }
    // resolving seven is over, next player's turn starts
    this->current_player_ = NextPlayerRoundRobin(this->previous_player_, this->catan_env_.players.size());
    return;
  }

  // card stealing with robber needs to be resolved, this is a chance node
  if (this->catan_env_.resolve_stealing) {
    this->previous_player_ = this->current_player_;
    this->current_player_ = kChancePlayerId;
    return;
  }

  // after chance node stealing a card, back to the player who's turn it is
  if (333 <= move && move <= 338) {
    this->current_player_ = this->previous_player_;
    return;
  }

  // if the player buys a development card we go to a chance node
  if (move == 339) {
    this->catan_env_.current_player = this->current_player_;
    this->previous_player_ = this->current_player_;
    this->current_player_ = kChancePlayerId;
    this->buying_a_development_card = true;
    return;
  }

  // after chance node drawing a development card, back to the player who's turn it is
  if (363 <= move && move <= 367) {
    this->buying_a_development_card = false;
    this->current_player_ = this->previous_player_;
    return;
  }

  // if we rolled the dice the next player's turn starts
  if (IsChanceNode()) {
    this->current_player_ = NextPlayerRoundRobin(this->previous_player_, this->catan_env_.players.size());
  }
  if (move == 0) {
    this->previous_player_ = this->current_player_;
    this->current_player_ = kChancePlayerId;
  }
}

std::string CatanState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, this->catan_env_.players.size());

  return this->ToString();
}

void CatanState::ObservationTensor(Player player, std::vector<double>* values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, this->catan_env_.players.size());

  auto obs = this->enc_.Encode(player);
  values->resize(obs.size());
  for (int i = 0; i < obs.size(); ++i) values->at(i) = obs[i];
}

std::unique_ptr<State> CatanState::Clone() const {
  return std::unique_ptr<State>(new CatanState(*this));
}

ActionsAndProbs CatanState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());

  if (this->catan_env_.resolve_stealing) {
    return catan_env_.random_card_actions_and_probs_from_victim();
  }
  else if (this->buying_a_development_card) {
    return catan_env_.dev_cards_actions_and_probs();
  }
  else {
    return this->catan_env_.dice_roll_actions_and_probs();
  }
}

std::string CatanState::ToString() const {
  return this->catan_env_.to_string();
}

bool CatanState::IsTerminal() const {
  return this->catan_env_.has_ended;
}

}  // namespace catan
}  // namespace open_spiel
