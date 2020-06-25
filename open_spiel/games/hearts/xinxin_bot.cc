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

#include "open_spiel/games/hearts/xinxin_bot.h"

namespace open_spiel {
namespace hearts {
namespace {

constexpr int kUCTNumRuns = 50;
constexpr int kUCTcVal = 0.4;
constexpr int kIIMCNumWorlds = 20;
constexpr Suit kXinxinSuits[] = {Suit::kSpades, Suit::kDiamonds, Suit::kClubs,
                                 Suit::kHearts};
constexpr ::hearts::tPassDir kXinxinPassDir[] = {
    ::hearts::tPassDir::kHold, ::hearts::tPassDir::kLeftDir,
    ::hearts::tPassDir::kAcrossDir, ::hearts::tPassDir::kRightDir};
}  // namespace

Action GetOpenSpielAction(::hearts::card card) {
  // xinxin keeps ranks in the reverse order of open_spiel
  int rank = kNumCardsPerSuit - ::hearts::Deck::getrank(card) - 1;
  Suit suit = kXinxinSuits[::hearts::Deck::getsuit(card)];
  return Card(suit, rank);
}

::hearts::card GetXinxinAction(Action action) {
  int rank = kNumCardsPerSuit - CardRank(action) - 1;
  int suit =
      std::find(kXinxinSuits, kXinxinSuits + kNumSuits, CardSuit(action)) -
      kXinxinSuits;
  return ::hearts::Deck::getcard(suit, rank);
}

std::unique_ptr<::hearts::SafeSimpleHeartsPlayer> XinxinBot::CreatePlayer() {
  xinxin_uct_.push_back(std::make_unique<::hearts::UCT>(kUCTNumRuns, kUCTcVal));
  xinxin_playouts_.push_back(std::make_unique<::hearts::HeartsPlayout>());
  xinxin_uct_.back()->setPlayoutModule(xinxin_playouts_.back().get());
  xinxin_mc_.push_back(std::make_unique<::hearts::iiMonteCarlo>(
      xinxin_uct_.back().get(), kIIMCNumWorlds));
  // single-threaded xinxin is currently broken
  // (https://github.com/nathansttt/hearts/issues/5)
  xinxin_mc_.back()->setUseThreads(true);
  auto player = std::make_unique<::hearts::SafeSimpleHeartsPlayer>(
      xinxin_mc_.back().get());
  player->setModelLevel(2);
  return player;
}

XinxinBot::XinxinBot(int rules, int num_players) : kNumPlayers(num_players) {
  pass_dir_ = ::hearts::tPassDir::kHold;
  num_cards_dealt_ = 0;
  game_state_ = std::make_unique<::hearts::HeartsGameState>();
  game_state_->setRules(rules);
  game_state_->deletePlayers();
  for (int i = 0; i < kNumPlayers; i++) {
    initial_deal_.push_back(std::vector<::hearts::card>());
    // The game state destructor deletes the players, so we do not manage their
    // memory in this bot.
    std::unique_ptr<::hearts::SafeSimpleHeartsPlayer> player = CreatePlayer();
    ::hearts::SafeSimpleHeartsPlayer* released_player = player.release();
    game_state_->addPlayer(released_player);
    released_player->setGameState(game_state_.get());
  }
  SPIEL_CHECK_EQ(xinxin_uct_.size(), kNumPlayers);
  SPIEL_CHECK_EQ(xinxin_mc_.size(), kNumPlayers);
  SPIEL_CHECK_EQ(xinxin_playouts_.size(), kNumPlayers);
}

void XinxinBot::Restart() {
  game_state_->Reset();
  pass_dir_ = ::hearts::tPassDir::kHold;
  num_cards_dealt_ = 0;
  for (auto& hand : initial_deal_) {
    hand.clear();
  }
}

void XinxinBot::NewDeal(std::vector<std::vector<::hearts::card>>* initial_cards,
                        ::hearts::tPassDir pass_dir, int first_player) {
  // the order in which these are called matters (e.g. setting the cards unsets
  // the pass dir)
  game_state_->Reset();
  game_state_->SetInitialCards(*initial_cards);
  game_state_->setPassDir(pass_dir);
  game_state_->setFirstPlayer(first_player);
}

Action XinxinBot::Step(const State&) {
  ::hearts::CardMove* move =
      static_cast<::hearts::CardMove*>(game_state_->getNextPlayer()->Play());
  game_state_->ApplyMove(move);
  Action act = GetOpenSpielAction(move->c);
  game_state_->freeMove(move);
  return act;
}

void XinxinBot::InformAction(const State& state, Player player_id,
                             Action action) {
  if (player_id == kChancePlayerId) {
    if (state.ChanceOutcomes().size() == 4 && num_cards_dealt_ == 0) {
      // this is guaranteed to be the pass dir selection action as long as
      // that remains as the first optional chance node in the Hearts
      // implementation
      pass_dir_ = kXinxinPassDir[action];
    } else {
      ::hearts::card card = GetXinxinAction(action);
      initial_deal_[num_cards_dealt_ % kNumPlayers].push_back(card);
      if (++num_cards_dealt_ == kNumCards) {
        NewDeal(&initial_deal_, pass_dir_, 0);
      }
    }
  } else {
    ::hearts::Move* move =
        new ::hearts::CardMove(GetXinxinAction(action), player_id);
    game_state_->ApplyMove(move);
    game_state_->freeMove(move);
  }
}

void XinxinBot::ForceAction(const State& state, Action action) {
  ::hearts::Move* move = new ::hearts::CardMove(
      GetXinxinAction(action), game_state_->getNextPlayerNum());
  game_state_->ApplyMove(move);
  game_state_->freeMove(move);
}

int XinxinBot::XinxinRules(GameParameters params) {
  int rules = 0;
  if (params["pass_cards"].bool_value()) rules |= ::hearts::kDoPassCards;
  if (params["no_pts_on_first_trick"].bool_value())
    rules |= ::hearts::kNoHeartsFirstTrick | ::hearts::kNoQueenFirstTrick;
  if (params["can_lead_any_club"].bool_value()) {
    rules |= ::hearts::kLeadClubs;
  } else {
    rules |= ::hearts::kLead2Clubs;
  }
  if (params["jd_bonus"].bool_value()) rules |= ::hearts::kJackBonus;
  if (!params["avoid_all_tricks_bonus"].bool_value())
    rules |= ::hearts::kNoShooting;
  if (params["qs_breaks_hearts"].bool_value())
    rules |= ::hearts::kQueenBreaksHearts;
  if (params["must_break_hearts"].bool_value())
    rules |= ::hearts::kMustBreakHearts;
  if (params["can_lead_hearts_instead_of_qs"].bool_value()) {
    SpielFatalError("Xinxin does not support leading hearts instead of qs");
  }
  return rules;
}

std::unique_ptr<Bot> MakeXinxinBot(GameParameters params, int num_players) {
  int rules = XinxinBot::XinxinRules(params);
  return std::make_unique<XinxinBot>(rules, num_players);
}

}  // namespace hearts
}  // namespace open_spiel
