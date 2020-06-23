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
constexpr Suit kXinxinSuits[] = { Suit::kSpades, Suit::kDiamonds, Suit::kClubs,
  Suit::kHearts };
constexpr ::hearts::tPassDir kXinxinPassDir[] = { ::hearts::tPassDir::kHold,
  ::hearts::tPassDir::kLeftDir, ::hearts::tPassDir::kAcrossDir,
  ::hearts::tPassDir::kRightDir};

::hearts::SafeSimpleHeartsPlayer* CreatePlayer() {
  ::hearts::UCT *uct = new ::hearts::UCT(kUCTNumRuns, kUCTcVal);
  uct->setPlayoutModule(new ::hearts::HeartsPlayout());
  ::hearts::iiMonteCarlo *mc = new ::hearts::iiMonteCarlo(uct, kIIMCNumWorlds);
  // single-threaded xinxin is currently broken
  // (https://github.com/nathansttt/hearts/issues/5)
  mc->setUseThreads(true);
  ::hearts::SafeSimpleHeartsPlayer *p;
  p = new ::hearts::SafeSimpleHeartsPlayer(mc);
  p->setModelLevel(2);
  return p;
}

int GetXinxinRules(bool pass_cards, bool no_pts_on_first_trick,
                   bool can_lead_any_club, bool jd_bonus,
                   bool avoid_all_tricks_bonus, bool qs_breaks_hearts,
                   bool must_break_hearts, bool can_lead_hearts_instead_of_qs) {
  int rules = 0;
  if (pass_cards) rules |= ::hearts::kDoPassCards;
  if (no_pts_on_first_trick)
    rules |= ::hearts::kNoHeartsFirstTrick | ::hearts::kNoQueenFirstTrick;
  if (can_lead_any_club) {
    rules |= ::hearts::kLeadClubs;
  } else {
    rules |= ::hearts::kLead2Clubs;
  }
  if (jd_bonus) rules |= ::hearts::kJackBonus;
  if (!avoid_all_tricks_bonus) rules |= ::hearts::kNoShooting;
  if (qs_breaks_hearts) rules |= ::hearts::kQueenBreaksHearts;
  if (must_break_hearts) rules |= ::hearts::kMustBreakHearts;
  if (can_lead_hearts_instead_of_qs) {
    SpielFatalError("Xinxin does not support leading hearts instead of qs");
  }
  return rules;
}
}  // namespace

Action GetOpenSpielAction(::hearts::card card) {
  // xinxin keeps ranks in the reverse order of open_spiel
  int rank  = kNumCardsPerSuit - ::hearts::Deck::getrank(card) - 1;
  Suit suit = kXinxinSuits[::hearts::Deck::getsuit(card)];
  return Card(suit, rank);
}

::hearts::card GetXinxinAction(Action action) {
  int rank = kNumCardsPerSuit - CardRank(action) - 1;
  int suit = std::find(kXinxinSuits, kXinxinSuits + kNumSuits,
      CardSuit(action)) - kXinxinSuits;
  return ::hearts::Deck::getcard(suit, rank);
}


XinxinBot::XinxinBot(int rules, int num_players) :
  kNumPlayers(num_players) {
  pass_dir_ = ::hearts::tPassDir::kHold;
  num_cards_dealt_ = 0;
  game_state_ = new ::hearts::HeartsGameState();
  game_state_->setRules(rules);
  game_state_->deletePlayers();
  for (int i = 0; i < kNumPlayers; i++) {
    initial_deal_.push_back(std::vector<::hearts::card>());
    ::hearts::SafeSimpleHeartsPlayer *p = CreatePlayer();
    game_state_->addPlayer(p);
    p->setGameState(game_state_);
  }
}

XinxinBot::~XinxinBot() {
  for (int i = 0; i < kNumPlayers; i++) {
    delete game_state_->getPlayer(i)->getAlgorithm();
  }
  game_state_->deletePlayers();
  delete game_state_;
}

void XinxinBot::Restart() {
  game_state_->Reset();
  pass_dir_ = ::hearts::tPassDir::kHold;
  num_cards_dealt_ = 0;
  for (auto& hand : initial_deal_) {
    hand.clear();
  }
}

void XinxinBot::NewDeal(std::vector<std::vector<::hearts::card>> *initial_cards,
                        ::hearts::tPassDir pass_dir, int first_player) {
  // the order in which these are called matters (e.g. setting the cards unsets
  // the pass dir)
  game_state_->Reset();
  game_state_->SetInitialCards(*initial_cards);
  game_state_->setPassDir(pass_dir);
  game_state_->setFirstPlayer(first_player);
}

Action XinxinBot::Step(const State&) {
  ::hearts::CardMove *move = static_cast<::hearts::CardMove*>(
      game_state_->getNextPlayer()->Play());
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
    ::hearts::Move* move = new ::hearts::CardMove(GetXinxinAction(action),
        player_id);
    game_state_->ApplyMove(move);
    game_state_->freeMove(move);
  }
}

void XinxinBot::ForceAction(const State& state, Action action) {
    ::hearts::Move* move = new ::hearts::CardMove(GetXinxinAction(action),
        game_state_->getNextPlayerNum());
    game_state_->ApplyMove(move);
    game_state_->freeMove(move);
}

int XinxinBot::XinxinRules(GameParameters params) {
  return GetXinxinRules(params["pass_cards"].bool_value(),
                        params["no_pts_on_first_trick"].bool_value(),
                        params["can_lead_any_club"].bool_value(),
                        params["jd_bonus"].bool_value(),
                        params["avoid_all_tricks_bonus"].bool_value(),
                        params["qs_breaks_hearts"].bool_value(),
                        params["must_break_hearts"].bool_value(),
                        params["can_lead_hearts_instead_of_qs"].bool_value());
}

std::unique_ptr<Bot> MakeXinxinBot(GameParameters params, int num_players) {
  int rules = XinxinBot::XinxinRules(params);
  return std::make_unique<XinxinBot>(rules, num_players);
}

}  // namespace hearts
}  // namespace open_spiel
