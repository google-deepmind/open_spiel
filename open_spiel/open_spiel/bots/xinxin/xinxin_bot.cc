// Copyright 2021 DeepMind Technologies Limited
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

#include "open_spiel/bots/xinxin/xinxin_bot.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace hearts {
namespace {

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
  xinxin_uct_.push_back(
      std::make_unique<::hearts::UCT>(uct_num_runs_, uct_c_val_));
  xinxin_playouts_.push_back(std::make_unique<::hearts::HeartsPlayout>());
  xinxin_uct_.back()->setPlayoutModule(xinxin_playouts_.back().get());
  xinxin_mc_.push_back(std::make_unique<::hearts::iiMonteCarlo>(
      xinxin_uct_.back().get(), iimc_num_worlds_));
  xinxin_mc_.back()->setUseThreads(use_threads_);
  auto player = std::make_unique<::hearts::SafeSimpleHeartsPlayer>(
      xinxin_mc_.back().get());
  player->setModelLevel(2);
  return player;
}

XinxinBot::XinxinBot(int rules, int uct_num_runs, double uct_c_val,
                     int iimc_num_worlds, bool use_threads)
    : uct_num_runs_(uct_num_runs),
      uct_c_val_(uct_c_val),
      iimc_num_worlds_(iimc_num_worlds),
      use_threads_(use_threads),
      initial_state_(nullptr) {
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

void XinxinBot::RestartAt(const State& state) {
  if (initial_state_ == nullptr) {
    initial_state_ = state.GetGame()->NewInitialState();
  }

  // TODO(author5): define a default operator== in State.
  if (state.ToString() != initial_state_->ToString()) {
    SpielFatalError("XinxinBot::RestartAt only supports restarts from the "
                    "initial state.");
  }

  Restart();
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

void XinxinBot::LogStateMismatchError(const State& state, std::string msg) {
  std::cout << "Begin error message: " << std::endl;
  std::cout << "xinxin game state: " << std::endl;
  game_state_->Print();
  std::cout << "xinxin legal moves: " << std::endl;
  ::hearts::Move* all_moves = game_state_->getAllMoves();
  if (all_moves != nullptr) all_moves->Print(1);
  std::cout << "xinxin points (N E S W): " << std::endl;
  for (Player p = 0; p < game_state_->getNumPlayers(); p++)
    std::cout << game_state_->score(p) << " ";
  std::cout << std::endl;
  std::cout << "OpenSpiel game state: " << std::endl;
  std::cout << state.ToString() << std::endl;
  std::cout << "OpenSpiel legal actions: " << std::endl;
  std::cout << state.LegalActions() << std::endl;
  std::cout << "OpenSpiel history: " << std::endl;
  std::cout << state.History() << std::endl;
  SpielFatalError(msg);
}

Action XinxinBot::Step(const State& state) {
  // check that xinxin and open_spiel agree on legal actions
  ::hearts::Move* all_moves = game_state_->getAllMoves();
  std::vector<Action> xinxin_actions;
  while (all_moves != nullptr) {
    ::hearts::card card = static_cast<::hearts::CardMove*>(all_moves)->c;
    xinxin_actions.push_back(GetOpenSpielAction(card));
    all_moves = all_moves->next;
  }
  absl::c_sort(xinxin_actions);
  std::vector<Action> legal_actions = state.LegalActions();
  if (legal_actions != xinxin_actions) {
    LogStateMismatchError(state,
                          "xinxin legal actions != OpenSpiel legal actions.");
  }
  // test passed!
  ::hearts::CardMove* move =
      static_cast<::hearts::CardMove*>(game_state_->getNextPlayer()->Play());
  game_state_->ApplyMove(move);
  Action act = GetOpenSpielAction(move->c);
  game_state_->freeMove(move);
  SPIEL_CHECK_TRUE(absl::c_binary_search(legal_actions, act));
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
    if (state.IsTerminal()) {
      if (!game_state_->Done()) {
        LogStateMismatchError(state, "xinxin state is not terminal.");
      }
      std::vector<double> returns = state.Returns();
      for (Player p = 0; p < returns.size(); p++) {
        // returns in open_spiel hearts are transformed from the score
        // to reflect that getting the least number of total points is better
        if (returns[p] != kTotalPositivePoints - game_state_->score(p)) {
          LogStateMismatchError(state, "xinxin score != OpenSpiel score");
        }
      }
    } else {
      ::hearts::Move* move =
          new ::hearts::CardMove(GetXinxinAction(action), player_id);
      game_state_->ApplyMove(move);
      game_state_->freeMove(move);
    }
  }
}

void XinxinBot::ForceAction(const State& state, Action action) {
  ::hearts::Move* move = new ::hearts::CardMove(
      GetXinxinAction(action), game_state_->getNextPlayerNum());
  game_state_->ApplyMove(move);
  game_state_->freeMove(move);
}

int XinxinBot::XinxinRules(GameParameters params) {
  int rules = ::hearts::kQueenPenalty;
  if (params["pass_cards"].bool_value()) rules |= ::hearts::kDoPassCards;
  if (params["no_pts_on_first_trick"].bool_value())
    rules |= ::hearts::kNoHeartsFirstTrick | ::hearts::kNoQueenFirstTrick;
  if (params["can_lead_any_club"].bool_value()) {
    rules |= ::hearts::kLeadClubs;
  } else {
    rules |= ::hearts::kLead2Clubs;
  }
  if (params["jd_bonus"].bool_value()) rules |= ::hearts::kJackBonus;
  if (params["avoid_all_tricks_bonus"].bool_value())
    rules |= ::hearts::kNoTrickBonus;
  if (params["qs_breaks_hearts"].bool_value())
    rules |= ::hearts::kQueenBreaksHearts;
  if (params["must_break_hearts"].bool_value())
    rules |= ::hearts::kMustBreakHearts;
  if (params["can_lead_hearts_instead_of_qs"].bool_value()) {
    SpielFatalError("Xinxin does not support leading hearts instead of qs");
  }
  return rules;
}

std::unique_ptr<Bot> MakeXinxinBot(GameParameters params, int uct_num_runs,
                                   double uct_c_val, int iimc_num_worlds,
                                   bool use_threads) {
  int rules = XinxinBot::XinxinRules(params);
  return std::make_unique<XinxinBot>(rules, uct_num_runs, uct_c_val,
                                     iimc_num_worlds, use_threads);
}

}  // namespace hearts
}  // namespace open_spiel
