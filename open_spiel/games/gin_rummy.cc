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

#include "open_spiel/games/gin_rummy.h"

#include <algorithm>
#include <map>
#include <utility>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/algorithms/observation_history.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/gin_rummy/gin_rummy_utils.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace gin_rummy {
namespace {

const GameType kGameType{
    /*short_name=*/"gin_rummy",
    /*long_name=*/"Gin Rummy",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
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
    {
        {"oklahoma", GameParameter(false)},
        {"knock_card", GameParameter(kDefaultKnockCard)},
        {"gin_bonus", GameParameter(kDefaultGinBonus)},
        {"undercut_bonus", GameParameter(kDefaultUndercutBonus)},
        {"num_ranks", GameParameter(kDefaultNumRanks)},
        {"num_suits", GameParameter(kDefaultNumSuits)},
        {"hand_size", GameParameter(kDefaultHandSize)},
    }};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new GinRummyGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

bool ObserverHasString(IIGObservationType iig_obs_type) {
  return !iig_obs_type.perfect_recall ||
         (iig_obs_type.public_info &&
          iig_obs_type.private_info == PrivateInfoType::kSinglePlayer);
}

bool ObserverHasTensor(IIGObservationType iig_obs_type) {
  return !iig_obs_type.perfect_recall;
}

}  // namespace


class GinRummyObserver : public Observer {
 public:
  explicit GinRummyObserver(IIGObservationType iig_obs_type)
      : Observer(/*has_string=*/ObserverHasString(iig_obs_type),
                 /*has_tensor=*/ObserverHasTensor(iig_obs_type)),
        iig_obs_type_(iig_obs_type) {}


  void WriteTensor(const State& observed_state, int player,
                   Allocator* allocator) const override {
    auto& state = open_spiel::down_cast<const GinRummyState&>(observed_state);
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, state.num_players_);

    if (iig_obs_type_.perfect_recall) {
      SpielFatalError(
          "GinRummyObserver: tensor with perfect recall not implemented.");
    }

    // Observing player.
    WriteObservingPlayer(state, player, allocator);

    // Private hand(s).
    if (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer) {
      WriteSinglePlayerHand(state, player, allocator);
    } else if (iig_obs_type_.private_info == PrivateInfoType::kAllPlayers) {
      WriteAllPlayerHands(state, allocator);
    }

    // Public information.
    if (iig_obs_type_.public_info) {
      WriteCurrentPlayer(state, allocator);
      WriteKnockCard(state, allocator);
      WriteUpcard(state, allocator);
      WriteDiscardPile(state, allocator);
      WriteStockSize(state, allocator);
      WriteLayedMelds(state, allocator);
    }
  }

  std::string StringFrom(const State& observed_state,
                         int player) const override {
    auto& state = open_spiel::down_cast<const GinRummyState&>(observed_state);
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, state.num_players_);

    if (iig_obs_type_.perfect_recall) {
      if (iig_obs_type_.public_info &&
          iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer) {
        return state.aohs_[player].ToString();
      } else {
        SpielFatalError(
            "GinRummyObserver: string with perfect recall is implemented only"
            " for the (default) info state observation type.");
      }
    }

    std::string rv;
    absl::StrAppend(&rv, "\nKnock card: ", state.knock_card_);
    absl::StrAppend(&rv, "\nPrev upcard: ",
                    state.utils_.CardString(state.prev_upcard_));
    absl::StrAppend(&rv, "\nRepeated move: ", state.repeated_move_);
    absl::StrAppend(&rv, "\nCurrent player: ", state.cur_player_);
    absl::StrAppend(&rv, "\nPhase: ",
                    GinRummyState::kPhaseString[static_cast<int>(state.phase_)],
                    "\n");
    if (iig_obs_type_.private_info == PrivateInfoType::kAllPlayers ||
        (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer &&
        player == 0)) {
      absl::StrAppend(&rv, "\nPlayer0: Deadwood=", state.deadwood_[0]);
    } else {
      absl::StrAppend(&rv, "\nPlayer0:");
    }
    if (state.knocked_[1] && !state.layoffs_.empty()) {
      absl::StrAppend(&rv, "\nLayoffs: ");
      for (int card : state.layoffs_) {
        absl::StrAppend(&rv, state.utils_.CardString(card));
      }
    }
    if (!state.layed_melds_[0].empty()) {
      absl::StrAppend(&rv, "\nLayed melds:");
      for (int meld_id : state.layed_melds_[0]) {
        absl::StrAppend(&rv, " ");
        std::vector<int> meld = state.utils_.int_to_meld.at(meld_id);
        for (int card : meld) {
          absl::StrAppend(&rv, state.utils_.CardString(card));
        }
      }
    }
    if (iig_obs_type_.private_info == PrivateInfoType::kAllPlayers ||
        (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer &&
        player == 0)) {
      absl::StrAppend(&rv, "\n", state.utils_.HandToString(state.hands_[0]));
    } else {
      absl::StrAppend(&rv, "\n", state.utils_.HandToString(std::vector<int>()));
    }

    absl::StrAppend(&rv, "\nStock size: ", state.stock_size_);
    absl::StrAppend(&rv, "  Upcard: ", state.utils_.CardString(state.upcard_));
    absl::StrAppend(&rv, "\nDiscard pile: ");
    for (int card : state.discard_pile_) {
      absl::StrAppend(&rv, state.utils_.CardString(card));
    }
    if (iig_obs_type_.private_info == PrivateInfoType::kAllPlayers ||
        (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer &&
        player == 1)) {
      absl::StrAppend(&rv, "\n\nPlayer1: Deadwood=", state.deadwood_[1]);
    } else {
      absl::StrAppend(&rv, "\n\nPlayer1:");
    }
    if (state.knocked_[0] && !state.layoffs_.empty()) {
      absl::StrAppend(&rv, "\nLayoffs: ");
      for (int card : state.layoffs_) {
        absl::StrAppend(&rv, state.utils_.CardString(card));
      }
    }
    if (!state.layed_melds_[1].empty()) {
      absl::StrAppend(&rv, "\nLayed melds:");
      for (int meld_id : state.layed_melds_[1]) {
        absl::StrAppend(&rv, " ");
        std::vector<int> meld = state.utils_.int_to_meld.at(meld_id);
        for (int card : meld) {
          absl::StrAppend(&rv, state.utils_.CardString(card));
        }
      }
    }

    if (iig_obs_type_.private_info == PrivateInfoType::kAllPlayers ||
        (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer &&
        player == 1)) {
      absl::StrAppend(&rv, "\n", state.utils_.HandToString(state.hands_[1]));
    } else {
      absl::StrAppend(&rv, "\n", state.utils_.HandToString(std::vector<int>()));
    }
    return rv;
  }

 private:
  static void WriteObservingPlayer(const GinRummyState& state, int player,
                                   Allocator* allocator) {
    auto out = allocator->Get("player", {kNumPlayers});
    out.at(player) = 1;
  }

  static void WriteSinglePlayerHand(const GinRummyState& state, int player,
                                    Allocator* allocator) {
    auto out = allocator->Get("private_hand", {kNumPlayers, kDefaultNumCards});
    for (auto card : state.hands_[player]) out.at(player, card) = 1;
  }

  static void WriteAllPlayerHands(const GinRummyState& state,
                                  Allocator* allocator) {
    auto out = allocator->Get("private_hands", {kNumPlayers, kDefaultNumCards});
    for (Player p = 0; p < kNumPlayers; ++p) {
      for (auto card : state.hands_[p]) out.at(p, card) = 1;
    }
  }

  static void WriteCurrentPlayer(const GinRummyState& state,
                                 Allocator* allocator) {
    auto out = allocator->Get("current_player", {kNumPlayers});
    if (state.cur_player_ >= 0) out.at(state.cur_player_) = 1;
  }

  static void WriteKnockCard(const GinRummyState& state,
                             Allocator* allocator) {
    auto out = allocator->Get("knock_card", {kDefaultKnockCard});
    for (int i = 0; i < state.knock_card_; ++i) out.at(i) = 1;
  }

  static void WriteUpcard(const GinRummyState& state, Allocator* allocator) {
    auto out = allocator->Get("upcard", {kDefaultNumCards});
    if (state.upcard_.has_value()) out.at(state.upcard_.value()) = 1;
  }

  static void WriteDiscardPile(const GinRummyState& state,
                               Allocator* allocator) {
    auto out = allocator->Get("discard_pile", {kDefaultNumCards});
    for (auto card : state.discard_pile_) out.at(card) = 1;
  }

  static void WriteStockSize(const GinRummyState& state,
                             Allocator* allocator) {
    auto out = allocator->Get("stock_size", {kDefaultNumCards});
    for (int i = 0; i < state.stock_size_; ++i) out.at(i) = 1;
  }

  static void WriteLayedMelds(const GinRummyState& state,
                              Allocator* allocator) {
    auto out = allocator->Get("layed_melds", {kNumPlayers, kNumMeldActions});
    for (Player p = 0; p < kNumPlayers; ++p) {
      for (auto meld : state.layed_melds_[p]) out.at(p, meld) = 1;
    }
  }

  IIGObservationType iig_obs_type_;
};

GinRummyState::GinRummyState(std::shared_ptr<const Game> game, bool oklahoma,
                             int knock_card, int gin_bonus, int undercut_bonus,
                             int num_ranks, int num_suits, int hand_size)
    : State(std::move(game)),
      oklahoma_(oklahoma),
      knock_card_(knock_card),
      gin_bonus_(gin_bonus),
      undercut_bonus_(undercut_bonus),
      num_ranks_(num_ranks),
      num_suits_(num_suits),
      num_cards_(num_ranks * num_suits),
      hand_size_(hand_size),
      utils_(GinRummyUtils(num_ranks, num_suits, hand_size)),
      stock_size_(num_ranks * num_suits),
      deck_(num_ranks * num_suits, true) {
  aohs_.reserve(kNumPlayers);
  for (Player player = 0; player < kNumPlayers; ++player) {
    std::vector<std::pair<absl::optional<Action>, std::string>> aoh;
    aoh.push_back({{}, ObservationString(player)});
    aohs_.push_back(open_spiel::ActionObservationHistory(player, aoh));
  }
}

int GinRummyState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  } else {
    return cur_player_;
  }
}

void GinRummyState::DoApplyAction(Action action) {
  Player current_player = CurrentPlayer();
  switch (phase_) {
    case Phase::kDeal:
      ApplyDealAction(action);
      break;
    case Phase::kFirstUpcard:
      ApplyFirstUpcardAction(action);
      break;
    case Phase::kDraw:
      ApplyDrawAction(action);
      break;
    case Phase::kDiscard:
      ApplyDiscardAction(action);
      break;
    case Phase::kKnock:
      ApplyKnockAction(action);
      break;
    case Phase::kLayoff:
      ApplyLayoffAction(action);
      break;
    case Phase::kWall:
      ApplyWallAction(action);
      break;
    case Phase::kGameOver:
      SpielFatalError("Cannot act in terminal states.");
      break;
    default:
      SpielFatalError("Invalid game phase.");
  }
  for (Player player = 0; player < NumPlayers(); ++player) {
    absl::optional<Action> a = {};
    if (current_player == player) a = action;
    aohs_[player].Extend(a, ObservationString(player));
  }
}

void GinRummyState::ApplyDealAction(Action action) {
  SPIEL_CHECK_TRUE(IsChanceNode());
  SPIEL_CHECK_GE(action, 0);
  SPIEL_CHECK_LT(action, num_cards_);
  // Deal cards to player 0.
  if (stock_size_ > num_cards_ - hand_size_) {
    StockToHand(0, action);
  } else if (stock_size_ > num_cards_ - 2 * hand_size_) {
    // Next deal to player 1.
    StockToHand(1, action);
  } else if (stock_size_ == num_cards_ - 2 * hand_size_) {
    // Set upcard.
    StockToUpcard(action);
    for (int i = 0; i < kNumPlayers; ++i) {
      deadwood_[i] = utils_.MinDeadwood(hands_[i]);
    }
    // Initial upcard determines the knock card if playing Oklahoma.
    if (oklahoma_) {
      knock_card_ = utils_.CardValue(action);
      // Ace upcard means we must play for gin!
      if (knock_card_ == 1) knock_card_ = 0;
    }
    prev_player_ = kChancePlayerId;
    // This implementation always starts the action with player 0.
    cur_player_ = 0;
    phase_ = Phase::kFirstUpcard;
  } else {
    // Previous player drew from stock, let's deal them a card.
    StockToHand(prev_player_, action);
    // Update deadwood total, used to see if knock is legal.
    deadwood_[prev_player_] = utils_.MinDeadwood(hands_[prev_player_]);
    cur_player_ = prev_player_;
    prev_player_ = kChancePlayerId;
    phase_ = Phase::kDiscard;
  }
}

// Unique rules apply to the first upcard. If the first player to act does not
// draw the upcard, the second player then has the option to pick it up. If
// both players pass, the first player draws from the stock.
void GinRummyState::ApplyFirstUpcardAction(Action action) {
  if (action == kDrawUpcardAction) {
    SPIEL_CHECK_TRUE(upcard_.has_value());
    prev_upcard_ = upcard_;
    UpcardToHand(cur_player_);
    deadwood_[cur_player_] = utils_.MinDeadwood(hands_[cur_player_]);
    prev_player_ = cur_player_;
    phase_ = Phase::kDiscard;
  } else if (action == kDrawStockAction) {
    SPIEL_CHECK_TRUE(pass_on_first_upcard_[0] && pass_on_first_upcard_[1]);
    prev_upcard_ = upcard_;
    discard_pile_.push_back(upcard_.value());
    upcard_ = absl::nullopt;
    prev_player_ = cur_player_;
    cur_player_ = kChancePlayerId;
    phase_ = Phase::kDeal;
  } else if (action == kPassAction) {
    SPIEL_CHECK_FALSE(pass_on_first_upcard_[0] && pass_on_first_upcard_[1]);
    pass_on_first_upcard_[cur_player_] = true;
    prev_player_ = cur_player_;
    cur_player_ = Opponent(prev_player_);
    phase_ = Phase::kFirstUpcard;
  } else {
    SpielFatalError("Invalid Action");
  }
}

void GinRummyState::ApplyDrawAction(Action action) {
  if (action == kDrawUpcardAction) {
    SPIEL_CHECK_TRUE(upcard_.has_value());
    if (++num_draw_upcard_actions_ == kMaxNumDrawUpcardActions) {
      phase_ = Phase::kGameOver;
      return;
    }
    prev_upcard_ = upcard_;
    UpcardToHand(cur_player_);
    deadwood_[cur_player_] = utils_.MinDeadwood(hands_[cur_player_]);
    prev_player_ = cur_player_;
    phase_ = Phase::kDiscard;
  } else if (action == kDrawStockAction) {
    // When a player chooses to draw from stock the upcard is no
    // longer in play and goes to the top of the discard pile.
    prev_upcard_ = upcard_;
    if (upcard_.has_value()) discard_pile_.push_back(upcard_.value());
    upcard_ = absl::nullopt;
    prev_player_ = cur_player_;
    cur_player_ = kChancePlayerId;
    phase_ = Phase::kDeal;
  } else {
    SpielFatalError("Invalid DrawAction");
  }
}

void GinRummyState::ApplyDiscardAction(Action action) {
  if (action == kKnockAction) {
    SPIEL_CHECK_LE(deadwood_[cur_player_], knock_card_);
    // The hand has been knocked, so now deadwood tracks the total card value.
    for (int i = 0; i < kNumPlayers; ++i)
      deadwood_[i] = utils_.TotalCardValue(hands_[i]);
    knocked_[cur_player_] = true;
    prev_player_ = cur_player_;
    phase_ = Phase::kKnock;
  } else {
    SPIEL_CHECK_TRUE(absl::c_linear_search(hands_[cur_player_], action));
    RemoveFromHand(cur_player_, action);
    deadwood_[cur_player_] = utils_.MinDeadwood(hands_[cur_player_]);
    upcard_ = action;
    prev_player_ = cur_player_;
    cur_player_ = Opponent(prev_player_);
    if (upcard_ == prev_upcard_) {
      if (repeated_move_) {
        phase_ = Phase::kGameOver;
        return;
      } else {
        repeated_move_ = true;
      }
    } else {
      repeated_move_ = false;
    }
    if (stock_size_ == kWallStockSize) {
      phase_ = Phase::kWall;
    } else {
      phase_ = Phase::kDraw;
    }
  }
}

void GinRummyState::ApplyKnockAction(Action action) {
  // First the knocking player must discard.
  if (hands_[cur_player_].size() == hand_size_ + 1) {
    SPIEL_CHECK_TRUE(absl::c_linear_search(hands_[cur_player_], action));
    RemoveFromHand(cur_player_, action);
    discard_pile_.push_back(action);
    deadwood_[cur_player_] = utils_.TotalCardValue(hands_[cur_player_]);
    phase_ = Phase::kKnock;
  } else if (action == kPassAction) {
    // Here the pass action indicates knocking player is finished laying the
    // hand. The player's deadwood is now final, and any cards not layed in a
    // meld are counted towards the deadwood total.
    deadwood_[cur_player_] = utils_.TotalCardValue(hands_[cur_player_]);
    // Make sure the knocking player has completed a legal knock.
    SPIEL_CHECK_LE(deadwood_[cur_player_], knock_card_);
    // If deadwood equals 0 then the player has gin. The opponent is not
    // allowed to lay off on gin.
    if (deadwood_[cur_player_] == 0) finished_layoffs_ = true;
    cur_player_ = Opponent(prev_player_);
    phase_ = Phase::kLayoff;
  } else {
    // Knocking player must declare or "lay" melds, one action at a time.
    SPIEL_CHECK_LE(action - kMeldActionBase, kNumMeldActions);
    SPIEL_CHECK_GE(action - kMeldActionBase, 0);
    layed_melds_[cur_player_].push_back(action - kMeldActionBase);
    // Upon laying a meld the cards are removed from the player's hand.
    for (int card : utils_.int_to_meld.at(action - kMeldActionBase)) {
      RemoveFromHand(cur_player_, card);
    }
    deadwood_[cur_player_] = utils_.TotalCardValue(hands_[cur_player_]);
    phase_ = Phase::kKnock;
  }
}

void GinRummyState::ApplyLayoffAction(Action action) {
  if (!finished_layoffs_) {
    if (action == kPassAction) {
      finished_layoffs_ = true;
      phase_ = Phase::kLayoff;
    } else {
      SPIEL_CHECK_TRUE(absl::c_linear_search(hands_[cur_player_], action));
      layoffs_.push_back(action);
      RemoveFromHand(cur_player_, action);
      deadwood_[cur_player_] = utils_.TotalCardValue(hands_[cur_player_]);
      phase_ = Phase::kLayoff;
    }
  } else {
    // Finished laying off individual cards, now lay melds.
    if (action == kPassAction) {
      deadwood_[cur_player_] = utils_.TotalCardValue(hands_[cur_player_]);
      phase_ = Phase::kGameOver;
    } else {
      // Lay melds one action at a time.
      SPIEL_CHECK_LE(action - kMeldActionBase, kNumMeldActions);
      SPIEL_CHECK_GE(action - kMeldActionBase, 0);
      layed_melds_[cur_player_].push_back(action - kMeldActionBase);
      // Upon laying a meld the cards are removed from the player's hand.
      for (int card : utils_.int_to_meld.at(action - kMeldActionBase))
        RemoveFromHand(cur_player_, card);
      deadwood_[cur_player_] = utils_.TotalCardValue(hands_[cur_player_]);
      phase_ = Phase::kLayoff;
    }
  }
}

void GinRummyState::ApplyWallAction(Action action) {
  if (action == kKnockAction) {
    // When we've reached the wall, a knock automatically includes upcard.
    UpcardToHand(cur_player_);
    deadwood_[cur_player_] = utils_.MinDeadwood(hands_[cur_player_]);
    // Make sure knock is legal.
    SPIEL_CHECK_LE(deadwood_[cur_player_], knock_card_);
    knocked_[cur_player_] = true;
    prev_player_ = cur_player_;
    phase_ = Phase::kKnock;
  } else if (action == kPassAction) {
    phase_ = Phase::kGameOver;
  } else {
    SpielFatalError("Invalid WallAction");
  }
}

std::vector<Action> GinRummyState::LegalActions() const {
  switch (phase_) {
    case Phase::kDeal:
      return DealLegalActions();
    case Phase::kFirstUpcard:
      return FirstUpcardLegalActions();
    case Phase::kDraw:
      return DrawLegalActions();
    case Phase::kDiscard:
      return DiscardLegalActions();
    case Phase::kKnock:
      return KnockLegalActions();
    case Phase::kLayoff:
      return LayoffLegalActions();
    case Phase::kWall:
      return WallLegalActions();
    default:
      return {};
  }
}

std::vector<Action> GinRummyState::DealLegalActions() const {
  std::vector<Action> legal_actions;
  for (int card = 0; card < num_cards_; ++card) {
    if (deck_[card]) legal_actions.push_back(card);
  }
  return legal_actions;
}

std::vector<Action> GinRummyState::FirstUpcardLegalActions() const {
  std::vector<Action> legal_actions;
  // If both players have passed then must draw from stock.
  if (pass_on_first_upcard_[0] && pass_on_first_upcard_[1]) {
    legal_actions.push_back(kDrawStockAction);
  } else {
    // Options are to draw upcard or pass to opponent.
    legal_actions.push_back(kDrawUpcardAction);
    legal_actions.push_back(kPassAction);
  }
  return legal_actions;
}

std::vector<Action> GinRummyState::DrawLegalActions() const {
  std::vector<Action> legal_actions;
  legal_actions.push_back(kDrawUpcardAction);
  legal_actions.push_back(kDrawStockAction);
  return legal_actions;
}

std::vector<Action> GinRummyState::DiscardLegalActions() const {
  // All cards in hand are legal discards.
  std::vector<Action> legal_actions(hands_[cur_player_].begin(),
                                    hands_[cur_player_].end());
  if (deadwood_[cur_player_] <= knock_card_) {
    legal_actions.push_back(kKnockAction);
  }
  std::sort(legal_actions.begin(), legal_actions.end());
  return legal_actions;
}

std::vector<Action> GinRummyState::KnockLegalActions() const {
  std::vector<Action> legal_actions;
  // After knocking, the player discards. This discard must not prevent
  // the player from arranging the hand in such a way that the deadwood
  // total is less than the knock card.
  if (hands_[cur_player_].size() == hand_size_ + 1) {
    for (int card : utils_.LegalDiscards(hands_[cur_player_], knock_card_)) {
      legal_actions.push_back(card);
    }
  } else {
    for (int meld_id : utils_.LegalMelds(hands_[cur_player_], knock_card_)) {
      legal_actions.push_back(meld_id + kMeldActionBase);
    }
    // Must keep laying melds until remaining deadwood is less than knock card.
    // Once that has been accomplished, the knocking player can pass.
    if (utils_.TotalCardValue(hands_[cur_player_]) <= knock_card_) {
      legal_actions.push_back(kPassAction);
    }
  }
  std::sort(legal_actions.begin(), legal_actions.end());
  return legal_actions;
}

std::vector<Action> GinRummyState::LayoffLegalActions() const {
  std::vector<Action> legal_actions;
  if (!finished_layoffs_) {
    // Always have the option not to lay off any cards.
    legal_actions.push_back(kPassAction);
    std::vector<int> all_possible_layoffs =
        utils_.AllLayoffs(layed_melds_[prev_player_], layoffs_);
    for (int card : all_possible_layoffs) {
      if (absl::c_linear_search(hands_[cur_player_], card)) {
        legal_actions.push_back(card);
      }
    }
  } else {
    // After laying off individual cards, now the player lays melds.
    // Always have the option not to declare any melds.
    legal_actions.push_back(kPassAction);
    // The non-knocking player does not have to arrange melds in a particular
    // way to get under the knock card. Therefore we use kMaxPossibleDeadwood
    // to ensure that all melds are legal.
    for (int meld_id : utils_.LegalMelds(hands_[cur_player_],
                                         kMaxPossibleDeadwood)) {
      legal_actions.push_back(meld_id + kMeldActionBase);
    }
  }
  std::sort(legal_actions.begin(), legal_actions.end());
  return legal_actions;
}

std::vector<Action> GinRummyState::WallLegalActions() const {
  std::vector<Action> legal_actions;
  legal_actions.push_back(kPassAction);
  // Player can either pass or knock (if legal).
  int deadwood = utils_.MinDeadwood(hands_[cur_player_], upcard_);
  if (deadwood <= knock_card_) {
    legal_actions.push_back(kKnockAction);
  }
  return legal_actions;
}

std::vector<std::pair<Action, double>> GinRummyState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  std::vector<std::pair<Action, double>> outcomes;
  outcomes.reserve(stock_size_);
  const double p = 1.0 / stock_size_;
  for (int card = 0; card < num_cards_; ++card) {
    // This card is still in the deck, prob is 1/stock_size_.
    if (deck_[card]) outcomes.push_back({card, p});
  }
  return outcomes;
}

std::string GinRummyState::ActionToString(Player player, Action action) const {
  if (player == kChancePlayerId) {
    return absl::StrCat("Chance outcome: ", utils_.CardString(action));
  } else {
    std::string action_str;
    if (action < num_cards_) {
      action_str = utils_.CardString(action);
    } else if (action == kDrawUpcardAction) {
      action_str = "Draw upcard";
    } else if (action == kDrawStockAction) {
      action_str = "Draw stock";
    } else if (action == kPassAction) {
      action_str = "Pass";
    } else if (action == kKnockAction) {
      action_str = "Knock";
    } else if (action < kMeldActionBase + kNumMeldActions) {
      std::vector<int> meld = utils_.int_to_meld.at(action - kMeldActionBase);
      std::vector<std::string> meld_str = utils_.CardIntsToCardStrings(meld);
      action_str = absl::StrJoin(meld_str, "");
    } else {
      SpielFatalError(
          absl::StrCat("Error in GinRummyState::ActionToString()."));
    }
    return absl::StrCat("Player: ", player, " Action: ", action_str);
  }
}

std::string GinRummyState::ToString() const {
  std::string rv;
  absl::StrAppend(&rv, "\nKnock card: ", knock_card_);
  absl::StrAppend(&rv, "\nPrev upcard: ", utils_.CardString(prev_upcard_));
  absl::StrAppend(&rv, "\nRepeated move: ", repeated_move_);
  absl::StrAppend(&rv, "\nCurrent player: ", cur_player_);
  absl::StrAppend(&rv, "\nPhase: ", kPhaseString[static_cast<int>(phase_)],
                  "\n");
  absl::StrAppend(&rv, "\nPlayer0: Deadwood=", deadwood_[0]);
  if (knocked_[1] && !layoffs_.empty()) {
    absl::StrAppend(&rv, "\nLayoffs: ");
    for (int card : layoffs_) absl::StrAppend(&rv, utils_.CardString(card));
  }
  if (!layed_melds_[0].empty()) {
    absl::StrAppend(&rv, "\nLayed melds:");
    for (int meld_id : layed_melds_[0]) {
      absl::StrAppend(&rv, " ");
      std::vector<int> meld = utils_.int_to_meld.at(meld_id);
      for (int card : meld) absl::StrAppend(&rv, utils_.CardString(card));
    }
  }
  absl::StrAppend(&rv, "\n", utils_.HandToString(hands_[0]));
  absl::StrAppend(&rv, "\nStock size: ", stock_size_);
  absl::StrAppend(&rv, "  Upcard: ", utils_.CardString(upcard_));
  absl::StrAppend(&rv, "\nDiscard pile: ");
  for (int card : discard_pile_) absl::StrAppend(&rv, utils_.CardString(card));
  absl::StrAppend(&rv, "\n\nPlayer1: Deadwood=", deadwood_[1]);
  if (knocked_[0] && !layoffs_.empty()) {
    absl::StrAppend(&rv, "\nLayoffs: ");
    for (int card : layoffs_) absl::StrAppend(&rv, utils_.CardString(card));
  }
  if (!layed_melds_[1].empty()) {
    absl::StrAppend(&rv, "\nLayed melds:");
    for (int meld_id : layed_melds_[1]) {
      absl::StrAppend(&rv, " ");
      std::vector<int> meld = utils_.int_to_meld.at(meld_id);
      for (int card : meld) absl::StrAppend(&rv, utils_.CardString(card));
    }
  }
  absl::StrAppend(&rv, "\n", utils_.HandToString(hands_[1]));
  return rv;
}

std::vector<double> GinRummyState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>(kNumPlayers, 0.0);
  }
  std::vector<double> returns(kNumPlayers, 0.0);
  // player 0 knocked
  if (knocked_[0]) {
    returns[0] = deadwood_[1] - deadwood_[0];
    if (deadwood_[0] == 0) {
      returns[0] += gin_bonus_;
    }
    if (returns[0] < 0) {
      returns[0] -= undercut_bonus_;
    }
    returns[1] = -returns[0];
  } else if (knocked_[1]) {
    // player 1 knocked
    returns[1] = deadwood_[0] - deadwood_[1];
    if (deadwood_[1] == 0) {
      returns[1] += gin_bonus_;
    }
    if (returns[1] < 0) {
      returns[1] -= undercut_bonus_;
    }
    returns[0] = -returns[1];
  }
  // If neither player knocked both players get 0.
  return returns;
}

void GinRummyState::StockToHand(Player player, Action card) {
  hands_[player].push_back(card);
  deck_[card] = false;
  --stock_size_;
}

void GinRummyState::StockToUpcard(Action card) {
  upcard_ = card;
  deck_[card] = false;
  --stock_size_;
}

void GinRummyState::UpcardToHand(Player player) {
  SPIEL_CHECK_TRUE(upcard_.has_value());
  hands_[player].push_back(upcard_.value());
  upcard_ = absl::nullopt;
}

void GinRummyState::RemoveFromHand(Player player, Action card) {
  hands_[player].erase(
      std::remove(hands_[player].begin(), hands_[player].end(), card),
      hands_[player].end());
}

std::unique_ptr<State> GinRummyState::Clone() const {
  return std::unique_ptr<State>(new GinRummyState(*this));
}

std::string GinRummyState::InformationStateString(Player player) const {
  const auto& game = open_spiel::down_cast<const GinRummyGame&>(*game_);
  return game.info_state_observer_->StringFrom(*this, player);
}

std::string GinRummyState::ObservationString(Player player) const {
  const GinRummyGame& game = open_spiel::down_cast<const GinRummyGame&>(*game_);
  return game.default_observer_->StringFrom(*this, player);
}

void GinRummyState::ObservationTensor(Player player,
                                      absl::Span<float> values) const {
  ContiguousAllocator allocator(values);
  const GinRummyGame& game = open_spiel::down_cast<const GinRummyGame&>(*game_);
  game.default_observer_->WriteTensor(*this, player, &allocator);
}

GinRummyGame::GinRummyGame(const GameParameters& params)
    : Game(kGameType, params),
      oklahoma_(ParameterValue<bool>("oklahoma")),
      knock_card_(ParameterValue<int>("knock_card")),
      gin_bonus_(ParameterValue<int>("gin_bonus")),
      undercut_bonus_(ParameterValue<int>("undercut_bonus")),
      num_ranks_(ParameterValue<int>("num_ranks")),
      num_suits_(ParameterValue<int>("num_suits")),
      hand_size_(ParameterValue<int>("hand_size")) {
  SPIEL_CHECK_GE(knock_card_, 0);
  SPIEL_CHECK_LE(knock_card_, kDefaultKnockCard);
  // Check that the game size makes sense.
  SPIEL_CHECK_GE(num_ranks_, 1);
  SPIEL_CHECK_LE(num_ranks_, kDefaultNumRanks);
  SPIEL_CHECK_GE(num_suits_, 1);
  SPIEL_CHECK_LE(num_suits_, kDefaultNumSuits);
  SPIEL_CHECK_GE(hand_size_, 1);
  SPIEL_CHECK_LE(hand_size_, kDefaultHandSize);
  // Must be able to deal both hands and have enough cards left over for one
  // round of play.
  SPIEL_CHECK_GE(num_ranks_ * num_suits_,
                 kNumPlayers * hand_size_ + kWallStockSize + 1);

  default_observer_ = std::make_shared<GinRummyObserver>(kDefaultObsType);
  info_state_observer_ = std::make_shared<GinRummyObserver>(kInfoStateObsType);
}

std::shared_ptr<Observer> GinRummyGame::MakeObserver(
    absl::optional<IIGObservationType> iig_obs_type,
    const GameParameters& params) const {
  if (!params.empty()) SpielFatalError("Observation params not supported");
  return std::make_shared<GinRummyObserver>(
      iig_obs_type.value_or(kDefaultObsType));
}

}  // namespace gin_rummy
}  // namespace open_spiel
