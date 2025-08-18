// Copyright 2023 DeepMind Technologies Limited
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

#include "open_spiel/games/crazy_eights/crazy_eights.h"

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"

namespace open_spiel {
namespace crazy_eights {

namespace {

constexpr char kRankChar[] = "23456789TJQKA";
constexpr char kSuitChar[] = "CDHS";

constexpr int kDefaultPlayers = 5;
constexpr int kDefaultMaxDrawCards = 5;
constexpr int kNumInitialCardsForTwoPlayers = 7;
constexpr int kNumInitialCards = 5;
constexpr int kDefaultMaxTurns = 100;

constexpr int kEightRank = 6;     // 8
constexpr int kSkipRank = 10;     // Q
constexpr int kReverseRank = 12;  // A
constexpr int kDrawTwoRank = 0;   // 2

const GameType kGameType{
    /*short_name=*/"crazy_eights",
    /*long_name=*/"Crazy Eights",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/15,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {{"players", GameParameter(kDefaultPlayers)},
     {"max_draw_cards", GameParameter(kDefaultMaxDrawCards)},
     {"max_turns", GameParameter(kDefaultMaxTurns)},
     {"use_special_cards", GameParameter(false)},
     {"reshuffle", GameParameter(false)}},
    /*default_loadable=*/true,
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new CrazyEightsGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

Suit GetSuit(int action) {
  SPIEL_CHECK_GE(action, 0);
  SPIEL_CHECK_LT(action, kNumCards);

  return static_cast<Suit>(action % kNumSuits);
}

int GetRank(int action) {
  SPIEL_CHECK_GE(action, 0);
  SPIEL_CHECK_LT(action, kNumCards);

  return action / kNumSuits;
}

int GetAction(Suit suit, int rank) {
  SPIEL_CHECK_LE(rank, kNumRanks);
  return rank * kNumSuits + static_cast<int>(suit);
}

std::string GetCardStr(int action) {
  SPIEL_CHECK_GE(action, 0);
  SPIEL_CHECK_LT(action, kNumCards);
  int rank = GetRank(action);
  int suit = static_cast<int>(GetSuit(action));
  return {kSuitChar[suit], kRankChar[rank]};
}

}  // namespace

CrazyEightsGame::CrazyEightsGame(const GameParameters& params)
    : Game(kGameType, params),
      num_players_(ParameterValue<int>("players")),
      max_draw_cards_(ParameterValue<int>("max_draw_cards")),
      max_turns_(ParameterValue<int>("max_turns")),
      use_special_cards_(ParameterValue<bool>("use_special_cards")),
      reshuffle_(ParameterValue<bool>("reshuffle")) {}

CrazyEightsState::CrazyEightsState(std::shared_ptr<const Game> game,
                                   int num_players, int max_draw_cards,
                                   int max_turns,
                                   bool use_special_cards, bool reshuffle)
    : State(game),
      num_players_(num_players),
      max_draw_cards_(max_draw_cards),
      max_turns_(max_turns),
      use_special_cards_(use_special_cards),
      reshuffle_(reshuffle){
  num_initial_cards_ =
      num_players == 2 ? kNumInitialCardsForTwoPlayers : kNumInitialCards;
  num_decks_ = num_players > 5 ? 2 : 1;
  num_cards_left_ = num_decks_ * kNumCards;
  absl::c_fill(dealer_deck_, num_decks_);
  for (int i = 0; i < num_players; ++i) {
    hands_.push_back(std::vector<int>(kNumCards, 0));
    returns_.push_back(0);
  }
}

std::string CrazyEightsState::ActionToString(Player player,
                                             Action action) const {
  if (player == kChancePlayerId) {
    if (action < kDraw) {
      return absl::StrFormat("Deal %s", GetCardStr(action));
    } else if (action < kDecideDealerActionBase + num_players_) {
      return absl::StrFormat("Decide Player %d to be the dealer",
                             action - kDecideDealerActionBase);
    } else {
      SpielFatalError(
          absl::StrFormat("Non action valid Id  %d for chance player", action));
    }
  }

  if (action < kDraw) {
    return absl::StrFormat("Play %s", GetCardStr(action));
  } else if (action == kDraw) {
    return "Draw";
  } else if (action == kPass) {
    return "Pass";
  } else if (action < kNominateSuitActionBase + kNumSuits) {
    return absl::StrFormat("Nominate suit %c",
                           kSuitChar[action - kNominateSuitActionBase]);
  } else {
    SpielFatalError(
        absl::StrFormat("Non valid Id %d for player: %d", action, player));
  }
}

std::vector<std::string> CrazyEightsState::FormatHand(Player player) const {
  std::vector<std::string> hand_str(kNumSuits,
                                    std::string(num_decks_ * kNumRanks, ' '));
  for (int suit = 0; suit < kNumSuits; ++suit) {
    for (int rank = 0; rank < kNumRanks; ++rank) {
      int card = GetAction(static_cast<Suit>(suit), rank);
      for (int i = 0; i < hands_[player][card]; ++i) {
        hand_str[suit][rank * num_decks_ + i] = kRankChar[rank];
      }
    }
  }
  return hand_str;
}

std::string CrazyEightsState::FormatAllHands() const {
  std::string hands_str;
  std::vector<std::vector<std::string>> all_hands;
  all_hands.reserve(num_players_);
  for (int player = 0; player < num_players_; ++player) {
    all_hands.push_back(FormatHand(player));
  }
  constexpr int kLongWidth = 40;

  for (int player = 0; player < num_players_; ++player) {
    std::string player_str = absl::StrFormat("Player %d:", player);
    if (player != num_players_ - 1) {
      absl::StrAppend(&player_str,
                      std::string(kLongWidth - player_str.length(), ' '));
    } else {
      absl::StrAppend(&player_str, "\n");
    }
    absl::StrAppend(&hands_str, player_str);
  }

  for (int suit = 0; suit < kNumSuits; ++suit) {
    std::string suit_row;
    for (int player = 0; player < num_players_; ++player) {
      std::string player_row;
      absl::StrAppend(&player_row,
                      absl::StrFormat("Suit %c: %s", kSuitChar[suit],
                                      all_hands[player][suit]));
      SPIEL_CHECK_GE(kLongWidth, player_row.length());
      if (player != num_players_ - 1) {
        absl::StrAppend(&player_row,
                        std::string(kLongWidth - player_row.length(), ' '));
      } else {
        absl::StrAppend(&player_row, "\n");
      }
      absl::StrAppend(&suit_row, player_row);
    }
    absl::StrAppend(&hands_str, suit_row);
  }
  return hands_str;
}

std::string CrazyEightsState::ToString() const {
  std::string str;
  int playing_player = dealer_;
  for (int i = 0; i < history_.size(); ++i) {
    if (i == 0) {
      absl::StrAppend(
          &str, absl::StrFormat("Player %d becomes the dealer\n", dealer_));
    } else if (i <= num_players_ * num_initial_cards_) {
      int player = (dealer_ + i) % num_players_;
      absl::StrAppend(&str, absl::StrFormat("Player %d is dealt %s\n", player,
                                            GetCardStr(history_[i].action)));
    } else {
      if (history_[i].player == kChancePlayerId) {
        absl::StrAppend(&str,
                        absl::StrFormat("Player %d draws %s\n", playing_player,
                                        GetCardStr(history_[i].action)));
      } else if (history_[i].player != kTerminalPlayerId) {
        playing_player = history_[i].player;
        if (history_[i].action == kDraw) {
          absl::StrAppend(&str, absl::StrFormat("Player %d starts drawing\n",
                                                playing_player));
        } else if (history_[i].action == kPass) {
          absl::StrAppend(
              &str, absl::StrFormat("Player %d passes\n", playing_player));
        } else if (history_[i].action >= kNominateSuitActionBase &&
                   history_[i].action < kNominateSuitActionBase + kNumSuits) {
          int suit = history_[i].action - kNominateSuitActionBase;
          absl::StrAppend(&str,
                          absl::StrFormat("Player %d nominates suit %c\n",
                                          playing_player, kSuitChar[suit]));
        } else {
          SPIEL_CHECK_GE(history_[i].action, 0);
          SPIEL_CHECK_LT(history_[i].action, kNumCards);
          absl::StrAppend(
              &str, absl::StrFormat("Player %d plays %s\n", playing_player,
                                    GetCardStr(history_[i].action)));
        }
      } else {
        absl::StrAppend(&str, "Final scores\n");
        for (int player = 0; player < num_players_; ++player) {
          absl::StrAppend(&str, absl::StrFormat("Player %d gets score %f\n",
                                                player, returns_[player]));
        }
      }
    }
  }
  if (last_card_ != kInvalidAction) {
    absl::StrAppend(&str,
                    absl::StrFormat("Last card: %s\n", GetCardStr(last_card_)));
    absl::StrAppend(&str,
                    absl::StrFormat("Last suit: %c\n", kSuitChar[last_suit_]));
  }
  absl::StrAppend(&str, absl::StrFormat("Number of cards left in deck: %d\n",
                                        num_cards_left_));
  absl::StrAppend(&str, FormatAllHands());
  return str;
}

std::string CrazyEightsState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  std::string str;
  if (phase_ == Phase::kDeal) return str;
  absl::StrAppend(&str, "Currently I have: \n");
  std::vector<std::string> hands = FormatHand(player);
  for (int suit = 0; suit < kNumSuits; ++suit) {
    absl::StrAppend(
        &str, absl::StrFormat("Suit %c: %s\n", kSuitChar[suit], hands[suit]));
  }
  absl::StrAppend(
      &str, absl::StrFormat("Previous card: %s\n", GetCardStr(last_card_)));
  absl::StrAppend(
      &str, absl::StrFormat("Previous suit: %c\n", kSuitChar[last_suit_]));
  absl::StrAppend(&str, "Starting counterclockwise, other players have: ");
  for (int i = 0; i <= num_players_ - 1; ++i) {
    int player_idx = (player + i) % num_players_;
    int player_num_cards = 0;
    for (int card = 0; card < kNumCards; ++card) {
      player_num_cards += hands_[player_idx][card];
    }
    if (i != num_players_ - 1) {
      absl::StrAppend(&str, absl::StrFormat("%d, ", player_num_cards));
    } else {
      absl::StrAppend(&str, absl::StrFormat("%d cards.\n", player_num_cards));
    }
  }
  if (use_special_cards_) {
    absl::StrAppend(&str, absl::StrFormat("The direction is %s\n",
                                          direction_ == 1 ? "counterclockwise"
                                                          : "clockwise"));
  }
  return str;
}

void CrazyEightsState::ObservationTensor(Player player,
                                         absl::Span<float> values) const {
  SPIEL_CHECK_EQ(values.size(), game_->ObservationTensorSize());
  WriteObservationTensor(player, values);
}

void CrazyEightsState::WriteObservationTensor(Player player,
                                              absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  absl::c_fill(values, 0.);
  if (phase_ == Phase::kDeal) return;

  for (int card = 0; card < kNumCards; ++card) {
    values[card * (num_decks_ + 1) + hands_[player][card]] = 1;
  }
  values[(num_decks_ + 1) * kNumCards + last_card_] = 1;
  values[(num_decks_ + 1) * kNumCards + kNumCards + last_suit_] = 1;
  int tmp_base = (num_decks_ + 1) * kNumCards + kNumCards + kNumSuits;
  for (int i = 1; i <= num_players_ - 1; ++i) {
    int num_cards = 0;
    for (int card = 0; card < kNumCards; ++card) {
      num_cards += hands_[(player + i) % num_players_][card];
    }
    values[tmp_base + (i - 1) * (num_decks_ * kNumCards + 1) + num_cards] = 1;
  }

  if (use_special_cards_) {
    tmp_base += (num_decks_ * kNumCards + 1) * (num_players_ - 1);
    values[tmp_base] = (direction_ + 1) / 2;
  }
}

std::vector<Action> CrazyEightsState::LegalActions() const {
  switch (phase_) {
    case Phase::kDeal:
      return DealLegalActions();
    case Phase::kPlay:
      return PlayLegalActions();
    default:
      return {};
  }
}

std::vector<std::pair<Action, double>> CrazyEightsState::ChanceOutcomes()
    const {
  std::vector<std::pair<Action, double>> outcomes;
  if (history_.empty()) {
    for (int player = 0; player < num_players_; ++player) {
        outcomes.emplace_back(player + kDecideDealerActionBase,
                            1.0 / num_players_);
    }
  } else {
    int num_cards_remaining = 0;
    for (int card = 0; card < kNumCards; ++card) {
      SPIEL_CHECK_GE(dealer_deck_[card], 0);
      SPIEL_CHECK_LE(dealer_deck_[card], num_decks_);
      num_cards_remaining += dealer_deck_[card];
    }
    outcomes.reserve(num_cards_remaining);
    for (int card = 0; card < kNumCards; ++card) {
      if (dealer_deck_[card]) {
        outcomes.emplace_back(card, static_cast<double>(dealer_deck_[card]) /
                              num_cards_remaining);
      }
    }
  }
  return outcomes;
}

void CrazyEightsState::DoApplyAction(Action action) {
  switch (phase_) {
    case Phase::kDeal:
      return ApplyDealAction(action);
    case Phase::kPlay:
      return ApplyPlayAction(action);
    case Phase::kGameOver:
      SpielFatalError("Cannot act in terminal states");
    default:
      SpielFatalError("Invalid Phase!");
  }
}

std::vector<Action> CrazyEightsState::DealLegalActions() const {
  std::vector<Action> legal_actions;
  if (history_.empty()) {
    for (int player = 0; player < num_players_; ++player) {
      legal_actions.push_back(kDecideDealerActionBase + player);
    }
  } else {
    for (int card = 0; card < kNumCards; ++card) {
      if (dealer_deck_[card]) {
        legal_actions.push_back(card);
      }
    }
  }
  return legal_actions;
}

void CrazyEightsState::Reshuffle() {
  SPIEL_CHECK_NE(last_card_, kInvalidAction);
  for (int card = 0; card < kNumCards; ++card) {
    dealer_deck_[card] = num_decks_;
    for (int player = 0; player < num_players_; ++player) {
      dealer_deck_[card] -= hands_[player][card];
    }
    if (card == last_card_) dealer_deck_[card]--;
    SPIEL_CHECK_GE(dealer_deck_[card], 0);
    SPIEL_CHECK_LE(dealer_deck_[card], num_decks_);
    num_cards_left_ += dealer_deck_[card];
  }
}

void CrazyEightsState::ApplyDealAction(int action) {
  // determine the dealer
  if (history_.empty()) {
    dealer_ = action - kDecideDealerActionBase;
    current_player_ = (dealer_ + 1) % num_players_;
    return;
  }

  SPIEL_CHECK_GE(action, 0);
  SPIEL_CHECK_LT(action, kDraw);

  num_cards_left_--;
  dealer_deck_[action]--;
  hands_[current_player_][action]++;

  SPIEL_CHECK_GE(dealer_deck_[action], 0);
  SPIEL_CHECK_LE(dealer_deck_[action], num_decks_);

  // reshuffle the discarded cards
  if (!num_cards_left_ && reshuffle_) {
    Reshuffle();
  }

  // redraw=true if we are examining the first card turned face up after the
  // initial dealing round, which cannot be Eights
  if (redraw_) {
    SPIEL_CHECK_EQ(current_player_, dealer_);
    int rank = GetRank(action);
    if (rank != kEightRank) {
      phase_ = Phase::kPlay;
      redraw_ = false;
      last_card_ = action;
      last_suit_ = GetSuit(action);
      hands_[current_player_][action]--;
      // if it is special card, act as if the dealer played this card
      if (use_special_cards_) {
        if (rank == kSkipRank) {
          current_player_ = (current_player_ + 2) % num_players_;
          return;
        } else if (rank == kReverseRank) {
          current_player_ = (current_player_ - 1 + num_players_) %
                             num_players_;
          direction_ *= -1;
          return;
        } else if (rank == kDrawTwoRank) {
          num_draws_from_twos_left_ += 2;
          current_player_ = (current_player_ + 1) % num_players_;
          return;
        }
      }
      current_player_ = (current_player_ + 1) % num_players_;
      return;
    } else {
      // put back
      dealer_deck_[action]++;
      num_cards_left_++;
      hands_[current_player_][action]--;
      return;
    }
  }

  SPIEL_CHECK_FALSE(redraw_);

  if (history_.size() < num_players_ * num_initial_cards_) {
    current_player_ = (current_player_ + 1) % num_players_;
    return;
  }

  if (history_.size() == num_players_ * num_initial_cards_) {
    SPIEL_CHECK_EQ(current_player_, dealer_);
    redraw_ = true;
    return;
  }

  if (!num_cards_left_) can_pass_action_ = true;

  // if has accumlated 2s and has decided to draw these 2s from previous plays
  if (start_draw_twos_) {
    SPIEL_CHECK_TRUE(use_special_cards_);
    num_draws_from_twos_left_--;
    // assume if there is no card in the pile then the liability is cleared
    if (!num_cards_left_) {
      // if it is due to that the pile is exhausted during drawing +2s,
      // counted as a pass
      if (!num_draws_from_twos_left_) num_passes_++;
      num_draws_from_twos_left_ = 0;
    }
    if (!num_draws_from_twos_left_) {
      start_draw_twos_ = false;
      phase_ = Phase::kPlay;
      current_player_ = (current_player_ + direction_ +
            num_players_) % num_players_;
    }
    return;
  }

  // lastly, consider when the player draws card without having a previous +2
  // card
  num_draws_before_play_++;
  phase_ = Phase::kPlay;

  if (!num_cards_left_) num_draws_before_play_ = max_draw_cards_;
  if (num_draws_before_play_ == max_draw_cards_) {
    can_pass_action_ = true;
  }
}

void SearchLegalCards(std::vector<Action>* legal_actions,
                      const std::vector<int>& hand, int last_rank,
                      int last_suit) {
  for (int card = 0; card < kNumCards; ++card) {
    if (hand[card] == 0) continue;
    Suit suit = GetSuit(card);
    int rank = GetRank(card);
    if (rank == kEightRank) {
      legal_actions->push_back(card);
    } else if (last_suit == suit || last_rank == rank) {
      legal_actions->push_back(card);
    }
  }
}

std::vector<Action> CrazyEightsState::PlayLegalActions() const {
  std::vector<Action> legal_actions;
  if (nominate_suits_) {
    for (int suit = kClubs; suit <= kSpades; ++suit) {
      legal_actions.push_back(suit + kNominateSuitActionBase);
    }
    return legal_actions;
  }

  if (can_pass_action_ || !num_cards_left_) {
    SPIEL_CHECK_TRUE(!start_draw_twos_);
    legal_actions.push_back(kPass);
  }

  if (num_draws_from_twos_left_) {
    SPIEL_CHECK_GT(num_cards_left_, 0);

    legal_actions.push_back(kDraw);
    // since we are able to draw
    SPIEL_CHECK_FALSE(can_pass_action_);
    SPIEL_CHECK_TRUE(use_special_cards_);

    if (!start_draw_twos_) {
      for (int suit = kClubs; suit <= kSpades; ++suit) {
        int duo_card = GetAction(static_cast<Suit>(suit), kDrawTwoRank);
        if (hands_[current_player_][duo_card])
          legal_actions.push_back(duo_card);
      }
    }
  } else {
    for (int card = 0; card < kNumCards; ++card) {
      if (hands_[current_player_][card] == 0) continue;
      Suit suit = GetSuit(card);
      int rank = GetRank(card);
      if (rank == kEightRank) {
        legal_actions.push_back(card);
      } else if (last_suit_ == suit || GetRank(last_card_) == rank) {
        legal_actions.push_back(card);
      }
    }
    if (num_cards_left_ && num_draws_before_play_ != max_draw_cards_) {
      SPIEL_CHECK_FALSE(can_pass_action_);
      legal_actions.push_back(kDraw);
    }
  }
  absl::c_sort(legal_actions);
  return legal_actions;
}

bool CrazyEightsState::CheckAllCardsPlayed(int action) {
  SPIEL_CHECK_GT(hands_[current_player_][action], 0);
  hands_[current_player_][action]--;
  bool all_played = true;
  for (int card = 0; card < kNumCards; ++card) {
    all_played &= !hands_[current_player_][card];
  }
  return all_played;
}

void CrazyEightsState::ApplyPlayAction(int action) {
  if (action == kPass) {
    if (!num_cards_left_) {
      num_passes_++;
    } else {
      num_passes_ = 0;
    }

    if (num_passes_ == num_players_ + 1) {
      phase_ = kGameOver;
      ScoreUp();
      return;
    }

    if (max_draw_cards_ == num_draws_before_play_) {
      num_draws_before_play_ = 0;
    }
    current_player_ =
        (current_player_ + direction_ + num_players_) % num_players_;
    if (num_cards_left_) can_pass_action_ = false;
    return;
  } else {
    num_passes_ = 0;
  }

  if (action == kDraw) {
    SPIEL_CHECK_FALSE(can_pass_action_);
    phase_ = kDeal;
    if (num_draws_from_twos_left_) { start_draw_twos_ = true; }
    return;
  } else if (nominate_suits_) {
    SPIEL_CHECK_LT(action, kNominateSuitActionBase + kNumSuits);
    SPIEL_CHECK_GE(action, kNominateSuitActionBase);
    last_suit_ = action - kNominateSuitActionBase;
    current_player_ =
       (current_player_ + direction_ + num_players_) % num_players_;
    nominate_suits_ = false;
    return;
  } else {
    num_plays++;
    can_pass_action_ = false;
    num_draws_before_play_ = 0;
    bool all_played = CheckAllCardsPlayed(action);
    if (all_played || num_plays >= max_turns_) {
      phase_ = kGameOver;
      ScoreUp();
    }

    last_card_ = action;
    last_suit_ = GetSuit(action);

    if (!num_cards_left_ && reshuffle_) {
      Reshuffle();
    }

    int rank = GetRank(action);

    if (rank == kEightRank) {
      nominate_suits_ = true;
      return;
    }
    if (use_special_cards_) {
      if (rank == kSkipRank) {
        current_player_ =
            (current_player_ + 2 * direction_ + num_players_) % num_players_;
        return;
      }
      if (rank == kReverseRank) {
        direction_ *= -1;
        current_player_ =
            (current_player_ + direction_ + num_players_) % num_players_;
        return;
      }
      if (rank == kDrawTwoRank) {
        // if there is no card currently available in the pile, assume
        // the next player doesn't have to draw cards in the next round,
        // and just view it played a normal card
        if (num_cards_left_) num_draws_from_twos_left_ += 2;
        current_player_ =
            (current_player_ + direction_ + num_players_) % num_players_;
        return;
      }
    }
    current_player_ =
        (current_player_ + direction_ + num_players_) % num_players_;
    return;
  }
}

Player CrazyEightsState::CurrentPlayer() const {
  if (phase_ == Phase::kDeal) {
    return kChancePlayerId;
  } else if (phase_ == Phase::kGameOver) {
    return kTerminalPlayerId;
  } else {
    return current_player_;
  }
}

void CrazyEightsState::ScoreUp() {
  for (int player = 0; player < num_players_; ++player) {
    for (int card = 0; card < kNumCards; ++card) {
      if (!hands_[player][card]) continue;
      int rank = GetRank(card);
      if (rank == kEightRank) {
        returns_[player] -= 50 * hands_[player][card];
      } else if (rank >= 9) {
        returns_[player] -= 10 * hands_[player][card];
      } else {
        returns_[player] -= (card + 2) * hands_[player][card];
      }
    }
  }
}

}  // namespace crazy_eights
}  // namespace open_spiel
