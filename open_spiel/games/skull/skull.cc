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

#include "open_spiel/games/skull/skull.h"

#include <algorithm>
#include <string>
#include <utility>

#include "absl/algorithm/algorithm.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "observer.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel_globals.h"
#include "spiel.h"
#include "spiel_utils.h"

namespace open_spiel {
namespace skull {

namespace {

const GameType kGameType{
    /*short_name=*/"skull",
    /*long_name=*/"Skull",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kRewards,
    /*max_num_players=*/kMaxPlayers,
    /*min_num_players=*/kMinPlayers,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {{"players", GameParameter(kDefaultPlayers)},
     {"handsize", GameParameter(kDefaultInitialHandSize)},
     {"winsneeded", GameParameter(kDefaultWinningScore)},
     {"obs_public_derived_info", GameParameter(kDefaultObserveDerivedInfo)},
     {"obs_egocentric", GameParameter(kDefaultEgocentric)},
     {"obs_partial_recall", GameParameter(kDefaultPartialRecall)}}};

std::shared_ptr<const Game> Factory(const GameParameters &params) {
  return std::make_shared<const SkullGame>(params);
}

constexpr int CalcMaxRoundLength(int max_total_cards, int num_players,
                                 bool has_card_loss_phase) {
  // Can be tightened somewhat now that flipping one's own cards are shortcutted
  // to play automatically

  int card_loss_actions = has_card_loss_phase ? 1 : 0;

  return max_total_cards + // every player plays every card they have, then bid:
         (max_total_cards - 1) * (num_players - 1) + // bid1 pass pass bid ...
         1 +                // until bid_max (not followed by passes)
         max_total_cards +  // max flip actions
         card_loss_actions; // if present.
}

REGISTER_SPIEL_GAME(kGameType, Factory);
RegisterSingleTensorObserver single_tensor(kGameType.short_name);
} // namespace

SkullGame::SkullGame(const GameParameters &params)
    : Game(kGameType, params), num_players_(ParameterValue<int>("players")),
      max_hand_size_(ParameterValue<int>("handsize")),
      wins_needed_(ParameterValue<int>("winsneeded")),
      obs_public_derived_info_(ParameterValue<bool>("obs_public_derived_info")),
      obs_egocentric_(ParameterValue<bool>("obs_egocentric")),
      obs_partial_recall_(ParameterValue<bool>("obs_partial_recall")),
      max_game_length_(
          CalcMaxGameLength(num_players_, max_hand_size_, wins_needed_)),
      flip_base_(kActionBidBase + num_players_ * max_hand_size_ + 1) {
  SPIEL_CHECK_GT(max_game_length_, 0);
  SPIEL_CHECK_GE(num_players_, kMinPlayers);
  SPIEL_CHECK_LE(num_players_, kMaxPlayers);
  SPIEL_CHECK_LE(max_hand_size_, kMaximumHandSize);

  /* TODO: `MakeObserver` Fix
    const GameParameters obs_params = {
        {"public_derived_info", GameParameter(obs_public_derived_info_)},
        {"egocentric", GameParameter(obs_egocentric_)},
        {"partial_recall", GameParameter(obs_partial_recall_)},
    };
      default_observer_ = MakeObserver(kDefaultObsType, obs_params);
      info_state_observer_ = MakeObserver(kInfoStateObsType, obs_params);
      private_observer_ = MakeObserver(kPrivateObsType, obs_params);
      public_observer_ = MakeObserver(kPublicObsType, obs_params);
    */

  default_observer_ =
      std::make_shared<SkullObserver>(kDefaultObsType, obs_public_derived_info_,
                                      obs_egocentric_, obs_partial_recall_);
  info_state_observer_ = std::make_shared<SkullObserver>(
      kInfoStateObsType, obs_public_derived_info_, obs_egocentric_,
      obs_partial_recall_);

  private_observer_ =
      std::make_shared<SkullObserver>(kPrivateObsType, obs_public_derived_info_,
                                      obs_egocentric_, obs_partial_recall_);
  public_observer_ =
      std::make_shared<SkullObserver>(kPublicObsType, obs_public_derived_info_,
                                      obs_egocentric_, obs_partial_recall_);

  IIGObservationType sees_all_privateObsType = {
      /*public_info*/ true,
      /*perfect_recall*/ false,
      /*private_info*/ PrivateInfoType::kAllPlayers};
  sees_all_private_observer_ = std::make_shared<SkullObserver>(
      sees_all_privateObsType, obs_public_derived_info_, obs_egocentric_,
      obs_partial_recall_);
}

SkullState::SkullState(std::shared_ptr<const Game> game)
    : State(game),
      max_hand_size_(down_cast<const SkullGame *>(game.get())->MaxHandSize()),
      wins_needed_(down_cast<const SkullGame *>(game.get())->WinsNeeded()),
      max_total_cards_(
          down_cast<const SkullGame *>(game.get())->MaxTotalCards()),
      flip_base_(down_cast<const SkullGame *>(game.get())->flip_base()),
      hands_(num_players_, InitialHand()), stacks_(num_players_),
      depth_flipped_(num_players_, 0), scores_(num_players_, 0),
      known_has_rose_(num_players_, true),
      known_has_only_roses_(num_players_, false),
      known_has_skull_(num_players_, true),
      known_has_only_skull_(num_players_, false),
      current_phase_(GamePhase::kPlacement), current_player_(kDefaultPlayerId),
      first_player_(kDefaultPlayerId), challenger_(kInvalidPlayer),
      current_bid_(0), total_cards_flipped_(0), winner_(kInvalidPlayer),
      start_of_last_round_idx_(history_.size()) {}

// =============================================================================
// API Functions + Helper String Functions
// =============================================================================

constexpr std::string SkullState::CardTypeToString(CardType c) {
  return c == CardType::kRose ? "R" : "S";
};

constexpr std::string SkullState::PhaseToString(GamePhase p) {
  switch (p) {
  case GamePhase::kPlacement:
    return "Placement";
  case GamePhase::kBidding:
    return "Bidding";
  case GamePhase::kFlipping:
    return "Flipping";
  case GamePhase::kCardLoss:
    return "CardLoss";
  default:
    return "Unknown";
  }
};

std::string SkullState::HandToString(Player p) const {
  auto card_formatter = [](std::string *ret, CardType c) {
    absl::StrAppend(ret, SkullState::CardTypeToString(c));
  };
  return absl::StrCat("[", absl::StrJoin(hands_[p], ", ", card_formatter), "]");
}

std::string SkullState::StackToString(Player p, bool full_info) const {
  std::string result = "[bot, ";
  for (int stack_idx = 0; stack_idx < max_hand_size_; stack_idx++) {

    if (stack_idx >= stack_size(p)) // not placed
      absl::StrAppend(&result, "-, ");

    else if (full_info || stack_idx >= stack_size(p) - depth_flipped_[p])
      absl::StrAppend(
          &result, SkullState::CardTypeToString(stacks_[p][stack_idx]), ", ");

    else // unrevealed cards
      absl::StrAppend(&result, "?, ");
  }
  return absl::StrCat(result, "top]");
}

std::string SkullState::DerivedPublicInfoString(Player p) const {
  std::string rose_status = "might have ";
  if (known_has_rose_[p])
    rose_status = "has ";
  else if (known_has_only_skull_[p])
    rose_status = "doesn't have ";

  std::string skull_status = "might have ";
  if (known_has_skull_[p])
    skull_status = "has ";
  else if (known_has_only_roses_[p])
    skull_status = "doesn't have ";

  return absl::StrCat(rose_status, "a rose, ", skull_status, "a skull");
}
std::string SkullState::FormatActionString(Player player, Action action,
                                           bool make_short) const {
  std::string result;
  if (player == kChancePlayerId) {
    absl::StrAppend(&result, "rng:");
  } else if (!make_short) {
    absl::StrAppend(&result, player, ": ");
  }

  if (action == kActionPlaceRose)
    absl::StrAppend(&result, make_short ? "pr" : "Place Rose");
  else if (action == kActionPlaceSkull)
    absl::StrAppend(&result, make_short ? "ps" : "Place Skull");
  else if (action == kActionPass)
    absl::StrAppend(&result, make_short ? "p" : "Pass");
  else if (action == kActionDiscardRose)
    absl::StrAppend(&result, make_short ? "dr" : "Discard Rose");
  else if (action == kActionDiscardSkull)
    absl::StrAppend(&result, make_short ? "ds" : "Discard Skull");
  else if (IsActionBid(action)) {
    absl::StrAppend(&result, make_short ? "b" : "Bid ",
                    action - kActionBidBase);
  } else if (IsActionFlip(action)) {
    absl::StrAppend(&result, make_short ? "f" : "Flip player ",
                    FlipTargetFromAction(action));
  } else {
    absl::StrAppend(&result, make_short ? "?" : "Unknown action ", action);
  }

  return result;
}

std::string SkullState::ActionToShortString(Player player,
                                            Action action) const {
  return FormatActionString(player, action, /*is_short=*/true);
}
std::string SkullState::ActionToString(Player player, Action action) const {
  return FormatActionString(player, action, /*is_short=*/false);
}

std::string SkullState::ToString() const {
  std::string result;

  absl::StrAppend(
      &result, "PUBLIC INFO:\nphase=", PhaseToString(current_phase_),
      " move=", MoveNumber(), " current_player=", current_player_,
      " challenger=", challenger_, " bid=", current_bid_,
      " first=", first_player_, " flipped=", total_cards_flipped_, "\n");

  absl::StrAppend(&result, "PRIVATE INFO:\n");
  for (Player p = 0; p < num_players_; ++p)
    absl::StrAppend(&result, "Player: ", p, HandToString(p),
                    StackToString(p, true), "\n");

  return result;
}

std::vector<double> SkullState::Returns() const {
  // TODO: Input wanted: For a game where winning 1 round better than zero
  // abstractly, but _overall_ the only thing that matters is winning 2, is
  // factoring it into the rewards like this reasonable, or could it give
  // negative incentives. Also, the math is a little convoluted to keep it
  // Zero-Sum, I'm also unsure how much this matters.
  std::vector<double> returns(num_players_, 0.0);
  if (IsTerminal()) {
    int winner_return = 0;
    for (Player p = 0; p < num_players_; ++p) {
      if (winner_ != p) {
        winner_return += wins_needed_ - score(p);
        returns[p] = score(p) - wins_needed_;
      }
    }
    returns[winner_] = winner_return;
  }
  return returns;
}

std::string SkullState::InformationStateString(Player player) const {
  const SkullGame &game = open_spiel::down_cast<const SkullGame &>(*game_);
  return game.info_state_observer_->StringFrom(*this, player);
}

std::string SkullState::ObservationString(Player player) const {
  const SkullGame &game = open_spiel::down_cast<const SkullGame &>(*game_);
  return game.default_observer_->StringFrom(*this, player);
}
void SkullState::InformationStateTensor(Player player,
                                        absl::Span<float> values) const {
  ContiguousAllocator allocator(values);
  const SkullGame &game = open_spiel::down_cast<const SkullGame &>(*game_);
  game.info_state_observer_->WriteTensor(*this, player, &allocator);
}

void SkullState::ObservationTensor(Player player,
                                   absl::Span<float> values) const {
  ContiguousAllocator allocator(values);
  const SkullGame &game = open_spiel::down_cast<const SkullGame &>(*game_);
  game.default_observer_->WriteTensor(*this, player, &allocator);
}

std::unique_ptr<State>
SkullState::ResampleFromInfostate(int player_id,
                                  std::function<double()> rng) const {
  auto cloned_state = std::make_unique<SkullState>(*this);

  for (Player p = 0; p < num_players_; ++p) {
    if (p == player_id)
      continue;

    int num_revealed = cloned_state->depth_flipped_[p];
    int num_unrevealed = cloned_state->stacks_[p].size() - num_revealed;
    if (num_unrevealed <= 0)
      continue;

    absl::InlinedVector<CardType, kMaximumHandSize> pool =
        cloned_state->hands_[p];

    for (int i = 0; i < num_revealed; ++i) {
      int revealed_idx = cloned_state->stacks_[p].size() - 1 - i;
      CardType revealed_card = cloned_state->stacks_[p][revealed_idx];

      auto it = std::find(pool.begin(), pool.end(), revealed_card);
      SPIEL_CHECK_TRUE(it != pool.end());
      pool.erase(it);
    }

    for (int i = 0; i < num_unrevealed; ++i) {
      int idx = static_cast<int>(rng() * pool.size());
      cloned_state->stacks_[p][i] = pool[idx];
      pool.erase(pool.begin() + idx);
    }
  }

  return cloned_state;
}

std::vector<Action>
SkullState::ActionsConsistentWithInformationFrom(Action action) const {
  if (action == kActionPlaceRose || action == kActionPlaceSkull) {
    return {kActionPlaceRose, kActionPlaceSkull};
  }
  if (action == kActionDiscardRose || action == kActionDiscardSkull) {
    return {kActionPlaceRose, kActionPlaceSkull};
  }
  return {action};
}

std::unique_ptr<State> SkullState::Clone() const {
  return std::unique_ptr<State>(new SkullState(*this));
}

std::vector<std::pair<Action, double>> SkullState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(current_phase_ == GamePhase::kCardLoss);
  SPIEL_CHECK_EQ(CurrentPlayer(), kChancePlayerId);
  const Player p = challenger_;
  SPIEL_CHECK_TRUE(p != kInvalidPlayer);
  if (has_rose(p) && has_skull(p)) {
    return {{kActionDiscardRose, 1.0 - (1.0 / (double)hand_size(p))},
            {kActionDiscardSkull, (1.0 / (double)hand_size(p))}};

  } else if (has_skull(p)) {
    SPIEL_DCHECK_TRUE(has_only_skull(p));
    return {{kActionDiscardSkull, 1.0}};

  } else {
    SPIEL_DCHECK_TRUE(has_only_roses(p));
    return {{kActionDiscardRose, 1.0}};
  }
}

std::vector<Action> SkullState::LegalBids() const {
  std::vector<Action> movelist;
  for (int i = current_bid_ + 1; i <= total_cards_on_table(); i++) {
    movelist.push_back(kActionBidBase + i);
  }
  return movelist;
}

std::vector<Action> SkullState::LegalActions() const {
  if (IsTerminal())
    return {};
  if (IsChanceNode())
    return LegalChanceOutcomes();
  switch (current_phase_) {
  case GamePhase::kPlacement: {
    const Player p = CurrentPlayer();
    std::vector<Action> movelist;

    bool skull_on_stack = false;
    for (const auto &card : stacks_[p]) {
      if (card == CardType::kSkull) {
        skull_on_stack = true;
        break;
      }
    }
    bool can_place_skull = has_skull(p) && !skull_on_stack;
    int cards_in_hand_count = hand_size(p) - stack_size(p);
    bool can_place_rose = (cards_in_hand_count > 1) ||
                          (cards_in_hand_count == 1 && !can_place_skull);

    if (cards_in_hand_count > 0) {
      if (can_place_rose)
        movelist.push_back(kActionPlaceRose);
      if (can_place_skull)
        movelist.push_back(kActionPlaceSkull);
    }

    if (stack_size(p) > 0) {
      auto bids = LegalBids();
      movelist.insert(movelist.end(), bids.begin(), bids.end());
    }
    return movelist;
  }
  case GamePhase::kBidding: {
    std::vector<Action> movelist = LegalBids();
    movelist.insert(movelist.begin(), kActionPass);
    return movelist;
  }
  case GamePhase::kFlipping: {
    const Player p = CurrentPlayer();
    std::vector<Action> movelist;
    for (Player target = 0; target < num_players_; target++) {
      if (is_active(target) &&
          (flipped_stack_depth(target) < stack_size(target))) {
        movelist.push_back(flip_base_ + target);
      }
    }
    return movelist;
  }
  case GamePhase::kCardLoss: {
    const Player p = challenger_;
    if (has_rose(p) && has_skull(p)) {
      return {kActionDiscardRose, kActionDiscardSkull};
    } else if (has_skull(p)) {
      SPIEL_DCHECK_TRUE(has_only_skull(p));
      return {kActionDiscardSkull};
    } else {
      SPIEL_DCHECK_TRUE(has_only_roses(p));
      return {kActionDiscardRose};
    }
  }
  default:
    SpielFatalError("Invalid GamePhase in Skull");
  }
}

// =============================================================================
// Game Info Helpers
// =============================================================================

bool SkullState::has_rose(Player p) const {
  return is_active(p) && absl::linear_search(hands_[p].begin(), hands_[p].end(),
                                             CardType::kRose);
}

bool SkullState::has_skull(Player p) const {
  return is_active(p) && absl::linear_search(hands_[p].begin(), hands_[p].end(),
                                             CardType::kSkull);
}

Action SkullState::highest_safe_bid_or_pass(Player p) const {
  SPIEL_CHECK_TRUE(kActionBidBase ==
                   kActionPass); // Takes advantage of the Current action
                                 // mapping to simplify our logic
  int safe_bid = kActionBidBase;
  bool skull_found = false;

  for (int i = static_cast<int>(stacks_[p].size()) - 1; i >= 0; --i) {
    if (stacks_[p][i] == CardType::kSkull) {
      skull_found = true;
      return safe_bid;
    }
    safe_bid++;
  }

  if (!skull_found) {
    for (Player target = 0; target < num_players_; ++target) {
      if (target != p && is_active(target) && known_has_only_roses_[target]) {
        safe_bid += stacks_[target].size();
      }
    }
  }

  return safe_bid;
}

int SkullState::total_cards_on_table() const {
  int total = 0;
  for (Player p = 0; p < num_players_; ++p) {
    if (is_active(p))
      total += static_cast<int>(stacks_[p].size());
  }
  return total;
}

int SkullState::num_active_players() const {
  int count = 0;
  for (Player p = 0; p < num_players_; ++p) {
    if (is_active(p))
      ++count;
  }
  return count;
}

// =============================================================================
// Game Aciton Processing
// =============================================================================

void SkullState::DoApplyAction(Action action) {
  if (action == kActionPlaceRose) {
    stacks_[CurrentPlayer()].push_back(CardType::kRose);
    AdvanceToNextPlayer();

  } else if (action == kActionPlaceSkull) {
    stacks_[CurrentPlayer()].push_back(CardType::kSkull);
    AdvanceToNextPlayer();

  } else if (action == kActionPass) {
    AdvanceToNextPlayer();
    if (current_player_ == challenger_)
      BeginFlipping();

  } else if (IsActionBid(action)) {
    ApplyBidAction(action);

  } else if (IsActionFlip(action)) {
    ApplyFlipAction(FlipTargetFromAction(action));

  } else if (action == kActionDiscardRose) {
    DiscardCard(challenger_, CardType::kRose);

  } else if (action == kActionDiscardSkull) {
    DiscardCard(challenger_, CardType::kSkull);

  } else {
    SpielFatalError("Invalid Action");
  }
}

void SkullState::ApplyBidAction(Action action) {
  current_phase_ = GamePhase::kBidding;
  current_bid_ = action - kActionBidBase;
  challenger_ = CurrentPlayer();

  if (current_bid_ == total_cards_on_table()) {
    BeginFlipping();
  } else {
    AdvanceToNextPlayer();
  }
}

void SkullState::BeginFlipping() {
  SPIEL_DCHECK_EQ(CurrentPlayer(), challenger_);
  current_phase_ = GamePhase::kFlipping;
  while (current_phase_ == GamePhase::kFlipping &&
         total_cards_flipped_ < stack_size(challenger_))
    ApplyFlipAction(challenger_);
}

void SkullState::ApplyFlipAction(Player flip_target) {
  SPIEL_DCHECK_EQ(CurrentPlayer(), challenger_);
  total_cards_flipped_++;
  depth_flipped_[flip_target]++;
  CardType revealed_card = stacks_[flip_target].at(stack_size(flip_target) -
                                                   depth_flipped_[flip_target]);
  UpdateDerivedHandInfo(flip_target);

  if (revealed_card == CardType::kSkull) { // Bet Lost
    current_phase_ = GamePhase::kCardLoss;
    if (flip_target != challenger_) {
      current_player_ = kChancePlayerId;
    }
  } else if (total_cards_flipped_ >= current_bid_) { // Bet Won
    ++scores_[challenger_];
    first_player_ = challenger_;
    CheckForWin();
    if (!IsTerminal())
      StartNewRound();
  }
}

const absl::InlinedVector<CardType, kMaximumHandSize>
SkullState::InitialHand() const {
  absl::InlinedVector<CardType, kMaximumHandSize> hand(max_hand_size_ - 1,
                                                       CardType::kRose);
  hand.push_back(CardType::kSkull);
  return hand;
}

void SkullState::StartNewRound() {
  for (Player p = 0; p < num_players_; ++p) {
    stacks_[p].clear();
    depth_flipped_[p] = 0;
  }
  current_bid_ = 0;
  total_cards_flipped_ = 0;
  challenger_ = kInvalidPlayer;
  current_player_ = first_player_;
  current_phase_ = GamePhase::kPlacement;
  start_of_last_round_idx_ = history_.size();
}

void SkullState::AdvanceToNextPlayer() {
  SPIEL_CHECK_GE(CurrentPlayer(), 0);
  const Player start = current_player_;
  do {
    current_player_ = (current_player_ + 1) % num_players_;
    if (!is_active(current_player_))
      continue;
    return;
  } while (current_player_ != start);

  SpielFatalError("AdvanceToNextPlayer: no eligible player found.");
}

void SkullState::CheckForWin() {
  if (scores_[challenger_] >= wins_needed_) {
    winner_ = challenger_;
    current_player_ = kTerminalPlayerId;
    return;
  }
  if (num_active_players() == 1) {
    for (Player q = 0; q < num_players_; ++q) {
      if (is_active(q)) {
        winner_ = q;
        current_player_ = kTerminalPlayerId;
        return;
      }
    }
    SPIEL_CHECK_TRUE_WSI(winner_ != kInvalidPlayer,
                         "Eliminated, found no surviving player", *game_,
                         *this);
  }
}

void SkullState::DiscardCard(Player p, CardType type) {
  SPIEL_CHECK_TRUE(current_phase_ == GamePhase::kCardLoss);
  SPIEL_CHECK_GE(p, kDefaultPlayerId);

  auto &hand = hands_[p];
  auto it = std::find(hand.begin(), hand.end(), type);
  SPIEL_CHECK_TRUE(it != hand.end()); // must be found
  hand.erase(it);

  if (!is_active(p))
    CheckForWin();
  else
    UpdateDerivedHandInfo(p);

  if (IsTerminal())
    return;

  if (!is_active(challenger_)) {
    current_player_ = challenger_;
    AdvanceToNextPlayer();
    first_player_ = current_player_;
  } else {
    first_player_ = challenger_;
  }
  StartNewRound();
}

void SkullState::UpdateDerivedHandInfo(Player p) {
  if (hand_size(p) <= 0)
    return;
  if (known_has_only_roses(p) || known_has_only_skull(p))
    return;
  if (current_phase_ == GamePhase::kCardLoss) {
    known_has_skull_[p] = false;
    known_has_rose_[p] = false;
    return;
  }

  SPIEL_CHECK_TRUE(current_phase_ == GamePhase::kFlipping);
  if (known_has_rose(p) && known_has_skull(p))
    return;
  bool rose_revealed = false;
  bool skull_revealed = false;
  for (int depth = 1; depth <= flipped_stack_depth(p); ++depth) {
    CardType card = stacks_[p].at(stack_size(p) - depth);
    if (card == CardType::kRose) {
      rose_revealed = true;
      known_has_rose_[p] = true;
    }
    if (card == CardType::kSkull) {
      skull_revealed = true;
      known_has_skull_[p] = true;
    }
  }
  if (flipped_stack_depth(p) == hand_size(p)) {
    if (!rose_revealed)
      known_has_only_skull_[p] = true;
    if (!skull_revealed)
      known_has_only_roses_[p] = true;
  }
  SPIEL_CHECK_TRUE((!known_has_rose_[p]) || has_rose(p));
  SPIEL_CHECK_TRUE((!known_has_only_roses_[p]) || has_only_roses(p));
  SPIEL_CHECK_TRUE((!known_has_skull_[p]) || has_skull(p));
  SPIEL_CHECK_TRUE((!known_has_only_skull_[p]) || has_only_skull(p));
}

// =============================================================================
// SkullGame Methods
// =============================================================================

std::unique_ptr<State> SkullGame::NewInitialState() const {
  return std::make_unique<SkullState>(shared_from_this());
}

constexpr int SkullGame::CalcMaxGameLength(int num_players, int max_hand_size,
                                           int wins_needed) {

  int max_total_cards = num_players * max_hand_size;
  int max_rounds_without_card_loss = num_players * (wins_needed - 1);
  int max_round_one_length =
      CalcMaxRoundLength(max_total_cards, num_players, false);
  int early_rounds_length = max_round_one_length * max_rounds_without_card_loss;
  int max_total_turns;
  int card_loss_rounds_length = 0;
  for (max_total_turns = max_total_cards; max_total_turns > num_players;
       max_total_turns--) {
    card_loss_rounds_length +=
        CalcMaxRoundLength(max_total_turns, num_players, true);
  }
  int players_left = num_players;
  while (max_total_turns > 1) {
    SPIEL_CHECK_TRUE(max_total_turns == players_left);
    card_loss_rounds_length +=
        CalcMaxRoundLength(max_total_turns, players_left, true);
    max_total_turns--;
    players_left--;
  }
  int out = early_rounds_length + card_loss_rounds_length;
  return out;
}

std::vector<int> SkullGame::InformationStateTensorShape() const {
  auto out =
      TensorShapeFromIIGObsType(kInfoStateObsType, obs_public_derived_info_,
                                obs_egocentric_, obs_partial_recall_);
  return out;
}

std::vector<int> SkullGame::ObservationTensorShape() const {
  auto out =
      TensorShapeFromIIGObsType(kDefaultObsType, obs_public_derived_info_,
                                obs_egocentric_, obs_partial_recall_);
  return out;
}

std::vector<int> SkullGame::TensorShapeFromIIGObsType(
    IIGObservationType iig_obs_type, bool include_public_derived_info,
    bool egocentric, bool partial_recall) const {
  int count = 0;

  if (iig_obs_type.perfect_recall) {
    count += num_players_; // Which player relative to us is first to move;
    const int action_width = NumDistinctActions() + 2;
    if (partial_recall) {
      // see `WritePartialHistory()`
      const int max_round_len =
          CalcMaxRoundLength(MaxTotalCards(), num_players_, true);
      count += max_round_len * action_width;
    } else {
      // see `WriteHistory()`
      const int max_history = MaxGameLength();
      count += max_history * action_width;
      return {count};
    }
  }

  SPIEL_DCHECK_TRUE(partial_recall || !iig_obs_type.perfect_recall);
  if (iig_obs_type.public_info) {
    { // see `WriteGlobalPublicInfo()`
      count += kGamePhaseCount;
      count += num_players_;
      count += num_players_;
      count += MaxTotalCards();
    }
    { // see `WritePerPlayerPublicInfo()`
      count += num_players_ * (wins_needed_ + 1);
      count += num_players_ * (max_hand_size_ + 1);
      constexpr int kStackCategories = kCardTypeCount + 1;
      count += num_players_ * max_hand_size_ * kStackCategories;
      if (include_public_derived_info) {
        count += num_players_ * 4;
      }
    }
  }

  if (iig_obs_type.private_info == PrivateInfoType::kSinglePlayer) {
    count += max_hand_size_ * kCardTypeCount * 2;
  } else if (iig_obs_type.private_info == PrivateInfoType::kAllPlayers) {
    count += max_hand_size_ * kCardTypeCount * 2 * num_players_;
  }

  return {count};
}

/*
std::shared_ptr<Observer>
SkullGame::MakeObserver(absl::optional<IIGObservationType> iig_obs_type,
                        const GameParameters &params) const {

  bool public_derived_info = obs_public_derived_info_;
  bool egocentric = obs_egocentric_;
  bool partial_recall = obs_partial_recall_;

  if (auto it = params.find("public_derived_info"); it != params.end())
    public_derived_info = it->second.value<bool>();
  if (auto it = params.find("egocentric"); it != params.end())
    egocentric = it->second.value<bool>();
  if (auto it = params.find("partial_recall"); it != params.end())
    partial_recall = it->second.value<bool>();

  return std::make_shared<SkullObserver>(iig_obs_type.value_or(kDefaultObsType),
                                         public_derived_info, egocentric,
                                         partial_recall);
}
*/

SkullObserver::SkullObserver(IIGObservationType iig_obs_type,
                             bool include_public_derived_info, bool egocentric,
                             bool partial_recall)
    : Observer(/*has_string=*/true, /*has_tensor=*/true),
      iig_obs_type_(iig_obs_type),
      include_public_derived_info_(include_public_derived_info),
      egocentric_(egocentric), partial_recall_(partial_recall) {}

void SkullObserver::WriteTensor(const State &observed_state, int player,
                                Allocator *allocator) const {
  const SkullState &state =
      open_spiel::down_cast<const SkullState &>(observed_state);
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, state.num_players_);

  if (iig_obs_type_.perfect_recall) {
    {
      auto out = allocator->Get("rel_starting_player", {state.num_players_});
      out.at(AbsoluteToRelativePlayer(kDefaultPlayerId, player,
                                      state.num_players_)) = 1.0;
    }
    if (partial_recall_) {
      WritePartialHistory(player, state, allocator);
    } else {
      WriteHistory(player, state, allocator);
      return; // Early exit, all information can be derived from full history
    }
  }

  if (iig_obs_type_.public_info) {
    WriteGlobalPublicInfo(player, state, allocator);
    WritePerPlayerPublicInfo(player, true, state, allocator);
  }

  if (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer) {
    WriteOnePlayerPrivateInfo(player, state, allocator);
  } else if (iig_obs_type_.private_info == PrivateInfoType::kAllPlayers) {
    WriteAllPlayersPrivateInfo(player, state, allocator);
  }
}

inline void SkullObserver::NextPlayer(int *count, Player *player,
                                      int num_players) const {
  *count += 1;
  *player = (*player + 1) % num_players;
}
inline Player SkullObserver::AbsoluteToRelativePlayer(Player player,
                                                      Player obs_player,
                                                      int num_players) const {
  if (!egocentric_)
    return player;

  if ((player < 0) || (obs_player < 0))
    return player;

  return (player + (num_players - obs_player)) % num_players;
}

void SkullObserver::WriteGlobalPublicInfo(Player obs_player,
                                          const SkullState &state,
                                          Allocator *allocator) const {
  {
    auto out = allocator->Get("phase", {kGamePhaseCount});
    out.at(static_cast<int>(state.current_phase_)) = 1.0;
  }
  {
    auto out = allocator->Get("current_player", {state.num_players_});
    if (state.current_player_ >= 0) {
      out.at(AbsoluteToRelativePlayer(state.current_player_, obs_player,
                                      state.num_players_)) = 1.0;
    }
  }
  {
    auto out = allocator->Get("challenger", {state.num_players_});
    if (state.challenger_ >= 0) {
      out.at(AbsoluteToRelativePlayer(state.challenger_, obs_player,
                                      state.num_players_)) = 1.0;
    }
  }
  {
    auto out = allocator->Get("current_bid", {state.max_total_cards_});
    if (state.current_bid_ > 0)
      out.at(state.current_bid_ - 1) = 1.0;
  }
}

void SkullObserver::WritePerPlayerPublicInfo(Player obs_player,
                                             bool skip_observing_player,
                                             const SkullState &state,
                                             Allocator *allocator) const {
  auto score_out = // we always write score, even for obs_player.
      allocator->Get("score", {state.num_players_, state.wins_needed_ + 1});

  int num_players_to_write =
      skip_observing_player ? state.num_players_ - 1 : state.num_players_;

  auto hand_size_out = allocator->Get(
      "hand_size", {num_players_to_write, state.max_hand_size_ + 1});

  //  Each stack slot is represented in the following way
  //  (bot) [?, ?, R, -] (top) <-- Example Stack.
  //   [0]  [0, 0, 1, 0] <- is the card at depth X REVEALED (as a rose)
  //   [1]  [1, 1, 1, 0] <- is the card at depth X PLACED at all?
  //
  // theoretically, we don't have to consider viewing board-states with a
  // skull revealed, because the only time there is a revealed skull and we
  // need to make a decision is choosing which card to discard, since
  // revealing the skull ends the round.
  // The only time one makes a decision in a state where a skull is revealed, is
  // when one revealed one's own skull, and so can be deduced by simply being
  // within the phase 'card_loss'
  //
  // right now, during the reveal the stack would look like this, but only for
  // time of the game_phase = card_loss, which can be interpreted as
  // last-revealed = skull
  //  (bot) [?, S, R, -] (top) <-- Example Stack.
  //   [0]  [0, 1, 1, 0] <- is the card at depth X REVEALED
  //   [1]  [1, 1, 1, 0] <- is the card at depth X PLACED at all?

  constexpr int kStackCategories = kCardTypeCount;
  auto stack_out =
      allocator->Get("public_stack", {num_players_to_write,
                                      state.max_hand_size_, kStackCategories});

  Player p = obs_player;
  for (int n = 0; n < state.num_players_;
       NextPlayer(&n, &p, state.num_players_)) {

    score_out.at(n, state.scores_[p]) = 1.0;

    // TODO: CLEANUP `skip_observing_player` logic, Tie it into `egocentric`
    if (skip_observing_player) {
      if (p == obs_player)
        continue;
      n--;
    }

    hand_size_out.at(n, state.hand_size(p)) = 1.0;

    const auto &stack = state.stacks_[p];
    const int size = state.stack_size(p);
    const int flipped = state.depth_flipped_[p];

    for (int c = 0; c < size; ++c) {
      if (c >= size - flipped) {
        stack_out.at(n, c, 0) = 1.0; // revealed
      }
      stack_out.at(n, c, kStackCategories - 1) = 1.0; // placed
    }
    if (skip_observing_player) {
      n++;
    }
  }

  if (include_public_derived_info_) {
    auto derived_out =
        allocator->Get("derived_public_info", {state.num_players_, 4});
    Player p = obs_player;
    for (int n = 0; n < state.num_players_;
         NextPlayer(&n, &p, state.num_players_)) {
      int i = 0;
      derived_out.at(n, i) = state.known_has_rose_[p] ? 1.0 : 0.0;
      i++;
      derived_out.at(n, i) = state.known_has_only_roses_[p] ? 1.0 : 0.0;
      i++;
      derived_out.at(n, i) = state.known_has_skull_[p] ? 1.0 : 0.0;
      i++;
      derived_out.at(n, i) = state.known_has_only_skull_[p] ? 1.0 : 0.0;
    }
  }
};
void SkullObserver::WriteOnePlayerPrivateInfo(Player player,
                                              const SkullState &state,
                                              Allocator *allocator) const {
  // NOTE: Room for space saving: make hand 'sorted' eg:
  // for hand [S R R -]
  // rather than:
  //         1 0 0 0 [skull no no no],
  //         0 1 1 0 [no rose rose no]
  // its just:
  //         1 1 1 0 [slot1, slot2, slot3, no] meaning [skull, rose, rose none]
  auto hand_out =
      allocator->Get("private_hand", {state.max_hand_size_, kCardTypeCount});
  auto stack_out =
      allocator->Get("private_stack", {state.max_hand_size_, kCardTypeCount});

  const auto &hand = state.hands_[player];
  const auto &stack = state.stacks_[player];
  for (int c = 0; c < hand.size(); ++c) {
    hand_out.at(c, static_cast<int>(hand[c])) = 1.0;
  }
  for (int c = 0; c < stack.size(); ++c) {
    stack_out.at(c, static_cast<int>(stack[c])) = 1.0;
  }
}

void SkullObserver::WriteAllPlayersPrivateInfo(Player obs_player,
                                               const SkullState &state,
                                               Allocator *allocator) const {
  auto hand_out =
      allocator->Get("private_hand", {state.num_players_, state.max_hand_size_,
                                      kCardTypeCount});
  auto stack_out =
      allocator->Get("private_stack", {state.num_players_, state.max_hand_size_,
                                       kCardTypeCount});

  Player p = obs_player;
  for (int n = 0; n < state.num_players_;
       NextPlayer(&n, &p, state.num_players_)) {

    const auto &hand = state.hands_[p];
    const auto &stack = state.stacks_[p];
    for (int c = 0; c < hand.size(); ++c) {
      hand_out.at(n, c, static_cast<int>(hand[c])) = 1.0;
    }
    for (int c = 0; c < stack.size(); ++c) {
      stack_out.at(n, c, static_cast<int>(stack[c])) = 1.0;
    }
  }
}

void SkullObserver::WritePartialHistory(Player obs_player,
                                        const SkullState &state,
                                        Allocator *allocator) const {
  const int num_distinct_actions = state.GetGame()->NumDistinctActions();
  const int action_width =
      num_distinct_actions + 2; // visible actions + hidden placement/discard
  const int max_round_len =
      CalcMaxRoundLength(state.max_total_cards_, state.num_players_, true);
  auto action_out =
      allocator->Get("history_actions", {max_round_len, action_width});

  size_t start_index = state.BeginningOfMostRecentRound();
  const auto &history = state.FullHistory();
  for (size_t i = start_index; i < history.size(); ++i) {
    const SkullState::PlayerAction &obs_player_action = history[i];
    Action a = obs_player_action.action;
    Player p = obs_player_action.player;
    action_out.at(i - start_index,
                  ObservedActionIndex(p, a, state, obs_player)) = 1.0;
  }
}

void SkullObserver::WriteHistory(Player obs_player, const SkullState &state,
                                 Allocator *allocator) const {
  const int num_distinct_actions = state.GetGame()->NumDistinctActions();
  const int action_width =
      num_distinct_actions + 2; // visible actions + hidden placement/discard
  const int max_history = state.GetGame()->MaxGameLength();
  auto action_out =
      allocator->Get("history_actions", {max_history, action_width});

  const auto &history = state.FullHistory();
  for (int i = 0; i < history.size() && i < max_history; ++i) {
    const Player p = history[i].player;
    const Action a = history[i].action;
    action_out.at(i, ObservedActionIndex(p, a, state, obs_player)) = 1.0;
  }
}

int SkullObserver::ObservedActionIndex(Player acting_player, Action action,
                                       const SkullState &state,
                                       Player obs_player) const {
  const int kHiddenPlacement = state.GetGame()->NumDistinctActions();
  const int kHiddenDiscard = kHiddenPlacement + 1;

  bool is_placement_action =
      action == kActionPlaceRose || action == kActionPlaceSkull;

  bool sees_private_card_placement =
      (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer &&
       acting_player == obs_player) ||
      iig_obs_type_.private_info == PrivateInfoType::kAllPlayers;

  if (is_placement_action) {
    if (sees_private_card_placement)
      return action;
    return kHiddenPlacement;
  }

  bool is_fully_public_action = action == kActionPass ||
                                state.IsActionBid(action) ||
                                state.IsActionFlip(action);

  if (is_fully_public_action)
    return action;

  bool is_discard_action =
      action == kActionDiscardRose || action == kActionDiscardSkull;

  if (is_discard_action) {
    if (obs_player == state.challenger_)
      return action;
    return kHiddenDiscard;
  }
  SpielFatalError("Uknown Action");
}

std::string SkullObserver::StringFrom(const State &observed_state,
                                      int obs_player) const {
  const SkullState &state =
      open_spiel::down_cast<const SkullState &>(observed_state);
  SPIEL_CHECK_LT(obs_player, state.num_players_);
  SPIEL_CHECK_GE(obs_player, 0);

  std::string result;

  if (iig_obs_type_.public_info) {
    absl::StrAppend(
        &result,
        "PUBLIC INFO:\nphase=", SkullState::PhaseToString(state.current_phase_),
        " move=", state.MoveNumber(), " current_player=", state.current_player_,
        " challenger=", state.challenger_, " bid=", state.current_bid_,
        " first=", state.first_player_, " flipped=", state.total_cards_flipped_,
        "\n");

    for (Player p = 0; p < state.num_players_; ++p) {
      absl::StrAppend(&result, "Player: ", p, " active=", state.is_active(p),
                      " score=", state.scores_[p], "/", state.wins_needed_);
      if (p == obs_player)
        continue;
      absl::StrAppend(&result, " hand_size=", state.hand_size(p), "/",
                      state.max_hand_size_,
                      " stack=", state.StackToString(p, false));

      if (include_public_derived_info_) {
        absl::StrAppend(&result, " derived_info={",
                        state.DerivedPublicInfoString(p), "}\n");
      }
    }
  }

  if (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer) {
    absl::StrAppend(&result, "PRIVATE INFO:\nPlayer: ", obs_player,
                    state.HandToString(obs_player),
                    state.StackToString(obs_player, true));
  } else if (iig_obs_type_.private_info == PrivateInfoType::kAllPlayers) {
    absl::StrAppend(&result, "PRIVATE INFO:\n");
    for (Player p = 0; p < state.num_players_; ++p)
      absl::StrAppend(&result, "Player: ", p, state.HandToString(p),
                      state.StackToString(p, true), "\n");
  }

  if (iig_obs_type_.perfect_recall) {
    absl::StrAppend(&result, "HISTORY:\n");
    const auto &history = state.FullHistory();

    if (partial_recall_) {

      for (size_t i = state.BeginningOfMostRecentRound(); i < history.size();
           ++i) {
        const SkullState::PlayerAction &obs_player_action = history[i];
        Action a = obs_player_action.action;
        Player p = obs_player_action.player;
        absl::StrAppend(&result, ObservedActionString(p, a, state, obs_player));
      }

    } else {
      for (const SkullState::PlayerAction &obs_player_action : history) {
        Action a = obs_player_action.action;
        Player p = obs_player_action.player;
        absl::StrAppend(&result, ObservedActionString(p, a, state, obs_player));
      }
    }
  }
  return result;
}

inline const std::string
SkullObserver::ObservedActionString(Player acting_player, Action action,
                                    const SkullState &state,
                                    Player obs_player) const {
  SPIEL_CHECK_TRUE(iig_obs_type_.perfect_recall);

  bool is_placement_action =
      action == kActionPlaceRose || action == kActionPlaceSkull;

  bool sees_private_card_placement =
      (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer &&
       acting_player == obs_player) ||
      iig_obs_type_.private_info == PrivateInfoType::kAllPlayers;

  if (is_placement_action) {
    if (sees_private_card_placement)
      return state.ActionToShortString(acting_player, action);

    return "pc"; // "place card"
  }

  bool is_fully_public_action = action == kActionPass ||
                                state.IsActionBid(action) ||
                                state.IsActionFlip(action);

  if (is_fully_public_action) {
    return state.ActionToShortString(acting_player, action);
  }

  bool is_discard_action =
      action == kActionDiscardRose || action == kActionDiscardSkull;

  if (is_discard_action) {
    if (obs_player == state.challenger_)
      return state.ActionToShortString(acting_player, action);

    return acting_player == kChancePlayerId ? "rng:dc" : "dc"; // discard card
  }
  SpielFatalError("Uknown Action");
  return "?";
}

} // namespace skull
} // namespace open_spiel
