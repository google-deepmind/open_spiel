// Copyright 2026 DeepMind Technologies Limited
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

#include "open_spiel/games/seep/seep.h"

#include <algorithm>
#include <array>
#include <memory>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace seep {
namespace {

constexpr char kRankChars[] = "A23456789TJQK";
constexpr char kSuitChars[] = "CDHS";

const GameType kGameType{
    /*short_name=*/"seep",
    /*long_name=*/"Seep",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/kNumPlayers,
    /*min_num_players=*/kNumPlayers,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{},
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new SeepGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

struct SubsetOption {
  int mask = 0;
  int points = 0;
  int card_count = 0;
};

// Enumerates all non-empty subsets of uncemented floor piles (excluding any
// piles already marked in `excluded_mask`) whose values sum to `target_sum`.
// When `allow_multi_card_house_with_others` is false, a multi-card uncemented
// house can only form a single-pile subset by itself (raising the house
// directly with a hand card) and cannot be combined with other floor piles to
// form the primary build sum.
std::vector<SubsetOption> FindUncementedSubsetsWithSum(
    const std::vector<House>& floor_piles, int target_sum, int excluded_mask,
    bool allow_multi_card_house_with_others = true,
    int forbidden_owner_mask = 0) {
  std::vector<SubsetOption> options;
  if (target_sum <= 0) return options;

  const int n = static_cast<int>(floor_piles.size());
  auto dfs = [&](auto& self, int idx, int current_sum, int current_mask,
                 int current_points, int current_cards, int pile_count,
                 bool has_multi_card_house) -> void {
    if (current_sum == target_sum) {
      if (allow_multi_card_house_with_others || !has_multi_card_house ||
          pile_count == 1) {
        options.push_back({current_mask, current_points, current_cards});
      }
      return;
    }
    if (idx >= n || current_sum > target_sum) return;

    // Option 1: include pile `idx` if eligible.
    // A player cannot break their own kachha (uncemented multi-card) house.
    const bool is_multi_card = floor_piles[idx].cards.size() > 1;
    const bool is_forbidden_own_house =
        is_multi_card &&
        ((floor_piles[idx].owners_mask & forbidden_owner_mask) != 0);
    if (((excluded_mask >> idx) & 1) == 0 && !floor_piles[idx].is_cemented &&
        !is_forbidden_own_house &&
        current_sum + floor_piles[idx].value <= target_sum) {
      self(self, idx + 1, current_sum + floor_piles[idx].value,
           current_mask | (1 << idx),
           current_points + floor_piles[idx].PointValue(),
           current_cards + static_cast<int>(floor_piles[idx].cards.size()),
           pile_count + 1, has_multi_card_house || is_multi_card);
    }
    // Option 2: skip pile `idx`.
    self(self, idx + 1, current_sum, current_mask, current_points,
         current_cards, pile_count, has_multi_card_house);
  };

  dfs(dfs, 0, 0, 0, 0, 0, 0, false);
  return options;
}

// Finds all maximal disjoint collections of subsets from `options`, sorted by
// (subset_count, total_points, total_cards) descending and mask ascending.
// Because only maximal disjoint collections are returned, any disjoint subset
// on the floor that can be included MUST be included (e.g., if both 8+2 and 7+3
// are on the floor, both must be taken together), whereas mutually exclusive
// overlapping combinations become distinct choices (choice_index = 0, 1, ...).
struct DisjointPartitionResult {
  int mask = 0;
  int points = 0;
  int card_count = 0;
  int subset_count = 0;
};

std::vector<DisjointPartitionResult> FindMaximalDisjointSubsets(
    const std::vector<SubsetOption>& options, int initial_excluded_mask) {
  std::vector<DisjointPartitionResult> raw_results;
  const int m = static_cast<int>(options.size());

  auto search = [&](auto& self, int idx, int used_mask, int pts, int cards,
                    int count) -> void {
    bool can_extend = false;
    for (int j = 0; j < m; ++j) {
      if ((used_mask & options[j].mask) == 0) {
        can_extend = true;
        break;
      }
    }
    if (!can_extend) {
      raw_results.push_back(
          {used_mask & ~initial_excluded_mask, pts, cards, count});
      return;
    }
    for (int i = idx; i < m; ++i) {
      if ((used_mask & options[i].mask) == 0) {
        self(self, i + 1, used_mask | options[i].mask, pts + options[i].points,
             cards + options[i].card_count, count + 1);
      }
    }
  };

  search(search, 0, initial_excluded_mask, 0, 0, 0);

  std::vector<DisjointPartitionResult> unique_results;
  for (const auto& r : raw_results) {
    bool found = false;
    for (auto& u : unique_results) {
      if (u.mask == r.mask) {
        found = true;
        if (std::tie(r.subset_count, r.points, r.card_count) >
            std::tie(u.subset_count, u.points, u.card_count)) {
          u = r;
        }
        break;
      }
    }
    if (!found) {
      unique_results.push_back(r);
    }
  }

  absl::c_sort(unique_results, [](const DisjointPartitionResult& a,
                                  const DisjointPartitionResult& b) {
    if (std::tie(a.subset_count, a.points, a.card_count) !=
        std::tie(b.subset_count, b.points, b.card_count)) {
      return std::tie(a.subset_count, a.points, a.card_count) >
             std::tie(b.subset_count, b.points, b.card_count);
    }
    return a.mask < b.mask;
  });

  if (unique_results.size() > kMaxSubsetChoices) {
    unique_results.resize(kMaxSubsetChoices);
  }
  return unique_results;
}

// Computes all maximal bitmasks of floor piles that can be captured when
// playing a card of `card_value` with `Intent::kCapture`.
std::vector<int> GetCaptureMasks(const std::vector<House>& floor_piles,
                                 int card_value) {
  int direct_mask = 0;
  const int n = static_cast<int>(floor_piles.size());
  for (int i = 0; i < n; ++i) {
    if (floor_piles[i].value == card_value) {
      direct_mask |= (1 << i);
    }
  }

  std::vector<SubsetOption> options =
      FindUncementedSubsetsWithSum(floor_piles, card_value, direct_mask);
  std::vector<DisjointPartitionResult> maximal =
      FindMaximalDisjointSubsets(options, direct_mask);

  std::vector<int> masks;
  for (const auto& r : maximal) {
    int mask = direct_mask | r.mask;
    if (mask != 0) {
      masks.push_back(mask);
    }
  }
  return masks;
}

struct BuildPlan {
  int merged_pile_mask = 0;
  bool is_cemented = false;
  int group_count = 0;
  int total_points = 0;
  int total_cards = 0;
};

// Computes all maximal floor pile merge plans when playing `card` with target
// `house_value` in 9..13.
std::vector<BuildPlan> GetBuildHousePlans(const std::vector<House>& floor_piles,
                                          int card, int house_value,
                                          Player player = kInvalidPlayer) {
  std::vector<BuildPlan> plans;
  const int card_value = CardCaptureValue(card);
  if (house_value < kMinBidValue || house_value > kMaxBidValue ||
      card_value > house_value) {
    return plans;
  }

  if (card_value == house_value) {
    // Playing a card of value H to build H cements existing pile(s)/subsets of
    // value H on the floor.
    std::vector<int> masks = GetCaptureMasks(floor_piles, house_value);
    for (int mask : masks) {
      BuildPlan plan;
      plan.merged_pile_mask = mask;
      plan.is_cemented = true;
      plans.push_back(plan);
    }
    return plans;
  }

  const int n = static_cast<int>(floor_piles.size());
  int existing_house_mask = 0;
  int existing_house_count = 0;
  for (int i = 0; i < n; ++i) {
    if (floor_piles[i].value == house_value) {
      existing_house_mask |= (1 << i);
      ++existing_house_count;
    }
  }

  // card_value < house_value: must combine `card` with a subset of uncemented
  // floor piles summing to `house_value - card_value`. An uncemented house can
  // only be raised directly by `card` (single-pile primary subset), never by
  // combining an uncemented house with loose floor cards, and a player can
  // never break their own uncemented house (though their partner or an
  // opponent can).
  const int needed = house_value - card_value;
  const int forbidden_owner_mask =
      (player >= 0 && player < kNumPlayers) ? (1 << player) : 0;
  std::vector<SubsetOption> primary_options = FindUncementedSubsetsWithSum(
      floor_piles, needed, existing_house_mask,
      /*allow_multi_card_house_with_others=*/false, forbidden_owner_mask);
  if (primary_options.empty()) {
    return plans;
  }

  for (const SubsetOption& primary : primary_options) {
    int base_mask = existing_house_mask | primary.mask;
    std::vector<SubsetOption> extra_options =
        FindUncementedSubsetsWithSum(floor_piles, house_value, base_mask);
    std::vector<DisjointPartitionResult> extra_results =
        FindMaximalDisjointSubsets(extra_options, base_mask);

    for (const DisjointPartitionResult& extra : extra_results) {
      int mask = base_mask | extra.mask;
      int total_points = primary.points + extra.points;
      int total_cards = primary.card_count + extra.card_count;
      int group_count = 1 + existing_house_count + extra.subset_count;
      bool is_cemented = (group_count > 1);

      bool found = false;
      for (BuildPlan& existing : plans) {
        if (existing.merged_pile_mask == mask) {
          found = true;
          if (std::tie(group_count, total_points, total_cards) >
              std::tie(existing.group_count, existing.total_points,
                       existing.total_cards)) {
            existing.group_count = group_count;
            existing.total_points = total_points;
            existing.total_cards = total_cards;
            existing.is_cemented = is_cemented;
          }
          break;
        }
      }
      if (!found) {
        plans.push_back(BuildPlan{mask, is_cemented, group_count, total_points,
                                  total_cards});
      }
    }
  }

  absl::c_sort(plans, [](const BuildPlan& a, const BuildPlan& b) {
    if (std::tie(a.group_count, a.total_points, a.total_cards) !=
        std::tie(b.group_count, b.total_points, b.total_cards)) {
      return std::tie(a.group_count, a.total_points, a.total_cards) >
             std::tie(b.group_count, b.total_points, b.total_cards);
    }
    return a.merged_pile_mask < b.merged_pile_mask;
  });

  if (plans.size() > kMaxSubsetChoices) {
    plans.resize(kMaxSubsetChoices);
  }
  return plans;
}

}  // namespace

std::string PhaseToString(Phase phase) {
  switch (phase) {
    case Phase::kDealInitial:
      return "DealInitial";
    case Phase::kBid:
      return "Bid";
    case Phase::kDealFloor:
      return "DealFloor";
    case Phase::kDealRemainder:
      return "DealRemainder";
    case Phase::kPlay:
      return "Play";
    case Phase::kGameOver:
      return "GameOver";
  }
  SpielFatalError("Unknown phase.");
}

std::string IntentToString(Intent intent) {
  switch (intent) {
    case Intent::kThrow:
      return "Throw";
    case Intent::kCapture:
      return "Capture";
    case Intent::kBuild9:
      return "Build9";
    case Intent::kBuild10:
      return "Build10";
    case Intent::kBuild11:
      return "Build11";
    case Intent::kBuild12:
      return "Build12";
    case Intent::kBuild13:
      return "Build13";
  }
  SpielFatalError("Unknown intent.");
}

std::ostream& operator<<(std::ostream& os, Suit suit) {
  return os << kSuitChars[static_cast<int>(suit)];
}

std::ostream& operator<<(std::ostream& os, Phase phase) {
  return os << PhaseToString(phase);
}

std::ostream& operator<<(std::ostream& os, Intent intent) {
  return os << IntentToString(intent);
}

int CardPointValue(int card) {
  SPIEL_CHECK_GE(card, 0);
  SPIEL_CHECK_LT(card, kNumCards);
  const Suit suit = CardSuit(card);
  const int capture_val = CardCaptureValue(card);
  if (suit == Suit::kSpades) {
    return capture_val;  // 1..13 (sum = 91)
  }
  if (capture_val == 1) {
    return 1;  // Non-Spade Aces = 1 point each (3 total)
  }
  if (suit == Suit::kDiamonds && capture_val == 10) {
    return 6;  // 10 of Diamonds = 6 points
  }
  return 0;
}

std::string CardString(int card) {
  SPIEL_CHECK_GE(card, 0);
  SPIEL_CHECK_LT(card, kNumCards);
  return {kRankChars[CardRank(card)],
          kSuitChars[static_cast<int>(CardSuit(card))]};
}

int CardFromString(absl::string_view str) {
  if (str.size() != 2) return kInvalidAction;
  const char rank_char = str[0];
  const char suit_char = str[1];
  int rank = -1;
  for (int r = 0; r < kNumRanks; ++r) {
    if (kRankChars[r] == rank_char) {
      rank = r;
      break;
    }
  }
  int suit = -1;
  for (int s = 0; s < kNumSuits; ++s) {
    if (kSuitChars[s] == suit_char) {
      suit = s;
      break;
    }
  }
  if (rank < 0 || suit < 0) return kInvalidAction;
  return Card(static_cast<Suit>(suit), rank);
}

int House::PointValue() const {
  int total = 0;
  for (int c : cards) {
    total += CardPointValue(c);
  }
  return total;
}

std::string House::ToString() const {
  if (cards.size() == 1 && !is_cemented &&
      CardCaptureValue(cards[0]) == value) {
    return CardString(cards[0]);
  }
  std::vector<std::string> card_strs;
  card_strs.reserve(cards.size());
  for (int c : cards) {
    card_strs.push_back(CardString(c));
  }
  return absl::StrCat(value, is_cemented ? "P[" : "[",
                      absl::StrJoin(card_strs, ","), "]");
}

SeepState::SeepState(std::shared_ptr<const Game> game) : State(game) {
  card_dealt_.fill(false);
}

Player SeepState::CurrentPlayer() const {
  if (phase_ == Phase::kGameOver) return kTerminalPlayerId;
  if (phase_ == Phase::kDealInitial || phase_ == Phase::kDealFloor ||
      phase_ == Phase::kDealRemainder) {
    return kChancePlayerId;
  }
  return current_player_;
}

std::string SeepState::ActionToString(Player player, Action action_id) const {
  if (player == kChancePlayerId) {
    return absl::StrCat("Deal:", CardString(static_cast<int>(action_id)));
  }
  if (action_id >= kBidActionBase && action_id < kPlayActionBase) {
    return absl::StrCat("Bid:", BidValueFromAction(action_id));
  }
  if (action_id >= kPlayActionBase && action_id < kNumDistinctActions) {
    auto [card, intent] = DecodePlayAction(action_id);
    int choice_index = DecodePlayActionChoiceIndex(action_id);
    if (choice_index == 0) {
      return absl::StrCat(CardString(card), ":", IntentToString(intent));
    }
    return absl::StrCat(CardString(card), ":", IntentToString(intent), ":",
                        choice_index);
  }
  return absl::StrCat("InvalidAction(", action_id, ")");
}

std::string SeepState::ToString() const {
  std::vector<std::string> floor_strs;
  floor_strs.reserve(floor_piles_.size());
  for (const House& h : floor_piles_) {
    floor_strs.push_back(h.ToString());
  }
  std::string result =
      absl::StrFormat("Phase: %s | CurrentPlayer: %d | Bid: %d | Floor: [%s]\n",
                      PhaseToString(phase_), CurrentPlayer(), bid_value_,
                      absl::StrJoin(floor_strs, " "));
  for (Player p = 0; p < kNumPlayers; ++p) {
    std::vector<std::string> hand_strs;
    hand_strs.reserve(hands_[p].size());
    for (int c : hands_[p]) {
      hand_strs.push_back(CardString(c));
    }
    absl::StrAppendFormat(&result, "P%d Hand: [%s]\n", p,
                          absl::StrJoin(hand_strs, " "));
  }
  const auto pts = TeamCardPoints();
  const auto totals = TeamTotalScores();
  absl::StrAppendFormat(&result,
                        "Team 0 (P0,P2): %d card pts, %d seeps, %d total | "
                        "Team 1 (P1,P3): %d card pts, %d seeps, %d total",
                        pts[0], team_seeps_[0], totals[0], pts[1],
                        team_seeps_[1], totals[1]);
  return result;
}

std::vector<double> SeepState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>(kNumPlayers, 0.0);
  }
  const auto totals = TeamTotalScores();
  const double diff = static_cast<double>(totals[0] - totals[1]);
  return {diff, -diff, diff, -diff};
}

std::string SeepState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, kNumPlayers);
  std::string str = ObservationString(player);
  std::vector<std::string> history_strs;
  for (const PlayerAction& pa : FullHistory()) {
    if (pa.player == kChancePlayerId) continue;
    history_strs.push_back(absl::StrCat("P", pa.player, ":",
                                        ActionToString(pa.player, pa.action)));
  }
  absl::StrAppend(&str, "\nPublicMoves: [", absl::StrJoin(history_strs, " "),
                  "]");
  return str;
}

std::string SeepState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, kNumPlayers);
  return ToObservationStruct(player)->ToJson();
}

void SeepState::ObservationTensor(Player player,
                                  absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, kNumPlayers);
  SPIEL_CHECK_EQ(values.size(), kObservationTensorSize);
  std::fill(values.begin(), values.end(), 0.0f);

  int offset = 0;
  // 1. Observing player one-hot (4).
  values[offset + player] = 1.0f;
  offset += kNumPlayers;

  // 2. Phase one-hot (6).
  values[offset + static_cast<int>(phase_)] = 1.0f;
  offset += kNumPhases;

  // 3. Bid value one-hot (5).
  if (bid_value_ >= kMinBidValue && bid_value_ <= kMaxBidValue) {
    values[offset + (bid_value_ - kMinBidValue)] = 1.0f;
  }
  offset += kNumBids;

  // 4. Observing player's private hand (52).
  for (int c : hands_[player]) {
    values[offset + c] = 1.0f;
  }
  offset += kNumCards;

  // 5. Cards currently on the floor (52).
  for (const House& h : floor_piles_) {
    for (int c : h.cards) {
      values[offset + c] = 1.0f;
    }
  }
  offset += kNumCards;

  // 6. Houses of values 9..13 (5 * 3 = 15): [exists, is_cemented, points/100].
  for (const House& h : floor_piles_) {
    if (h.value >= kMinBidValue && h.value <= kMaxBidValue) {
      int h_idx = h.value - kMinBidValue;
      values[offset + h_idx * 3 + 0] = 1.0f;
      if (h.is_cemented) {
        values[offset + h_idx * 3 + 1] = 1.0f;
      }
      values[offset + h_idx * 3 + 2] +=
          static_cast<float>(h.PointValue()) / kDeckTotalPoints;
    }
  }
  offset += kNumBids * 3;

  // 7. Captured cards per team (2 * 52 = 104).
  for (int t = 0; t < kNumTeams; ++t) {
    for (int c : team_captured_cards_[t]) {
      values[offset + t * kNumCards + c] = 1.0f;
    }
  }
  offset += kNumTeams * kNumCards;

  // 8. Normalized card points and seep counts per team (2 * 2 = 4).
  const auto card_pts = TeamCardPoints();
  for (int t = 0; t < kNumTeams; ++t) {
    values[offset + t * 2 + 0] =
        static_cast<float>(card_pts[t]) / kDeckTotalPoints;
    values[offset + t * 2 + 1] = static_cast<float>(team_seeps_[t]);
  }
  offset += kNumTeams * 2;
  SPIEL_CHECK_EQ(offset, kObservationTensorSize);
}

std::unique_ptr<StateStruct> SeepState::ToStruct() const {
  auto rv = std::make_unique<SeepStateStruct>();
  rv->phase = PhaseToString(phase_);
  rv->current_player = CurrentPlayer();
  rv->bid_value = bid_value_;
  rv->hands.resize(kNumPlayers);
  for (Player p = 0; p < kNumPlayers; ++p) {
    for (int c : hands_[p]) {
      rv->hands[p].push_back(CardString(c));
    }
  }
  for (const House& h : floor_piles_) {
    rv->floor_piles.push_back(h.ToString());
  }
  const auto pts = TeamCardPoints();
  rv->team_card_points = {pts[0], pts[1]};
  rv->team_seeps = {team_seeps_[0], team_seeps_[1]};
  return rv;
}

std::unique_ptr<ObservationStruct> SeepState::ToObservationStruct(
    Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, kNumPlayers);
  auto rv = std::make_unique<SeepObservationStruct>(ToStruct()->ToJson());
  rv->observing_player = player;
  for (Player p = 0; p < kNumPlayers; ++p) {
    if (p != player) {
      rv->hands[p].clear();
    }
  }
  return rv;
}

std::vector<int> SeepState::Hand(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, kNumPlayers);
  return hands_[player];
}

std::array<int, kNumTeams> SeepState::TeamCardPoints() const {
  std::array<int, kNumTeams> pts = {0, 0};
  for (int t = 0; t < kNumTeams; ++t) {
    for (int c : team_captured_cards_[t]) {
      pts[t] += CardPointValue(c);
    }
  }
  return pts;
}

std::array<int, kNumTeams> SeepState::TeamTotalScores() const {
  const auto pts = TeamCardPoints();
  return {pts[0] + team_seeps_[0] * kSeepBonusPoints,
          pts[1] + team_seeps_[1] * kSeepBonusPoints};
}

void SeepState::SetCustomPlayState(
    Player current_player, int bid_value, int num_cards_played,
    const std::array<std::vector<int>, kNumPlayers>& hands,
    const std::vector<House>& floor_piles,
    const std::array<std::vector<int>, kNumTeams>& team_captured_cards,
    const std::array<int, kNumTeams>& team_seeps, Player last_capture_player) {
  phase_ = Phase::kPlay;
  current_player_ = current_player;
  bid_value_ = bid_value;
  num_cards_dealt_ = kNumCards;
  num_cards_played_ = num_cards_played;
  num_redeals_ = 0;
  card_dealt_.fill(true);
  hands_ = hands;
  for (auto& h : hands_) {
    absl::c_sort(h);
  }
  floor_piles_ = floor_piles;
  for (House& h : floor_piles_) {
    absl::c_sort(h.cards);
  }
  SortFloorPiles();
  team_captured_cards_ = team_captured_cards;
  for (auto& captured : team_captured_cards_) {
    absl::c_sort(captured);
  }
  team_seeps_ = team_seeps;
  last_capture_player_ = last_capture_player;
}

std::vector<Action> SeepState::LegalActions() const {
  switch (phase_) {
    case Phase::kDealInitial:
    case Phase::kDealFloor:
    case Phase::kDealRemainder:
      return DealLegalActions();
    case Phase::kBid:
      return BidLegalActions();
    case Phase::kPlay:
      return PlayLegalActions();
    case Phase::kGameOver:
      return {};
  }
  SpielFatalError("Unknown phase in LegalActions.");
}

ActionsAndProbs SeepState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  std::vector<Action> actions = DealLegalActions();
  SPIEL_CHECK_FALSE(actions.empty());
  const double prob = 1.0 / static_cast<double>(actions.size());
  ActionsAndProbs outcomes;
  outcomes.reserve(actions.size());
  for (Action a : actions) {
    outcomes.emplace_back(a, prob);
  }
  return outcomes;
}

std::vector<Action> SeepState::DealLegalActions() const {
  // Normal initial deals draw uniformly from all undealt cards; if Player 0's
  // 4-card hand has no card >= 9, those 4 cards are reshuffled and redealt.
  // Only if `kMaxRedeals` consecutive redeals have already occurred do we cap
  // the 4th card to >= 9 to keep `MaxChanceNodesInHistory()` finite.
  bool require_bid_card = false;
  if (phase_ == Phase::kDealInitial && num_redeals_ >= kMaxRedeals &&
      num_cards_dealt_ == kInitialBidderCards - 1) {
    require_bid_card = !absl::c_any_of(
        hands_[0], [](int c) { return CardCaptureValue(c) >= kMinBidValue; });
  }

  std::vector<Action> actions;
  actions.reserve(kNumCards - num_cards_dealt_);
  for (int c = 0; c < kNumCards; ++c) {
    if (!card_dealt_[c]) {
      if (!require_bid_card || CardCaptureValue(c) >= kMinBidValue) {
        actions.push_back(c);
      }
    }
  }
  return actions;
}

std::vector<Action> SeepState::BidLegalActions() const {
  std::vector<Action> actions;
  for (int bid = kMinBidValue; bid <= kMaxBidValue; ++bid) {
    for (int c : hands_[0]) {
      if (CardCaptureValue(c) == bid) {
        actions.push_back(MakeBidAction(bid));
        break;
      }
    }
  }
  return actions;
}

std::vector<Action> SeepState::PlayLegalActions() const {
  std::vector<Action> actions;
  const bool is_first_turn = (num_cards_played_ == 0);

  for (int card : hands_[current_player_]) {
    const int card_val = CardCaptureValue(card);

    // 1. Throw intent.
    if ((!is_first_turn || card_val == bid_value_) && CanThrowCard(card)) {
      actions.push_back(MakePlayAction(card, Intent::kThrow));
    }

    // 2. Capture intent (one action per maximal floor-pile combination).
    if (!is_first_turn || card_val == bid_value_) {
      std::vector<int> masks = GetCaptureMasks(floor_piles_, card_val);
      for (int idx = 0; idx < static_cast<int>(masks.size()); ++idx) {
        actions.push_back(MakePlayAction(card, Intent::kCapture, idx));
      }
    }

    // 3. Build 9..13 intents (one action per maximal floor-pile combination).
    for (int h_val = kMinBidValue; h_val <= kMaxBidValue; ++h_val) {
      if (is_first_turn && h_val != bid_value_) continue;
      if (CanBuildHouseWithCard(current_player_, card, h_val)) {
        Intent build_intent = static_cast<Intent>(
            static_cast<int>(Intent::kBuild9) + (h_val - kMinBidValue));
        std::vector<BuildPlan> plans =
            GetBuildHousePlans(floor_piles_, card, h_val, current_player_);
        for (int idx = 0; idx < static_cast<int>(plans.size()); ++idx) {
          actions.push_back(MakePlayAction(card, build_intent, idx));
        }
      }
    }
  }
  return actions;
}

void SeepState::DoApplyAction(Action action_id) {
  switch (phase_) {
    case Phase::kDealInitial:
    case Phase::kDealFloor:
    case Phase::kDealRemainder:
      SPIEL_CHECK_GE(action_id, 0);
      SPIEL_CHECK_LT(action_id, kNumCards);
      ApplyDealAction(static_cast<int>(action_id));
      return;
    case Phase::kBid:
      SPIEL_CHECK_GE(action_id, kBidActionBase);
      SPIEL_CHECK_LT(action_id, kPlayActionBase);
      ApplyBidAction(BidValueFromAction(action_id));
      return;
    case Phase::kPlay: {
      SPIEL_CHECK_GE(action_id, kPlayActionBase);
      SPIEL_CHECK_LT(action_id, kNumDistinctActions);
      auto [card, intent] = DecodePlayAction(action_id);
      int choice_index = DecodePlayActionChoiceIndex(action_id);
      ApplyPlayAction(card, intent, choice_index);
      return;
    }
    case Phase::kGameOver:
      SpielFatalError("Cannot apply action in kGameOver phase.");
  }
}

void SeepState::ApplyDealAction(int card) {
  SPIEL_CHECK_GE(card, 0);
  SPIEL_CHECK_LT(card, kNumCards);
  SPIEL_CHECK_FALSE(card_dealt_[card]);
  card_dealt_[card] = true;

  if (phase_ == Phase::kDealInitial) {
    hands_[0].push_back(card);
    absl::c_sort(hands_[0]);
    ++num_cards_dealt_;
    if (num_cards_dealt_ == kInitialBidderCards) {
      const bool has_bid_card = absl::c_any_of(
          hands_[0], [](int c) { return CardCaptureValue(c) >= kMinBidValue; });
      if (has_bid_card) {
        phase_ = Phase::kBid;
        current_player_ = 0;
      } else {
        // Player 0 has no card >= 9 in their initial 4 cards: return the 4
        // cards to the deck, reshuffle, and deal again from the beginning.
        hands_[0].clear();
        card_dealt_.fill(false);
        num_cards_dealt_ = 0;
        ++num_redeals_;
      }
    }
    return;
  }

  if (phase_ == Phase::kDealFloor) {
    floor_piles_.push_back(House{CardCaptureValue(card), false, {card}});
    SortFloorPiles();
    ++num_cards_dealt_;
    if (num_cards_dealt_ == kInitialDealCards) {
      phase_ = Phase::kPlay;
      current_player_ = 0;
    }
    return;
  }

  SPIEL_CHECK_EQ(phase_, Phase::kDealRemainder);
  // Deal 12 cards to Players 1, 2, 3 and 8 more cards to Player 0:
  // Cards 8..19 (12 cards): 4 cards each to Players 1, 2, 3.
  // Cards 20..51 (32 cards): 8 cards each to Players 0, 1, 2, 3.
  Player recipient = 0;
  if (num_cards_dealt_ < 20) {
    recipient = 1 + (num_cards_dealt_ - kInitialDealCards) / 4;
  } else {
    recipient = (num_cards_dealt_ - 20) / 8;
  }
  hands_[recipient].push_back(card);
  absl::c_sort(hands_[recipient]);
  ++num_cards_dealt_;

  if (num_cards_dealt_ == kNumCards) {
    phase_ = Phase::kPlay;
    current_player_ = 1;
  }
}

void SeepState::ApplyBidAction(int bid_value) {
  SPIEL_CHECK_GE(bid_value, kMinBidValue);
  SPIEL_CHECK_LE(bid_value, kMaxBidValue);
  SPIEL_CHECK_TRUE(absl::c_any_of(hands_[0], [bid_value](int c) {
    return CardCaptureValue(c) == bid_value;
  }));
  bid_value_ = bid_value;
  // Once the bid is announced, the 4 initial floor cards are dealt face-up
  // before Player 0 makes their opening play (turn 0).
  phase_ = Phase::kDealFloor;
  current_player_ = kChancePlayerId;
}

void SeepState::ApplyPlayAction(int card, Intent intent, int choice_index) {
  auto& hand = hands_[current_player_];
  auto it = absl::c_find(hand, card);
  SPIEL_CHECK_TRUE(it != hand.end());
  SPIEL_CHECK_GE(choice_index, 0);

  const int card_val = CardCaptureValue(card);
  const bool is_first_turn = (num_cards_played_ == 0);
  int capture_mask = 0;
  BuildPlan build_plan;

  if (intent == Intent::kThrow) {
    SPIEL_CHECK_EQ(choice_index, 0);
    SPIEL_CHECK_TRUE(!is_first_turn || card_val == bid_value_);
    SPIEL_CHECK_TRUE(CanThrowCard(card));
  } else if (intent == Intent::kCapture) {
    SPIEL_CHECK_TRUE(!is_first_turn || card_val == bid_value_);
    std::vector<int> masks = GetCaptureMasks(floor_piles_, card_val);
    SPIEL_CHECK_LT(choice_index, static_cast<int>(masks.size()));
    capture_mask = masks[choice_index];
  } else {
    const int house_value = IntentHouseValue(intent);
    SPIEL_CHECK_TRUE(!is_first_turn || house_value == bid_value_);
    SPIEL_CHECK_TRUE(CanBuildHouseWithCard(current_player_, card, house_value));
    std::vector<BuildPlan> plans =
        GetBuildHousePlans(floor_piles_, card, house_value, current_player_);
    SPIEL_CHECK_LT(choice_index, static_cast<int>(plans.size()));
    build_plan = plans[choice_index];
  }

  hand.erase(it);

  if (intent == Intent::kThrow) {
    floor_piles_.push_back(House{card_val, false, {card}});
    SortFloorPiles();
  } else if (intent == Intent::kCapture) {
    const int team = PlayerTeam(current_player_);
    team_captured_cards_[team].push_back(card);

    std::vector<House> remaining_piles;
    const int n = static_cast<int>(floor_piles_.size());
    for (int i = 0; i < n; ++i) {
      if ((capture_mask >> i) & 1) {
        for (int c : floor_piles_[i].cards) {
          team_captured_cards_[team].push_back(c);
        }
      } else {
        remaining_piles.push_back(std::move(floor_piles_[i]));
      }
    }
    absl::c_sort(team_captured_cards_[team]);
    floor_piles_ = std::move(remaining_piles);
    SortFloorPiles();
    last_capture_player_ = current_player_;

    // Clearing the floor awards a Seep (+50 bonus), except on the final play
    // of the deal.
    if (floor_piles_.empty() && num_cards_played_ + 1 < kTotalPlayTurns) {
      ++team_seeps_[team];
    }
  } else {
    const int house_value = IntentHouseValue(intent);

    House built_house;
    built_house.value = house_value;
    built_house.is_cemented = build_plan.is_cemented;
    built_house.cards.push_back(card);
    int existing_same_value_owners = 0;

    std::vector<House> remaining_piles;
    const int n = static_cast<int>(floor_piles_.size());
    for (int i = 0; i < n; ++i) {
      if ((build_plan.merged_pile_mask >> i) & 1) {
        if (floor_piles_[i].cards.size() > 1 &&
            floor_piles_[i].value == house_value) {
          existing_same_value_owners |= floor_piles_[i].owners_mask;
        }
        for (int c : floor_piles_[i].cards) {
          built_house.cards.push_back(c);
        }
      } else {
        remaining_piles.push_back(std::move(floor_piles_[i]));
      }
    }
    const Player partner = current_player_ ^ 2;
    if (((existing_same_value_owners >> partner) & 1) != 0) {
      built_house.owners_mask = existing_same_value_owners;
    } else {
      built_house.owners_mask =
          existing_same_value_owners | (1 << current_player_);
    }
    absl::c_sort(built_house.cards);
    remaining_piles.push_back(std::move(built_house));
    floor_piles_ = std::move(remaining_piles);
    SortFloorPiles();
  }

  ++num_cards_played_;
  if (num_cards_dealt_ < kNumCards) {
    phase_ = Phase::kDealRemainder;
    current_player_ = kChancePlayerId;
  } else if (num_cards_played_ >= kTotalPlayTurns) {
    SweepRemainingFloorCardsAtEnd();
    phase_ = Phase::kGameOver;
    current_player_ = kTerminalPlayerId;
  } else {
    current_player_ = (current_player_ + 1) % kNumPlayers;
  }
}

bool SeepState::CanThrowCard(int card) const {
  return !CanCaptureWithCard(card);
}

bool SeepState::CanCaptureWithCard(int card) const {
  return !GetCaptureMasks(floor_piles_, CardCaptureValue(card)).empty();
}

bool SeepState::CanBuildHouseWithCard(Player player, int card,
                                      int house_value) const {
  const Player partner = player ^ 2;
  bool partner_owns_target_house = false;
  for (const House& h : floor_piles_) {
    if (h.cards.size() > 1 && h.value == house_value &&
        (((h.owners_mask >> partner) & 1) != 0)) {
      partner_owns_target_house = true;
      break;
    }
  }
  if (!partner_owns_target_house &&
      !PlayerHoldsAnotherCardOfValue(player, card, house_value)) {
    return false;
  }
  const int card_val = CardCaptureValue(card);
  if (card_val < house_value) {
    for (const House& h : floor_piles_) {
      const bool player_is_owner =
          (h.owners_mask == 0) || (((h.owners_mask >> player) & 1) != 0);
      if (h.cards.size() > 1 && h.value == card_val && player_is_owner &&
          !PlayerHoldsAnotherCardOfValue(player, card, card_val)) {
        return false;
      }
    }
  }
  return !GetBuildHousePlans(floor_piles_, card, house_value, player).empty();
}

bool SeepState::PlayerHoldsAnotherCardOfValue(Player player, int excluded_card,
                                              int value) const {
  for (int c : hands_[player]) {
    if (c != excluded_card && CardCaptureValue(c) == value) {
      return true;
    }
  }
  return false;
}

void SeepState::SortFloorPiles() {
  absl::c_sort(floor_piles_, [](const House& a, const House& b) {
    return std::tie(a.value, a.is_cemented, a.cards) <
           std::tie(b.value, b.is_cemented, b.cards);
  });
}

void SeepState::SweepRemainingFloorCardsAtEnd() {
  if (floor_piles_.empty()) return;
  if (last_capture_player_ >= 0 && last_capture_player_ < kNumPlayers) {
    const int team = PlayerTeam(last_capture_player_);
    for (const House& h : floor_piles_) {
      for (int c : h.cards) {
        team_captured_cards_[team].push_back(c);
      }
    }
    absl::c_sort(team_captured_cards_[team]);
  }
  floor_piles_.clear();
}

SeepGame::SeepGame(const GameParameters& params) : Game(kGameType, params) {}

}  // namespace seep
}  // namespace open_spiel
