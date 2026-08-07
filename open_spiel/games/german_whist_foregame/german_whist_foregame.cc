
#include "open_spiel/games/german_whist_foregame/german_whist_foregame.h"

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
// define BMI2 only if your system supports BMI2 intrinsics, modify compiler
// flags so that bmi2 instructions are compiled// #define __BMI2__
#ifdef __BMI2__
#include <x86intrin.h>
#endif
namespace open_spiel {
namespace german_whist_foregame {

uint32_t tzcnt_u32(uint32_t a) { return __builtin_ctz(a); }
uint64_t tzcnt_u64(uint64_t a) { return __builtin_ctzll(a); }
uint32_t bzhi_u32(uint32_t a, uint32_t b) { return a & ((1u << b) - 1); }
uint64_t bzhi_u64(uint64_t a, uint64_t b) { return a & ((1ULL << b) - 1); }
uint32_t blsr_u32(uint32_t a) { return (a - 1) & a; }
uint64_t blsr_u64(uint64_t a) { return (a - 1) & a; }
uint32_t popcnt_u32(uint32_t a) { return __builtin_popcount(a); }
uint64_t popcnt_u64(uint64_t a) { return __builtin_popcountll(a); }
// the pext bithack is a lot slower than the bmi2 intrinsic, and even with bmi2
// support enabled this will not compile down to a pext instruction//
uint64_t pext_u64(uint64_t x, uint64_t m) {
#ifdef __BMI2__
  return _pext_u64(x, m);
#endif
#ifndef __BMI2__
  uint64_t r = 0;
  uint64_t s = 0;
  uint64_t b = 0;
  do {
    b = m & 1;
    r = r | ((x & b) << s);
    s = s + b;
    x = x >> 1;
    m = m >> 1;
  } while (m != 0);
  return r;
#endif
}

inline int CardRank(int card, int suit) {
  uint64_t card_mask = ((uint64_t)1 << card);
  card_mask = (card_mask >> (suit * kNumRanks));
  return tzcnt_u64(card_mask);
}
inline int CardSuit(int card) {
  uint64_t card_mask = ((uint64_t)1 << card);
  for (int i = 0; i < kNumSuits; ++i) {
    if (popcnt_u64(card_mask & kSuitMasks[i]) == 1) {
      return i;
    }
  }
  return kNumSuits;
}
std::string CardString(int card) {
  int suit = CardSuit(card);
  return {kSuitChar[suit], kRankChar[CardRank(card, suit)]};
}

std::vector<uint32_t> GenQuads(int size_endgames) {
  // Generates Suit splittings for endgames of a certain size//
  std::vector<uint32_t> v;
  for (uint8_t i = 0; i <= std::min(size_endgames * 2, kNumRanks); ++i) {
    int sum = size_endgames * 2 - i;
    for (uint8_t j = 0; j <= std::min(sum, kNumRanks); ++j) {
      for (uint8_t k = static_cast<uint8_t>( std::max(static_cast<int>(j),
                                                     sum - j - kNumRanks));
           k <= std::min(sum - j, kNumRanks); ++k) {
        uint8_t l = static_cast<uint8_t>(sum - j - k);
        if (l < k) {
          break;
        } else {
          uint32_t num = 0;
          num = num | (i);
          num = num | (j << 4);
          num = num | (k << 8);
          num = num | (l << 12);
          v.push_back(num);
        }
      }
    }
  }
  return v;
}

uint32_t Colex(uint32_t cards) {
  // sum NCR(S[i],i+1)
  uint32_t out = 0;
#pragma unroll
  for (uint32_t i = 0; i < 32; ++i) {
    uint32_t mask = (1 << i) - 1;
    uint32_t count = popcnt_u32(cards & mask);
    uint32_t ind = tzcnt_u32(cards);
    uint32_t val = ((cards >> i) & 0b1) ? BIN_COEFFS_LUT[i][count + 1] : 0;
    out += val;
  }
  return out;
}
void GenSuitRankingsRel(uint32_t size,
                        std::unordered_map<uint32_t, uint32_t>* Ranks) {
  // Generates ranking Table for suit splittings for endgames of a certain
  // size//
  std::vector<uint32_t> v = GenQuads(size);
  for (uint32_t i = 0; i < v.size(); ++i) {
    Ranks->insert({v[i], i});
  }
}

vectorNa::vectorNa(uint32_t suit_splits, uint32_t card_combs, uint8_t val) {
  data = std::vector<uint8_t>(
      ((card_combs * suit_splits) >> 1) + ((card_combs * suit_splits) & 0b1),
      val);
  inner_size = card_combs >> 1;
}
vectorNa::vectorNa() {
  data = {};
  inner_size = 0;
}
uint32_t vectorNa::size() const { return data.size(); }
uint32_t vectorNa::GetInnerSize() const { return inner_size; }
uint8_t const& vectorNa::operator[](uint32_t idx) const { return data[idx]; }
uint8_t vectorNa::GetChar(uint32_t idx) const { return data[idx]; }
void vectorNa::SetChar(uint32_t idx, uint8_t value) { data[idx] = value; }
uint8_t vectorNa::Get(uint32_t suit_idx, uint32_t card_idx) const {
  uint32_t idx = suit_idx * 2 * inner_size + (card_idx);
  uint8_t val = data[idx >> 1];
  uint8_t uval = (val & u_sel_mask) >> u_shift;
  uint8_t lval = (val & l_sel_mask) >> l_shift;
  uint8_t ret = (idx & 0b1) ? uval : lval;
  return ret;
}
void vectorNa::Set(uint32_t suit_idx, uint32_t card_idx, uint8_t value) {
  uint32_t idx = suit_idx * 2 * inner_size + (card_idx);
  uint32_t real_idx = idx >> 1;
  uint32_t u = idx & 0b1;
  uint8_t s_val = data[real_idx];
  s_val = (u) ? (s_val & u_del_mask) : (s_val & l_del_mask);
  s_val = (u) ? (s_val | (value << u_shift)) : (s_val | (value << l_shift));
  data[real_idx] = s_val;
  return;
}
vectorNa InitialiseTTable(int size) {
  // initialises TTable for a certain depth//
  uint32_t suit_size = GenQuads(size).size();
  return vectorNa(suit_size, BIN_COEFFS_LUT[2 * size][size], 0);
}
vectorNa LoadTTable(const std::string filename, int depth) {
  // loads solution from a text file into a vector for use//
  std::cout << "Loading Tablebase"
            << "\n";
  vectorNa v = InitialiseTTable(depth);
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cout << "Failed to load Tablebase"
              << "\n";
    std::cout << "Tablebase will be set to all 0"
              << "\n";
    file.close();
    return v;
  } else {
    char c;
    for (int i = 0; i < v.size(); ++i) {
      file.get(c);
      v.SetChar(i, static_cast<uint8_t>(c));
    }
    file.close();
    std::cout << "Tablebase Loaded\n";
    return v;
  }
}

// Default parameters.

namespace {  // namespace
// Facts about the game
const GameType kGameType{/*short_name=*/"german_whist_foregame",
                         /*long_name=*/"german_whist_foregame",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kImperfectInformation,
                         GameType::Utility::kZeroSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/2,
                         /*min_num_players=*/2,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/false,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/false,
                         {{"ttable_path", GameParameter(std::string(""))}}

};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new GWhistFGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace

GWhistFGame::GWhistFGame(const GameParameters& params)
    : Game(kGameType, params) {
  std::string ttable_path = ParameterValue<std::string>("ttable_path");
  std::unordered_map<uint32_t, uint32_t> temp;
  GenSuitRankingsRel(13, &temp);
  suit_ranks_ = temp;
  ttable_ = LoadTTable(ttable_path, 13);
};
std::unique_ptr<State> GWhistFGame::NewInitialState() const {
  const auto ptr =
      std::dynamic_pointer_cast<const GWhistFGame>(shared_from_this());
  return std::make_unique<GWhistFState>(ptr);
}

GWhistFState::GWhistFState(std::shared_ptr<const GWhistFGame> game)
    : State(game) {
  player_ = kChancePlayerId;
  move_number_ = 0;
  trump_ = -1;
  deck_ = bzhi_u64(~0, kNumRanks * kNumSuits);
  discard_ = 0;
  hands_ = {0, 0};
  history_.reserve(78);
  ttable_ = &(game->ttable_);
  suit_ranks_ = &(game->suit_ranks_);
}
bool GWhistFState::Trick(int lead, int follow) const {
  int lead_suit = CardSuit(lead);
  int follow_suit = CardSuit(follow);
  int lead_rank = CardRank(lead, lead_suit);
  int follow_rank = CardRank(follow, follow_suit);
  return (lead_suit == follow_suit && lead_rank < follow_rank) ||
         (lead_suit != follow_suit && follow_suit != trump_);
}
bool GWhistFState::IsTerminal() const { return (popcnt_u64(deck_) == 0); }

std::pair<uint32_t, uint32_t> GWhistFState::EndgameKey(
    int player_to_move) const {
  // Generates Endgame Key for accessing Endgame Tablebase//
  uint64_t cards_in_play = hands_[0] | hands_[1];
  int opp = (player_to_move == 0) ? 1 : 0;
  using suit_info = std::tuple<bool, uint64_t, uint64_t, uint8_t>;
  std::vector<suit_info> suit_infos = {
      {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
  for (uint8_t i = 0; i < kNumSuits; ++i) {
    bool is_trump = (i == trump_);
    uint64_t active_suit_cards = kSuitMasks[i] & cards_in_play;
    uint64_t suit_length = popcnt_u64(active_suit_cards);
    uint64_t suit_sig =
        pext_u64(hands_[opp] & kSuitMasks[i], active_suit_cards);
    uint8_t suit_idx = i;
    suit_infos[i] = {is_trump, suit_length, suit_sig, suit_idx};
  }
  auto custom_cmp = [&](const suit_info& lhs, const suit_info& rhs) {
    const auto& [t1, l1, s1, i1] = lhs;
    const auto& [t2, l2, s2, i2] = rhs;
    return std::tie(t2, l1, i1) < std::tie(t1, l2, i2);
  };
  std::sort(suit_infos.begin(), suit_infos.end(), custom_cmp);
  uint32_t bitpacked_suit_lengths = 0;
  uint32_t card_mask = 0;
  uint32_t total_cards = 0;
  for (uint32_t i = 0; i < kNumSuits; ++i) {
    uint32_t suit_length = (uint32_t)std::get<1>(suit_infos[i]);
    bitpacked_suit_lengths =
        (bitpacked_suit_lengths | (suit_length << (4 * i)));
    card_mask =
        (card_mask) | (((uint32_t)std::get<2>(suit_infos[i])) << (total_cards));
    total_cards += suit_length;
  }
  return {card_mask, bitpacked_suit_lengths};
}

std::vector<double> GWhistFState::Returns() const {
  if (IsTerminal()) {
    std::vector<double> out = {0, 0};
    int lead_win = Trick(history_[move_number_ - 3].action,
                         history_[move_number_ - 2].action);
    int player_to_move = (lead_win) ? history_[move_number_ - 3].player
                                    : history_[move_number_ - 2].player;
    int opp = (player_to_move == 0) ? 1 : 0;
    std::pair<uint32_t, uint32_t> key = EndgameKey(player_to_move);
    uint32_t colex = Colex(key.first);
    uint32_t suit_rank = suit_ranks_->at(key.second);
    uint8_t value = ttable_->Get(suit_rank, colex);
    out[player_to_move] = 2 * value - kNumRanks;
    out[opp] = -out[player_to_move];
    return out;
  } else {
    std::vector<double> out = {0, 0};
    return out;
  }
}

int GWhistFState::CurrentPlayer() const { return player_; }

std::vector<std::pair<Action, double>> GWhistFState::ChanceOutcomes() const {
  std::vector<std::pair<Action, double>> outcomes;
  std::vector<Action> legal_actions = LegalActions();
  for (int i = 0; i < legal_actions.size(); ++i) {
    std::pair<Action, double> pair;
    pair.first = legal_actions[i];
    pair.second = 1.0 / legal_actions.size();
    outcomes.push_back(pair);
  }
  return outcomes;
}
std::string GWhistFState::ActionToString(Player player, Action move) const {
  return CardString(move);
}
std::string GWhistFState::ToString() const {
  std::string out;
  for (int i = 0; i < history_.size(); ++i) {
    out += ActionToString(history_[i].player, history_[i].action);
    out += "\n";
  }
  return out;
}
std::unique_ptr<State> GWhistFState::Clone() const {
  return std::unique_ptr<State>(new GWhistFState(*this));
}

std::string GWhistFState::StateToString() const {
  // doesnt use history in case of a resampled state with unreconciled history//
  std::string out;
  uint64_t copy_deck = deck_;
  uint64_t copy_discard = discard_;
  std::array<uint64_t, 2> copy_hands = hands_;
  std::vector<int> deck_cards;
  std::vector<int> player0_cards;
  std::vector<int> player1_cards;
  std::vector<int> discard;
  while (copy_deck != 0) {
    deck_cards.push_back(tzcnt_u64(copy_deck));
    copy_deck = blsr_u64(copy_deck);
  }
  while (copy_discard != 0) {
    discard.push_back(tzcnt_u64(copy_discard));
    copy_discard = blsr_u64(copy_discard);
  }

  while (copy_hands[0] != 0) {
    player0_cards.push_back(tzcnt_u64(copy_hands[0]));
    copy_hands[0] = blsr_u64(copy_hands[0]);
  }
  while (copy_hands[1] != 0) {
    player1_cards.push_back(tzcnt_u64(copy_hands[1]));
    copy_hands[1] = blsr_u64(copy_hands[1]);
  }
  out += "Deck \n";
  for (int i = 0; i < deck_cards.size(); ++i) {
    out += CardString(deck_cards[i]) + "\n";
  }
  out += "Discard \n";
  for (int i = 0; i < discard.size(); ++i) {
    out += CardString(discard[i]) + "\n";
  }

  for (int i = 0; i < 2; ++i) {
    out += "Player " + std::to_string(i) + "\n";
    std::vector<int> var;
    if (i == 0) {
      var = player0_cards;
    } else {
      var = player1_cards;
    }
    for (int j = 0; j < var.size(); ++j) {
      out += CardString(var[j]) + "\n";
    }
  }
  return out;
}
std::string GWhistFState::InformationStateString(Player player) const {
  // THIS IS WHAT A PLAYER IS SHOWN WHEN PLAYING//
  SPIEL_CHECK_TRUE(player >= 0 && player < 2);
  std::string p = std::to_string(player) + ",";
  std::string cur_hand = "";
  std::string observations = "";
  std::vector<int> v_hand = {};
  uint64_t p_hand = hands_[player];
  while (p_hand != 0) {
    v_hand.push_back(tzcnt_u64(p_hand));
    p_hand = blsr_u64(p_hand);
  }
  std::sort(v_hand.begin(), v_hand.end());
  for (int i = 0; i < v_hand.size(); ++i) {
    cur_hand = cur_hand + CardString(v_hand[i]);
    cur_hand = cur_hand + ",";
  }
  cur_hand += "\n";
  for (int i = 2 * kNumRanks; i < history_.size(); ++i) {
    int index = (i - 2 * kNumRanks) % 4;
    switch (index) {
      case 0:
        observations =
            observations + "c_public:" + CardString(history_[i].action) + ",";
        break;
      case 1:
        observations = observations + "p" + std::to_string(history_[i].player) +
                       ":" + CardString(history_[i].action) + ",";
        break;
      case 2:
        observations = observations + "p" + std::to_string(history_[i].player) +
                       ":" + CardString(history_[i].action) + ",";
        break;
      case 3:
        int lead_win = Trick(history_[i - 2].action, history_[i - 1].action);
        int loser = ((lead_win) ^ (history_[i - 2].player == 0)) ? 0 : 1;
        if (loser == player) {
          observations = observations +
                         "c_observed:" + CardString(history_[i].action) + "\n";
        } else {
          observations = observations + "c_unobserved:" + "\n";
        }
        break;
    }
  }
  return p + cur_hand + observations;
}
std::unique_ptr<State> GWhistFState::ResampleFromInfostate(
    int player_id, std::function<double()> rng) const {
  // only valid when called from a position where a player can act//
  auto resampled_state = std::unique_ptr<GWhistFState>(new GWhistFState(*this));
  // seeding mt19937//
  std::random_device rd;
  std::mt19937 gen(rd());
  uint64_t necessary_cards = 0;
  for (int i = 2 * kNumRanks; i < history_.size(); i += 4) {
    // face up cards from deck//
    necessary_cards = (necessary_cards | (uint64_t(1) << history_[i].action));
  }
  int move_index = move_number_ - ((kNumRanks * kNumSuits) / 2);
  int move_remainder = move_index % 4;
  int opp = (player_id == 0) ? 1 : 0;
  int recent_faceup = move_number_ - move_remainder;
  uint64_t recent_faceup_card = (uint64_t(1) << history_[recent_faceup].action);
  // if a face up card from the deck is not in players hand or discard it must
  // be in opps unless it is the most recent face up//
  necessary_cards = (necessary_cards &
                     (~(hands_[player_id] | discard_ | recent_faceup_card)));
  // sufficient cards are all cards not in players hand,the discard, or the
  // recent face up//
  uint64_t sufficient_cards =
      (bzhi_u64(~0, kNumRanks * kNumSuits) ^
       (hands_[player_id] | discard_ | recent_faceup_card));
  // sufficient_cards are not necessary //
  sufficient_cards = (sufficient_cards & (~(necessary_cards)));
  // we must now take into account the observation of voids//
  std::array<int, kNumSuits> when_voided = {0, 0, 0, 0};
  std::array<int, kNumSuits> voids = {-1, -1, -1, -1};
  std::vector<int> opp_dealt_hidden;
  for (int i = 2 * kNumRanks; i < history_.size(); ++i) {
    if (history_[i - 1].player == player_id && history_[i].player == (opp) &&
        CardSuit(history_[i - 1].action) != CardSuit(history_[i].action)) {
      when_voided[CardSuit(history_[i - 1].action)] = i - 1;
    }
    if (history_[i - 1].player == player_id && history_[i].player == (opp) &&
        Trick(history_[i - 1].action, history_[i].action)) {
      opp_dealt_hidden.push_back(i - 1);
    }
    if (history_[i - 1].player == (opp) && history_[i].player == (player_id) &&
        !Trick(history_[i - 1].action, history_[i].action)) {
      opp_dealt_hidden.push_back(i - 1);
    }
  }
  // now voids contains the number of hidden cards dealt to opp since it showed
  // a void in that suit, i.e the maximum number of cards held in that suit// if
  // the suit is unvoided, then this number is -1//
  for (int i = 0; i < kNumSuits; ++i) {
    if (when_voided[i] != 0) {
      voids[i] = 0;
      for (int j = 0; j < opp_dealt_hidden.size(); ++j) {
        if (opp_dealt_hidden[j] >= when_voided[i]) {
          voids[i] += 1;
        }
      }
    }
  }
  // we now perform a sequence of shuffles to generate a possible opponent hand,
  // and make no attempt to reconcile the history with this new deal//
  int nec = popcnt_u64(necessary_cards);
  for (int i = 0; i < kNumSuits; ++i) {
    if (voids[i] != -1 &&
        popcnt_u64(sufficient_cards & kSuitMasks[i]) > voids[i]) {
      uint64_t suit_subset = (sufficient_cards & kSuitMasks[i]);
      std::vector<int> temp;
      while (suit_subset != 0) {
        temp.push_back(tzcnt_u64(suit_subset));
        suit_subset = blsr_u64(suit_subset);
      }
      std::shuffle(temp.begin(), temp.end(), gen);
      sufficient_cards = (sufficient_cards & ~(kSuitMasks[i]));
      for (int j = 0; j < voids[i]; ++j) {
        sufficient_cards = (sufficient_cards | (uint64_t(1) << temp[j]));
      }
    }
  }
  // finally generating a possible hand for opponent//
  std::vector<int> hand_vec;
  while (sufficient_cards != 0) {
    hand_vec.push_back(tzcnt_u64(sufficient_cards));
    sufficient_cards = blsr_u64(sufficient_cards);
  }
  std::shuffle(hand_vec.begin(), hand_vec.end(), gen);
  uint64_t suff_hand = 0;
  uint64_t opp_hand = 0;
  for (int i = 0; i < popcnt_u64(hands_[opp]) - nec; ++i) {
    suff_hand = suff_hand | (uint64_t(1) << hand_vec[i]);
  }
  opp_hand = suff_hand | necessary_cards;
  resampled_state->hands_[opp] = opp_hand;
  resampled_state->deck_ =
      bzhi_u64(~0, kNumRanks * kNumSuits) ^
      (discard_ | opp_hand | hands_[player_id] | recent_faceup_card);
  return resampled_state;
}
std::string GWhistFState::ObservationString(Player player) const {
  // note this is a lie, this is not the observation state string but it is used
  // for ISMCTS to label nodes//
  SPIEL_CHECK_TRUE(player >= 0 && player < 2);
  std::string p = "p" + std::to_string(player) + ",";
  std::string cur_hand = "";
  std::string public_info = "";
  uint64_t p_hand = hands_[player];
  std::vector<int> v_hand = {};
  while (p_hand != 0) {
    v_hand.push_back(tzcnt_u64(p_hand));
    p_hand = blsr_u64(p_hand);
  }
  std::sort(v_hand.begin(), v_hand.end());
  for (int i = 0; i < v_hand.size(); ++i) {
    cur_hand = cur_hand + CardString(v_hand[i]) + ",";
  }
  for (int i = 2 * kNumRanks; i < history_.size(); ++i) {
    int index = (i - 2 * kNumRanks) % 4;
    if (index != 3) {
      public_info = public_info + std::to_string(history_[i].player) + ":" +
                    CardString(history_[i].action) + ",";
    }
  }
  return p + cur_hand + public_info;
}

std::vector<Action> GWhistFState::LegalActions() const {
  std::vector<Action> actions;
  if (IsTerminal()) return {};
  if (IsChanceNode()) {
    actions.reserve(popcnt_u64(deck_));
    uint64_t copy_deck = deck_;
    while (copy_deck != 0) {
      actions.push_back(tzcnt_u64(copy_deck));
      copy_deck = blsr_u64(copy_deck);
    }
  } else {
    // lead//
    actions.reserve(kNumRanks);
    if (history_.back().player == kChancePlayerId) {
      uint64_t copy_hand = hands_[player_];
      while (copy_hand != 0) {
        actions.push_back(tzcnt_u64(copy_hand));
        copy_hand = blsr_u64(copy_hand);
      }
    }

    // follow//
    else {
      uint64_t copy_hand =
          hands_[player_] & kSuitMasks[CardSuit(history_.back().action)];
      if (copy_hand == 0) {
        copy_hand = hands_[player_];
      }
      while (copy_hand != 0) {
        actions.push_back(tzcnt_u64(copy_hand));
        copy_hand = blsr_u64(copy_hand);
      }
    }
  }
  return actions;
}

void GWhistFState::DoApplyAction(Action move) {
  // initial deal//
  int player_start = player_;
  if (move_number_ < (kNumSuits * kNumRanks) / 2) {
    hands_[move_number_ % 2] =
        (hands_[move_number_ % 2] | ((uint64_t)1 << move));
    deck_ = (deck_ ^ ((uint64_t)1 << move));
  } else if (move_number_ == (kNumSuits * kNumRanks / 2)) {
    trump_ = CardSuit(move);
    deck_ = (deck_ ^ ((uint64_t)1 << move));
    player_ = 0;
  }
  // cardplay//
  else if (move_number_ > (kNumSuits * kNumRanks) / 2) {
    int move_index = (move_number_ - ((kNumSuits * kNumRanks) / 2)) % 4;
    switch (move_index) {
      bool lead_win;
      int winner;
      int loser;
      case 0:
        // revealing face up card//
        deck_ = (deck_ ^ ((uint64_t)1 << move));
        lead_win = Trick(history_[move_number_ - 3].action,
                         history_[move_number_ - 2].action);
        winner =
            ((lead_win) ^ (history_[move_number_ - 3].player == 0)) ? 1 : 0;
        player_ = winner;
        break;
      case 1:
        // establishing lead//
        discard_ = (discard_ | ((uint64_t)1 << move));
        hands_[player_] = (hands_[player_] ^ ((uint64_t)1 << move));
        (player_ == 0) ? player_ = 1 : player_ = 0;
        break;
      case 2:
        // following and awarding face up//
        discard_ = (discard_ | ((uint64_t)1 << move));
        hands_[player_] = (hands_[player_] ^ ((uint64_t)1 << move));
        lead_win = Trick(history_[move_number_ - 1].action, move);
        winner =
            ((lead_win) ^ (history_[move_number_ - 1].player == 0)) ? 1 : 0;
        hands_[winner] = (hands_[winner] |
                          ((uint64_t)1 << history_[move_number_ - 2].action));
        player_ = kChancePlayerId;
        break;
      case 3:
        // awarding face down//
        deck_ = (deck_ ^ ((uint64_t)1 << move));
        lead_win = Trick(history_[move_number_ - 2].action,
                         history_[move_number_ - 1].action);
        loser = ((lead_win) ^ (history_[move_number_ - 2].player == 0)) ? 0 : 1;
        hands_[loser] = (hands_[loser] | ((uint64_t)1 << move));
        if (IsTerminal()) {
          player_ = kTerminalPlayerId;
        }
        break;
    }
  }
#ifdef DEBUG
  std::cout << ActionToString(player_start, move) << std::endl;
  std::cout << move << std::endl;
#endif
}

}  // namespace german_whist_foregame
}  // namespace open_spiel
