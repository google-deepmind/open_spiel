// Source Code for an Executable Generating an Endgame Tablebase for German
// Whist

#include <cassert>
#include <thread>

#include "open_spiel/games/german_whist_foregame/german_whist_foregame.h"

// #define DEBUG
namespace open_spiel {
namespace german_whist_foregame {

struct Pair {
  char index;
  char value;
  Pair(char index_, char value_) {
    index = index_;
    value = value_;
  }
  bool operator<(const Pair& pair) const { return value < pair.value; }
};
struct ActionStruct {
  uint32_t index;
  unsigned char suit;
  bool player;
  ActionStruct(uint32_t index_, unsigned char suit_, bool player_) {
    index = index_;
    suit = suit_;
    player = player_;
  }
};
struct ActionValue {
  ActionStruct action;
  int value;
  bool operator<(const ActionValue& aval) const { return value < aval.value; }
};

class Node {
 private:
  uint32_t cards_;
  std::array<uint32_t, kNumSuits> suit_masks_;
  char total_tricks_;
  char trump_;
  char score_;
  char moves_;
  bool player_;
  std::vector<ActionStruct> history_;
  uint64_t key_;

 public:
  Node(uint32_t cards, std::array<uint32_t, kNumSuits> suit_masks, char trump,
       bool player) {
    cards_ = cards;
    suit_masks_ = suit_masks;
    total_tricks_ = popcnt_u32(cards);
    trump_ = trump;
    moves_ = 0;
    player_ = player;
    score_ = 0;
    history_ = {};
  };
  bool Player() { return player_; };
  char Score() { return score_; };
  char Moves() { return moves_; };
  bool IsTerminal() { return (moves_ == 2 * total_tricks_); }
  char RemainingTricks() { return (char)(total_tricks_ - (moves_ >> 1)); }
  char TotalTricks() { return total_tricks_; }
  uint32_t Cards() { return cards_; }
  std::array<uint32_t, kNumSuits> SuitMasks() { return suit_masks_; }
  uint64_t GetNodeKey() { return key_; }
  bool Trick(ActionStruct lead, ActionStruct follow) {
    // true if leader won//
    return (lead.suit != follow.suit && lead.suit == trump_) ||
           (lead.suit == follow.suit && lead.index <= follow.index);
  }

  void RemoveCard(ActionStruct action) {
    // Removes card from cards_//
    uint32_t mask_b = ~0;
    mask_b = bzhi_u32(mask_b, action.index);
    uint32_t mask_a = ~mask_b;
    mask_a = blsr_u32(mask_a);
    uint32_t copy_a = cards_ & mask_a;
    uint32_t copy_b = cards_ & mask_b;
    copy_a = copy_a >> 1;
    cards_ = copy_a | copy_b;
    // decrements appropriate suits//
    suit_masks_[action.suit] = blsr_u32(suit_masks_[action.suit]) >> 1;
    char suit = action.suit;
    suit++;
    while (suit < kNumSuits) {
      suit_masks_[suit] = suit_masks_[suit] >> 1;
      suit++;
    }
  }
  void InsertCard(ActionStruct action) {
    // inserts card into cards_//
    uint32_t mask_b = ~0;
    mask_b = bzhi_u32(mask_b, action.index);
    uint32_t mask_a = ~mask_b;
    uint32_t copy_b = cards_ & mask_b;
    uint32_t copy_a = cards_ & mask_a;
    copy_a = copy_a << 1;
    uint32_t card = action.player << action.index;
    cards_ = card | copy_a | copy_b;
    // increments appropriate suits//
    uint32_t new_suit =
        (suit_masks_[action.suit] & mask_b) | (1 << action.index);
    suit_masks_[action.suit] =
        ((suit_masks_[action.suit] & mask_a) << 1) | new_suit;
    char suit = action.suit;
    suit++;
    while (suit < kNumSuits) {
      suit_masks_[suit] = suit_masks_[suit] << 1;
      suit++;
    }
  }
  void UpdateNodeKey() {
    // recasts the cards and suitlengths into quasi-canonical form//
    // least sig part of 32bit card is trump, then suits in ascending length//

    // note this canonical form does not take advantage of all isomorphisms//
    // suppose a game is transformed as follows: all card bits flipped and the
    // player bit flipped, ie player 1 has the lead and has player 0s cards from
    // the original game// this implies player 1 achieves the minimax value of
    // the original game ie the value is remaining tricks - value of the
    // original game for this transformed game// also does not take advantage of
    // single suit isomorphism. Namely all single suit games with the same card
    // distribution are isomorphic. Currently this considers all trump, all no
    // trump games as distinct//
    uint64_t suit_sig = 0;
    char trump_length = popcnt_u32(suit_masks_[trump_]);
    if (trump_length > kNumRanks) {
      throw;
    }
    std::vector<Triple> non_trump_lengths;
    for (char i = 0; i < kNumSuits; ++i) {
      if (i != trump_) {
        char length = popcnt_u32(suit_masks_[i]);
        uint32_t sig = suit_masks_[i] & cards_;
        if (suit_masks_[i] != 0) {
          sig = (sig >> (tzcnt_u32(suit_masks_[i])));
        }
        if (length > kNumRanks) {
          throw 1;
        }
        non_trump_lengths.push_back(Triple{i, length, sig});
      }
    }
    // sorting takes advantage of two isomorphisms namely nontrump suits of
    // nonequal length can be exchanged and the value of the game does not
    // change// and this more complicated suppose two games with two or more
    // (non_trump)suits of equal length, permuting those suits should not change
    // the value of solved game ie it is an isomorphism//
    std::sort(non_trump_lengths.begin(), non_trump_lengths.end());
    suit_sig = suit_sig | trump_length;
    for (size_t i = 0; i < non_trump_lengths.size(); ++i) {
      suit_sig =
          suit_sig | ((uint64_t)non_trump_lengths[i].length << (4 * (i + 1)));
    }
    suit_sig = suit_sig << 32;
    std::array<uint32_t, kNumSuits> suit_cards;
    suit_cards[0] = cards_ & suit_masks_[trump_];
    if (suit_masks_[trump_] != 0) {
      suit_cards[0] = suit_cards[0] >> tzcnt_u32(suit_masks_[trump_]);
    }
    uint32_t sum = popcnt_u32(suit_masks_[trump_]);
    uint32_t cards = 0 | suit_cards[0];
    for (size_t i = 0; i < non_trump_lengths.size(); ++i) {
      suit_cards[i] = cards_ & suit_masks_[non_trump_lengths[i].index];
      uint32_t val = 0;
      if (suit_masks_[non_trump_lengths[i].index] != 0) {
        val = tzcnt_u32(suit_masks_[non_trump_lengths[i].index]);
      }
      suit_cards[i] = suit_cards[i] >> val;
      suit_cards[i] = suit_cards[i] << sum;
      sum += popcnt_u32(suit_masks_[non_trump_lengths[i].index]);
      cards = cards | suit_cards[i];
    }
    // cards = cards | (player_ << 31);
    key_ = suit_sig | (uint64_t)cards;
#ifdef DEBUG_KEY
    std::cout << "CARDS_ " << cards_ << std::endl;
    std::cout << "CARDS " << cards << std::endl;
    std::cout << "SUIT MASKS " << std::endl;
    for (int i = 0; i < kNumSuits; ++i) {
      std::cout << suit_masks_[i] << std::endl;
    }
    std::cout << "SUIT_SIG " << suit_sig << std::endl;
    std::cout << "KEY " << key_ << std::endl;
#endif
  }
  uint64_t AltKey() {
    uint32_t mask = bzhi_u32(~0, 2 * RemainingTricks());
    return key_ ^ (uint64_t)mask;
  }
  // Move Ordering Heuristics//
  // These could Definitely be improved, very hacky//
  int LeadOrdering(ActionStruct action) {
    char suit = action.suit;
    uint32_t copy_cards = cards_;
    if (player_ == 0) {
      copy_cards = ~copy_cards;
    }
    uint32_t suit_cards = copy_cards & suit_masks_[suit];
    uint32_t mask = suit_cards & ~(suit_cards >> 1);
    // represents out of the stategically inequivalent cards in a suit that a
    // player holds, what rank is it, rank 0 is highest rank etc//
    int suit_rank = popcnt_u32(bzhi_u32(mask, action.index));
    ApplyAction(action);
    std::vector<ActionStruct> moves = LegalActions();
    UndoAction(action);
    int sum = 0;
    for (size_t i = 0; i < moves.size(); ++i) {
      sum += Trick(action, moves[i]);
    }
    if (sum == moves.size()) {
      return action.suit == trump_
                 ? 0 - suit_rank
                 : -1 * kNumRanks -
                       suit_rank;  // intriguing this seems to produce small
                                   // perfomance increase//
    }
    if (sum == 0) {
      return 2 * kNumRanks - suit_rank;
    } else {
      return 1 * kNumRanks - suit_rank;
    }
  }
  int FollowOrdering(ActionStruct action) {
    ActionStruct lead = history_.back();
    // follow ordering for fast cut offs//
    // win as cheaply as possible, followed by lose as cheaply as possible
    char suit = action.suit;
    uint32_t copy_cards = cards_;
    if (player_ == 0) {
      copy_cards = ~copy_cards;
    }
    uint32_t suit_cards = copy_cards & suit_masks_[suit];
    uint32_t mask = suit_cards & ~(suit_cards >> 1);
    // represents out of the stategically inequivalent cards in a suit that a
    // player holds, what rank is it, rank 0 is highest rank etc//
    int suit_rank = popcnt_u32(bzhi_u32(mask, action.index));
    if (!Trick(lead, action)) {
      return -kNumRanks - suit_rank;
    } else {
      return -suit_rank;
    }
  }

  std::vector<ActionStruct> LegalActions() {
    // Features//
    // Move fusion//
    std::vector<ActionStruct> out;
    out.reserve(kNumRanks);
    uint32_t copy_cards = cards_;
    std::array<uint32_t, kNumSuits> player_suit_masks;
    if (player_ == 0) {
      copy_cards = ~copy_cards;
    }
    for (size_t i = 0; i < kNumSuits; ++i) {
      uint32_t suit_cards = copy_cards & suit_masks_[i];
      player_suit_masks[i] = suit_cards & ~(suit_cards >> 1);
#ifdef DEBUG
      std::cout << "Cards " << cards_ << std::endl;
      std::cout << "Suit Mask " << i << " " << suit_masks_[i] << std::endl;
      std::cout << "Player " << player_ << " suit mask " << (int)i << " "
                << player_suit_masks[i] << std::endl;
#endif
    }
    for (char i = 0; i < kNumSuits; ++i) {
      uint32_t suit_mask = player_suit_masks[i];
      bool lead = (moves_ % 2 == 0);
      bool follow = (moves_ % 2 == 1);
      bool correct_suit = 0;
      bool void_in_suit = 0;
      if (follow == true) {
        correct_suit = (history_.back().suit == i);
        void_in_suit = (player_suit_masks[history_.back().suit] == 0);
      }
      if ((lead || (follow && (correct_suit || void_in_suit)))) {
        while (suit_mask != 0) {
          uint32_t best = tzcnt_u32(suit_mask);
          out.push_back(ActionStruct(best, i, player_));
          suit_mask = blsr_u32(suit_mask);
        }
      }
    }
#ifdef DEBUG
    std::cout << "Player " << player_ << " MoveGen " << std::endl;
    for (size_t i = 0; i < out.size(); ++i) {
      std::cout << out[i].index << " " << (int)out[i].suit << std::endl;
    }
#endif
    return out;
  }
  void ApplyAction(ActionStruct action) {
#ifdef DEBUG
    std::cout << "Player " << player_ << " ApplyAction " << action.index << " "
              << (int)action.suit << std::endl;
#endif
    if (moves_ % 2 == 1) {
      ActionStruct lead = history_.back();
      bool winner = !((Trick(lead, action)) ^ lead.player);
#ifdef DEBUG
      std::cout << "Player " << winner << " won this trick" << std::endl;
#endif
      score_ += (winner == 0);
      player_ = (winner);
    } else {
      player_ = !player_;
    }
#ifdef DEBUG
    assert((suit_masks_[0] & suit_masks_[1]) == 0);
    assert((suit_masks_[0] & suit_masks_[2]) == 0);
    assert((suit_masks_[0] & suit_masks_[3]) == 0);
    assert((suit_masks_[1] & suit_masks_[2]) == 0);
    assert((suit_masks_[1] & suit_masks_[3]) == 0);
    assert((suit_masks_[2] & suit_masks_[3]) == 0);
#endif
    RemoveCard(action);
    moves_++;
    history_.push_back(action);
  }
  void UndoAction(ActionStruct action) {
    if (moves_ % 2 == 0) {
      ActionStruct lead = history_[history_.size() - 2];
      ActionStruct follow = history_[history_.size() - 1];
      bool winner = !(Trick(lead, follow) ^ lead.player);
      score_ -= (winner == 0);
    }
    InsertCard(action);
    moves_--;
    player_ = history_.back().player;
    history_.pop_back();
#ifdef DEBUG
    std::cout << "Player " << player_ << " UndoAction " << action.index << " "
              << (int)action.suit << std::endl;
#endif
  }
};

// solvers below
int AlphaBeta(Node* node, int alpha, int beta) {
  // fail soft ab search//
  // uses move ordering to speed up search//
  if (node->IsTerminal()) {
    return node->Score();
  }
  // move ordering code//
  std::vector<ActionStruct> actions = node->LegalActions();
  std::vector<ActionValue> temp;
  temp.reserve(kNumRanks);
  for (int i = 0; i < actions.size(); ++i) {
    if (node->Moves() % 2 == 0) {
      temp.push_back({actions[i], node->LeadOrdering(actions[i])});
    } else {
      temp.push_back({actions[i], node->FollowOrdering(actions[i])});
    }
  }
  std::sort(temp.begin(), temp.end());
  for (int i = 0; i < temp.size(); ++i) {
    actions[i] = temp[i].action;
  }
  // alpha beta search//
  if (node->Player() == 0) {
    int val = 0;
    for (int i = 0; i < actions.size(); ++i) {
      node->ApplyAction(actions[i]);
      val = std::max(val, AlphaBeta(node, alpha, beta));
      node->UndoAction(actions[i]);
      alpha = std::max(val, alpha);
      if (val >= beta) {
        break;
      }
    }
    return val;
  } else if (node->Player() == 1) {
    int val = node->TotalTricks();
    for (int i = 0; i < actions.size(); ++i) {
      node->ApplyAction(actions[i]);
      val = std::min(val, AlphaBeta(node, alpha, beta));
      node->UndoAction(actions[i]);
      beta = std::min(val, beta);
      if (val <= alpha) {
        break;
      }
    }
    return val;
  }
  return -1;
};

// Helper Functions//

// Credit to computationalcombinatorics.wordpress.com
// hideous code for generating the next colexicographical combination//
bool NextColex(std::vector<int>& v, int k) {
  int num = 0;
  for (int i = 0; i < v.size(); ++i) {
    if (i == v.size() - 1) {
      v[i] = v[i] + 1;
      if (v[i] > k - v.size() + i) {
        return false;
      }
      num = i;
      break;
    } else if (v[i + 1] - v[i] > 1 && v[i + 1] != i) {
      v[i] = v[i] + 1;
      if (v[i] > k - v.size() + i) {
        return false;
      }
      num = i;
      break;
    }
  }
  for (int i = 0; i < num; ++i) {
    v[i] = i;
  }
  return true;
}

char IncrementalAlphaBetaMemoryIso(
    Node* node, char alpha, char beta, int depth, vectorNa* TTable,
    std::unordered_map<uint32_t, uint32_t>* SuitRanks,
    const std::vector<std::vector<uint32_t>>& bin_coeffs) {
  // fail soft ab search
  char val = 0;
  uint64_t key = 0;
  bool player = node->Player();
  if (node->IsTerminal()) {
    return node->Score();
  }
  if (node->Moves() % 2 == 0 && depth == 0) {
    node->UpdateNodeKey();
    key = (player) ? node->AltKey() : node->GetNodeKey();
    uint32_t cards = key & bzhi_u64(~0, 32);
    uint32_t colex = HalfColexer(cards, &bin_coeffs);
    uint32_t suits = (key & (~0 ^ bzhi_u64(~0, 32))) >> 32;
    uint32_t suit_rank = SuitRanks->at(suits);
    char value = (player)
                     ? node->RemainingTricks() - TTable->Get(colex, suit_rank)
                     : TTable->Get(colex, suit_rank);
    return value + node->Score();
  } else if (node->Player() == 0) {
    val = 0;
    std::vector<ActionStruct> actions = node->LegalActions();
    for (int i = 0; i < actions.size(); ++i) {
      node->ApplyAction(actions[i]);
      val = std::max(
          val, IncrementalAlphaBetaMemoryIso(node, alpha, beta, depth - 1,
                                             TTable, SuitRanks, bin_coeffs));
      node->UndoAction(actions[i]);
      alpha = std::max(val, alpha);
      if (val >= beta) {
        break;
      }
    }
  } else if (node->Player() == 1) {
    val = node->TotalTricks();
    std::vector<ActionStruct> actions = node->LegalActions();
    for (int i = 0; i < actions.size(); ++i) {
      node->ApplyAction(actions[i]);
      val = std::min(
          val, IncrementalAlphaBetaMemoryIso(node, alpha, beta, depth - 1,
                                             TTable, SuitRanks, bin_coeffs));
      node->UndoAction(actions[i]);
      beta = std::min(val, beta);
      if (val <= alpha) {
        break;
      }
    }
  }
  return val;
};

std::vector<Node> GWhistGenerator(int num, unsigned int seed) {
  // generates pseudorandom endgames//
  std::vector<Node> out;
  out.reserve(num);
  std::mt19937 g(seed);
  std::array<int, 2 * kNumRanks> nums;
  for (int i = 0; i < 2 * kNumRanks; ++i) {
    nums[i] = i;
  }
  for (int i = 0; i < num; ++i) {
    std::shuffle(nums.begin(), nums.end(), g);
    uint32_t cards = 0;
    std::array<uint32_t, kNumSuits> suits;
    for (int j = 0; j < kNumRanks; ++j) {
      cards = cards | (1 << nums[j]);
    }
    int sum = 0;
    std::vector<int> suit_lengths = {0, 0, 0, 0};
    for (int j = 0; j < kNumSuits - 1; ++j) {
      int max = std::min(kNumRanks, 2 * kNumRanks - sum);
      int min = std::max(0, (j - 1) * kNumRanks - sum);
      std::uniform_int_distribution<> distrib(min, max);
      suit_lengths[j] = distrib(g);
      sum += suit_lengths[j];
    }
    suit_lengths[kNumSuits - 1] = 2 * kNumRanks - sum;
    sum = 0;
    for (int j = 0; j < kNumSuits; ++j) {
      sum += suit_lengths[j];
      if (suit_lengths[j] > kNumRanks) {
        throw;
      }
    }
    if (sum != 2 * kNumRanks) {
      for (int j = 0; j < suit_lengths.size(); ++j) {
        std::cout << suit_lengths[j] << " " << std::endl;
      }
      throw;
    }
    int cum_sum = 0;
    for (int j = 0; j < kNumSuits; ++j) {
      if (j == 0) {
        suits[j] = bzhi_u32(~0, suit_lengths[j]);
      } else {
        suits[j] =
            (bzhi_u32(~0, suit_lengths[j] + cum_sum)) ^ bzhi_u32(~0, cum_sum);
      }
      cum_sum += suit_lengths[j];
    }
    out.push_back(Node(cards, suits, 0, false));
#ifdef DEBUG
    std::cout << popcnt_u32(cards) << " "
              << popcnt_u32(suits[0]) + popcnt_u32(suits[1]) +
                     popcnt_u32(suits[2]) + popcnt_u32(suits[3])
              << std::endl;
    std::cout << cards << " " << suits[0] << " " << suits[1] << " " << suits[2]
              << " " << suits[3] << std::endl;
#endif
  }
  return out;
}

void ThreadSolver(int size_endgames, vectorNa* outTTable, vectorNa* TTable,
                  const std::vector<std::vector<uint32_t>>& bin_coeffs,
                  const std::vector<uint32_t>& suit_splits,
                  const std::unordered_map<uint32_t, uint32_t>& SuitRanks,
                  size_t start_id, size_t end_id) {
  // takes endgames solved to depth d-1 and returns endgames solved to depth d
  // //
  std::vector<int> combination;
  combination.reserve(size_endgames);
  for (int i = 0; i < size_endgames; ++i) {
    combination.push_back(i);
  }
  bool control = true;
  int count = 0;
  uint32_t cards = 0;
  for (int i = 0; i < combination.size(); ++i) {
    cards = cards | (1 << combination[i]);
  }
  while (count < start_id) {
    NextColex(combination, 2 * size_endgames);
    count++;
  }
  while (count < end_id && control) {
    uint32_t cards = 0;
    for (int i = 0; i < combination.size(); ++i) {
      cards = cards | (1 << combination[i]);
    }
    for (int i = 0; i < suit_splits.size(); ++i) {
      std::array<uint32_t, kNumSuits> suit_arr;
      suit_arr[0] = bzhi_u32(~0, suit_splits[i] & 0b1111);
      uint32_t sum = suit_splits[i] & 0b1111;
      for (int j = 1; j < kNumSuits; ++j) {
        uint32_t mask = bzhi_u32(~0, sum);
        sum += (suit_splits[i] & (0b1111 << (4 * j))) >> 4 * j;
        suit_arr[j] = bzhi_u32(~0, sum);
        suit_arr[j] = suit_arr[j] ^ mask;
      }
      Node node(cards, suit_arr, 0, false);
      char result = IncrementalAlphaBetaMemoryIso(
          &node, 0, size_endgames, 2, TTable, &SuitRanks, bin_coeffs);
      outTTable->Set(count, i, result);
    }
    control = NextColex(combination, 2 * size_endgames);
    count++;
  }
}
vectorNa RetroSolver(int size_endgames, vectorNa* TTable,
                     const std::vector<std::vector<uint32_t>>& bin_coeffs) {
  // takes endgames solved to depth d-1 and returns endgames solved to depth d
  // //
  vectorNa outTTable = InitialiseTTable(size_endgames, bin_coeffs);
  std::vector<uint32_t> suit_splits = GenQuads(size_endgames);
  std::unordered_map<uint32_t, uint32_t> SuitRanks;
  GenSuitRankingsRel(size_endgames - 1, &SuitRanks);
  std::vector<int> combination;
  combination.reserve(size_endgames);
  for (int i = 0; i < size_endgames; ++i) {
    combination.push_back(i);
  }
  uint32_t v_length = (suit_splits.size() >> 1) + 1;
  uint32_t min_block_size = 256;
  uint32_t hard_threads = std::thread::hardware_concurrency();
  uint32_t num_threads = 1;
  uint32_t num_outers = outTTable.GetOuterSize();
  // a haphazard attempt to mitigate false sharing//
  for (uint32_t i = hard_threads; i >= 1; i--) {
    if ((num_outers * v_length / i) >= min_block_size) {
      num_threads = i;
      break;
    }
  }
  std::vector<std::thread> threads = {};
  for (int i = 0; i < num_threads; ++i) {
    uint32_t block_size = num_outers / num_threads;
    uint32_t start_id;
    uint32_t end_id;
    if (num_threads == 1) {
      start_id = 0;
      end_id = num_outers;
    } else if (i == num_threads - 1) {
      start_id = block_size * (num_threads - 1);
      end_id = num_outers;
    } else {
      start_id = block_size * i;
      end_id = block_size * (i + 1);
    }
    threads.push_back(std::thread(
        ThreadSolver, size_endgames, &outTTable, TTable, std::ref(bin_coeffs),
        std::ref(suit_splits), std::ref(SuitRanks), start_id, end_id));
  }
  for (int i = 0; i < num_threads; ++i) {
    threads[i].join();
  }
  return outTTable;
}

bool TestRetroSolve(int samples, int depth, uint32_t seed,
                    const std::vector<std::vector<uint32_t>>& bin_coeffs) {
  // Tests endgame solution with TTable vs raw seach
  std::vector<Node> nodes = GWhistGenerator(samples, seed);
  vectorNa v;
  for (int i = 1; i <= depth; ++i) {
    v = RetroSolver(i, &v, bin_coeffs);
  }
  std::unordered_map<uint32_t, uint32_t> SuitRanks;
  GenSuitRankingsRel(depth, &SuitRanks);
  for (auto it = nodes.begin(); it != nodes.end(); ++it) {
    char abm_unsafe = IncrementalAlphaBetaMemoryIso(&*it, 0, kNumRanks,
                                                    2 * (kNumRanks - depth), &v,
                                                    &SuitRanks, bin_coeffs);
    char abm_safe = AlphaBeta(&*it, 0, kNumRanks);
    if (abm_unsafe != abm_safe) {
      return false;
    }
  }
  return true;
}
vectorNa BuildTablebase(const std::vector<std::vector<uint32_t>>& bin_coeffs) {
  vectorNa v;
  std::cout << "Building Tablebase"
            << "\n";
  for (int i = 1; i <= kNumRanks; ++i) {
    v = RetroSolver(i, &v, bin_coeffs);
    std::cout << "Done " << i << "\n";
  }
  std::cout << "Built Tablebase"
            << "\n";
  return v;
}
bool TestTablebase(int samples, uint32_t seed, const vectorNa& table_base,
                   const std::vector<std::vector<uint32_t>>& bin_coeffs) {
  std::vector<Node> nodes = GWhistGenerator(samples, seed);
  std::unordered_map<uint32_t, uint32_t> SuitRanks;
  GenSuitRankingsRel(kNumRanks, &SuitRanks);
  for (auto it = nodes.begin(); it != nodes.end(); ++it) {
    char abm_unsafe = IncrementalAlphaBetaMemoryIso(
        &*it, 0, kNumRanks, 0, &table_base, &SuitRanks, bin_coeffs);
    char abm_safe = AlphaBeta(&*it, 0, kNumRanks);
    if (abm_unsafe != abm_safe) {
      return false;
    }
  }
  return true;
}
void StoreTTable(const std::string filename, const vectorNa& solution) {
  // stores solution into a text file//
  std::ofstream file(filename);
  for (int i = 0; i < solution.GetOuterSize(); ++i) {
    for (int j = 0; j < solution.GetInnerSize(); ++j) {
      file.put(solution.GetChar(i, j));
    }
  }
  file.close();
}

bool TestTTableStorage(std::string filename, const vectorNa& v, int depth,
                       const std::vector<std::vector<uint32_t>>& bin_coeffs) {
  // Tests storage fidelity//
  StoreTTable(filename, v);
  vectorNa new_v = LoadTTable(filename, depth, bin_coeffs);
  for (int i = 0; i < v.GetOuterSize(); ++i) {
    for (int j = 0; j < v.GetInnerSize(); ++j) {
      if (v.GetChar(i, j) != new_v.GetChar(i, j)) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace german_whist_foregame
}  // namespace open_spiel

int main() {
  std::vector<std::vector<uint32_t>> bin_coeffs =
      open_spiel::german_whist_foregame::BinCoeffs(
          2 * open_spiel::german_whist_foregame::kNumRanks);
  open_spiel::german_whist_foregame::vectorNa tablebase =
      open_spiel::german_whist_foregame::BuildTablebase(bin_coeffs);
  std::random_device rd;
  int num_samples = 100;
  if (open_spiel::german_whist_foregame::TestTablebase(num_samples, rd(),
                                                       tablebase, bin_coeffs)) {
    std::cout << "Tablebase accurate" << std::endl;
  } else {
    std::cout << "Tablebase inaccurate" << std::endl;
  }
  std::cout << "Starting Saving Tablebase" << std::endl;
  open_spiel::german_whist_foregame::StoreTTable("TTable13.txt", tablebase);
  std::cout << "Finished Saving Tablebase" << std::endl;
}
