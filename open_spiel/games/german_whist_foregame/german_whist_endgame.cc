// Source Code for an Executable Generating an Endgame Tablebase for German
// Whist

#include <cassert>
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "german_whist_foregame.h"
#include "open_spiel/games/german_whist_foregame/german_whist_foregame.h"
#include "open_spiel/utils/file.h"
#include "open_spiel/utils/thread.h"

ABSL_FLAG(std::string, output_path, "TTable13.txt",
          "Where to write the generated tablebase.");

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
  bool Trick(ActionStruct lead, ActionStruct follow) {
    // true if leader won//
    return (lead.suit != follow.suit && follow.suit != trump_) ||
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
  std::tuple<uint32_t,uint32_t,uint32_t> Canonicalise() {
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
    using suit_info = std::tuple<bool,uint32_t,uint32_t,uint32_t>;
    std::vector<suit_info> suit_infos={{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};
    for(uint8_t i=0;i<kNumSuits;++i){
      bool is_trump = (i==trump_);
      uint32_t suit_length = popcnt_u32(suit_masks_[i]);
      uint32_t suit_sig =(suit_masks_[i]&cards_)>>(tzcnt_u32(suit_masks_[i]));
      uint32_t suit_idx = i;
      suit_infos[i]={is_trump,suit_length,suit_sig,suit_idx};
    }
    auto custom_cmp = [&](const suit_info&lhs,const suit_info&rhs){
      const auto& [t1, l1,s1,i1] = lhs;
      const auto& [t2, l2,s2,i2] = rhs;
      return std::tie(t2, l1,i1) < std::tie(t1, l2,i2);
    };
    std::sort(suit_infos.begin(),suit_infos.end(),custom_cmp);
    uint32_t bitpacked_suit_lengths = 0;
    uint32_t card_mask=0;
    uint32_t total_cards=0;
    for(uint32_t i=0;i<kNumSuits;++i){
      uint32_t suit_length = std::get<1>(suit_infos[i]);
      bitpacked_suit_lengths = (bitpacked_suit_lengths|(suit_length<<(4*i)));
      card_mask = (card_mask)|(std::get<2>(suit_infos[i])<<(total_cards));
      total_cards+=suit_length;
    }
    uint32_t sel_mask = ((0b1<<(total_cards))-1);
    uint32_t alt_card_mask = (~card_mask)&sel_mask;
    return {card_mask,alt_card_mask,bitpacked_suit_lengths};
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
    for (uint32_t i = 0; i < moves.size(); ++i) {
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
    for (uint32_t i = 0; i < kNumSuits; ++i) {
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
    for (uint32_t i = 0; i < out.size(); ++i) {
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

//CREDIT TO GOSPERS//
uint32_t NextColex(uint32_t n){
  const uint32_t c = n & -n;
  const uint32_t r = n + c;
  return ( ( ( r ^ n ) >> 2 ) / c ) | r;
}

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
    Node* node, char alpha, char beta, int depth, const vectorNa* TTable,
    const std::unordered_map<uint32_t, uint32_t>* SuitRanks) {
  // fail soft ab search
  char val = 0;
  uint64_t key = 0;
  bool player = node->Player();
  if (node->IsTerminal()) {
    return node->Score();
  }
  if (node->Moves() % 2 == 0 && depth == 0) {
    std::tuple<uint32_t,uint32_t,uint32_t> canonical = node->Canonicalise();
    uint32_t suits = std::get<2>(canonical);
    uint32_t cards = std::get<0>(canonical);
    uint32_t alt_cards = std::get<1>(canonical);

    cards  = (player) ? alt_cards : cards;
    uint32_t colex = Colex(cards);
    uint32_t suit_rank = SuitRanks->at(suits);
    char value = (player)
                     ? node->RemainingTricks() - TTable->Get(suit_rank,colex)
                     : TTable->Get(suit_rank,colex);
    return value + node->Score();
  } else if (node->Player() == 0) {
    val = 0;
    std::vector<ActionStruct> actions = node->LegalActions();
    for (int i = 0; i < actions.size(); ++i) {
      node->ApplyAction(actions[i]);
      val = std::max(
          val, IncrementalAlphaBetaMemoryIso(node, alpha, beta, depth - 1,
                                             TTable, SuitRanks));
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
                                             TTable, SuitRanks));
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

void ThreadSolver(uint32_t size_endgames, vectorNa* outTTable,
                  const vectorNa* TTable,
                  const std::vector<uint32_t>& suit_splits,
                  const std::unordered_map<uint32_t, uint32_t>& SuitRanks,
                  uint32_t start_id, uint32_t end_id) {
  // takes endgames solved to depth d-1 and returns endgames solved to depth d
  //NEW//
  uint32_t cards = (1<<size_endgames)-1;
  for (uint32_t i = start_id; i < end_id; ++i) {
    std::array<uint32_t, kNumSuits> suit_arr;
    suit_arr[0] = bzhi_u32(~0, suit_splits[i] & 0b1111);
    uint32_t sum = suit_splits[i] & 0b1111;
    for (uint32_t j = 1; j < kNumSuits; ++j) {
      uint32_t mask = bzhi_u32(~0, sum);
      sum += (suit_splits[i] & (0b1111 << (4 * j))) >> 4 * j;
      suit_arr[j] = bzhi_u32(~0, sum);
      suit_arr[j] = suit_arr[j] ^ mask;
    }
    for(uint32_t colex_rank =0;colex_rank<BIN_COEFFS_LUT[2*size_endgames][size_endgames];++colex_rank){
      Node node(cards, suit_arr, 0, false);
      char result = IncrementalAlphaBetaMemoryIso(
        &node, 0, size_endgames, 2, TTable, &SuitRanks);
      outTTable->Set(i,colex_rank,result);
      cards = NextColex(cards);
    }
  }
  return;
}

vectorNa RetroSolver(int size_endgames, vectorNa* TTable,
                     const uint32_t hard_threads) {
  // takes endgames solved to depth d-1 and returns endgames solved to depth d
  // //
  vectorNa outTTable = InitialiseTTable(size_endgames);
  std::vector<uint32_t> suit_splits = GenQuads(size_endgames);
  std::unordered_map<uint32_t, uint32_t> SuitRanks;
  GenSuitRankingsRel(size_endgames - 1, &SuitRanks);
  std::vector<int> combination;
  combination.reserve(size_endgames);
  for (int i = 0; i < size_endgames; ++i) {
    combination.push_back(i);
  }

  uint32_t min_block_size = 256;
  uint32_t num_threads = 1;
  uint32_t num_outers = suit_splits.size();
  // a haphazard attempt to mitigate false sharing//
  for (uint32_t i = hard_threads; i >= 1; i--) {
    if ((outTTable.size() / i) >= min_block_size) {
      num_threads = i;
      break;
    }
  }
  std::vector<Thread> threads = {};
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
    threads.emplace_back([&, start_id, end_id]() {
      ThreadSolver(size_endgames, &outTTable, TTable,
                   std::ref(suit_splits), std::ref(SuitRanks), start_id,
                   end_id);
    });
  }
  for (int i = 0; i < num_threads; ++i) {
    threads[i].join();
  }
  return outTTable;
}

bool TestRetroSolve(int samples, int depth, uint32_t seed,
                    const uint32_t hard_threads) {
  // Tests endgame solution with TTable vs raw seach
  std::vector<Node> nodes = GWhistGenerator(samples, seed);
  vectorNa v;
  for (int i = 1; i <= depth; ++i) {
    v = RetroSolver(i, &v,hard_threads);
  }
  std::unordered_map<uint32_t, uint32_t> SuitRanks;
  GenSuitRankingsRel(depth, &SuitRanks);
  for (auto it = nodes.begin(); it != nodes.end(); ++it) {
    char abm_unsafe = IncrementalAlphaBetaMemoryIso(&*it, 0, kNumRanks,
                                                    2 * (kNumRanks - depth), &v,
                                                    &SuitRanks);
    char abm_safe = AlphaBeta(&*it, 0, kNumRanks);
    if (abm_unsafe != abm_safe) {
      return false;
    }
  }
  return true;
}
vectorNa BuildTablebase(const uint32_t hard_threads) {
  vectorNa v;
  std::cout << "Building Tablebase"
            << "\n";
  for (int i = 1; i <= kNumRanks; ++i) {
    v = RetroSolver(i, &v,hard_threads);
    std::cout << "Done " << i << "\n";
  }
  std::cout << "Built Tablebase"
            << "\n";
  return v;
}
bool TestTablebase(int samples, uint32_t seed, const vectorNa& table_base) {
  std::vector<Node> nodes = GWhistGenerator(samples, seed);
  std::unordered_map<uint32_t, uint32_t> SuitRanks;
  GenSuitRankingsRel(kNumRanks, &SuitRanks);
  for (auto it = nodes.begin(); it != nodes.end(); ++it) {
    char abm_unsafe = IncrementalAlphaBetaMemoryIso(
        &*it, 0, kNumRanks, 0, &table_base, &SuitRanks);
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
  for (int i = 0; i < solution.size(); ++i) {
      file.put(solution.GetChar(i));
  }
  file.close();
}

bool TestTTableStorage(std::string filename, const vectorNa& v, int depth) {
  // Tests storage fidelity//
  StoreTTable(filename, v);
  vectorNa new_v = LoadTTable(filename, depth);
  for (int i = 0; i < v.size(); ++i) {
      if (v.GetChar(i) != new_v.GetChar(i)) {
        return false;
      }
  }
  return true;
}

}  // namespace german_whist_foregame
}  // namespace open_spiel

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  const uint32_t hard_threads = 8;//set this to take advantage of more cores on your machine//
  open_spiel::german_whist_foregame::vectorNa tablebase =
  open_spiel::german_whist_foregame::BuildTablebase(hard_threads);
  std::random_device rd;
  int num_samples = 100;
  if (open_spiel::german_whist_foregame::TestTablebase(num_samples, rd(),
    tablebase)) {
    std::cout << "Tablebase accurate" << std::endl;
    } else {
      std::cout << "Tablebase inaccurate" << std::endl;
    }
    std::cout << "Starting Saving Tablebase" << std::endl;
  open_spiel::german_whist_foregame::StoreTTable(
    absl::GetFlag(FLAGS_output_path), tablebase);
  std::cout << "Finished Saving Tablebase" << std::endl;
}
