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

#include "open_spiel/games/universal_poker.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/universal_poker/logic/betting_tree.h"
#include "open_spiel/games/universal_poker/logic/card_set.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace universal_poker {

const absl::string_view kNoGameDef = "NoGameDef";

const GameType kGameType{
    /*short_name=*/"universal_poker",
    /*long_name=*/"Universal Poker",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/10,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    //
    // The ACPC code uses a specific configuration file to describe the game.
    // The following has been copied from ACPC documentation:
    // """
    // Game definitions can have the following fields (case is ignored):
    //
    // gamedef - the starting tag for a game definition
    // end gamedef - ending tag for a game definition
    // stack - the stack size for each player at the start of each hand (for
    //   no-limit)
    // blind - the size of the blinds for each player (relative to the dealer)
    // raisesize - the size of raises on each round (for limit games)
    // limit - specifies a limit game
    // nolimit - specifies a no-limit game
    // numplayers - number of players in the game
    // numrounds - number of betting rounds per hand of the game
    // firstplayer - the player that acts first (relative to the dealer) on each
    //   round
    // maxraises - the maximum number of raises on each round
    // numsuits - the number of different suits in the deck
    // numranks - the number of different ranks in the deck
    // numholecards - the number of private cards to  deal to each player
    // numboardcards - the number of cards revealed on each round
    //
    // Empty lines or lines with '#' as the very first character will be ignored
    //
    // If you are creating your own game definitions, please note that game.h
    // defines some constants for maximums in games (e.g., number of rounds).
    // These may need to be changed for games outside of the what is being run
    // for the Annual Computer Poker Competition.
    // """
    {// The ACPC gamedef string.  When present, it will take precedence over
     // everything and no other argument should be provided.
     {"gamedef", GameParameter(std::string(kNoGameDef))},
     // Number of Players (up to 10)
     {"players", GameParameter(2)},
     // Betting Type "limit" "nolimit" (currently only nolimit supported)
     {"bettingType", GameParameter(std::string("nolimit"))},
     // Stack of money per Player
     {"stackPerPlayer", GameParameter(1200)},
     {"bigBlind", GameParameter(100)},
     {"smallBlind", GameParameter(100)},
     // Count of Rounds
     {"rounds", GameParameter(2)},
     // Who is the first player by round?
     {"firstPlayer", GameParameter(std::string("1 1"))},
     {"numSuits", GameParameter(4)},
     {"numRanks", GameParameter(6)},
     // Hole Cards (Private Cards) per Player
     {"numHoleCards", GameParameter(1)},
     // Board Cards (Public Cards) per Player
     {"numBoardCards", GameParameter(std::string("0 1"))}}};

std::shared_ptr<const Game> Factory(const GameParameters &params) {
  return std::shared_ptr<const Game>(new UniversalPokerGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

// namespace universal_poker
UniversalPokerState::UniversalPokerState(std::shared_ptr<const Game> game)
    : State(game),
      acpc_game_(
          static_cast<const UniversalPokerGame *>(game.get())->GetACPCGame()),
      betting_node_(acpc_game_),
      deck_(/*num_suits=*/acpc_game_->NumSuitsDeck(),
            /*num_ranks=*/acpc_game_->NumRanksDeck()),
      hole_cards_(acpc_game_->GetNbPlayers()) {
  SPIEL_CHECK_EQ(betting_node_.GetNodeType(),
                 logic::BettingNode::NODE_TYPE_CHANCE);
}

std::string UniversalPokerState::ToString() const {
  std::ostringstream buf;

  for (int p = 0; p < acpc_game_->GetNbPlayers(); ++p) {
    buf << "P" << p << " Cards: " << hole_cards_[p].ToString()
        << std::endl;
  }
  buf << "BoardCards " << board_cards_.ToString() << std::endl;

  if (IsChanceNode()) {
    buf << "PossibleCardsToDeal " << deck_.ToString() << std::endl;
  }
  if (betting_node_.GetNodeType() ==
          logic::BettingNode::NODE_TYPE_TERMINAL_FOLD ||
      betting_node_.GetNodeType() ==
          logic::BettingNode::NODE_TYPE_TERMINAL_SHOWDOWN) {
    for (int p = 0; p < acpc_game_->GetNbPlayers(); ++p) {
      buf << "P" << p << " Reward: " << GetTotalReward(p) << std::endl;
    }
  }
  buf << betting_node_.ToString();

  return buf.str();
}

bool UniversalPokerState::IsTerminal() const {
  bool finished = betting_node_.GetNodeType() ==
                      logic::BettingNode::NODE_TYPE_TERMINAL_SHOWDOWN ||
                  betting_node_.GetNodeType() ==
                      logic::BettingNode::NODE_TYPE_TERMINAL_FOLD;
  assert(betting_node_.IsFinished() || !finished);
  return finished;
}

std::string UniversalPokerState::ActionToString(Player player,
                                                Action move) const {
  return absl::StrCat("player=", player, " move=", move);
}

Player UniversalPokerState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  }
  if (betting_node_.GetNodeType() == logic::BettingNode::NODE_TYPE_CHANCE) {
    return kChancePlayerId;
  }

  return Player(betting_node_.CurrentPlayer());
}

std::vector<double> UniversalPokerState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>(NumPlayers(), 0.0);
  }

  std::vector<double> returns(NumPlayers());
  for (Player player = 0; player < NumPlayers(); ++player) {
    // Money vs money at start.
    returns[player] = GetTotalReward(player);
  }

  return returns;
}

void UniversalPokerState::InformationStateTensor(
    Player player, std::vector<double> *values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  values->resize(game_->InformationStateTensorShape()[0]);
  std::fill(values->begin(), values->end(), 0.);

  // Layout of observation:
  //   my player number: num_players bits
  //   my cards: Initial deck size bits (1 means you have the card), i.e.
  //             MaxChanceOutcomes() = NumSuits * NumRanks
  //   public cards: Same as above, but for the public cards.
  //   NumRounds() round sequence: (max round seq length)*2 bits
  int offset = 0;

  // Mark who I am.
  (*values)[player] = 1;
  offset += NumPlayers();

  const logic::CardSet full_deck(acpc_game_->NumSuitsDeck(),
                                 acpc_game_->NumRanksDeck());
  const std::vector<uint8_t> deckCards = full_deck.ToCardArray();
  logic::CardSet holeCards = hole_cards_[player];
  // TODO(author2): it should be way more efficient to iterate over the cards
  // of the player, rather than iterating over all the cards.
  for (uint32_t i = 0; i < full_deck.NumCards(); i++) {
    (*values)[i + offset] = holeCards.ContainsCards(deckCards[i]) ? 1.0 : 0.0;
  }
  offset += full_deck.NumCards();

  // Public cards
  for (int i = 0; i < full_deck.NumCards(); ++i) {
    (*values)[i + offset] =
        board_cards_.ContainsCards(deckCards[i]) ? 1.0 : 0.0;
  }
  offset += full_deck.NumCards();

  const std::string actionSeq = betting_node_.GetActionSequence();
  const int length = actionSeq.length();
  SPIEL_CHECK_LT(length, game_->MaxGameLength());

  for (int i = 0; i < length; ++i) {
    SPIEL_CHECK_LT(offset + i + 1, values->size());
    if (actionSeq[i] == 'c') {
      // Encode call as 10.
      (*values)[offset + (2 * i)] = 1;
      (*values)[offset + (2 * i) + 1] = 0;
    } else if (actionSeq[i] == 'p') {
      // Encode raise as 01.
      (*values)[offset + (2 * i)] = 0;
      (*values)[offset + (2 * i) + 1] = 1;
    } else if (actionSeq[i] == 'a') {
      // Encode raise as 01.
      (*values)[offset + (2 * i)] = 1;
      (*values)[offset + (2 * i) + 1] = 1;
    } else if (actionSeq[i] == 'f') {
      // Encode fold as 00.
      // TODO(author2): Should this be 11?
      (*values)[offset + (2 * i)] = 0;
      (*values)[offset + (2 * i) + 1] = 0;
    } else if (actionSeq[i] == 'd') {
      (*values)[offset + (2 * i)] = 0;
      (*values)[offset + (2 * i) + 1] = 0;
    } else {
      SPIEL_CHECK_EQ(actionSeq[i], 'd');
    }
  }

  // Move offset up to the next round: 2 bits per move.
  offset += game_->MaxGameLength() * 2;
  SPIEL_CHECK_EQ(offset, game_->InformationStateTensorShape()[0]);
}

void UniversalPokerState::ObservationTensor(Player player,
                                            std::vector<double> *values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, NumPlayers());

  values->resize(game_->ObservationTensorShape()[0]);
  std::fill(values->begin(), values->end(), 0.);

  // Layout of observation:
  //   my player number: num_players bits
  //   my cards: Initial deck size bits (1 means you have the card), i.e.
  //             MaxChanceOutcomes() = NumSuits * NumRanks
  //   public cards: Same as above, but for the public cards.
  //   the contribution of each player to the pot. num_players integers.
  int offset = 0;

  // Mark who I am.
  (*values)[player] = 1;
  offset += NumPlayers();

  const logic::CardSet full_deck(acpc_game_->NumSuitsDeck(),
                                 acpc_game_->NumRanksDeck());
  const std::vector<uint8_t> all_cards = full_deck.ToCardArray();
  logic::CardSet holeCards = hole_cards_[player];

  for (uint32_t i = 0; i < full_deck.NumCards(); i++) {
    (*values)[i + offset] = holeCards.ContainsCards(all_cards[i]) ? 1.0 : 0.0;
  }
  offset += full_deck.NumCards();

  for (uint32_t i = 0; i < full_deck.NumCards(); i++) {
    (*values)[i + offset] =
        board_cards_.ContainsCards(all_cards[i]) ? 1.0 : 0.0;
  }
  offset += full_deck.NumCards();

  // Adding the contribution of each players to the pot.
  for (auto p = Player{0}; p < NumPlayers(); p++) {
    (*values)[offset + p] = betting_node_.Ante(p);
  }
  offset += NumPlayers();
  SPIEL_CHECK_EQ(offset, game_->ObservationTensorShape()[0]);
}

std::string UniversalPokerState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, acpc_game_->GetNbPlayers());
  const uint32_t pot = betting_node_.MaxSpend() *
                       (acpc_game_->GetNbPlayers() - betting_node_.NumFolded());
  std::vector<int> money;
  for (auto p = Player{0}; p < acpc_game_->GetNbPlayers(); p++) {
    money.emplace_back(betting_node_.Money(p));
  }
  std::vector<std::string> sequences;
  for (auto r = 0; r <= betting_node_.GetRound(); r++) {
    sequences.emplace_back(betting_node_.BettingSequence(r));
  }

  return absl::StrFormat(
      "[Round %i][Player: %i][Pot: %i][Money: %s][Private: %s][Public: "
      "%s][Sequences: %s]",
      betting_node_.GetRound(), CurrentPlayer(), pot, absl::StrJoin(money, " "),
      hole_cards_[player].ToString(), board_cards_.ToString(),
      absl::StrJoin(sequences, "¦"));
}

std::string UniversalPokerState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, acpc_game_->GetNbPlayers());
  std::string result;

  const uint32_t pot = betting_node_.MaxSpend() *
                       (acpc_game_->GetNbPlayers() - betting_node_.NumFolded());
  absl::StrAppend(&result, "[Round ", betting_node_.GetRound(),
                  "][Player: ", CurrentPlayer(), "][Pot: ", pot, "][Money:");
  for (auto p = Player{0}; p < acpc_game_->GetNbPlayers(); p++) {
    absl::StrAppend(&result, " ", betting_node_.Money(p));
  }
  // Add the player's private cards
  if (player != kChancePlayerId) {
    absl::StrAppend(&result, "[Private: ", hole_cards_[player].ToString(), "]");
  }
  // Adding the contribution of each players to the pot
  absl::StrAppend(&result, "[Ante:");
  for (auto p = Player{0}; p < num_players_; p++) {
    absl::StrAppend(&result, " ", betting_node_.Ante(p));
  }
  absl::StrAppend(&result, "]");

  return result;
}

std::unique_ptr<State> UniversalPokerState::Clone() const {
  return std::unique_ptr<State>(new UniversalPokerState(*this));
}

std::vector<std::pair<Action, double>> UniversalPokerState::ChanceOutcomes()
    const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  std::vector<uint8_t> available_cards = deck_.ToCardArray();
  const int num_cards = available_cards.size();
  const double p = 1.0 / num_cards;

  // We need to cast std::vector<uint8_t> into std::vector<Action>.
  std::vector<std::pair<Action, double>> outcomes;
  outcomes.reserve(num_cards);
  for (const auto &card : available_cards) {
    outcomes.push_back({Action{card}, p});
  }
  return outcomes;
}

std::vector<Action> UniversalPokerState::LegalActions() const {
  if (betting_node_.GetNodeType() == logic::BettingNode::NODE_TYPE_CHANCE) {
    std::vector<uint8_t> available_cards = deck_.ToCardArray();
    std::vector<Action> actions;
    actions.reserve(available_cards.size());
    for (const auto &card : available_cards) {
      actions.push_back(card);
    }
    return actions;
  }

  int num_actions = betting_node_.GetPossibleActionCount();
  std::vector<Action> actions(num_actions, 0);
  std::iota(actions.begin(), actions.end(), 0);
  return actions;
}

void UniversalPokerState::DoApplyAction(Action action_id) {
  if (IsChanceNode()) {
    betting_node_.ApplyDealCards();
    // In chance nodes, the action_id is exactly the card being dealt.
    uint8_t card = action_id;
    deck_.RemoveCard(card);

    // Check where to add this card
    for (int p = 0; p < acpc_game_->GetNbPlayers(); ++p) {
      if (hole_cards_[p].NumCards() < acpc_game_->GetNbHoleCardsRequired()) {
        hole_cards_[p].AddCard(card);
        break;
      }
    }

    if (board_cards_.NumCards() <
        acpc_game_->GetNbBoardCardsRequired(betting_node_.GetRound())) {
      board_cards_.AddCard(card);
    }
  } else {
    uint32_t idx = 0;
    for (auto action : logic::BettingNode::ALL_ACTIONS) {
      if (action & betting_node_.GetPossibleActionsMask()) {
        if (idx == action_id) {
          betting_node_.ApplyChoiceAction(action);
          break;
        }
        idx++;
      }
    }
  }
}

double UniversalPokerState::GetTotalReward(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, acpc_game_->GetNbPlayers());
  // Copy Board Cards and Hole Cards
  uint8_t holeCards[10][3], boardCards[7], nbHoleCards[10];

  for (size_t p = 0; p < hole_cards_.size(); ++p) {
    auto cards = hole_cards_[p].ToCardArray();
    for (size_t c = 0; c < cards.size(); ++c) {
      holeCards[p][c] = cards[c];
    }
    nbHoleCards[p] = cards.size();
  }

  auto bc = board_cards_.ToCardArray();
  for (size_t c = 0; c < bc.size(); ++c) {
    boardCards[c] = bc[c];
  }

  betting_node_.SetHoleAndBoardCards(holeCards, boardCards, nbHoleCards,
                                     /*nbBoardCards=*/bc.size());

  return betting_node_.ValueOfState(player);
}

/**
 * Universal Poker Game Constructor
 * @param params
 */
UniversalPokerGame::UniversalPokerGame(const GameParameters &params)
    : Game(kGameType, params),
      gameDesc_(parseParameters(params)),
      acpc_game_(gameDesc_) {}

std::unique_ptr<State> UniversalPokerGame::NewInitialState() const {
  return std::unique_ptr<State>(new UniversalPokerState(shared_from_this()));
}

std::vector<int> UniversalPokerGame::InformationStateTensorShape() const {
  // One-hot encoding for player number (who is to play).
  // 2 slots of cards (total_num_cards bits each): private card, public card
  // Followed by maximum game length * 2 bits each (call / raise)
  const int num_players = acpc_game_.GetNbPlayers();
  const int gameLength = MaxGameLength();
  const int total_num_cards = MaxChanceOutcomes();

  return {num_players + 2 * total_num_cards + 2 * gameLength};
}

std::vector<int> UniversalPokerGame::ObservationTensorShape() const {
  // One-hot encoding for player number (who is to play).
  // 2 slots of cards (total_cards_ bits each): private card, public card
  // Followed by the contribution of each player to the pot
  const int num_players = acpc_game_.GetNbPlayers();
  const int total_num_cards = MaxChanceOutcomes();
  return {2 * (num_players + total_num_cards)};
}

double UniversalPokerGame::MaxUtility() const {
  // In poker, the utility is defined as the money a player has at the end of
  // the game minus then money the player had before starting the game.
  // The most a player can win *per opponent* is the most each player can put
  // into the pot, which is the raise amounts on each round times the maximum
  // number raises, plus the original chip they put in to play.

  return (double)acpc_game_.StackSize(0) * (acpc_game_.GetNbPlayers() - 1);
}

double UniversalPokerGame::MinUtility() const {
  // In poker, the utility is defined as the money a player has at the end of
  // the game minus then money the player had before starting the game.
  // The most any single player can lose is the maximum number of raises per
  // round times the amounts of each of the raises, plus the original chip they
  // put in to play.
  return -1 * (double)acpc_game_.StackSize(0);
}

int UniversalPokerGame::MaxChanceOutcomes() const {
  return acpc_game_.NumSuitsDeck() * acpc_game_.NumRanksDeck();
}

int UniversalPokerGame::NumPlayers() const { return acpc_game_.GetNbPlayers(); }

int UniversalPokerGame::NumDistinctActions() const {
  return logic::GetMaxBettingActions(acpc_game_);
}

std::shared_ptr<const Game> UniversalPokerGame::Clone() const {
  return std::shared_ptr<const Game>(new UniversalPokerGame(*this));
}

int UniversalPokerGame::MaxGameLength() const {
  // Make a good guess here because bruteforcing the tree is far too slow
  // One Terminal Action
  int length = 1;

  // Deal Actions
  length += acpc_game_.GetTotalNbBoardCards() +
            acpc_game_.GetNbHoleCardsRequired() * acpc_game_.GetNbPlayers();

  // Check Actions
  length += (NumPlayers() * acpc_game_.NumRounds());

  // Bet Actions
  double maxStack = 0;
  double maxBlind = 0;
  for (uint32_t p = 0; p < NumPlayers(); p++) {
    maxStack =
        acpc_game_.StackSize(p) > maxStack ? acpc_game_.StackSize(p) : maxStack;
    maxBlind =
        acpc_game_.BlindSize(p) > maxStack ? acpc_game_.BlindSize(p) : maxBlind;
  }

  while (maxStack > maxBlind) {
    maxStack /= 2.0;         // You have always to bet the pot size
    length += NumPlayers();  // Each player has to react
  }

  return length;
}

/**
 * Parses the Game Paramters and makes a gameDesc out of it
 * @param map
 * @return
 */
std::string UniversalPokerGame::parseParameters(const GameParameters &map) {
  if (map.find("gamedef") != map.end()) {
    // We check for sanity that all parameters are empty
    if (map.size() != 1) {
      std::vector<std::string> game_parameter_keys;
      game_parameter_keys.reserve(map.size());
      for (auto const &imap : map) {
        game_parameter_keys.push_back(imap.first);
      }
      SpielFatalError(
          absl::StrCat("When loading a 'universal_poker' game, the 'gamedef' "
                       "field was present, but other fields were present too: ",
                       absl::StrJoin(game_parameter_keys, ", "),
                       "gamedef is exclusive with other paraemters."));
    }
    std::string retreived_gamedef = ParameterValue<std::string>("gamedef");
    std::cerr << "gamedef directly passed for Universal Poker:\n"
              << retreived_gamedef << std::endl;
    return retreived_gamedef;
  }

  std::string generated_gamedef = "GAMEDEF\n";
  absl::StrAppend(
      &generated_gamedef, ParameterValue<std::string>("bettingType"), "\n",
      "numPlayers = ", ParameterValue<int>("players"), "\n",
      "numRounds = ", ParameterValue<int>("rounds"), "\n",
      "numSuits = ", ParameterValue<int>("numSuits"), "\n",
      "firstPlayer = ", ParameterValue<std::string>("firstPlayer"), "\n",
      "numRanks = ", ParameterValue<int>("numRanks"), "\n",
      "numHoleCards = ", ParameterValue<int>("numHoleCards"), "\n",
      "numBoardCards = ", ParameterValue<std::string>("numBoardCards"), "\n");

  absl::StrAppend(&generated_gamedef, "stack = ");
  for (int p = 0; p < ParameterValue<int>("players"); p++) {
    absl::StrAppend(&generated_gamedef, ParameterValue<int>("stackPerPlayer"),
                    " ");
  }
  absl::StrAppend(&generated_gamedef, "\n");

  absl::StrAppend(&generated_gamedef, "blind = ");
  for (int p = 0; p < ParameterValue<int>("players"); p++) {
    if (p == 0) {
      absl::StrAppend(&generated_gamedef, ParameterValue<int>("bigBlind"), " ");
    } else if (p == 1) {
      absl::StrAppend(&generated_gamedef, ParameterValue<int>("smallBlind"),
                      " ");
    } else {
      absl::StrAppend(&generated_gamedef, "0 ");
    }
  }
  absl::StrAppend(&generated_gamedef, "\n");

  absl::StrAppend(&generated_gamedef, "END GAMEDEF\n");
  std::cerr << "Generated gamedef for Universal Poker:\n"
            << generated_gamedef << std::endl;
  return generated_gamedef;
}

}  // namespace universal_poker
}  // namespace open_spiel
