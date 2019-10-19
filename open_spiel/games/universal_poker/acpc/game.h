/*
Copyright (C) 2011 by the Computer Poker Research Group, University of Alberta
*/

#ifndef _GAME_H
#define _GAME_H
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include "rng.h"
#include "net.h"


#define VERSION_MAJOR 2
#define VERSION_MINOR 0
#define VERSION_REVISION 0


#define MAX_ROUNDS 4
#define MAX_PLAYERS 10
#define MAX_BOARD_CARDS 7
#define MAX_HOLE_CARDS 3
#define MAX_NUM_ACTIONS 64
#define MAX_SUITS 4
#define MAX_RANKS 13
#define MAX_LINE_LEN READBUF_LEN

#define NUM_ACTION_TYPES 3


enum BettingType { limitBetting, noLimitBetting };
enum ActionType { a_fold = 0, a_call = 1, a_raise = 2,
		  a_invalid = NUM_ACTION_TYPES };

typedef struct {
  enum ActionType type; /* is action a fold, call, or raise? */
  int32_t size; /* for no-limit raises, we need a size
		   MUST BE 0 IN ALL CASES WHERE IT IS NOT USED */
} Action;

typedef struct {

  /* stack sizes for each player */
  int32_t stack[ MAX_PLAYERS ];

  /* entry fee for game, per player */
  int32_t blind[ MAX_PLAYERS ];

  /* size of fixed raises for limitBetting games */
  int32_t raiseSize[ MAX_ROUNDS ];

  /* general class of game */
  enum BettingType bettingType;

  /* number of players in the game */
  uint8_t numPlayers;

  /* number of betting rounds */
  uint8_t numRounds;

  /* first player to act in a round */
  uint8_t firstPlayer[ MAX_ROUNDS ];

  /* number of bets/raises that may be made in each round */
  uint8_t maxRaises[ MAX_ROUNDS ];

  /* number of suits and ranks in the deck of cards */
  uint8_t numSuits;
  uint8_t numRanks;

  /* number of private player cards */
  uint8_t numHoleCards;

  /* number of shared public cards each round */
  uint8_t numBoardCards[ MAX_ROUNDS ];
} Game;

typedef struct {
  uint32_t handId;

  /* largest bet so far, including all previous rounds */
  int32_t maxSpent;

  /* minimum number of chips a player must have spend in total to raise
     only used for noLimitBetting games */
  int32_t minNoLimitRaiseTo;

  /* spent[ p ] gives the total amount put into the pot by player p */
  int32_t spent[ MAX_PLAYERS ];

  /* action[ r ][ i ] gives the i'th action in round r */
  Action action[ MAX_ROUNDS ][ MAX_NUM_ACTIONS ];

  /* actingPlayer[ r ][ i ] gives the player who made action i in round r
     we can always figure this out from the actions taken, but it's
     easier to just remember this in multiplayer (because of folds) */
  uint8_t actingPlayer[ MAX_ROUNDS ][ MAX_NUM_ACTIONS ];

  /* numActions[ r ] gives the number of actions made in round r */
  uint8_t numActions[ MAX_ROUNDS ];

  /* current round: a value between 0 and game.numRounds-1
     a showdown is still in numRounds-1, not a separate round */
  uint8_t round;

  /* finished is non-zero if and only if the game is over */
  uint8_t finished;

  /* playerFolded[ p ] is non-zero if and only player p has folded */
  uint8_t playerFolded[ MAX_PLAYERS ];

  /* public cards (including cards which may not yet be visible to players) */
  uint8_t boardCards[ MAX_BOARD_CARDS ];

  /* private cards */
  uint8_t holeCards[ MAX_PLAYERS ][ MAX_HOLE_CARDS ];
} State;

typedef struct {
  State state;
  uint8_t viewingPlayer;
} MatchState;


/* returns a game structure, or NULL on failure */
Game *readGame( FILE *file );

void printGame( FILE *file, const Game *game );

/* initialise a state so that it is at the beginning of a hand
   DOES NOT DEAL OUT CARDS */
void initState( const Game *game, const uint32_t handId, State *state );

/* shuffle a deck of cards and deal them out, writing the results to state */
void dealCards( const Game *game, rng_state_t *rng, State *state );

int statesEqual( const Game *game, const State *a, const State *b );

int matchStatesEqual( const Game *game, const MatchState *a,
		      const MatchState *b );

/* check if a raise is possible, and the range of valid sizes
   returns non-zero if raise is a valid action and sets *minSize
   and maxSize, or zero if raise is not valid */
int raiseIsValid( const Game *game, const State *curState,
		  int32_t *minSize, int32_t *maxSize );

/* check if an action is valid

   if tryFixing is non-zero, try modifying the given action to produce
   a valid action, as in the AAAI rules.  Currently this only means
   that a no-limit raise will be modified to the nearest valid raise size

   returns non-zero if final action/size is valid for state, 0 otherwise */
int isValidAction( const Game *game, const State *curState,
		   const int tryFixing, Action *action );

/* record the given action in state
    does not check that action is valid */
void doAction( const Game *game, const Action *action, State *state );

/* returns non-zero if hand is finished, zero otherwise */
#define stateFinished( constStatePtr ) ((constStatePtr)->finished)

/* get the current player to act in the state */
uint8_t currentPlayer( const Game *game, const State *state );

/* number of raises in the current round */
uint8_t numRaises( const State *state );

/* number of players who have folded */
uint8_t numFolded( const Game *game, const State *state );

/* number of players who have called the current bet (or initiated it)
   doesn't count non-acting players who are all-in */
uint8_t numCalled( const Game *game, const State *state );

/* number of players who are all-in */
uint8_t numAllIn( const Game *game, const State *state );

/* number of players who can still act (ie not all-in or folded) */
uint8_t numActingPlayers( const Game *game, const State *state );

/* get the index into array state.boardCards[] for the first board
   card in round (where the first round is round 0) */
uint8_t bcStart( const Game *game, const uint8_t round );

/* get the total number of board cards dealt out after (zero based) round */
uint8_t sumBoardCards( const Game *game, const uint8_t round );

/* return the value of a finished hand for a player
   returns a double because pots may be split when players tie
   WILL HAVE UNDEFINED BEHAVIOUR IF HAND ISN'T FINISHED
   (stateFinished(state)==0) */
double valueOfState( const Game *game, const State *state,
		      const uint8_t player );

/* returns number of characters consumed on success, -1 on failure
   state will be modified even on a failure to read */
int readState( const char *string, const Game *game, State *state );

/* returns number of characters consumed on success, -1 on failure
   state will be modified even on a failure to read */
int readMatchState( const char *string, const Game *game, MatchState *state );

/* print a state to a string, as viewed by viewingPlayer
   returns the number of characters in string, or -1 on error
   DOES NOT COUNT FINAL 0 TERMINATOR IN THIS COUNT!!! */
int printState( const Game *game, const State *state,
		const int maxLen, char *string );

/* print a state to a string, as viewed by viewingPlayer
   returns the number of characters in string, or -1 on error
   DOES NOT COUNT FINAL 0 TERMINATOR IN THIS COUNT!!! */
int printMatchState( const Game *game, const MatchState *state,
		     const int maxLen, char *string );

/* read an action, returning the action in the passed pointer
   action and size will be modified even on a failure to read
   returns number of characters consumed on succes, -1 on failure */
int readAction( const char *string, const Game *game, Action *action );

/* print an action to a string
   returns the number of characters in string, or -1 on error
   DOES NOT COUNT FINAL 0 TERMINATOR IN THIS COUNT!!! */
int printAction( const Game *game, const Action *action,
		 const int maxLen, char *string );

/* returns number of characters consumed, or -1 on error
   on success, returns the card in *card */
int readCard( const char *string, uint8_t *card );

/* read up to maxCards cards
   returns number of cards successfully read
   returns number of characters consumed in charsConsumed */
int readCards( const char *string, const int maxCards,
	       uint8_t *cards, int *charsConsumed );

/* print a card to a string
   returns the number of characters in string, or -1 on error
   DOES NOT COUNT FINAL 0 TERMINATOR IN THIS COUNT!!! */
int printCard( const uint8_t card, const int maxLen, char *string );

/* print out numCards cards to a string
   returns the number of characters in string
   DOES NOT COUNT FINAL 0 TERMINATOR IN THIS COUNT!!! */
int printCards( const int numCards, const uint8_t *cards,
		const int maxLen, char *string );

#define rankOfCard( card ) ((card)/MAX_SUITS)
#define suitOfCard( card ) ((card)%MAX_SUITS)
#define makeCard( rank, suit ) ((rank)*MAX_SUITS+(suit))

#endif
