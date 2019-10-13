/*
Copyright (C) 2011 by the Computer Poker Research Group, University of Alberta
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
#define __STDC_LIMIT_MACROS
#include <stdint.h>
#include "game.h"
#include "rng.h"

#include "evalHandTables"


static enum ActionType charToAction[ 256 ] = {
  /* 0x0X */
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  /* 0x1X */
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  /* 0x2X */
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  /* 0x3X */
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  /* 0x4X */
  a_invalid, a_invalid, a_raise, a_call,
  a_invalid, a_invalid, a_fold, a_invalid,
  a_invalid, a_invalid, a_invalid, a_call,
  a_invalid, a_invalid, a_invalid, a_invalid,
  /* 0x5X */
  a_invalid, a_invalid, a_raise, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  /* 0x6X */
  a_invalid, a_invalid, a_raise, a_call,
  a_invalid, a_invalid, a_fold, a_invalid,
  a_invalid, a_invalid, a_invalid, a_call,
  a_invalid, a_invalid, a_invalid, a_invalid,
  /* 0x7X */
  a_invalid, a_invalid, a_raise, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  /* 0x8X */
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  /* 0x9X */
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  /* 0xAX */
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  /* 0xBX */
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  /* 0xCX */
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  /* 0xDX */
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  /* 0xEX */
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  /* 0xFX */
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid,
  a_invalid, a_invalid, a_invalid, a_invalid
};

static char actionChars[ a_invalid + 1 ] = "fcr";

static char suitChars[ MAX_SUITS + 1 ] = "cdhs";
static char rankChars[ MAX_RANKS + 1 ] = "23456789TJQKA";


static int consumeSpaces( const char *string, int consumeEqual )
{
  int i;

  for( i = 0; string[ i ] != 0
	 && ( isspace( string[ i ] )
	      || ( consumeEqual && string[ i ] == '=' ) );
       ++i ) {
  }

  return i;
}

/* reads up to numItems with scanf format itemFormat from string,
   returning item i in *items[ i ]
   ignore the '=' character if consumeEqual is non-zero
   returns the number of characters consumed doing this in charsConsumed
   returns the number of items read */
static int readItems( const char *itemFormat, const int numItems,
		      const char *string, const int consumeEqual,
		      void *items, const size_t itemSize,
		      int *charsConsumed )
{
  int i, c, r;
  char *fmt;

  i = strlen( itemFormat );
  fmt = (char*)malloc( i + 3 );
  assert( fmt != 0 );
  strcpy( fmt, itemFormat );
  fmt[ i ] = '%';
  fmt[ i + 1 ] = 'n';
  fmt[ i + 2 ] = 0;

  c = 0;
  for( i = 0; i < numItems; ++i ) {

    c += consumeSpaces( &string[ c ], consumeEqual );
    if( sscanf( &string[ c ], fmt, items + i * itemSize, &r ) < 1 ) {
      break;
    }
    c += r;
  }

  free( fmt );

  *charsConsumed = c;
  return i;
}

Game *readGame( FILE *file )
{
  int stackRead, blindRead, raiseSizeRead, boardCardsRead, c, t;
  char line[ MAX_LINE_LEN ];
  Game *game;

  game = (Game*)malloc( sizeof( *game ) );
  assert( game != 0 );
  stackRead = 4;
  for( c = 0; c < MAX_ROUNDS; ++c ) {
    game->stack[ c ] = INT32_MAX;
  }
  blindRead = 0;
  raiseSizeRead = 0;
  game->bettingType = limitBetting;
  game->numPlayers = 0;
  game->numRounds = 0;
  for( c = 0; c < MAX_ROUNDS; ++c ) {
    game->firstPlayer[ c ] = 1;
  }
  for( c = 0; c < MAX_ROUNDS; ++c ) {
    game->maxRaises[ c ] = UINT8_MAX;
  }
  game->numSuits = 0;
  game->numRanks = 0;
  game->numHoleCards = 0;
  boardCardsRead = 0;

  while( fgets( line, MAX_LINE_LEN, file ) ) {

    if( line[ 0 ] == '#' || line[ 0 ] == '\n' ) {
      continue;
    }

    if( !strncasecmp( line, "end gamedef", 11 ) ) {

      break;
    } else if( !strncasecmp( line, "gamedef", 7 ) ) {

      continue;
    } else if( !strncasecmp( line, "stack", 5 ) ) {

      stackRead = readItems( "%"SCNd32, MAX_PLAYERS, &line[ 5 ],
				 1, game->stack, 4, &c );
    } else if( !strncasecmp( line, "blind", 5 ) ) {

      blindRead = readItems( "%"SCNd32, MAX_PLAYERS, &line[ 5 ],
			     1, game->blind, 4, &c );
    } else if( !strncasecmp( line, "raisesize", 9 ) ) {

      raiseSizeRead = readItems( "%"SCNd32, MAX_PLAYERS, &line[ 9 ],
				 1, game->raiseSize, 4, &c );
    } else if( !strncasecmp( line, "limit", 5 ) ) {

      game->bettingType = limitBetting;
    } else if( !strncasecmp( line, "nolimit", 7 ) ) {

      game->bettingType = noLimitBetting;
    } else if( !strncasecmp( line, "numplayers", 10 ) ) {

      readItems( "%"SCNu8, 1, &line[ 10 ], 1, &game->numPlayers, 1, &c );
    } else if( !strncasecmp( line, "numrounds", 9 ) ) {

      readItems( "%"SCNu8, 1, &line[ 9 ], 1, &game->numRounds, 1, &c );
    } else if( !strncasecmp( line, "firstplayer", 11 ) ) {

      readItems( "%"SCNu8, MAX_ROUNDS, &line[ 11 ],
		 1, game->firstPlayer, 1, &c );
    } else if( !strncasecmp( line, "maxraises", 9 ) ) {

      readItems( "%"SCNu8, MAX_ROUNDS, &line[ 9 ],
		 1, game->maxRaises, 1, &c );
    } else if( !strncasecmp( line, "numsuits", 8 ) ) {

      readItems( "%"SCNu8, 1, &line[ 8 ], 1, &game->numSuits, 1, &c );
    } else if( !strncasecmp( line, "numranks", 8 ) ) {

      readItems( "%"SCNu8, 1, &line[ 8 ], 1, &game->numRanks, 1, &c );
    } else if( !strncasecmp( line, "numholecards", 12 ) ) {

      readItems( "%"SCNu8, 1, &line[ 12 ], 1, &game->numHoleCards, 1, &c );
    } else if( !strncasecmp( line, "numboardcards", 13 ) ) {

      boardCardsRead = readItems( "%"SCNu8, MAX_ROUNDS, &line[ 13 ],
				  1, game->numBoardCards, 1, &c );
    }
  }

  /* do sanity checks */
  if( game->numRounds == 0 || game->numRounds > MAX_ROUNDS ) {

    fprintf( stderr, "invalid number of rounds: %"PRIu8"\n", game->numRounds );
    free( game );
    return NULL;
  }

  if( game->numPlayers < 2 || game->numPlayers > MAX_PLAYERS ) {

    fprintf( stderr, "invalid number of players: %"PRIu8"\n",
	     game->numPlayers );
    free( game );
    return NULL;
  }

  if( stackRead < game->numPlayers ) {

    fprintf( stderr, "only read %"PRIu8" stack sizes, need %"PRIu8"\n",
	    stackRead, game->numPlayers );
    free( game );
    return NULL;
  }

  if( blindRead < game->numPlayers ) {

    fprintf( stderr, "only read %"PRIu8" blinds, need %"PRIu8"\n",
	    blindRead, game->numPlayers );
    free( game );
    return NULL;
  }
  for( c = 0; c < game->numPlayers; ++c ) {

    if( game->blind[ c ] > game->stack[ c ] ) {
      fprintf( stderr, "blind for player %d is greater than stack size\n",
	       c + 1 );
      free( game );
      return NULL;
    }
  }

  if( game->bettingType == limitBetting
      && raiseSizeRead < game->numRounds ) {

    fprintf( stderr, "only read %"PRIu8" raise sizes, need %"PRIu8"\n",
	     raiseSizeRead, game->numRounds );
    free( game );
    return NULL;
  }

  for( c = 0; c < game->numRounds; ++c ) {

    if( game->firstPlayer[ c ] == 0
	|| game->firstPlayer[ c ] > game->numPlayers ) {

      fprintf( stderr, "invalid first player %"PRIu8" on round %d\n",
	      game->firstPlayer[ c ], c + 1 );
      free( game );
      return NULL;
    }

    --game->firstPlayer[ c ];
  }

  if( game->numSuits == 0 || game->numSuits > MAX_SUITS ) {

    fprintf( stderr, "invalid number of suits: %"PRIu8"\n", game->numSuits );
    free( game );
    return NULL;
  }

  if( game->numRanks == 0 || game->numRanks > MAX_RANKS ) {

    fprintf( stderr, "invalid number of ranks: %"PRIu8"\n", game->numRanks );
    free( game );
    return NULL;
  }

  if( game->numHoleCards == 0 || game->numHoleCards > MAX_HOLE_CARDS ) {

    fprintf( stderr, "invalid number of hole cards: %"PRIu8"\n",
	     game->numHoleCards );
    free( game );
    return NULL;
  }

  if( boardCardsRead < game->numRounds ) {

    fprintf( stderr, "only read %"PRIu8" board card numbers, need %"PRIu8"\n",
	    boardCardsRead, game->numRounds );
    free( game );
    return NULL;
  }

  t = game->numHoleCards * game->numPlayers;
  for( c = 0; c < game->numRounds; ++c ) {
    t += game->numBoardCards[ c ];
  }
  if( t > game->numSuits * game->numRanks ) {

    fprintf( stderr, "too many hole and board cards for specified deck\n" );
    free( game );
    return NULL;
  }

  return game;
}

void printGame( FILE *file, const Game *game )
{
  int i;

  fprintf( file, "GAMEDEF\n" );

  if( game->bettingType == noLimitBetting ) {
    fprintf( file, "nolimit\n" );
  } else {
    fprintf( file, "limit\n" );
  }

  fprintf( file, "numPlayers = %"PRIu8"\n", game->numPlayers );

  fprintf( file, "numRounds = %"PRIu8"\n", game->numRounds );

  for( i = 0; i < game->numPlayers; ++i ) {
    if( game->stack[ i ] < INT32_MAX ) {

      fprintf( file, "stack =" );
      for( i = 0; i < game->numPlayers; ++i ) {
	fprintf( file, " %"PRId32, game->stack[ i ] );
      }
      fprintf( file, "\n" );

      break;
    }
  }

  fprintf( file, "blind =" );
  for( i = 0; i < game->numPlayers; ++i ) {
    fprintf( file, " %"PRId32, game->blind[ i ] );
  }
  fprintf( file, "\n" );

  if( game->bettingType == limitBetting ) {

    fprintf( file, "raiseSize =" );
    for( i = 0; i < game->numRounds; ++i ) {
      fprintf( file, " %"PRId32, game->raiseSize[ i ] );
    }
    fprintf( file, "\n" );
  }

  for( i = 0; i < game->numRounds; ++i ) {
    if( game->firstPlayer[ i ] != 0 ) {

      fprintf( file, "firstPlayer =" );
      for( i = 0; i < game->numRounds; ++i ) {
	fprintf( file, " %"PRIu8, game->firstPlayer[ i ] + 1 );
      }
      fprintf( file, "\n" );

      break;
    }
  }

  for( i = 0; i < game->numRounds; ++i ) {
    if( game->maxRaises[ i ] != UINT8_MAX ) {

      fprintf( file, "maxRaises =" );
      for( i = 0; i < game->numRounds; ++i ) {
	fprintf( file, " %"PRIu8, game->maxRaises[ i ] );
      }
      fprintf( file, "\n" );

      break;
    }
  }

  fprintf( file, "numSuits = %"PRIu8"\n", game->numSuits );

  fprintf( file, "numRanks = %"PRIu8"\n", game->numRanks );

  fprintf( file, "numHoleCards = %"PRIu8"\n", game->numHoleCards );

  fprintf( file, "numBoardCards =" );
  for( i = 0; i < game->numRounds; ++i ) {
    fprintf( file, " %"PRIu8, game->numBoardCards[ i ] );
  }
  fprintf( file, "\n" );

  fprintf( file, "END GAMEDEF\n" );
}

uint8_t bcStart( const Game *game, const uint8_t round )
{
  int r;
  uint8_t start;

  start = 0;
  for( r = 0; r < round; ++r ) {

    start += game->numBoardCards[ r ];
  }

  return start;
}

uint8_t sumBoardCards( const Game *game, const uint8_t round )
{
  int r;
  uint8_t total;

  total = 0;
  for( r = 0; r <= round; ++r ) {
    total += game->numBoardCards[ r ];
  }

  return total;
}

static uint8_t nextPlayer( const Game *game, const State *state,
			   const uint8_t curPlayer )
{
  uint8_t n;

  n = curPlayer;
  do {
    n = ( n + 1 ) % game->numPlayers;
  } while( state->playerFolded[ n ]
	   || state->spent[ n ] >= game->stack[ n ] );

  return n;
}

uint8_t currentPlayer( const Game *game, const State *state )
{
  /* if action has already been made, compute next player from last player */
  if( state->numActions[ state->round ] ) {
    return nextPlayer( game, state, state->actingPlayer[ state->round ]
		       [ state->numActions[ state->round ] - 1 ] );
  }

  /* first player in a round is determined by the game and round
     use nextPlayer() because firstPlayer[round] might be unable to act */
  return nextPlayer( game, state, game->firstPlayer[ state->round ]
		     + game->numPlayers - 1 );
}

uint8_t numRaises( const State *state )
{
  int i;
  uint8_t ret;

  ret = 0;
  for( i = 0; i < state->numActions[ state->round ]; ++i ) {
    if( state->action[ state->round ][ i ].type == a_raise ) {
      ++ret;
    }
  }

  return ret;
}

uint8_t numFolded( const Game *game, const State *state )
{
  int p;
  uint8_t ret;

  ret = 0;
  for( p = 0; p < game->numPlayers; ++p ) {
    if( state->playerFolded[ p ] ) {
      ++ret;
    }
  }

  return ret;
}

uint8_t numCalled( const Game *game, const State *state )
{
  int i;
  uint8_t ret, p;

  ret = 0;
  for( i = state->numActions[ state->round ]; i > 0; --i ) {

    p = state->actingPlayer[ state->round ][ i - 1 ];

    if( state->action[ state->round ][ i - 1 ].type == a_raise ) {
      /* player initiated the bet, so they've called it */

      if( state->spent[ p ] < game->stack[ p ] ) {
	/* player is not all-in, so they're still acting */

	++ret;
      }

      /* this is the start of the current bet, so we're finished */
      return ret;
    } else if( state->action[ state->round ][ i - 1 ].type == a_call ) {

      if( state->spent[ p ] < game->stack[ p ] ) {
	/* player is not all-in, so they're still acting */

	++ret;
      }
    }
  }

  return ret;
}

uint8_t numAllIn( const Game *game, const State *state )
{
  int p;
  uint8_t ret;

  ret = 0;
  for( p = 0; p < game->numPlayers; ++p ) {
    if( state->spent[ p ] >= game->stack[ p ] ) {
      ++ret;
    }
  }

  return ret;
}

uint8_t numActingPlayers( const Game *game, const State *state )
{
  int p;
  uint8_t ret;

  ret = 0;
  for( p = 0; p < game->numPlayers; ++p ) {
    if( state->playerFolded[ p ] == 0
	&& state->spent[ p ] < game->stack[ p ] ) {
      ++ret;
    }
  }

  return ret;
}

void initState( const Game *game, const uint32_t handId, State *state )
{
  int p, r;

  state->handId  = handId;

  state->maxSpent = 0;
  for( p = 0; p < game->numPlayers; ++p ) {

    state->spent[ p ] = game->blind[ p ];
    if( game->blind[ p ] > state->maxSpent ) {

      state->maxSpent = game->blind[ p ];
    }
  }

  if( game->bettingType == noLimitBetting ) {
    /* no-limit games need to keep track of the minimum bet */

    if( state->maxSpent ) {
      /* we'll have to call the big blind and then raise by that
	 amount, so the minimum raise-to is 2*maximum blinds */

      state->minNoLimitRaiseTo = state->maxSpent * 2;
    } else {
      /* need to bet at least one chip, and there are no blinds/ante */

      state->minNoLimitRaiseTo = 1;
    }
  } else {
    /* no need to worry about minimum raises outside of no-limit games */

    state->minNoLimitRaiseTo = 0;
  }

  for( p = 0; p < game->numPlayers; ++p ) {

    state->spent[ p ] = game->blind[ p ];

    if( game->blind[ p ] > state->maxSpent ) {
      state->maxSpent = game->blind[ p ];
    }

    state->playerFolded[ p ] = 0;
  }

  for( r = 0; r < game->numRounds; ++r ) {

    state->numActions[ r ] = 0;
  }

  state->round = 0;

  state->finished = 0;
}

static uint8_t dealCard( rng_state_t *rng, uint8_t *deck, const int numCards )
{
  int i;
  uint8_t ret;

  i = genrand_int32( rng ) % numCards;
  ret = deck[ i ];
  deck[ i ] = deck[ numCards - 1 ];

  return ret;
}

void dealCards( const Game *game, rng_state_t *rng, State *state )
{
  int r, s, numCards, i, p;
  uint8_t deck[ MAX_RANKS * MAX_SUITS ];

  numCards = 0;
  for( s = MAX_SUITS - game->numSuits; s < MAX_SUITS; ++s ) {

    for( r = MAX_RANKS - game->numRanks; r < MAX_RANKS; ++r ) {

      deck[ numCards ] = makeCard( r, s );
      ++numCards;
    }
  }

  for( p = 0; p < game->numPlayers; ++p ) {

    for( i = 0; i < game->numHoleCards; ++i ) {

      state->holeCards[ p ][ i ] = dealCard( rng, deck, numCards );
      --numCards;
    }
  }

  s = 0;
  for( r = 0; r < game->numRounds; ++r ) {

    for( i = 0; i < game->numBoardCards[ r ]; ++i ) {

      state->boardCards[ s ] = dealCard( rng, deck, numCards );
      --numCards;
      ++s;
    }
  }
}

/* check whether some portions of a state are equal,
   common to both statesEqual and matchStatesEqual */
static int statesEqualCommon( const Game *game, const State *a,
			      const State *b )
{
  int r, i, t;

  /* is it the same hand? */
  if( a->handId != b->handId ) {
    return 0;
  }

  /* make sure the betting is the same */
  if( a->round != b->round ) {
    return 0;
  }

  for( r = 0; r <= a->round; ++r ) {

    if( a->numActions[ r ] != b->numActions[ r ] ) {
      return 0;
    }

    for( i = 0; i < a->numActions[ r ]; ++i ) {

      if( a->action[ r ][ i ].type != b->action[ r ][ i ].type ) {
	return 0;
      }
      if( a->action[ r ][ i ].size != b->action[ r ][ i ].size ) {
	return 0;
      }
    }
  }

  /* spent, maxSpent, actingPlayer, finished, and playerFolded are
     all determined by the betting taken, so if it's equal, so are
     they (at least for valid states) */

  /* are the board cards the same? */
  t = sumBoardCards( game, a->round );
  for( i = 0; i < t; ++i ) {

    if( a->boardCards[ i ] != b->boardCards[ i ] ) {
      return 0;
    }
  }

  /* all tests say states are equal */
  return 1;
}

int statesEqual( const Game *game, const State *a, const State *b )
{
  int p, i;

  if( !statesEqualCommon( game, a, b ) ) {
    return 0;
  }

  /* are all the hole cards the same? */
  for( p = 0; p < game->numPlayers; ++p ) {

    for( i = 0; i < game->numHoleCards; ++i ) {
      if( a->holeCards[ p ][ i ] != b->holeCards[ p ][ i ] ) {
	return 0;
      }
    }
  }

  return 1;
}

int matchStatesEqual( const Game *game, const MatchState *a,
		      const MatchState *b )
{
  int p, i;

  if( a->viewingPlayer != b->viewingPlayer ) {
    return 0;
  }

  if( !statesEqualCommon( game, &a->state, &b->state ) ) {
    return 0;
  }

  /* are the viewing player's hole cards the same? */
  p = a->viewingPlayer;
  for( i = 0; i < game->numHoleCards; ++i ) {
    if( a->state.holeCards[ p ][ i ] != b->state.holeCards[ p ][ i ] ) {
      return 0;
    }
  }

  return 1;
}

int raiseIsValid( const Game *game, const State *curState,
		  int32_t *minSize, int32_t *maxSize )
{
  int p;

  if( numRaises( curState ) >= game->maxRaises[ curState->round ] ) {
    /* already made maximum number of raises */

    return 0;
  }

  if( curState->numActions[ curState->round ] + game->numPlayers
      > MAX_NUM_ACTIONS ) {
    /* 1 raise + NUM PLAYERS-1 calls is too many actions */

    fprintf( stderr, "WARNING: #actions in round is too close to MAX_NUM_ACTIONS, forcing call/fold\n" );
    return 0;
  }

  if( numActingPlayers( game, curState ) <= 1 ) {
    /* last remaining player can't bet if there's no one left to call
       (this check is needed if the 2nd last player goes all in, and
       the last player has enough stack left to bet) */

    return 0;
  }

  if( game->bettingType != noLimitBetting ) {
    /* if it's not no-limit betting, don't worry about sizes */

    *minSize = 0;
    *maxSize = 0;
    return 1;
  }

  p = currentPlayer( game, curState );
  *minSize = curState->minNoLimitRaiseTo;
  *maxSize = game->stack[ p ];

  /* handle case where remaining player stack is too small */
  if( *minSize > game->stack[ p ] ) {
    /* can't handle the minimum bet size - can we bet at all? */

    if( curState->maxSpent >= game->stack[ p ] ) {
      /* not enough money to increase current bet */

      return 0;
    } else {
      /* can raise by going all-in */

      *minSize = *maxSize;
      return 1;
    }
  }

  return 1;
}

int isValidAction( const Game *game, const State *curState,
		   const int tryFixing, Action *action )
{
  int min, max, p;

  if( stateFinished( curState ) || action->type == a_invalid ) {
    return 0;
  }

  p = currentPlayer( game, curState );

  if( action->type == a_raise ) {

    if( !raiseIsValid( game, curState, &min, &max ) ) {
      /* there are no valid raise sizes */

      return 0;
    }

    if( game->bettingType == noLimitBetting ) {
      /* no limit games have a size */

      if( action->size < min ) {
	/* bet size is too small */

	if( !tryFixing ) {

	  return 0;
	}
	fprintf( stderr, "WARNING: raise of %d increased to %d\n",
		 action->size, min );
	action->size = min;
      } else if( action->size > max ) {
	/* bet size is too big */

	if( !tryFixing ) {

	  return 0;
	}
	fprintf( stderr, "WARNING: raise of %d decreased to %d\n",
		 action->size, max );
	action->size = max;
      }
    } else {

    }
  } else if( action->type == a_fold ) {

    if( curState->spent[ p ] == curState->maxSpent
	|| curState->spent[ p ] == game->stack[ p ] ) {
      /* player has already called all bets, or is all-in */

      return 0;
    }

    if( action->size != 0 ) {

      fprintf( stderr, "WARNING: size given for fold\n" );
      action->size = 0;
    }
  } else {
    /* everything else */

    if( action->size != 0 ) {

      fprintf( stderr, "WARNING: size given for something other than a no-limit raise\n" );
      action->size = 0;
    }
  }

  return 1;
}

void doAction( const Game *game, const Action *action, State *state )
{
  int p = currentPlayer( game, state );

  assert( state->numActions[ state->round ] < MAX_NUM_ACTIONS );

  state->action[ state->round ][ state->numActions[ state->round ] ] = *action;
  state->actingPlayer[ state->round ][ state->numActions[ state->round ] ] = p;
  ++state->numActions[ state->round ];

  switch( action->type ) {
  case a_fold:

    state->playerFolded[ p ] = 1;
    break;

  case a_call:

    if( state->maxSpent > game->stack[ p ] ) {
      /* calling puts player all-in */

      state->spent[ p ] = game->stack[ p ];
    } else {
      /* player matches the bet by spending same amount of money */

      state->spent[ p ] = state->maxSpent;
    }
    break;

  case a_raise:

    if( game->bettingType == noLimitBetting ) {
      /* no-limit betting uses size in action */

      assert( action->size > state->maxSpent );
      assert( action->size <= game->stack[ p ] );

      /* next raise must call this bet, and raise by at least this much */
      if( action->size + action->size - state->maxSpent
	  > state->minNoLimitRaiseTo ) {

	state->minNoLimitRaiseTo
	  = action->size + action->size - state->maxSpent;
      }
      state->maxSpent = action->size;
    } else {
      /* limit betting uses a fixed amount on top of current bet size */

      if( state->maxSpent + game->raiseSize[ state->round ]
	  > game->stack[ p ] ) {
	/* raise puts player all-in */

	state->maxSpent = game->stack[ p ];
      } else {
	/* player raises by the normal limit size */

	state->maxSpent += game->raiseSize[ state->round ];
      }
    }

    state->spent[ p ] = state->maxSpent;
    break;

  default:
    fprintf( stderr, "ERROR: trying to do invalid action %d", action->type );
    assert( 0 );
  }

  /* see if the round or game has ended */
  if( numFolded( game, state ) + 1 >= game->numPlayers ) {
    /* only one player left - game is immediately over, no showdown */

    state->finished = 1;
  } else if( numCalled( game, state ) >= numActingPlayers( game, state ) ) {
    /* >= 2 non-folded players, all acting players have called */

    if( numActingPlayers( game, state ) > 1 ) {
      /* there are at least 2 acting players */

      if( state->round + 1 < game->numRounds ) {
	/* active players move onto next round */

	++state->round;

	/* minimum raise-by is reset to minimum of big blind or 1 chip */
	state->minNoLimitRaiseTo = 1;
	for( p = 0; p < game->numPlayers; ++p ) {

	  if( game->blind[ p ] > state->minNoLimitRaiseTo ) {

	    state->minNoLimitRaiseTo = game->blind[ p ];
	  }
	}

	/* we finished at least one round, so raise-to = raise-by + maxSpent */
	state->minNoLimitRaiseTo += state->maxSpent;
      } else {
	/* no more betting rounds, so we're totally finished */

	state->finished = 1;
      }
    } else {
      /* not enough players for more betting, but still need a showdown */

      state->finished = 1;
      state->round = game->numRounds - 1;
    }
  }
}

static int rankHand( const Game *game, const State *state,
		     const uint8_t player )
{
  int i;
  Cardset c = emptyCardset();

  for( i = 0; i < game->numHoleCards; ++i ) {

    addCardToCardset( &c, suitOfCard( state->holeCards[ player ][ i ] ),
		      rankOfCard( state->holeCards[ player ][ i ] ) );
  }

  for( i = 0; i < sumBoardCards( game, state->round ); ++i ) {

    addCardToCardset( &c, suitOfCard( state->boardCards[ i ] ),
		      rankOfCard( state->boardCards[ i ] ) );
  }

  return rankCardset( c );
}

double valueOfState( const Game *game, const State *state,
		     const uint8_t player )
{
  double value;
  int p, numPlayers, playerIdx, numWinners, newNumPlayers;
  int32_t size, spent[ MAX_PLAYERS ];
  int rank[ MAX_PLAYERS ], winRank;

  if( state->playerFolded[ player ] ) {
    /* folding player loses all spent money */

    return (double)-state->spent[ player ];
  }

  if( numFolded( game, state ) + 1 == game->numPlayers ) {
    /* everyone else folded, so player takes the pot */

    value = 0.0;
    for( p = 0; p < game->numPlayers; ++p ) {
      if( p == player ) { continue; }

      value += (double)state->spent[ p ];
    }

    return value;
  }

  /* there's a showdown, and player is particpating.  Exciting! */

  /* make up a list of players */
  numPlayers = 0;
  playerIdx = -1; /* useless, but gets rid of a warning */
  for( p = 0; p < game->numPlayers; ++p ) {

    if( state->spent[ p ] == 0 ) {
      continue;
    }

    if( state->playerFolded[ p ] ) {
      /* folding players have a negative rank so they lose to a real hand
         we have also tested for fold, so p can't be the player of interest */

      rank[ numPlayers ] = -1;
    } else {
      /* p is participating in a showdown */

      if( p == player ) {
	playerIdx = numPlayers;
      }
      rank[ numPlayers ] = rankHand( game, state, p );
    }

    spent[ numPlayers ] = state->spent[ p ];
    ++numPlayers;
  }
  assert( numPlayers > 1 );

  /* go through all the sidepots player is participating in */
  value = 0.0;
  while( 1 ) {

    /* find the smallest remaining sidepot, largest rank,
        and number of winners with largest rank */
    size = INT32_MAX;
    winRank = 0;
    numWinners = 0;
    for( p = 0; p < numPlayers; ++p ) {
      assert( spent[ p ] > 0 );

      if( spent[ p ] < size ) {
	size = spent[ p ];
      }

      if( rank[ p ] > winRank ) {
	/* new largest rank - only one player with this rank so far */

	winRank = rank[ p ];
	numWinners = 1;
      } else if( rank[ p ] == winRank ) {
	/* another player with highest rank */

	++numWinners;
      }
    }

    if( rank[ playerIdx ] == winRank ) {
      /* player has spent size, and splits pot with other winners */

      value += (double)( size * ( numPlayers - numWinners ) )
	/ (double)numWinners;
    } else {
      /* player loses this pot */

      value -= (double)size;
    }

    /* update list of players for next pot */
    newNumPlayers = 0;
    for( p = 0; p < numPlayers; ++p ) {

      spent[ p ] -= size;
      if( spent[ p ] == 0 ) {
	/* player p is not participating in next side pot */

	if( p == playerIdx ) {
	  /* p is the player of interest, so we're done */

	  return value;
	}

	continue;
      }

      if( p == playerIdx ) {
	/* p is the player of interest, so note the new index */

	playerIdx = newNumPlayers;
      }

      if( p != newNumPlayers ) {
	/* put entry p into new position */

	spent[ newNumPlayers ] = spent[ p ];
	rank[ newNumPlayers ] = rank[ p ];
      }
      ++newNumPlayers;
    }
    numPlayers = newNumPlayers;
  }
}

/* read actions from a string, updating state with the actions
   reading is terminated by '\0' and ':'
   returns number of characters consumed, or -1 on failure
   state will be modified, even on failure */
static int readBetting( const char *string, const Game *game, State *state )
{
  int c, r;
  Action action;

  c = 0;
  while( 1 ) {

    if( string[ c ] == 0 ) {
      break;
    }

    if( string[ c ] == ':' ) {
      ++c;
      break;
    }

    /* ignore / character */
    if( string[ c ] == '/' ) {
      ++c;
      continue;
    }

    r = readAction( &string[ c ], game, &action );
    if( r < 0 ) {
      return -1;
    }

    if( !isValidAction( game, state, 0, &action ) ) {
      return -1;
    }

    doAction( game, &action, state );
    c += r;
  }

  return c;
}

/* print actions to a string
   returns number of characters printed to string, or -1 on failure
   DOES NOT COUNT FINAL 0 TERMINATOR IN THIS COUNT!!! */
static int printBetting( const Game *game, const State *state,
			 const int maxLen, char *string )
{
  int i, a, c, r;

  c = 0;
  for( i = 0; i <= state->round; ++i ) {

    /* print state separator */
    if( i != 0 ) {

      if( c >= maxLen ) {
	return -1;
      }
      string[ c ] = '/';
      ++c;
    }

    /* print betting for round */
    for( a = 0; a < state->numActions[ i ]; ++a ) {

      r = printAction( game, &state->action[ i ][ a ],
		       maxLen - c, &string[ c ] );
      if( r < 0 ) {
	return -1;
      }
      c += r;
    }
  }

  if( c >= maxLen ) {
    return -1;
  }
  string[ c ] = 0;

  return c;
}

static int readHoleCards( const char *string, const Game *game,
			  State *state )
{
  int p, c, r, num;

  c = 0;
  for( p = 0; p < game->numPlayers; ++p ) {

    /* check for player separator '|' */
    if( p != 0 ) {
      if( string[ c ] == '|' ) {
	++c;
      }
    }

    num = readCards( &string[ c ], game->numHoleCards,
		     state->holeCards[ p ], &r );
    if( num == 0 ) {
      /* no cards for player p */

      continue;
    }
    if( num != game->numHoleCards ) {
      /* read some cards, but not enough - bad! */

      return -1;
    }
    c += r;
  }

  return c;
}

static int printAllHoleCards( const Game *game, const State *state,
			      const int maxLen, char *string )
{
  int p, c, r;

  c = 0;
  for( p = 0; p < game->numPlayers; ++p ) {

    /* print player separator '|' */
    if( p != 0 ) {

      if( c >= maxLen ) {
	return -1;
      }
      string[ c ] = '|';
      ++c;
    }

    r = printCards( game->numHoleCards, state->holeCards[ p ],
		    maxLen - c, &string[ c ] );
    if( r < 0 ) {
      return -1;
    }
    c += r;
  }

  if( c >= maxLen ) {
    return -1;
  }
  string[ c ] = 0;

  return c;
}

static int printPlayerHoleCards( const Game *game, const State *state,
				 const uint8_t player,
				 const int maxLen, char *string )
{
  int p, c, r;

  c = 0;
  for( p = 0; p < game->numPlayers; ++p ) {

    /* print player separator '|' */
    if( p != 0 ) {

      if( c >= maxLen ) {
	return -1;
      }
      string[ c ] = '|';
      ++c;
    }

    if( p != player ) {
      /* don't print other player's cards unless there was a showdown
          and they didn't fold */

      if( !stateFinished( state ) ) {
	continue;
      }

      if( state->playerFolded[ p ] ) {
	continue;
      }

      if( numFolded( game, state ) + 1 == game->numPlayers ) {
	continue;
      }
    }

    r = printCards( game->numHoleCards, state->holeCards[ p ],
		    maxLen - c, &string[ c ] );
    if( r < 0 ) {
      return -1;
    }
    c += r;
  }

  if( c >= maxLen ) {
    return -1;
  }
  string[ c ] = 0;

  return c;
}

static int readBoardCards( const char *string, const Game *game,
			   State *state )
{
  int i, c, r;

  c = 0;
  for( i = 0; i <= state->round; ++i ) {

    /* check for round separator '/' */
    if( i != 0 ) {
      if( string[ c ] == '/' ) {
	++c;
      }
    }

    if( readCards( &string[ c ], game->numBoardCards[ i ],
		   &state->boardCards[ bcStart( game, i ) ], &r )
	!= game->numBoardCards[ i ] ) {
      /* couldn't read the required number of cards - bad! */

      return -1;
    }
    c += r;
  }

  return c;
}

static int printBoardCards( const Game *game, const State *state,
			    const int maxLen, char *string )
{
  int i, c, r;

  c = 0;
  for( i = 0; i <= state->round; ++i ) {

    /* print round separator '/' */
    if( i != 0 ) {

      if( c >= maxLen ) {
	return -1;
      }
      string[ c ] = '/';
      ++c;
    }

    r = printCards( game->numBoardCards[ i ],
		    &state->boardCards[ bcStart( game, i ) ],
		    maxLen - c, &string[ c ] );
    if( r < 0 ) {
      return -1;
    }
    c += r;
  }

  if( c >= maxLen ) {
    return -1;
  }
  string[ c ] = 0;

  return c;
}


static int readStateCommon( const char *string, const Game *game,
			    State *state )
{
  uint32_t handId;
  int c, r;

  /* HEADER */
  c = 0;

  /* HEADER:handId */
  if( sscanf( string, ":%"SCNu32"%n", &handId, &r ) < 1 ) {
    return -1;
  }
  c += r;

  initState( game, handId, state );

  /* HEADER:handId: */
  if( string[ c ] != ':' ) {
    return -1;
  }
  ++c;

  /* HEADER:handId:betting: */
  r = readBetting( &string[ c ], game, state );
  if( r < 0 ) {
    return -1;
  }
  c += r;

  /* HEADER:handId:betting:holeCards */
  r = readHoleCards( &string[ c ], game, state );
  if( r < 0 ) {
    return -1;
  }
  c += r;

  /* HEADER:handId:betting:holeCards boardCards */
  r = readBoardCards( &string[ c ], game, state );
  if( r < 0 ) {
    return -1;
  }
  c += r;

  return c;
}

int readState( const char *string, const Game *game, State *state )
{
  int c, r;

  /* HEADER = STATE */
  if( strncmp( string, "STATE", 5 ) != 0 ) {
    return -1;
  }
  c = 5;

  /* read rest of state */
  r = readStateCommon( &string[ 5 ], game, state );
  if( r < 0 ) {
    return -1;
  }
  c += r;

  return c;
}

int readMatchState( const char *string, const Game *game,
		    MatchState *state )
{
  int c, r;

  /* HEADER = MATCHSTATE:player */
  if( sscanf( string, "MATCHSTATE:%"SCNu8"%n",
	      &state->viewingPlayer, &c ) < 1
      || state->viewingPlayer >= game->numPlayers )  {
    return -1;
  }

  /* read rest of state */
  r = readStateCommon( &string[ c ], game, &state->state );
  if( r < 0 ) {
    return -1;
  }
  c += r;

  return c;
}

static int printStateCommon( const Game *game, const State *state,
			     const int maxLen, char *string )
{
  int c, r;

  /* HEADER */
  c = 0;

  /* HEADER:handId: */
  r = snprintf( &string[ c ], maxLen - c, ":%"PRIu32":", state->handId );
  if( r < 0 ) {
    return -1;
  }
  c += r;

  /* HEADER:handId:betting */
  r = printBetting( game, state, maxLen - c, &string[ c ] );
  if( r < 0 ) {
    return -1;
  }
  c += r;

  /* HEADER:handId:betting: */
  if( c >= maxLen ) {
    return -1;
  }
  string[ c ] = ':';
  ++c;

  return c;
}

int printState( const Game *game, const State *state,
		const int maxLen, char *string )
{
  int c, r;

  c = 0;

  /* STATE */
  r = snprintf( &string[ c ], maxLen - c, "STATE" );
  if( r < 0 ) {
    return -1;
  }
  c += r;

  /* STATE:handId:betting: */
  r = printStateCommon( game, state, maxLen - c, &string[ c ] );
  if( r < 0 ) {
    return -1;
  }
  c += r;

  /* STATE:handId:betting:holeCards */
  r = printAllHoleCards( game, state, maxLen - c, &string[ c ] );
  if( r < 0 ) {
    return -1;
  }
  c += r;

  /* STATE:handId:betting:holeCards boardCards */
  r = printBoardCards( game, state, maxLen - c, &string[ c ] );
  if( r < 0 ) {
    return -1;
  }
  c += r;

  if( c >= maxLen ) {
    return -1;
  }
  string[ c ] = 0;

  return c;
}

int printMatchState( const Game *game, const MatchState *state,
		     const int maxLen, char *string )
{
  int c, r;

  c = 0;

  /* MATCHSTATE:player */
  r = snprintf( &string[ c ], maxLen - c, "MATCHSTATE:%"PRIu8,
		state->viewingPlayer );
  if( r < 0 ) {
    return -1;
  }
  c += r;

  /* MATCHSTATE:player:handId:betting: */
  r = printStateCommon( game, &state->state, maxLen - c, &string[ c ] );
  if( r < 0 ) {
    return -1;
  }
  c += r;

  /* MATCHSTATE:player:handId:betting:holeCards */
  r = printPlayerHoleCards( game, &state->state, state->viewingPlayer,
			    maxLen - c, &string[ c ] );
  if( r < 0 ) {
    return -1;
  }
  c += r;

  /* MATCHSTATE:player:handId:betting:holeCards boardCards */
  r = printBoardCards( game, &state->state, maxLen - c, &string[ c ] );
  if( r < 0 ) {
    return -1;
  }
  c += r;

  if( c >= maxLen ) {
    return -1;
  }
  string[ c ] = 0;

  return c;
}

int readAction( const char *string, const Game *game, Action *action )
{
  int c, r;

  action->type = charToAction[ (uint8_t)string[ 0 ] ];
  if( action->type < 0 ) {
    return -1;
  }
  c = 1;

  if( action->type == a_raise && game->bettingType == noLimitBetting ) {
    /* no-limit bet/raise needs to read a size */

    if( sscanf( &string[ c ], "%"SCNd32"%n", &action->size, &r ) < 1 ) {
      return -1;
    }
    c += r;
  } else {
    /* size is zero for anything but a no-limit raise */

    action->size = 0;
  }

  return c;
}

int printAction( const Game *game, const Action *action,
		 const int maxLen, char *string )
{
  int c, r;

  if( maxLen == 0 ) {
    return -1;
  }

  c = 0;

  string[ c ] = actionChars[ action->type ];
  ++c;

  if( game->bettingType == noLimitBetting && action->type == a_raise ) {
    /* 2010 AAAI no-limit format has a size for bet/raise */

    r = snprintf( &string[ c ], maxLen - c, "%"PRId32, action->size );
    if( r < 0 ) {
      return -1;
    }
    c += r;
  }

  if( c >= maxLen ) {
    return -1;
  }
  string[ c ] = 0;

  return c;
}

int readCard( const char *string, uint8_t *card )
{
  char *spos;
  uint8_t c;

  if( string[ 0 ] == 0 ) {
    return -1;
  }
  spos = strchr( rankChars, toupper( string[ 0 ] ) );
  if( spos == 0 ) {
    return -1;
  }
  c = spos - rankChars;

  if( string[ 1 ] == 0 ) {
    return -1;
  }
  spos = strchr( suitChars, tolower( string[ 1 ] ) );
  if( spos == 0 ) {
    return -1;
  }

  *card = makeCard( c, spos - suitChars );

  return 2;
}

int readCards( const char *string, const int maxCards,
	       uint8_t *cards, int *charsConsumed )
{
  int i, c, r;

  c = 0;
  for( i = 0; i < maxCards; ++i ) {

    r = readCard( &string[ c ], &cards[ i ] );
    if( r < 0 ) {
      break;
    }
    c += r;
  }

  *charsConsumed = c;
  return i;
}

int printCard( const uint8_t card, const int maxLen, char *string  )
{
  if( 3 > maxLen ) {
    return -1;
  }

  string[ 0 ] = rankChars[ rankOfCard( card ) ];
  string[ 1 ] = suitChars[ suitOfCard( card ) ];
  string[ 2 ] = 0;

  return 2;
}

int printCards( const int numCards, const uint8_t *cards,
		const int maxLen, char *string )
{
  int i, c, r;

  c = 0;

  for( i = 0; i < numCards; ++i ) {

    r = printCard( cards[ i ], maxLen - c, &string[ c ] );
    if( r < 0 ) {
      return -1;
    }
    c += r;
  }

  /* no need to null terminate, we know for sure that printCard does */

  return c;
}
