/*
   Copyright (C) 2011 by the Computer Poker Research Group, University of 
   Alberta
   Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
   All rights reserved.                          
*/

#ifndef _RNG_H
#define _RNG_H
#define __STDC_FORMAT_MACROS
#include <inttypes.h>


/* functions included in Takuji Nishimura and Makoto Matsumoto's RNG code */
/* NOTE changes made on 2005/9/7 by Neil Burch - if you have problems
   with this code, DON'T complain to Makoto Matsumoto... */


/* Period parameters */  
#define RNG_N 624
#define RNG_M 397
#define MATRIX_A 0x9908b0dfUL   /* constant vector a */
#define UPPER_MASK 0x80000000UL /* most significant w-r bits */
#define LOWER_MASK 0x7fffffffUL /* least significant r bits */


typedef struct {
uint32_t mt[ RNG_N ];
int mti;
} rng_state_t;


/* initializes rng state using an integer seed */
void init_genrand( rng_state_t *state, uint32_t s );

/* initialize by an array with array-length */
/* init_key is the array for initializing keys */
/* key_length is its length */
void init_by_array( rng_state_t *state, uint32_t init_key[], int key_length );


/* all of the functions below can not be called until init_* has been called */

/* generates a random number on [0,0xffffffff]-interval */
uint32_t genrand_int32( rng_state_t *state );

/* generates a random number on [0,0xffffffff]-interval */
#define genrand_int31(state) ((int32_t)(genrand_int32(state)>>1))

/* These real versions are due to Isaku Wada, 2002/01/09 added */
/* generates a random number on [0,1]-real-interval */
#define genrand_real1(state) (genrand_int32(state)*(1.0/4294967295.0))

/* generates a random number on [0,1)-real-interval */
#define genrand_real2(state) (genrand_int32(state)*(1.0/4294967296.0))

/* generates a random number on (0,1)-real-interval */
#define genrand_real3(state) ((((double)genrand_int32(state))+0.5)*(1.0/4294967296.0))

/* generates a random number on [0,1) with 53-bit resolution*/
#define genrand_res53(state) (((genrand_int32(state)>>5)*67108864.0+(genrand_int32(state)>>6))*(1.0/9007199254740992.0))

#endif
