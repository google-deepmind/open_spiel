/* 
   A C-program for MT19937, with initialization improved 2002/1/26.
   Coded by Takuji Nishimura and Makoto Matsumoto.

   Before using, initialize the state by using init_genrand(seed)  
   or init_by_array(init_key, key_length).

   Copyright (C) 2011 by the Computer Poker Research Group, University of 
   Alberta

   This file is a modification of work by Makoto Matsumoto and Takuji
   Nishimura.  That work was provided with the following license:

   Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
   All rights reserved.                          

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

     3. The names of its contributors may not be used to endorse or promote 
        products derived from this software without specific prior written 
        permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
   OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
   OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
   ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


   Any feedback is very welcome.
   http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
   email: m-mat @ math.sci.hiroshima-u.ac.jp (remove space)
*/

#include "rng.h"

/* NOTE changes made on 2005/9/7 by Neil Burch - if you have problems
   with this code, DON'T complain to Makoto Matsumoto... */


/* initializes mt[RNG_N] with a seed */
void init_genrand( rng_state_t *state, uint32_t s )
{
  state->mt[0]= s & 0xffffffffUL;
  for (state->mti=1; state->mti<RNG_N; state->mti++) {
    state->mt[state->mti] = 
      (1812433253UL * (state->mt[state->mti-1]
		       ^ (state->mt[state->mti-1] >> 30))
       + state->mti); 
    /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
    /* In the previous versions, MSBs of the seed affect   */
    /* only MSBs of the array mt[].                        */
    /* 2002/01/09 modified by Makoto Matsumoto             */
  }
}

/* initialize by an array with array-length */
/* init_key is the array for initializing keys */
/* key_length is its length */
/* slight change for C++, 2004/2/26 */
void init_by_array( rng_state_t *state, uint32_t init_key[], int key_length )
{
  int i, j, k;

  init_genrand(state, 19650218UL);
  i=1; j=0;
  k = (RNG_N>key_length ? RNG_N : key_length);
  for (; k; k--) {
    state->mt[i] = ( state->mt[i]
		     ^ ( ( state->mt[i-1] ^ ( state->mt[i-1] >> 30 ) )
			 * 1664525UL ) )
      + init_key[j] + j; /* non linear */
    i++; j++;
    if (i>=RNG_N) { state->mt[0] = state->mt[RNG_N-1]; i=1; }
    if (j>=key_length) j=0;
  }
  for (k=RNG_N-1; k; k--) {
    state->mt[i] = ( state->mt[i]
		     ^ ( ( state->mt[i-1] ^ ( state->mt[i-1] >> 30 ) )
			 * 1566083941UL ) )
      - i; /* non linear */
    state->mt[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
    i++;
    if (i>=RNG_N) { state->mt[0] = state->mt[RNG_N-1]; i=1; }
  }

  state->mt[0]|= 0x80000000UL; /* MSB is 1; assuring non-zero initial array */ 
}

/* generates a random number on [0,0xffffffff]-interval */
uint32_t genrand_int32( rng_state_t *state )
{
    uint32_t y;
    static uint32_t mag01[2]={0x0UL, MATRIX_A};
    /* mag01[x] = x * MATRIX_A  for x=0,1 */

    if (state->mti == RNG_N) { /* generate RNG_N words at one time */
        int kk;

        for (kk=0;kk<RNG_N-RNG_M;kk++) {
            y = (state->mt[kk]&UPPER_MASK)|(state->mt[kk+1]&LOWER_MASK);
            state->mt[kk] = state->mt[kk+RNG_M] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        for (;kk<RNG_N-1;kk++) {
            y = (state->mt[kk]&UPPER_MASK)|(state->mt[kk+1]&LOWER_MASK);
            state->mt[kk] =
	      state->mt[kk+(RNG_M-RNG_N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        y = (state->mt[RNG_N-1]&UPPER_MASK)|(state->mt[0]&LOWER_MASK);
        state->mt[RNG_N-1] = state->mt[RNG_M-1] ^ (y >> 1) ^ mag01[y & 0x1UL];

        state->mti = 0;
    }
  
    y = state->mt[state->mti++];

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return y;
}
