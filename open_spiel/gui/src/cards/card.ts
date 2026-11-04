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

// The card model. On the wire a card is a short code, rank then suit: "As",
// "Td", "10h", "2C". A face-down card, or one the viewer may not see, is
// null. Ranks and suits are read case-insensitively; "10" is "T".

/** A suit: spades, hearts, diamonds or clubs. */
export type Suit = 's' | 'h' | 'd' | 'c';

/** A rank, highest first; "T" is ten. */
export type Rank =
  'A' | 'K' | 'Q' | 'J' | 'T' | '9' | '8' | '7' | '6' | '5' | '4' | '3' | '2';

/** A face-up card. */
export interface Card {
  readonly rank: Rank;
  readonly suit: Suit;
}

/** The suits, in bridge order from the highest. */
export const SUITS: readonly Suit[] = ['s', 'h', 'd', 'c'];

/** The ranks, from the highest. */
export const RANKS: readonly Rank[] = [
  'A',
  'K',
  'Q',
  'J',
  'T',
  '9',
  '8',
  '7',
  '6',
  '5',
  '4',
  '3',
  '2',
];

function isSuit(text: string): text is Suit {
  return (SUITS as readonly string[]).includes(text);
}

function isRank(text: string): text is Rank {
  return (RANKS as readonly string[]).includes(text);
}

/** The card a code names; throws on a code that names none. */
export function parseCard(code: string): Card {
  const suit = code.slice(-1).toLowerCase();
  const rankText = code.slice(0, -1).toUpperCase();
  const rank = rankText === '10' ? 'T' : rankText;
  if (!isSuit(suit) || !isRank(rank)) {
    throw new Error(`"${code}" is not a card.`);
  }
  return {rank, suit};
}

/** A card's canonical code: "As", "Td". */
export function cardCode({rank, suit}: Card): string {
  return `${rank}${suit}`;
}

/** The card a code names, or null for a hidden card. */
export function parseShown(code: string | null): Card | null {
  return code === null ? null : parseCard(code);
}
