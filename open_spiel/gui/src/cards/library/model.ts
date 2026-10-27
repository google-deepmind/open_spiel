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

// Card model helpers and suit data, exported once for every card UI.
import type {Card, ColorScheme, Rank, Suit} from './types.ts';

/** The suits. */
export const SUITS: readonly Suit[] = ['S', 'H', 'D', 'C'];
/** The ranks, ace low. */
export const RANKS: readonly Rank[] = [
  'A',
  '2',
  '3',
  '4',
  '5',
  '6',
  '7',
  '8',
  '9',
  '10',
  'J',
  'Q',
  'K',
];

/** Each suit's symbol. */
export const SUIT_GLYPH: Record<Suit, string> = {
  S: '♠',
  H: '♥',
  D: '♦',
  C: '♣',
};
/** Each suit's name. */
export const SUIT_NAME: Record<Suit, string> = {
  S: 'Spades',
  H: 'Hearts',
  D: 'Diamonds',
  C: 'Clubs',
};
/** Each rank's name. */
export const RANK_NAME: Record<Rank, string> = {
  A: 'Ace',
  '2': 'Two',
  '3': 'Three',
  '4': 'Four',
  '5': 'Five',
  '6': 'Six',
  '7': 'Seven',
  '8': 'Eight',
  '9': 'Nine',
  '10': 'Ten',
  J: 'Jack',
  Q: 'Queen',
  K: 'King',
};

/** Ink colours, as token references (src/styles/tokens.css). `two`: ♠♣ black, ♥♦ red. `four`: ♠ black, ♥ red, ♦ blue, ♣ green. */
export const INK = {
  black: 'var(--card-ink-black)',
  red: 'var(--card-ink-red)',
  blue: 'var(--card-ink-blue)',
  green: 'var(--card-ink-green)',
} as const;

/** A suit's ink in `scheme`. */
export function suitColor(suit: Suit, scheme: ColorScheme): string {
  switch (suit) {
    case 'S':
      return INK.black;
    case 'H':
      return INK.red;
    case 'D':
      return scheme === 'four' ? INK.blue : INK.red;
    case 'C':
      return scheme === 'four' ? INK.green : INK.black;
    default: {
      const unreachable: never = suit;
      throw new Error(`Unhandled suit: ${JSON.stringify(unreachable)}`);
    }
  }
}

/** Whether `card` is a joker. */
export function isJoker(card: Card): card is {joker: 'red' | 'black'} {
  return 'joker' in card;
}

/** A joker's ink. */
export function jokerColor(color: 'red' | 'black'): string {
  return color === 'red' ? INK.red : INK.black;
}

/** Stable short key: 'QH', '10S', 'JKR' (red joker), 'JKB' (black joker). */
export function cardKey(card: Card): string {
  if (isJoker(card)) {
    return card.joker === 'red' ? 'JKR' : 'JKB';
  }
  return `${card.rank}${card.suit}`;
}

/** Human label for aria: 'Queen of Hearts', 'Red joker'. */
export function cardLabel(card: Card): string {
  if (isJoker(card)) {
    return card.joker === 'red' ? 'Red joker' : 'Black joker';
  }
  return `${RANK_NAME[card.rank]} of ${SUIT_NAME[card.suit]}`;
}

/** The 52 cards, suit-major in SUITS order, then RANKS order. */
export function standardDeck(): Card[] {
  return SUITS.flatMap(suit => RANKS.map(rank => ({rank, suit})));
}

/** The two jokers. */
export const JOKERS: readonly Card[] = [{joker: 'red'}, {joker: 'black'}];
