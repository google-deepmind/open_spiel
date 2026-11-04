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

// Card types and utilities for Hearts game
import type {
  Card as SharedCard,
  Suit as SharedSuit,
} from '../../cards/library/index.ts';

/** A suit: hearts, diamonds, clubs, spades. */
export type Suit = 'h' | 'd' | 'c' | 's';
/** A rank, T for ten. */
export type Rank =
  'A' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' | 'T' | 'J' | 'Q' | 'K';

/** Card format: "Ah" = Ace of hearts, "Ts" = 10 of spades, "Qd" = Queen of diamonds. */
export type CardCode = `${Rank}${Suit}`;

/** A seat as the table draws it: yours at the bottom. */
export type PlayerPosition = 'top' | 'left' | 'right' | 'bottom';

/** The phase of the hand. */
export type GamePhase = 'pass' | 'play';

/** Where the hand's pass goes. */
export type PassDirection = 'left' | 'right' | 'across' | 'none';

/** A card of the trick, at the seat that played it. */
export interface TrickCard {
  card: CardCode;
  position: PlayerPosition;
}

/** A seat at the table: its name, hand, position and points. */
export interface Player {
  name: string;
  hand: CardCode[];
  position: PlayerPosition;
  score?: number;
  /** The hand is concealed from this viewer: `hand` holds HIDDEN_CARD backs. */
  hidden?: boolean;
}

/**
 * A hand the backend withheld from this viewer (game mode): the
 * struct carries one `null` per card, since the count is public and the table
 * draws a back for each. `hidden` players carry this placeholder in `hand`,
 * and nothing may render it face up.
 */
export const HIDDEN_CARD = '??' as unknown as CardCode;

/** True when the served hand is the masked (card-back) form. */
export function isConcealedHand(hand: ReadonlyArray<string | null>): boolean {
  return hand.some(card => card === null);
}

/** Suit metadata. */
export const SUIT_META: Record<Suit, {name: string; symbol: string}> = {
  h: {name: 'Hearts', symbol: '♥'},
  d: {name: 'Diamonds', symbol: '♦'},
  c: {name: 'Clubs', symbol: '♣'},
  s: {name: 'Spades', symbol: '♠'},
};

/** Rank metadata. */
export const RANK_META: Record<Rank, {name: string; value: number}> = {
  A: {name: 'Ace', value: 14},
  '2': {name: '2', value: 2},
  '3': {name: '3', value: 3},
  '4': {name: '4', value: 4},
  '5': {name: '5', value: 5},
  '6': {name: '6', value: 6},
  '7': {name: '7', value: 7},
  '8': {name: '8', value: 8},
  '9': {name: '9', value: 9},
  T: {name: '10', value: 10},
  J: {name: 'Jack', value: 11},
  Q: {name: 'Queen', value: 12},
  K: {name: 'King', value: 13},
};

/** All ranks, for iteration. */
export const RANKS: Rank[] = [
  'A',
  '2',
  '3',
  '4',
  '5',
  '6',
  '7',
  '8',
  '9',
  'T',
  'J',
  'Q',
  'K',
];
/** The suits as a hand shows them: clubs, diamonds, spades, hearts (left to right). */
export const SUITS: Suit[] = ['c', 'd', 's', 'h'];

/**
 * The shared card model's card for a hearts card code, for rendering with
 * cards/library. e.g. "Th" -> { rank: '10', suit: 'H' }. Throws on anything
 * that is not a card code: a malformed code from the backend must not render
 * as some other card.
 */
export function toSharedCard(code: string): SharedCard {
  const rank = code[0] as Rank;
  const suit = code[1] as Suit;
  if (code.length !== 2 || !(rank in RANK_META) || !(suit in SUIT_META)) {
    throw new Error(`hearts: not a card code: ${JSON.stringify(code)}`);
  }
  return {
    rank: rank === 'T' ? '10' : rank,
    suit: suit.toUpperCase() as SharedSuit,
  };
}

/**
 * Parse a card code into its components
 */
export function parseCard(card: CardCode): {rank: Rank; suit: Suit} {
  return {
    rank: card[0] as Rank,
    suit: card[1] as Suit,
  };
}

/**
 * Check if a card is a heart
 */
export function isHeart(card: CardCode): boolean {
  return card[1] === 'h';
}

/**
 * Check if a card is the Queen of Spades
 */
export function isQueenOfSpades(card: CardCode): boolean {
  return card === 'Qs';
}

/**
 * Get points value of a card in Hearts
 * Hearts = 1 point each, Queen of Spades = 13 points
 */
export function getPointValue(card: CardCode): number {
  if (isQueenOfSpades(card)) return 13;
  if (isHeart(card)) return 1;
  return 0;
}

/**
 * Sort cards by suit then rank (for display)
 * Suit order: clubs, diamonds, spades, hearts (left to right)
 * Rank order: low to high within each suit (2, 3, 4... K, A)
 */
export function sortCards(cards: CardCode[]): CardCode[] {
  return [...cards].sort((a, b) => {
    const {rank: rankA, suit: suitA} = parseCard(a);
    const {rank: rankB, suit: suitB} = parseCard(b);

    // Sort by suit first (c, d, s, h)
    const suitOrder = SUITS.indexOf(suitA) - SUITS.indexOf(suitB);
    if (suitOrder !== 0) return suitOrder;

    // Then by rank value (low to high)
    return RANK_META[rankA].value - RANK_META[rankB].value;
  });
}
