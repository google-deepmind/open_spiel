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

// Card types and utilities for Euchre game

/** A suit: hearts, diamonds, clubs or spades. */
export type Suit = 'h' | 'd' | 'c' | 's';
/** A rank of the 24-card euchre deck. */
export type EuchreRank = '9' | 'T' | 'J' | 'Q' | 'K' | 'A';

/** Card format: "Ah" = Ace of hearts, "Ts" = 10 of spades, "Qd" = Queen of diamonds */
export type CardCode = `${EuchreRank}${Suit}`;

/** A seat's place at the table; the viewer sits at the bottom. */
export type PlayerPosition = 'top' | 'left' | 'right' | 'bottom';

/** The stage of the hand, as the adapter's struct names it. */
export type GamePhase =
  'dealing' | 'bidding1' | 'bidding2' | 'alone' | 'playing' | 'terminal';

/** The kind of a legal action. */
export type ActionType = 'card' | 'pass' | 'call_suit' | 'alone' | 'partner';

/** A card on the table and the place of the seat that played it. */
export interface TrickCard {
  card: CardCode;
  position: PlayerPosition;
}

/** One seat at the table, as the board draws it. */
export interface Player {
  name: string;
  hand: CardCode[];
  position: PlayerPosition;
  tricksWon?: number;
  isDealer?: boolean;
  isDeclarer?: boolean;
  /** The hand is concealed from this viewer: `hand` holds HIDDEN_CARD backs. */
  hidden?: boolean;
}

/**
 * A hand the server withheld from this viewer (game mode): the
 * struct carries one `null` per card, since the count is public and the table
 * draws a back for each. `hidden` players carry this placeholder in `hand`,
 * and nothing may render it face up.
 */
export const HIDDEN_CARD = '??' as unknown as CardCode;

/** True when the served hand is the masked (card-back) form. */
export function isConcealedHand(hand: ReadonlyArray<string | null>): boolean {
  return hand.some(card => card === null);
}

/** The table's state. */
export interface EuchreGameState {
  players: Player[];
  trickPile: TrickCard[];
  phase: GamePhase;
  upcard: CardCode | null;
  trumpSuit: Suit | null;
  dealer: PlayerPosition;
  declarer: PlayerPosition | null;
  goingAlone: boolean;
  currentPlayer: PlayerPosition;
  tricksWon: {NS: number; EW: number};
}

/** Suit metadata */
export const SUIT_META: Record<
  Suit,
  {name: string; symbol: string; svgName: string; color: string}
> = {
  h: {name: 'Hearts', symbol: '♥', svgName: 'heart', color: '#e53e3e'},
  d: {name: 'Diamonds', symbol: '♦', svgName: 'diamond', color: '#e53e3e'},
  c: {name: 'Clubs', symbol: '♣', svgName: 'club', color: '#1a202c'},
  s: {name: 'Spades', symbol: '♠', svgName: 'spade', color: '#1a202c'},
};

/** Rank metadata - SVG uses 1 for Ace */
export const RANK_META: Record<
  EuchreRank,
  {name: string; svgName: string; value: number}
> = {
  '9': {name: '9', svgName: '9', value: 9},
  T: {name: '10', svgName: '10', value: 10},
  J: {name: 'Jack', svgName: 'jack', value: 11},
  Q: {name: 'Queen', svgName: 'queen', value: 12},
  K: {name: 'King', svgName: 'king', value: 13},
  A: {name: 'Ace', svgName: '1', value: 14},
};

/** All ranks for iteration (Euchre uses 9-A only) */
export const RANKS: EuchreRank[] = ['9', 'T', 'J', 'Q', 'K', 'A'];
/** All suits for iteration: clubs, diamonds, spades, hearts */
export const SUITS: Suit[] = ['c', 'd', 's', 'h'];

/**
 * Convert our card code format to SVG ID format
 * e.g., "Ah" -> "heart_1", "Qs" -> "spade_queen"
 */
export function cardCodeToSvgId(card: CardCode): string {
  const rank = card[0] as EuchreRank;
  const suit = card[1] as Suit;
  return `${SUIT_META[suit].svgName}_${RANK_META[rank].svgName}`;
}

/**
 * Parse a card code into its components
 */
export function parseCard(card: CardCode): {rank: EuchreRank; suit: Suit} {
  return {
    rank: card[0] as EuchreRank,
    suit: card[1] as Suit,
  };
}

/**
 * Get display name for a card
 * e.g., "Ah" -> "Ace of Hearts"
 */
export function getCardDisplayName(card: CardCode): string {
  const {rank, suit} = parseCard(card);
  return `${RANK_META[rank].name} of ${SUIT_META[suit].name}`;
}

/**
 * Get short display for a card
 * e.g., "Ah" -> "A♥"
 */
export function getCardShortDisplay(card: CardCode): string {
  const {rank, suit} = parseCard(card);
  const displayRank = rank === 'T' ? '10' : rank;
  return `${displayRank}${SUIT_META[suit].symbol}`;
}

/**
 * Get the "same color" suit for bower purposes
 * Hearts <-> Diamonds, Clubs <-> Spades
 */
export function getSameColorSuit(suit: Suit): Suit {
  const colorPairs: Record<Suit, Suit> = {
    h: 'd',
    d: 'h',
    c: 's',
    s: 'c',
  };
  return colorPairs[suit];
}

/**
 * Check if a card is the Right Bower (Jack of trump)
 */
export function isRightBower(card: CardCode, trumpSuit: Suit | null): boolean {
  if (!trumpSuit) return false;
  const {rank, suit} = parseCard(card);
  return rank === 'J' && suit === trumpSuit;
}

/**
 * Check if a card is the Left Bower (Jack of same color as trump)
 */
export function isLeftBower(card: CardCode, trumpSuit: Suit | null): boolean {
  if (!trumpSuit) return false;
  const {rank, suit} = parseCard(card);
  return rank === 'J' && suit === getSameColorSuit(trumpSuit);
}

/**
 * Get the effective suit of a card (Left Bower counts as trump)
 */
export function getEffectiveSuit(card: CardCode, trumpSuit: Suit | null): Suit {
  if (isLeftBower(card, trumpSuit) && trumpSuit) {
    return trumpSuit;
  }
  return parseCard(card).suit;
}

/**
 * Generate the Euchre deck (24 cards: 9-A in all suits)
 */
export function generateDeck(): CardCode[] {
  const deck: CardCode[] = [];
  for (const suit of SUITS) {
    for (const rank of RANKS) {
      deck.push(`${rank}${suit}`);
    }
  }
  return deck;
}

/**
 * Sort cards by suit then rank (for display)
 * Suit order: clubs, diamonds, spades, hearts (left to right)
 * Rank order: low to high within each suit (9, T, J, Q, K, A)
 */
export function sortCards(
  cards: CardCode[],
  trumpSuit?: Suit | null,
): CardCode[] {
  return [...cards].sort((a, b) => {
    const {rank: rankA, suit: suitA} = parseCard(a);
    const {rank: rankB, suit: suitB} = parseCard(b);

    // If there's a trump suit, sort trump cards last (rightmost)
    if (trumpSuit) {
      const aIsTrump = getEffectiveSuit(a, trumpSuit) === trumpSuit;
      const bIsTrump = getEffectiveSuit(b, trumpSuit) === trumpSuit;
      if (aIsTrump !== bIsTrump) {
        return aIsTrump ? 1 : -1;
      }
    }

    // Sort by suit first (c, d, s, h)
    const suitOrder = SUITS.indexOf(suitA) - SUITS.indexOf(suitB);
    if (suitOrder !== 0) return suitOrder;

    // Then by rank value (low to high)
    return RANK_META[rankA].value - RANK_META[rankB].value;
  });
}

/**
 * Get the team for a position
 */
export function getTeam(position: PlayerPosition): 'NS' | 'EW' {
  return position === 'top' || position === 'bottom' ? 'NS' : 'EW';
}

/**
 * Check if two positions are on the same team
 */
export function arePartners(
  pos1: PlayerPosition,
  pos2: PlayerPosition,
): boolean {
  return getTeam(pos1) === getTeam(pos2);
}
