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

// Card sets: what a game deals
// from, as data. A CardSetSpec says which ranks and suits make one pack, how
// many packs are shuffled together and how many jokers join them; makeCardSet
// turns it into physical card INSTANCES, each with a stable id that is unique
// even when two instances show the same face (two Eights of Hearts). The face
// (a Card) is drawn by whichever registered Deck is active, unchanged: a card
// set says nothing about art, and a Deck says nothing about composition.
// Pure: no React, no DOM.
import {cardKey, cardLabel, isJoker, RANKS, SUITS} from './model.ts';
import type {Card, Rank, Suit} from './types.ts';

/** A joker's colour. */
export type JokerColor = 'red' | 'black';

/** Which cards a set holds: packs, ranks, suits and jokers. */
export interface CardSetSpec {
  /** Copies of the ranks × suits pack shuffled together. Default 1. */
  packs?: number | undefined;
  /** The ranks one pack holds, in pack order. Default all 13 (RANKS). */
  ranks?: readonly Rank[] | undefined;
  /** The suits one pack holds, in pack order. Default all 4 (SUITS). */
  suits?: readonly Suit[] | undefined;
  /** Jokers in the whole set (not per pack). Default 0. */
  jokers?: number | undefined;
  /** One colour per joker, in order. Default: red, black, red, black, ... */
  jokerColors?: readonly JokerColor[] | undefined;
}

/** An instance id: `${cardKey(face)}.${copy}`. '8H.2' is the second Eight of Hearts;
 *  'JKR.1' is the first red joker. Parses without the set. */
export type CardInstanceId = string;

/** One physical card of a set: its id, its face and which copy it is. */
export interface CardInstance {
  id: CardInstanceId;
  /** The face; hand it to any Deck (PlayingCard, TableCard.card). */
  card: Card;
  /** 1-based: which of the set's copies of this face (a suited card's copy is
   *  its pack). */
  copy: number;
}

/** A spec with every default filled in. */
export interface ResolvedCardSetSpec {
  packs: number;
  ranks: readonly Rank[];
  suits: readonly Suit[];
  jokers: number;
  jokerColors: readonly JokerColor[];
}

/** A resolved card set: its spec and every card instance in it. */
export interface CardSet {
  spec: ResolvedCardSetSpec;
  /** Every instance, pack by pack (suit-major in `suits` order, then `ranks`
   *  order), then the jokers. */
  cards: readonly CardInstance[];
  size: number;
  /** The distinct faces, in first-seen order. */
  faces: readonly Card[];
  has(id: CardInstanceId): boolean;
  /** The instance for an id; throws on an id the set does not hold. */
  get(id: CardInstanceId): CardInstance;
  /** Every instance showing `card`, copy 1 first (empty: not in the set). */
  copiesOf(card: Card): readonly CardInstance[];
  /** How many instances show `card`. */
  copies(card: Card): number;
}

const RANK_SET = new Set<string>(RANKS);
const SUIT_SET = new Set<string>(SUITS);

function assertCount(name: string, n: unknown, min: number): number {
  if (typeof n !== 'number' || !Number.isInteger(n) || n < min) {
    throw new Error(
      `CardSetSpec.${name} must be an integer >= ${String(min)}, got ${JSON.stringify(n)}`,
    );
  }
  return n;
}

function assertList<T extends string>(
  name: string,
  list: readonly T[],
  valid: Set<string>,
): readonly T[] {
  const given: unknown = list;
  if (!Array.isArray(given) || given.length === 0) {
    throw new Error(`CardSetSpec.${name} must be a non-empty array`);
  }
  const seen = new Set<string>();
  for (const v of list) {
    if (!valid.has(v)) {
      throw new Error(
        `CardSetSpec.${name}: unknown value ${JSON.stringify(v)}`,
      );
    }
    if (seen.has(v)) {
      throw new Error(
        `CardSetSpec.${name}: '${v}' is listed twice (use packs for copies)`,
      );
    }
    seen.add(v);
  }
  return [...list];
}

/** Fills defaults and validates; throws on anything malformed. */
export function resolveCardSetSpec(
  spec: CardSetSpec = {},
): ResolvedCardSetSpec {
  const known = new Set(['packs', 'ranks', 'suits', 'jokers', 'jokerColors']);
  for (const k of Object.keys(spec)) {
    if (!known.has(k)) throw new Error(`CardSetSpec: unknown field '${k}'`);
  }
  const packs = assertCount('packs', spec.packs ?? 1, 1);
  const ranks = assertList('ranks', spec.ranks ?? RANKS, RANK_SET);
  const suits = assertList('suits', spec.suits ?? SUITS, SUIT_SET);
  const jokers = assertCount('jokers', spec.jokers ?? 0, 0);
  const jokerColors: JokerColor[] = spec.jokerColors
    ? [...spec.jokerColors]
    : Array.from({length: jokers}, (_, i) => (i % 2 === 0 ? 'red' : 'black'));
  if (jokerColors.length !== jokers) {
    throw new Error(
      `CardSetSpec.jokerColors has ${String(jokerColors.length)} colours for ${String(jokers)} jokers`,
    );
  }
  for (const c of jokerColors as readonly string[]) {
    if (c !== 'red' && c !== 'black') {
      throw new Error(
        `CardSetSpec.jokerColors: unknown colour ${JSON.stringify(c)}`,
      );
    }
  }
  return {packs, ranks, suits, jokers, jokerColors};
}

// ---------------------------------------------------------------------------
// Instance ids.

const SUITED_ID = /^(10|[2-9AJQK])([SHDC])\.([1-9][0-9]*)$/;
const JOKER_ID = /^JK([RB])\.([1-9][0-9]*)$/;

/** The id of copy `copy` (1-based) of `card`: '8H.2', 'JKB.1'. */
export function instanceId(card: Card, copy: number): CardInstanceId {
  if (!Number.isInteger(copy) || copy < 1) {
    throw new Error(
      `instanceId: copy must be an integer >= 1, got ${String(copy)}`,
    );
  }
  return `${cardKey(card)}.${String(copy)}`;
}

/** Inverse of instanceId; throws on anything that is not an instance id. */
export function parseInstanceId(id: CardInstanceId): {
  card: Card;
  copy: number;
} {
  const s = SUITED_ID.exec(id);
  if (s) {
    return {card: {rank: s[1] as Rank, suit: s[2] as Suit}, copy: Number(s[3])};
  }
  const j = JOKER_ID.exec(id);
  if (j) {
    return {card: {joker: j[1] === 'R' ? 'red' : 'black'}, copy: Number(j[2])};
  }
  throw new Error(
    `not a card instance id: ${JSON.stringify(id)} (expected e.g. '8H.2' or 'JKR.1')`,
  );
}

/** The face an instance id shows. */
export function faceOf(id: CardInstanceId): Card {
  return parseInstanceId(id).card;
}

/** Same face, whatever the copy ('8H.1' and '8H.2'). */
export function sameFace(a: CardInstanceId, b: CardInstanceId): boolean {
  return cardKey(faceOf(a)) === cardKey(faceOf(b));
}

// ---------------------------------------------------------------------------
// Building a set.

/** The instances `spec` describes, with lookups. Throws on a malformed spec. */
export function makeCardSet(spec: CardSetSpec = {}): CardSet {
  const r = resolveCardSetSpec(spec);
  const cards: CardInstance[] = [];
  const byFace = new Map<string, CardInstance[]>();
  const faces: Card[] = [];
  const add = (card: Card) => {
    const key = cardKey(card);
    let list = byFace.get(key);
    if (!list) {
      list = [];
      byFace.set(key, list);
      faces.push(card);
    }
    const inst: CardInstance = {
      id: instanceId(card, list.length + 1),
      card,
      copy: list.length + 1,
    };
    list.push(inst);
    cards.push(inst);
  };
  for (let p = 0; p < r.packs; p++) {
    for (const suit of r.suits) for (const rank of r.ranks) add({rank, suit});
  }
  for (const color of r.jokerColors) add({joker: color});
  const byId = new Map(cards.map(c => [c.id, c] as const));
  if (byId.size !== cards.length) {
    throw new Error('makeCardSet: duplicate instance id (library bug)');
  }
  return {
    spec: r,
    cards,
    size: cards.length,
    faces,
    has: id => byId.has(id),
    get(id) {
      const c = byId.get(id);
      if (!c) {
        throw new Error(
          `card set has no instance '${id}' (${String(cards.length)} cards)`,
        );
      }
      return c;
    },
    copiesOf: card => byFace.get(cardKey(card)) ?? [],
    copies: card => byFace.get(cardKey(card))?.length ?? 0,
  };
}

// ---------------------------------------------------------------------------
// Counting and naming.

type Countable = Card | CardInstance;
function faceOfCountable(c: Countable): Card {
  return 'copy' in c ? c.card : c;
}

/** Cards per rank (jokers are not counted). */
export function countByRank(
  cards: readonly Countable[],
): Partial<Record<Rank, number>> {
  const out: Partial<Record<Rank, number>> = {};
  for (const c of cards.map(faceOfCountable)) {
    if (!isJoker(c)) out[c.rank] = (out[c.rank] ?? 0) + 1;
  }
  return out;
}

/** Cards per suit (jokers are not counted). */
export function countBySuit(
  cards: readonly Countable[],
): Partial<Record<Suit, number>> {
  const out: Partial<Record<Suit, number>> = {};
  for (const c of cards.map(faceOfCountable)) {
    if (!isJoker(c)) out[c.suit] = (out[c.suit] ?? 0) + 1;
  }
  return out;
}

/** Cards per face, keyed by cardKey ('8H', 'JKR'). */
export function countByFace(
  cards: readonly Countable[],
): Record<string, number> {
  const out: Record<string, number> = {};
  for (const c of cards.map(faceOfCountable)) {
    out[cardKey(c)] = (out[cardKey(c)] ?? 0) + 1;
  }
  return out;
}

const ORDINALS = [
  'first',
  'second',
  'third',
  'fourth',
  'fifth',
  'sixth',
  'seventh',
  'eighth',
  'ninth',
  'tenth',
];

function ordinal(n: number): string {
  const named = ORDINALS[n - 1];
  if (named !== undefined) return named;
  const suffixes: Record<number, string> = {1: 'st', 2: 'nd', 3: 'rd'};
  const s = n % 100 >= 11 && n % 100 <= 13 ? 'th' : (suffixes[n % 10] ?? 'th');
  return `${String(n)}${s}`;
}

/**
 * The accessible name of an instance: the face's name, plus which copy when
 * the face repeats — 'Eight of Hearts (second pack)', 'Red joker (second
 * copy)'. With `set`, a face the set holds once gets no suffix. Games hand it
 * to InteractionPolicy.label; the face itself never shows the copy.
 */
export function instanceLabel(id: CardInstanceId, set?: CardSet): string {
  const {card, copy} = parseInstanceId(id);
  if (set) {
    if (!set.has(id)) {
      throw new Error(`instanceLabel: the card set has no instance '${id}'`);
    }
    if (set.copies(card) === 1) return cardLabel(card);
  }
  return `${cardLabel(card)} (${ordinal(copy)} ${isJoker(card) ? 'copy' : 'pack'})`;
}

/** An instance as a TableCard (Table, Pile, Hand items): the instance id is
 *  the item id, so duplicate faces are distinct items for motion, drag, focus
 *  and a11y. `extra` sets the rest (faceUp, tags, interactive, ...). */
export function instanceItem<E extends object>(
  inst: CardInstance,
  extra?: E,
): {id: CardInstanceId; card: Card} & E {
  return Object.assign({}, extra, {id: inst.id, card: inst.card});
}

// ---------------------------------------------------------------------------
// Presets: each is only a spec. Named by composition, never by game.

function ranksFrom(low: Rank): readonly Rank[] {
  const up: Rank[] = [
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
    'A',
  ];
  const i = up.indexOf(low);
  if (i < 0) throw new Error(`ranksFrom: unknown rank ${low}`);
  return up.slice(i);
}

/** The card sets games deal from, by name. */
export const CARD_SETS = {
  /** 52: one full pack. */
  STANDARD_52: {packs: 1},
  /** 54: one full pack and a red and a black joker. */
  STANDARD_54: {packs: 1, jokers: 2},
  /** 104: two full packs. */
  DOUBLE_52: {packs: 2},
  /** 108: two full packs and four jokers (two red, two black). */
  DOUBLE_54: {packs: 2, jokers: 4},
  /** 36: one pack of Six through Ace (short deck). */
  SHORT_36: {packs: 1, ranks: ranksFrom('6')},
  /** 32: one pack of Seven through Ace. */
  SHORT_32: {packs: 1, ranks: ranksFrom('7')},
  /** 24: one pack of Nine through Ace. */
  SHORT_24: {packs: 1, ranks: ranksFrom('9')},
  /** 48: two packs of Nine through Ace. */
  DOUBLE_24: {packs: 2, ranks: ranksFrom('9')},
} as const satisfies Record<string, CardSetSpec>;

/** A named card set in CARD_SETS. */
export type CardSetPreset = keyof typeof CARD_SETS;
