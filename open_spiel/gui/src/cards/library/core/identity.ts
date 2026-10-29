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

// Card identity: the items a table holds. Pure: no React, no DOM.
import type {Card} from '../types.ts';

/** An item's id, unique on its table. */
export type CardId = string;
/** A point, px. */
export interface Point {
  x: number;
  y: number;
}
/** An axis-aligned rectangle, px. */
export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}
/** A rect plus rotation (degrees clockwise about its centre). */
export interface Box extends Rect {
  rotate: number;
}

/** A card on a table. */
export interface TableCard {
  kind?: 'card';
  /** Required. Unique within a CardScene, and stable across renders and slots.
   *  Motion, drag, focus, measurement and a11y are all keyed on it.
   *  Missing or duplicate -> throws. */
  id: CardId;
  /** null: the client does not know this card (face-down, hidden hand, unrevealed
   *  stock). Always drawn as a back. There are no placeholder identities. */
  card: Card | null;
  /** Default: card !== null. `faceUp: true` with `card: null` throws. */
  faceUp?: boolean;
  /** Identity continuity: this is the same physical card that had id `was` in
   *  the previous render. Motion treats the two as one object: it flies and
   *  flips instead of fading. */
  was?: CardId;
  /** Open state tags. Styled by CardStyle.tags; emitted as data-tags and as the
   *  class pcard--t-<tag>. The library gives a tag no meaning beyond its style
   *  (aria-pressed is the game's word: InteractionPolicy.pressed). */
  tags?: readonly string[];
  /** Default: card !== null && faceUp (known face-up cards are interactive).
   *  false: no activation, no focus, not a tab stop; role="img" when its face
   *  is drawn or the game names it, else decorative (InteractionPolicy.role).
   *  It NEVER changes the look (the card stays opaque, never dimmed); to
   *  tell playable cards apart, raise them (playable.ts). Hover and menus on it: InteractionPolicy.observe. */
  interactive?: boolean;
  /** Turn in play, degrees clockwise. In 'squared' and 'free' slots the
   *  card turns in place. In a fan (a row of cards in play) it takes its
   *  turned footprint and the step past it clears its turned index, so a
   *  tapped card sits anywhere in a row. A quarter turn at position 0 of a
   *  vertical fan is a base lying under the upright pile. Hand slots throw. */
  turn?: 0 | 90 | 180 | 270;
}

/** Non-card objects that share the table: chips, dealer button, markers. */
export interface TableToken {
  kind: 'token';
  /** Same id namespace as cards. */
  id: CardId;
  /** In card widths. */
  size: {width: number; height: number};
  /** Accessible name (the game owns the text). */
  label: string;
  /** Default true. */
  interactive?: boolean;
  tags?: readonly string[];
}
/** Anything on a table: a card or a token. */
export type TableItem = TableCard | TableToken;

/** A slot's items list entry: an item, or `null` — a positional hole. Only a
 *  fan slot takes holes: a place that keeps a face-up card's room while
 *  empty, so later cards never move. A hole draws nothing and is no target;
 *  positions (ItemRef.position, data-position) count it. */
export type SlotEntryItem = TableItem | null;

/** Whether the item is a token rather than a card. */
export function isToken(i: TableItem): i is TableToken {
  return i.kind === 'token';
}

/** Is the card's face drawn? Default: card !== null. */
export function isFaceUp(c: TableCard): boolean {
  return c.faceUp ?? c.card !== null;
}

/** The item's own interactive default (before policy): known face-up cards
 *  and tokens are interactive. */
export function itemInteractive(i: TableItem): boolean {
  if (isToken(i)) return i.interactive ?? true;
  return i.interactive ?? (i.card !== null && isFaceUp(i));
}

const TURNS: readonly unknown[] = [0, 90, 180, 270];

// The checks of one token; `it` is the item as given, not yet trusted.
function assertToken(it: Record<string, unknown>, id: string): void {
  const size = it['size'] as {width?: unknown; height?: unknown} | undefined;
  const ok =
    typeof size === 'object' &&
    typeof size.width === 'number' &&
    typeof size.height === 'number' &&
    size.width > 0 &&
    size.height > 0;
  if (!ok) {
    throw new Error(
      `token '${id}' size must be > 0 card widths, got ${JSON.stringify(size)}`,
    );
  }
  if (typeof it['label'] !== 'string' || it['label'] === '') {
    throw new Error(`token '${id}' needs a label`);
  }
}

// The checks of one card; `it` is the item as given, not yet trusted.
function assertCard(it: Record<string, unknown>, id: string): void {
  if (it['kind'] !== undefined && it['kind'] !== 'card') {
    throw new Error(
      `item '${id}' has unknown kind ${JSON.stringify(it['kind'])}`,
    );
  }
  if (it['card'] === undefined) {
    throw new Error(`card '${id}' has no card (use null for an unknown face)`);
  }
  if (it['card'] === null && it['faceUp'] === true) {
    throw new Error(`card '${id}' is faceUp with card: null`);
  }
  if (it['turn'] !== undefined && !TURNS.includes(it['turn'])) {
    throw new Error(
      `card '${id}' turn must be one of 0/90/180/270, got ${JSON.stringify(it['turn'])}`,
    );
  }
}

/** Throws on a duplicate id, faceUp with a null card, or an unknown slot.
 *  Holes (null entries) are skipped; which slots may hold one is the scene's
 *  rule (fan slots only). */
export function assertItems(
  items: Readonly<Record<string, readonly SlotEntryItem[]>>,
  slotIds: readonly string[],
): void {
  const slots = new Set(slotIds);
  const seen = new Map<CardId, string>();
  for (const [slot, given] of Object.entries(items)) {
    if (!slots.has(slot)) {
      throw new Error(
        `items name slot '${slot}', which the spec does not name`,
      );
    }
    const list: unknown = given;
    if (!Array.isArray(list)) {
      throw new Error(`items['${slot}'] must be an array`);
    }
    list.forEach((entry: unknown, pos) => {
      if (entry === null) return;
      if (typeof entry !== 'object') {
        throw new Error(
          `items['${slot}'][${String(pos)}] is not an item (null is a hole)`,
        );
      }
      const it = entry as Record<string, unknown>;
      const id = it['id'];
      if (typeof id !== 'string' || id === '') {
        throw new Error(
          `items['${slot}'][${String(pos)}] has no id (ids are required and unique)`,
        );
      }
      const prev = seen.get(id);
      if (prev !== undefined) {
        throw new Error(
          `duplicate item id '${id}' (in slots '${prev}' and '${slot}')`,
        );
      }
      seen.set(id, slot);
      if (it['kind'] === 'token') assertToken(it, id);
      else assertCard(it, id);
    });
  }
}
