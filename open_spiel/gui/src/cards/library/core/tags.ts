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

// State tags. Pure.

/** Tags the library sets transiently. Games never set these; they only style them. */
export const LIBRARY_TAGS = [
  'focus',
  'hover',
  'dragging',
  'drop-legal',
  'drop-over',
  'held',
] as const;
/** A tag the library sets. */
export type LibraryTag = (typeof LIBRARY_TAGS)[number];
/** Tags with built-in default styles that games set. 'hide-empty' is for
 *  slots: an empty slot carrying it draws nothing (see TagStyle.empty).
 *  There is no 'dimmed': a card that may not be played keeps full contrast
 *  and the playable ones stand up instead (playable.ts). */
export const STYLED_TAGS = ['selected', 'highlighted', 'hide-empty'] as const;
/** A tag with a built-in style that games set. */
export type StyledTag = (typeof STYLED_TAGS)[number];
/** A slot tag (SlotSpec.tags): a symbol-only card (TagStyle.symbol) in the
 *  slot sits at the index corner, rank over suit, instead of the centre. For
 *  a slot whose card another may cover past its index (an overlapped grid):
 *  the corner is what an overlap always leaves in view (cards.css). The
 *  game sets --pcard-symbol-backing to its surface colour, so the box covers
 *  the card under it as a real card would. */
export const SYMBOL_CORNER_TAG = 'symbol-corner';

/** How a tag styles a card or a slot. */
export interface TagStyle {
  /** --pcard-ring (box-shadow ring colour). */
  ring?: string;
  /** --pcard-glow. */
  glow?: string;
  /** --pcard-tint: an overlay colour ABOVE an opaque face; the face stays opaque. */
  tint?: string;
  /** --pcard-band: an edge band colour (e.g. a meld colour). */
  band?: string;
  /** --pcard-filter (e.g. 'grayscale(1)'). */
  filter?: string;
  /** px toward the viewer's up; included in the legibility check. */
  lift?: number;
  cursor?: string;
  /** Stacking bump within the slot. */
  z?: number;
  /** Escape hatch: extra class while the tag is present. */
  className?: string;
  /** Slots only (cards ignore it): how an EMPTY slot carrying the tag draws.
   *  'outline' (the default): the deck's empty placeholder. 'hidden':
   *  nothing, the box kept (still a drop target; a legal drop still marks
   *  it). A play position or a play area that should not read as a gap. */
  empty?: 'outline' | 'hidden';
  /** Cards only: draw the card as its symbol alone (rank and suit, e.g.
   *  '7♥'), centred in the card's box, flat in this ONE colour whatever the
   *  suit's ink. No card body: no face, edge, shadow, ring, glow, tint,
   *  band or filter; the box is transparent. For a card a game names
   *  without holding (a dead or a known card). A CSS colour, typically a
   *  theme token ('var(--color-card-dead-ink)'). A face-down card ignores
   *  it (its face is unknown). In a slot tagged SYMBOL_CORNER_TAG the
   *  symbol sits at the index corner instead, on a box filled with the CSS
   *  variable --pcard-symbol-backing (the surface colour, set by the game).
   *  A library ring (drop-legal, drop-over) still draws on it. */
  symbol?: string;
  // Deliberately NO opacity and NO transform: opacity is how the old dimmed-card
  // hack happened, and transform fights FLIP and turn.
}

/** Every field a TagStyle may have. */
export const TAG_STYLE_FIELDS = [
  'ring',
  'glow',
  'tint',
  'band',
  'filter',
  'lift',
  'cursor',
  'z',
  'className',
  'empty',
  'symbol',
] as const;

/** Whether `t` is a tag the library sets. */
export function isLibraryTag(t: string): t is LibraryTag {
  return (LIBRARY_TAGS as readonly string[]).includes(t);
}

/** A valid tag name: lower-case letters, digits and '-', starting with a letter
 *  (it becomes a class name, pcard--t-<tag>, and a CSS selector). */
export const TAG_NAME = /^[a-z][a-z0-9-]*$/;

/** Throws unless `t` is a valid tag name; `where` names the caller. */
export function assertTagName(t: string, where: string): void {
  if (!TAG_NAME.test(t)) {
    throw new Error(
      `${where}: tag ${JSON.stringify(t)} must match ${String(TAG_NAME)}`,
    );
  }
}

/** Game tags first in item order, then LIBRARY_TAGS present, in LIBRARY_TAGS order. */
export function orderTags(tags: readonly string[]): string[] {
  const game = tags.filter(t => !isLibraryTag(t));
  const lib = LIBRARY_TAGS.filter(t => tags.includes(t));
  return [...new Set(game), ...lib];
}

/** Merge rule: tags apply in item order, then LIBRARY_TAGS; the later tag wins per field.
 *  className accumulates (every present tag's class applies). */
export function resolveTagStyle(
  tags: readonly string[],
  styles: Readonly<Record<string, TagStyle>>,
): TagStyle {
  const out: TagStyle = {};
  const classes: string[] = [];
  for (const t of orderTags(tags)) {
    const s = styles[t];
    if (!s) continue;
    for (const f of TAG_STYLE_FIELDS) {
      const v = s[f];
      if (v === undefined) continue;
      if (f === 'className') classes.push(v as string);
      else (out as Record<string, unknown>)[f] = v;
    }
  }
  if (classes.length) out.className = classes.join(' ');
  return out;
}

/** Throws on a TagStyle field the contract does not have (opacity, transform, …). */
export function assertTagStyle(name: string, s: TagStyle): void {
  for (const k of Object.keys(s)) {
    if (!(TAG_STYLE_FIELDS as readonly string[]).includes(k)) {
      throw new Error(
        `CardStyle.tags['${name}'] has no field '${k}' (TagStyle has no opacity or transform by design)`,
      );
    }
  }
  if (s.lift !== undefined && !Number.isFinite(s.lift)) {
    throw new Error(`CardStyle.tags['${name}'].lift must be finite`);
  }
  if (s.z !== undefined && !Number.isFinite(s.z)) {
    throw new Error(`CardStyle.tags['${name}'].z must be finite`);
  }
  const empty: unknown = s.empty;
  if (empty !== undefined && empty !== 'outline' && empty !== 'hidden') {
    throw new Error(
      `CardStyle.tags['${name}'].empty must be 'outline' or 'hidden', got ${JSON.stringify(s.empty)}`,
    );
  }
  if (
    s.symbol !== undefined &&
    (typeof s.symbol !== 'string' || s.symbol.trim() === '')
  ) {
    throw new Error(
      `CardStyle.tags['${name}'].symbol must be a CSS colour, got ${JSON.stringify(s.symbol)}`,
    );
  }
}
