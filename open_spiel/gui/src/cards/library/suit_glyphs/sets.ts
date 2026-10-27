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

// Suit-symbol sets (CardStyle.suitGlyphs): shapes for the four suits, each
// drawn in code in the same 100x100 box so its extent is exact and any deck
// that draws suit shapes can swap one set for another without moving its
// layout. Every consumer picks a set up through SUIT_GLYPH_SETS.
import type {Suit} from '../types.ts';

/** One filled shape of a glyph, in the 100x100 box. */
export type SuitShape = {d: string} | {cx: number; cy: number; r: number};

/** A suit-symbol set: its label, its source, and its four suits' shapes. */
export interface SuitGlyphSetSpec {
  label: string;
  /** One line: where the shapes come from. */
  source: string;
  shapes: Readonly<Record<Suit, readonly SuitShape[]>>;
}

// 'drawn': the classic suit symbols, drawn for this library from circular
// arcs and cubic Beziers, each mirrored about the box's vertical centre line
// so it is exactly symmetric. A heart is two circular lobes meeting in a
// cleft, their sides running down to a point; a diamond a rhombus taller
// than wide with slightly concave sides; a spade an inverted heart whose
// lobes curl up into a stem that flares into a narrow foot; a club three
// circular lobes (the top one centred (50, 25.5), radius 24; the sides
// (26, 62) and (74, 62), radius 22) on the same stem, traced as one closed
// outline.
const DRAWN: Record<Suit, SuitShape[]> = {
  H: [
    {
      d: 'M50 26C53 12 62 4 73 4C86 4 96 15 96 30C96 56 62 76 50 96C38 76 4 56 4 30C4 15 14 4 27 4C38 4 47 12 50 26Z',
    },
  ],
  D: [
    {
      d: 'M50 1C60.89 17.72 72.32 34.05 84.3 50C72.32 65.95 60.89 82.28 50 99C39.11 82.28 27.68 65.95 15.7 50C27.68 34.05 39.11 17.72 50 1Z',
    },
  ],
  S: [
    {
      d: 'M50 1C55 19 89 42 89 60.5C89 73 80.5 82.5 71 82.5C62.5 82.5 55.5 78.5 52.6 73.5C52.4 86 54 95.5 58.5 99L50 99L41.5 99C46 95.5 47.6 86 47.4 73.5C44.5 78.5 37.5 82.5 29 82.5C19.5 82.5 11 73 11 60.5C11 42 45 19 50 1Z',
    },
  ],
  C: [
    {
      d: 'M50 1.5A24 24 0 0 1 68.59 40.67A22 22 0 1 1 56 74.65C51.5 85.5 53 95.5 58.5 99H41.5C47 95.5 48.5 85.5 44 74.65A22 22 0 1 1 31.41 40.67A24 24 0 0 1 50 1.5Z',
    },
  ],
};

/** Every suit-symbol set, by id. */
export const SUIT_GLYPH_SET_SPECS = {
  drawn: {
    label: 'Drawn',
    source: 'Drawn in code for this library: the classic suit shapes',
    shapes: DRAWN,
  },
} as const satisfies Record<string, SuitGlyphSetSpec>;

/** A suit-symbol set's id. */
export type SuitGlyphSet = keyof typeof SUIT_GLYPH_SET_SPECS;

/** Every suit-symbol set id. */
export const SUIT_GLYPH_SETS: readonly SuitGlyphSet[] = ['drawn'];

/** The set a card draws with unless its CardStyle picks another. */
export const DEFAULT_SUIT_GLYPHS: SuitGlyphSet = 'drawn';

/** Whether `v` is a suit-symbol set id. */
export function isSuitGlyphSet(v: unknown): v is SuitGlyphSet {
  return (
    typeof v === 'string' &&
    Object.prototype.hasOwnProperty.call(SUIT_GLYPH_SET_SPECS, v)
  );
}

/** Throws, naming the choices, unless `v` is a suit-glyph set id. */
export function assertSuitGlyphSet(
  v: unknown,
  where = 'suitGlyphs',
): asserts v is SuitGlyphSet {
  if (!isSuitGlyphSet(v)) {
    throw new Error(
      `${where} must be one of ${SUIT_GLYPH_SETS.join(', ')}, got ${JSON.stringify(v)}`,
    );
  }
}

/** The shapes of `suit` in `set`. */
export function suitShapes(
  set: SuitGlyphSet,
  suit: Suit,
): readonly SuitShape[] {
  assertSuitGlyphSet(set);
  return SUIT_GLYPH_SET_SPECS[set].shapes[suit];
}
