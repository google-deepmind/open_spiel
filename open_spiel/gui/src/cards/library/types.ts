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

// The shared playing-card contract: cards, decks and their geometry, which
// every layout and deck builds against.
import type {ReactNode} from 'react';
import type {SuitGlyphSet} from './suit_glyphs/sets.ts';

/** A card's suit: spades, hearts, diamonds or clubs. */
export type Suit = 'S' | 'H' | 'D' | 'C';
/** A card's rank. */
export type Rank =
  'A' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' | '10' | 'J' | 'Q' | 'K';
/** A card: a rank and suit, or a joker. */
export type Card = {rank: Rank; suit: Suit} | {joker: 'red' | 'black'};
/** four: ♠ black, ♥ red, ♦ blue, ♣ green. */
export type ColorScheme = 'two' | 'four';

/** Box that must stay visible for the card to be identifiable (corner rank + suit),
 *  as fractions of card width/height, measured from the top-left corner. */
export interface IndexGeometry {
  top: number;
  left: number;
  width: number;
  height: number;
}

/** What renderFace draws a face with (from the CardStyle in effect). */
export interface FaceOptions {
  scheme: ColorScheme;
  indexScale?: number | undefined;
  suitGlyphs?: SuitGlyphSet | undefined;
}

/** A deck: how its faces and backs are drawn, and their geometry. */
export interface Deck {
  /** 'simple' | 'simple-corner' | 'basic' | ... */
  id: string;
  /** The picker's name for the face style: what it looks like. */
  label: string;
  /** One line under the picker saying what the style is best for. */
  description?: string | undefined;
  /** Licence line; 'private-only' decks say so. */
  provenance: string;
  /** width / height */
  aspect: number;
  index: IndexGeometry;
  /** The index box when drawn at `scale` (CardStyle.indexScale).
   *  Absent: the deck cannot redraw its index, which stays at `index`. */
  indexAt?(scale: number): IndexGeometry;
  /** Fills its box. `indexScale` (default 1) enlarges the corner index on
   *  decks that define indexAt; others ignore it. `suitGlyphs`
   *  (CardStyle.suitGlyphs, default DEFAULT_SUIT_GLYPHS) is the suit-shape
   *  set for decks that draw their suits with SuitGlyph; decks whose art
   *  carries its own suits ignores it. */
  renderFace(card: Card, opts: FaceOptions): ReactNode;
  renderBack(opts: {variant?: string | undefined}): ReactNode;
  /** The face draws its index a second time, rotated 180° about the card's
   *  centre, so the bottom-right twin's box is `index` mirrored.
   *  fan-left and fan-up piles cover the top-left index and read the twin;
   *  they refuse a deck that does not set this. */
  rotatedTwin?: boolean | undefined;
  /** Back variant ids renderBack accepts (the View panel's Card back picker). */
  backs?: readonly string[] | undefined;
  /** Each back's name in the picker. Absent for a back: its id, title-cased. */
  backLabels?: Readonly<Record<string, string>> | undefined;
  /** The face draws its suits with SuitGlyph, so CardStyle.suitGlyphs
   *  changes it (basic, simple). Absent/false: the art carries its own
   *  suits and the suit-symbol setting is not offered. */
  drawsSuitGlyphs?: boolean | undefined;
  /** The card's edge line. Absent/false (the rule): the art draws
   *  NO outline of its own, and the shared container draws one from
   *  CardStyle.outline along the rounded rect that clips the face, so the
   *  clip cannot cut it and every such deck gets the same line at every
   *  size. true: the art's own border is the outline (bitmap decks such as
   *  a bitmap deck, whose border is part of the pixels) and the container draws
   *  none, so there is never a doubled line. */
  drawsOwnOutline?: boolean | undefined;
  /** Optional synchronous fast path for canvas painting (bitmap/sprite decks).
   *  card === null draws the back. Windows implements it with one drawImage.
   *  `r` is in the ctx's current units. */
  draw?(
    ctx: CanvasRenderingContext2D,
    card: Card | null,
    r: {x: number; y: number; width: number; height: number},
    o: {scheme: ColorScheme; indexScale: number; back?: string | undefined},
  ): void;
  /** What `draw` needs loaded first (a sprite sheet). preloadCards awaits it
   *  before the first synchronous drawCard on this deck; draw throws until it
   *  has resolved. Only meaningful with `draw`. */
  preload?(): Promise<void>;
}
