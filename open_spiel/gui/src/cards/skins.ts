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

// The deck and skin registry: how a card's face is drawn. A skin maps a card
// to the text and colour tone of its face, so art can replace the glyphs
// later by registering a skin, without touching any renderer.

import type {Card, Rank, Suit} from './card.ts';

/** A colour tone a skin may give a suit; each is a token in tokens.css. */
export type SuitTone = 'red' | 'black' | 'blue' | 'green';

/** How a deck's faces and backs are drawn. */
export interface Skin {
  readonly name: string;
  /** The text of a rank's index: "A", "10". */
  rankLabel(rank: Rank): string;
  /** The suit's glyph. */
  suitGlyph(suit: Suit): string;
  /** The suit's colour tone. */
  suitTone(suit: Suit): SuitTone;
  /** The accessible name of a card: "Ace of spades". */
  cardName(card: Card): string;
}

const GLYPHS: Record<Suit, string> = {s: '♠', h: '♥', d: '♦', c: '♣'};

const SUIT_NAMES: Record<Suit, string> = {
  s: 'spades',
  h: 'hearts',
  d: 'diamonds',
  c: 'clubs',
};

const RANK_NAMES: Record<Rank, string> = {
  A: 'Ace',
  K: 'King',
  Q: 'Queen',
  J: 'Jack',
  T: 'Ten',
  '9': 'Nine',
  '8': 'Eight',
  '7': 'Seven',
  '6': 'Six',
  '5': 'Five',
  '4': 'Four',
  '3': 'Three',
  '2': 'Two',
};

const TWO_COLOUR: Record<Suit, SuitTone> = {
  s: 'black',
  h: 'red',
  d: 'red',
  c: 'black',
};

const FOUR_COLOUR: Record<Suit, SuitTone> = {
  s: 'black',
  h: 'red',
  d: 'blue',
  c: 'green',
};

function glyphSkin(name: string, tones: Record<Suit, SuitTone>): Skin {
  return {
    name,
    rankLabel: rank => (rank === 'T' ? '10' : rank),
    suitGlyph: suit => GLYPHS[suit],
    suitTone: suit => tones[suit],
    cardName: ({rank, suit}) => `${RANK_NAMES[rank]} of ${SUIT_NAMES[suit]}`,
  };
}

/** The default skin: Unicode suit glyphs, red hearts and diamonds. */
export const STANDARD_SKIN = glyphSkin('standard', TWO_COLOUR);

const skins = new Map<string, Skin>([
  [STANDARD_SKIN.name, STANDARD_SKIN],
  ['four_colour', glyphSkin('four_colour', FOUR_COLOUR)],
]);

/** Registers `skin` under its name; throws if the name is taken. */
export function registerSkin(skin: Skin): void {
  if (skins.has(skin.name)) {
    throw new Error(`Skin "${skin.name}" is already registered.`);
  }
  skins.set(skin.name, skin);
}

/** The skin named `name`; the standard skin when there is none. */
export function skinFor(name: string | null | undefined): Skin {
  return (
    (name === null || name === undefined ? undefined : skins.get(name)) ??
    STANDARD_SKIN
  );
}
