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

// Table and card settings as data: the
// schema, its defaults, where it persists on this browser, and how the layers
// merge. No React here; the hooks are in tableCards.tsx.
//
// localStorage IS the state. Every read goes to it (parsed once per distinct
// stored string), and every write notifies the subscribers, so two panels
// that show the same setting can never disagree.
import {
  DEFAULT_BACK,
  DEFAULT_SUIT_GLYPHS,
  getDeck,
  isSuitGlyphSet,
  listDecks,
  SUIT_GLYPH_SETS,
  type SuitGlyphSet,
} from '../library/index.ts';

/** A table texture: plain, or a pattern over the colour (surface.ts). */
export type Texture = 'plain' | 'felt' | 'cloth' | 'wood';
/** Every table texture. */
export const TEXTURES: readonly Texture[] = ['plain', 'felt', 'cloth', 'wood'];

/** The light fall-off over the surface: 'soft', each game's own gradient and
 *  vignette (today's look), or 'flat', none: with the plain texture the
 *  table is one uniform colour. */
export type Shading = 'soft' | 'flat';
/** Every table shading. */
export const SHADINGS: readonly Shading[] = ['soft', 'flat'];

/** The table surface settings: texture, shading and colour. */
export interface SurfaceSettings {
  texture: Texture;
  shading: Shading;
  /** '#rrggbb', or null: the theme's own felt. */
  color: string | null;
}

/** The card settings: deck, suit symbols and back. */
export interface CardSettings {
  /** A registered deck id. */
  deck: string;
  suitGlyphs: SuitGlyphSet;
  /** One of the deck's own back ids, or 'default'. */
  back: string;
}

/** The table and card settings: the surface and the cards. */
export interface TableCardSettings {
  surface: SurfaceSettings;
  cards: CardSettings;
}

/** Some fields of TableCardSettings: a stored layer, a game's defaults, a pin. */
export interface TableCardPatch {
  surface?: Partial<SurfaceSettings> | undefined;
  cards?: Partial<CardSettings> | undefined;
}

/** Today's look: the theme's felt, no texture, the basic deck. */
export const DEFAULT_TABLE_CARDS: TableCardSettings = Object.freeze({
  surface: Object.freeze({texture: 'plain', shading: 'soft', color: null}),
  cards: Object.freeze({
    deck: 'simple',
    suitGlyphs: DEFAULT_SUIT_GLYPHS,
    back: DEFAULT_BACK,
  }),
});

/** Where the app-wide table and card settings are kept. */
export const APP_KEY = 'open_spiel.gui.table-cards';
/** Where a game's override lives. Present (even `{}`) means the game has one. */
export function gameKey(game: string): string {
  if (!/^[a-z0-9_]+$/.test(game)) {
    throw new Error(
      `table-cards: game id must be [a-z0-9_]+, got ${JSON.stringify(game)}`,
    );
  }
  return `${APP_KEY}.${game}`;
}

const HEX = /^#[0-9a-f]{6}$/i;

/** The table is a dark surface. The games draw their on-table text light in
 *  fixed colours (white name plates), so a
 *  colour must keep white text at this contrast or better. A light table would
 *  need every adopting game's chrome re-coloured: shoehorning, so refused. */
export const MIN_TEXT_CONTRAST = 4.5;

/** Relative luminance of '#rrggbb' (WCAG 2.x). */
export function luminance(hex: string): number {
  const m = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex);
  if (!m) {
    throw new Error(
      `luminance: expected '#rrggbb', got ${JSON.stringify(hex)}`,
    );
  }
  const [r = 0, g = 0, b = 0] = m.slice(1).map(h => {
    const c = parseInt(h, 16) / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** White text's contrast on '#rrggbb'. */
export function whiteTextContrast(hex: string): number {
  return 1.05 / (luminance(hex) + 0.05);
}

const FIELDS: {
  surface: ReadonlyArray<keyof SurfaceSettings>;
  cards: ReadonlyArray<keyof CardSettings>;
} = {
  surface: ['texture', 'shading', 'color'],
  cards: ['deck', 'suitGlyphs', 'back'],
};

/** Why `value` is not a valid `section.field`, or null when it is. The back
 *  is checked against `deck` when one is given. */
function fieldError(
  section: 'surface' | 'cards',
  field: string,
  value: unknown,
  deck?: string,
): string | null {
  const got = JSON.stringify(value);
  if (section === 'surface' && field === 'texture') {
    return (TEXTURES as readonly unknown[]).includes(value)
      ? null
      : `must be one of ${TEXTURES.join(', ')}, got ${got}`;
  }
  if (section === 'surface' && field === 'shading') {
    return (SHADINGS as readonly unknown[]).includes(value)
      ? null
      : `must be one of ${SHADINGS.join(', ')}, got ${got}`;
  }
  if (section === 'surface' && field === 'color') {
    if (value === null) return null;
    if (typeof value !== 'string' || !HEX.test(value)) {
      return `must be null or '#rrggbb', got ${got}`;
    }
    const c = whiteTextContrast(value);
    return c >= MIN_TEXT_CONTRAST
      ? null
      : `${got} is too light: white table text on it is ${c.toFixed(2)}:1, under ${String(MIN_TEXT_CONTRAST)}:1`;
  }
  if (section === 'cards' && field === 'deck') {
    const ids = listDecks().map(d => d.id);
    return typeof value === 'string' && ids.includes(value)
      ? null
      : `must be one of ${ids.join(', ')}, got ${got}`;
  }
  if (section === 'cards' && field === 'suitGlyphs') {
    return isSuitGlyphSet(value)
      ? null
      : `must be one of ${SUIT_GLYPH_SETS.join(', ')}, got ${got}`;
  }
  if (section === 'cards' && field === 'back') {
    if (typeof value !== 'string' || value === '') {
      return `must be a non-empty string, got ${got}`;
    }
    if (deck === undefined || value === DEFAULT_BACK) return null;
    const backs = getDeck(deck).backs ?? [];
    return backs.includes(value)
      ? null
      : `must be '${DEFAULT_BACK}' or one of deck '${deck}''s backs (${backs.join(', ') || 'none'}), got ${got}`;
  }
  return `is not a field (have surface.${FIELDS.surface.join(', surface.')}, cards.${FIELDS.cards.join(', cards.')})`;
}

/** Throws, naming the field, unless every field of `patch` is valid. A back
 *  is checked against the patch's own deck, or `deck` when it has none. */
export function assertPatch(patch: TableCardPatch, deck?: string): void {
  const given: unknown = patch;
  if (given === null || typeof given !== 'object') {
    throw new Error(
      `table-cards: a patch must be an object, got ${JSON.stringify(given)}`,
    );
  }
  for (const [section, fields] of Object.entries(given) as Array<
    [string, unknown]
  >) {
    if (section !== 'surface' && section !== 'cards') {
      throw new Error(
        `table-cards: no section '${section}' (have surface, cards)`,
      );
    }
    if (fields === undefined) continue;
    if (fields === null || typeof fields !== 'object') {
      throw new Error(
        `table-cards.${section} must be an object, got ${JSON.stringify(fields)}`,
      );
    }
    const d =
      section === 'cards'
        ? ((fields as Partial<CardSettings>).deck ?? deck)
        : undefined;
    for (const [field, value] of Object.entries(fields) as Array<
      [string, unknown]
    >) {
      if (value === undefined) continue;
      const err = fieldError(section, field, value, d);
      if (err) throw new Error(`table-cards.${section}.${field} ${err}`);
    }
  }
}

/** A stored layer, read leniently: a field a newer or older build wrote that
 *  this one does not know (an unknown deck, set, texture or back) is reported
 *  with console.error and dropped, so the field falls back to the layer below.
 *  A stored value never breaks a page (as readStoredPreference). */
export function parseStored(raw: string, where: string): TableCardPatch {
  let data: unknown;
  try {
    data = JSON.parse(raw);
  } catch (e) {
    console.error(
      `[table-cards] ignoring ${where}: not JSON (${String(e)}): ${raw.slice(0, 80)}`,
    );
    return {};
  }
  if (data === null || typeof data !== 'object' || Array.isArray(data)) {
    console.error(
      `[table-cards] ignoring ${where}: not an object: ${raw.slice(0, 80)}`,
    );
    return {};
  }
  const out: TableCardPatch = {};
  for (const [section, fields] of Object.entries(
    data as Record<string, unknown>,
  )) {
    if (section !== 'surface' && section !== 'cards') {
      console.error(
        `[table-cards] ignoring ${where}.${section}: not a section`,
      );
      continue;
    }
    if (fields === null || typeof fields !== 'object') {
      console.error(
        `[table-cards] ignoring ${where}.${section}: not an object`,
      );
      continue;
    }
    const f = fields as Record<string, unknown>;
    const kept: Record<string, unknown> = {};
    for (const [field, value] of Object.entries(f)) {
      // A back is not checked against a deck here: it is the back
      // preferred wherever the deck in effect has it (resolveTableCards).
      const err = fieldError(section, field, value);
      if (err) {
        console.error(
          `[table-cards] ignoring ${where}.${section}.${field}: ${err}`,
        );
      } else kept[field] = value;
    }
    if (Object.keys(kept).length) out[section] = kept;
  }
  return out;
}

/** `base` with each layer's fields on top, in order. A back the resulting
 *  deck does not have (an app-wide red back under a game's own deck)
 *  falls back to the deck's own: layers may disagree, the result may not. */
export function resolveTableCards(
  ...layers: Array<TableCardPatch | null | undefined>
): TableCardSettings {
  const surface = {...DEFAULT_TABLE_CARDS.surface};
  const cards = {...DEFAULT_TABLE_CARDS.cards};
  for (const l of layers) {
    if (!l) continue;
    Object.assign(surface, strip(l.surface));
    Object.assign(cards, strip(l.cards));
  }
  if (
    cards.back !== DEFAULT_BACK &&
    !(getDeck(cards.deck).backs ?? []).includes(cards.back)
  ) {
    cards.back = DEFAULT_BACK;
  }
  return {surface, cards};
}

function strip<T extends object>(o: T | undefined): Partial<T> {
  return Object.fromEntries(
    Object.entries(o ?? {}).filter(([, v]) => v !== undefined),
  ) as Partial<T>;
}
