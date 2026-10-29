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

// The stored layers (settings.ts) on localStorage, with subscribe for
// useSyncExternalStore. A write from another tab arrives as a 'storage' event.
import {DEFAULT_BACK, getDeck} from '../library/index.ts';
import {
  APP_KEY,
  assertPatch,
  gameKey,
  parseStored,
  type CardSettings,
  type TableCardPatch,
} from './settings.ts';

const EMPTY: TableCardPatch = Object.freeze({});
// key -> [the raw string last read, what it parsed to]: one parse (and at most
// one console.error) per distinct stored value, and a stable object for React.
const cache = new Map<string, [string | null, TableCardPatch | null]>();
const listeners = new Set<() => void>();

// The browser's storage; null where there is none (a server render).
function storage(): Storage | null {
  return typeof window === 'undefined' ? null : window.localStorage;
}

/** The layer stored under `key`, or null when none is. */
function readKey(key: string): TableCardPatch | null {
  const raw = storage()?.getItem(key) ?? null;
  const hit = cache.get(key);
  if (hit?.[0] === raw) return hit[1];
  const parsed = raw === null ? null : parseStored(raw, key);
  cache.set(key, [raw, parsed]);
  return parsed;
}

/** The app-wide layer ({} when nothing is stored). */
export function readApp(): TableCardPatch {
  return readKey(APP_KEY) ?? EMPTY;
}
/** A game's override, or null when it has none (it inherits the app-wide value). */
export function readOverride(game: string): TableCardPatch | null {
  return readKey(gameKey(game));
}

/** The games whose own override sets the table colour, sorted, joined by
 *  ',' ('' when none): a string, so React can compare snapshots. The
 *  app-wide colour does not reach these games' tables. */
export function readColourOverrides(): string {
  const s = storage();
  if (!s) return '';
  const prefix = `${APP_KEY}.`;
  const games: string[] = [];
  for (let i = 0; i < s.length; i++) {
    const key = s.key(i);
    if (!key?.startsWith(prefix)) continue;
    const game = key.slice(prefix.length);
    if (!/^[a-z0-9_]+$/.test(game)) continue;
    if (readKey(key)?.surface?.color !== undefined) games.push(game);
  }
  return games.sort().join(',');
}

function writeKey(key: string, value: TableCardPatch | null): void {
  const s = storage();
  if (!s) throw new Error('table-cards: no localStorage to write to');
  if (value === null) s.removeItem(key);
  else s.setItem(key, JSON.stringify(value));
  for (const l of [...listeners]) l();
}

function merge(a: TableCardPatch, b: TableCardPatch): TableCardPatch {
  const out: TableCardPatch = {};
  for (const k of ['surface', 'cards'] as const) {
    const m = {...(a[k] ?? {}), ...(b[k] ?? {})};
    if (Object.keys(m).length) (out as Record<string, unknown>)[k] = m;
  }
  return out;
}

/**
 * Sets fields of the app-wide layer (`game` undefined) or of `game`'s
 * override (created if absent). Throws on an invalid value; a back is checked
 * against the deck the user sees (`shown`, every layer and pin applied). A
 * new deck that lacks the back shown resets the back to 'default'.
 */
export function writeTableCards(
  game: string | undefined,
  patch: TableCardPatch,
  shown: CardSettings,
): void {
  const key = game === undefined ? APP_KEY : gameKey(game);
  let p = patch;
  const deck = p.cards?.deck;
  if (
    deck !== undefined &&
    p.cards?.back === undefined &&
    shown.back !== DEFAULT_BACK &&
    !(getDeck(deck).backs ?? []).includes(shown.back)
  ) {
    p = merge(p, {cards: {back: DEFAULT_BACK}});
  }
  assertPatch(p, deck ?? shown.deck);
  writeKey(key, merge(readKey(key) ?? EMPTY, p));
}

/** Replaces `game`'s override wholesale with `patch` (validated), or
 *  deletes it (null): putting back an override saved earlier, exactly as it
 *  was. A back is checked only against a deck the patch itself names. */
export function replaceOverride(
  game: string,
  patch: TableCardPatch | null,
): void {
  if (patch !== null) assertPatch(patch);
  writeKey(gameKey(game), patch);
}

/** Turns `game`'s override on (an empty one: it still inherits every field
 *  until one is set) or off (deleted). */
export function setOverrideEnabled(game: string, on: boolean): void {
  const has = readOverride(game) !== null;
  if (on && !has) writeKey(gameKey(game), {});
  if (!on && has) writeKey(gameKey(game), null);
}

/** Calls `l` whenever a stored setting changes; returns the unsubscribe. */
export function subscribe(l: () => void): () => void {
  listeners.add(l);
  const onStorage = (e: StorageEvent) => {
    if (e.key === null || e.key.startsWith(APP_KEY)) l();
  };
  if (typeof window !== 'undefined') {
    window.addEventListener('storage', onStorage);
  }
  return () => {
    listeners.delete(l);
    if (typeof window !== 'undefined') {
      window.removeEventListener('storage', onStorage);
    }
  };
}
