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

// Reading the gin view's loose values: card codes in any spelling the struct
// uses, numbers that may arrive as strings, player names, and the words the
// board shows for the phase and the player to move.

import {GIN_STATUS} from './gin_turn_line.ts';

/** `value` if it is an object (not an array), else null. */
export function asObject(value: unknown): Record<string, unknown> | null {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return null;
  }
  return Object.fromEntries(Object.entries(value));
}

/** The strings of `value`, when it is an array; [] otherwise. */
export function asStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === 'string');
}

/** The finite numbers of `value`, numeric strings read as numbers. */
export function asNumberArray(value: unknown): number[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((item: unknown) => {
      if (typeof item === 'number') return item;
      if (typeof item === 'string') {
        const parsed = Number(item);
        return Number.isFinite(parsed) ? parsed : null;
      }
      return null;
    })
    .filter((item): item is number => item !== null);
}

/** The seat a struct player name names ("Player_1"), -1 for chance. */
export function normalizePlayerId(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value !== 'string') return null;
  if (value.startsWith('Player_')) {
    const parsed = Number(value.split('_')[1]);
    return Number.isFinite(parsed) ? parsed : null;
  }
  if (value.toLowerCase() === 'chance') return -1;
  return null;
}

/**
 * A card code as "Ts" for any of "10s", "Ts", "ts"; any other non-empty
 * string is returned trimmed (a hidden "XX" stays "XX"); null for nothing.
 */
export function normaliseCardCode(card: unknown): string | null {
  if (typeof card !== 'string') return null;
  const trimmed = card.trim();
  if (trimmed.length === 0) return null;
  const match = /^(10|[0-9TJQKA])([shdc])$/i.exec(trimmed);
  const rankToken = match?.[1];
  const suitToken = match?.[2];
  if (rankToken === undefined || suitToken === undefined) return trimmed;
  const rank = rankToken.toUpperCase();
  return `${rank === '10' ? 'T' : rank}${suitToken.toLowerCase()}`;
}

/** The codes of `cards` that are card codes, normalised. */
export function normalisedCodes(cards: readonly unknown[]): string[] {
  return cards
    .map(card => normaliseCardCode(card))
    .filter((card): card is string => card !== null);
}

/** The phase as the board names it: "FirstUpcard" is "First upcard". */
export function formatPhase(value: unknown): string | null {
  if (typeof value !== 'string' || value.length === 0) return null;
  const spaced = value.replace(/_/g, ' ').toLowerCase();
  if (spaced === 'firstupcard') return 'First upcard';
  return spaced.replace(/\b\w/g, char => char.toUpperCase());
}

/**
 * The phase as the turn line says it: what the player can do, read from the
 * legal actions, where the engine's phase name says something else. After
 * both players pass the first upcard, OpenSpiel stays in FirstUpcard for the
 * forced stock draw; with Draw stock the only legal move the line says to
 * draw from the stock, as on every later draw, not "First upcard".
 */
export function turnPhaseLabel(
  phase: unknown,
  legalActions: ReadonlyArray<{readonly label: string}>,
): string | null {
  const label = formatPhase(phase);
  const only = legalActions.length === 1 ? legalActions[0] : undefined;
  if (only !== undefined && isDrawStockLabel(only.label)) {
    return GIN_STATUS.drawStock;
  }
  return label;
}

/** Whether an action label is the stock draw ("Draw stock"). */
export function isDrawStockLabel(label: string): boolean {
  const lower = label.toLowerCase();
  return lower.includes('draw') && lower.includes('stock');
}

/** Whose turn it is, in the board's words; `fallbackLabel` otherwise. */
export function formatCurrentPlayer(
  structValue: unknown,
  fallbackLabel: string | null,
): string | null {
  if (typeof structValue === 'string') {
    if (structValue.startsWith('Player_')) {
      const index = Number(structValue.split('_')[1]);
      if (index === 0) return 'Your turn';
      if (index === 1) return 'Opponent turn';
    }
    if (structValue === 'Chance') return 'Chance dealing';
    return structValue.replace(/_/g, ' ');
  }
  return fallbackLabel === null || fallbackLabel === '' ? null : fallbackLabel;
}

/** A figure for the stats: the number, or a dash when there is none. */
export function formatNumber(value: unknown): string {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value.toString();
  }
  if (typeof value === 'string' && value.trim().length > 0) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed.toString();
  }
  return '—';
}

/** A meld's signature: its codes, lower-cased, sorted, joined by commas. */
export function serializeMeldSignature(cards: readonly string[]): string {
  return cards
    .map(card => card.toLowerCase())
    .sort()
    .join(',');
}

/** The seat key's number, or null when it is not a seat. */
export function seatOf(key: string): number | null {
  const seat = Number(key);
  return Number.isFinite(seat) && seat >= 0 ? seat : null;
}

/** Each meld of `melds` as normalised codes, empty melds dropped. */
export function normalisedMelds(melds: unknown): string[][] {
  if (!Array.isArray(melds)) return [];
  return melds
    .filter((meld): meld is unknown[] => Array.isArray(meld))
    .map(meld => normalisedCodes(meld))
    .filter(meld => meld.length > 0);
}
