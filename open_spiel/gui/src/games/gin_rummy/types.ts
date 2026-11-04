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

import type {LegalAction} from './gin_session.ts';

/** A struct record, as the adapter sends it. */
export type GinStructRecord = Record<string, unknown>;

/** Each suit's glyph and colour class, by suit code. */
export const SUIT_META: Readonly<
  Record<
    string,
    {symbol: string; colorClass: 'gin-card--red' | 'gin-card--black'}
  >
> = {
  s: {symbol: '♠', colorClass: 'gin-card--black'},
  h: {symbol: '♥', colorClass: 'gin-card--red'},
  d: {symbol: '♦', colorClass: 'gin-card--red'},
  c: {symbol: '♣', colorClass: 'gin-card--black'},
};

/** The suits, in the grid's row order. */
export const SUIT_ROWS = [
  {code: 's', label: 'Spades', short: '♠'},
  {code: 'h', label: 'Hearts', short: '♥'},
  {code: 'd', label: 'Diamonds', short: '♦'},
  {code: 'c', label: 'Clubs', short: '♣'},
] as const;

/** The ranks, in the grid's column order. */
export const RANK_CODES = [
  'A',
  '2',
  '3',
  '4',
  '5',
  '6',
  '7',
  '8',
  '9',
  'T',
  'J',
  'Q',
  'K',
] as const;

/** A meld's kind: a run in one suit, or a set of one rank. */
export type MeldType = 'run' | 'set';

/** A meld the player can lay, or has laid (`locked`). */
export interface PlayerMeld {
  id: string;
  type: MeldType;
  cards: string[];
  locked: boolean;
}

/** One lay action of a lay-down plan. */
export interface LaydownStep {
  cards: string[];
  action: number;
  signature: string;
}

/** A set of lay actions the seat can lay, then pass. */
export interface LaydownPlan {
  sequence: LaydownStep[];
  signatureSet: Set<string>;
}

function isRecord(value: unknown): value is GinStructRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * `value` as a struct record, or null. The adapter sends the struct as an
 * object; a string here means the view regressed, so it is logged loudly
 * rather than parsed.
 */
export function asRecord(value: unknown): GinStructRecord | null {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.length > 0) {
    console.error(
      'asRecord: received a string where a parsed struct object was ' +
        'expected; the view contract has regressed. Rendering nothing for it.',
      value.slice(0, 120),
    );
  }
  return null;
}

const RANK_SET: ReadonlySet<string> = new Set(RANK_CODES);

/** A card code as "As", "Td"; null when `card` is not one. */
export function normaliseCardCode(card: unknown): string | null {
  if (typeof card !== 'string') return null;
  const trimmed = card.trim();
  if (trimmed.length < 2 || trimmed.length > 3) return null;
  const rank = trimmed.slice(0, -1).toUpperCase();
  const suit = trimmed.slice(-1).toLowerCase();
  if (!RANK_SET.has(rank)) return null;
  if (!['s', 'h', 'd', 'c'].includes(suit)) return null;
  return `${rank}${suit}`;
}

/** The first action whose lower-cased label `matcher` accepts, or null. */
export function findActionId(
  actions: readonly LegalAction[],
  matcher: (label: string) => boolean,
): number | null {
  for (const action of actions) {
    if (matcher(action.label.toLowerCase())) return action.action;
  }
  return null;
}

/** Every card code a label names ("Player: 0 Action: Qd" names "Qd"). */
export function extractCardCodesFromLabel(label: string): string[] {
  const matches = label.match(/\b([AKQJT2-9][shdc])\b/gi);
  if (matches === null) return [];
  return matches
    .map(m => normaliseCardCode(m))
    .filter((c): c is string => c !== null);
}
