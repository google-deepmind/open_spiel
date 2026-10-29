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

// Gin's view settings as data: the hand view and grid style, their defaults,
// and where a choice persists on this browser. The hooks that read and set
// them are in gin_rummy_gameplay_config_extras.tsx (useHandView,
// useGridStyle).

import {storedSettingKey} from '../shared/view_settings.ts';

export {readStoredSetting, storedSettingKey} from '../shared/view_settings.ts';

/** How the player's hand is drawn: the suit x rank grid, a fan, an arc. */
export type HandView = 'grid' | 'flat' | 'arc';

/**
 * 'normal': a panel behind each grid and outlines of its empty cells;
 * 'slots': the outlines, no panel; 'no-bg': neither; 'compact': overlapped.
 */
export type GridStyle = 'normal' | 'slots' | 'no-bg' | 'compact';

/**
 * The board a new player sees: the suit x rank grid, spread (each card in
 * its own fixed cell). Compact is an option, never the default.
 */
export const DEFAULT_HAND_VIEW: HandView = 'grid';
/** The grid style a new player sees. */
export const DEFAULT_GRID_STYLE: GridStyle = 'normal';
/** Every hand view, in the menu's order. */
export const HAND_VIEWS: readonly HandView[] = ['grid', 'flat', 'arc'];
/** Every grid style, in the menu's order. */
export const GRID_STYLES: readonly GridStyle[] = [
  'normal',
  'slots',
  'no-bg',
  'compact',
];
/**
 * The Grid style menu's words for each style. 'slots' names what it
 * removes: the panel behind the hand (the "hand area"), cells kept.
 */
export const GRID_STYLE_LABELS: Readonly<Record<GridStyle, string>> = {
  normal: 'Normal',
  slots: 'Slots only (no hand area)',
  'no-bg': 'No background',
  compact: 'Compact',
};

/**
 * Card size, percent of the size the board fits the panel at. One scale for
 * every card on the board: hand, upcard, stock, melds and the opponent's
 * hand. 100 is the fitted size; past it the board scrolls.
 */
export const CARD_SCALE_MIN = 70;
/** The largest card size, percent. */
export const CARD_SCALE_MAX = 150;
/** The card size slider's step, percent. */
export const CARD_SCALE_STEP = 5;
/** The card size a new player sees, percent. */
export const DEFAULT_CARD_SCALE = 100;

/** Whether `v` is a card size the slider can set. */
export function isCardScale(v: number): boolean {
  return (
    Number.isInteger(v) &&
    v >= CARD_SCALE_MIN &&
    v <= CARD_SCALE_MAX &&
    (v - CARD_SCALE_MIN) % CARD_SCALE_STEP === 0
  );
}

/**
 * The card size last chosen on this browser, or the default when none was
 * (or what is stored is not a size the slider can set).
 */
export function readStoredCardScale(key: string): number {
  const raw = window.localStorage.getItem(storedSettingKey(key));
  const v = raw === null || raw.trim() === '' ? NaN : Number(raw);
  return isCardScale(v) ? v : DEFAULT_CARD_SCALE;
}
