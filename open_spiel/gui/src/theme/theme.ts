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

// The colour theme. The user picks a preference; the page shows the theme it
// resolves to, written to <html data-theme> where src/styles/tokens.css reads
// it. The inline script in index.html applies the same resolution before the
// first paint; keep the two in step (theme_test.ts checks that they agree).

/** What the user asks for: a fixed theme, or whatever the system prefers. */
export type ThemePreference = 'system' | 'light' | 'dark';

/** What the page shows. */
export type Theme = 'light' | 'dark';

/** Every preference, in the order a choice of them is offered. */
export const THEME_PREFERENCES: readonly ThemePreference[] = [
  'system',
  'light',
  'dark',
];

/** The localStorage key holding an explicit preference. */
export const THEME_STORAGE_KEY = 'open_spiel.theme';

/** The media query that reports a system preference for dark colours. */
export const DARK_QUERY = '(prefers-color-scheme: dark)';

/** Whether `value` is a preference. */
export function isThemePreference(value: unknown): value is ThemePreference {
  return THEME_PREFERENCES.some(preference => preference === value);
}

/** The theme `preference` shows while the system does or does not prefer dark. */
export function resolveTheme(
  preference: ThemePreference,
  systemPrefersDark: boolean,
): Theme {
  if (preference === 'system') {
    return systemPrefersDark ? 'dark' : 'light';
  }
  return preference;
}

/**
 * The preference stored in `storage`, or 'system' when there is none.
 *
 * An unknown stored value is reported and removed rather than kept, and
 * storage the browser refuses to open is reported and read as empty: the
 * theme is a convenience, never a reason for the page to fail.
 */
export function readThemePreference(storage: () => Storage): ThemePreference {
  let stored: string | null;
  try {
    stored = storage().getItem(THEME_STORAGE_KEY);
  } catch (e) {
    console.error('Cannot read the stored theme preference.', e);
    return 'system';
  }
  if (stored === null || isThemePreference(stored)) {
    return stored ?? 'system';
  }
  console.error(
    `Discarding the unknown stored theme preference ${JSON.stringify(stored)};` +
      ` expected one of ${THEME_PREFERENCES.join(', ')}.`,
  );
  try {
    storage().removeItem(THEME_STORAGE_KEY);
  } catch (e) {
    console.error('Cannot discard the stored theme preference.', e);
  }
  return 'system';
}

/**
 * Stores `preference` in `storage`. Storage the browser refuses is reported,
 * and the preference then lasts until the page is reloaded.
 */
export function writeThemePreference(
  storage: () => Storage,
  preference: ThemePreference,
): void {
  try {
    storage().setItem(THEME_STORAGE_KEY, preference);
  } catch (e) {
    console.error('Cannot store the theme preference.', e);
  }
}

/** Shows `theme` on the page rooted at `root`, native controls included. */
export function applyTheme(root: HTMLElement, theme: Theme): void {
  root.dataset['theme'] = theme;
  root.style.colorScheme = theme;
}
