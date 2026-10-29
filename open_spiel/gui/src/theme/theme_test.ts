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

/// <reference types="node" />

import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {runInThisContext} from 'node:vm';

import {afterEach, describe, expect, it, vi} from 'vitest';

import {fakeColorScheme} from '../testing/color_scheme.ts';
import type {ThemePreference} from './theme.ts';
import {
  THEME_PREFERENCES,
  THEME_STORAGE_KEY,
  applyTheme,
  readThemePreference,
  resolveTheme,
  writeThemePreference,
} from './theme.ts';

function storage(): Storage {
  return window.localStorage;
}
function refusingStorage(): Storage {
  throw new DOMException('Storage is disabled.', 'SecurityError');
}

afterEach(() => {
  vi.unstubAllGlobals();
  window.localStorage.clear();
  delete document.documentElement.dataset['theme'];
  document.documentElement.style.colorScheme = '';
  vi.restoreAllMocks();
});

describe('resolveTheme', () => {
  it.each([
    ['system', false, 'light'],
    ['system', true, 'dark'],
    ['light', true, 'light'],
    ['dark', false, 'dark'],
  ] as const)(
    'resolves %s with a dark system %s to %s',
    (preference, dark, theme) => {
      expect(resolveTheme(preference, dark)).toBe(theme);
    },
  );
});

describe('readThemePreference', () => {
  it('is system when nothing is stored', () => {
    expect(readThemePreference(storage)).toBe('system');
  });

  it.each(THEME_PREFERENCES)('reads a stored %s', preference => {
    window.localStorage.setItem(THEME_STORAGE_KEY, preference);
    expect(readThemePreference(storage)).toBe(preference);
  });

  it('reports and discards an unknown stored value', () => {
    const error = vi
      .spyOn(console, 'error')
      .mockImplementation(() => undefined);
    window.localStorage.setItem(THEME_STORAGE_KEY, 'sepia');
    expect(readThemePreference(storage)).toBe('system');
    expect(error).toHaveBeenCalledWith(expect.stringContaining('"sepia"'));
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBeNull();
  });

  it('reports an unknown stored value it cannot discard', () => {
    const error = vi
      .spyOn(console, 'error')
      .mockImplementation(() => undefined);
    const readOnly = {
      getItem: () => 'sepia',
      removeItem: () => {
        throw new DOMException('Storage is read-only.', 'SecurityError');
      },
    } as unknown as Storage;
    expect(readThemePreference(() => readOnly)).toBe('system');
    expect(error).toHaveBeenLastCalledWith(
      'Cannot discard the stored theme preference.',
      expect.any(DOMException),
    );
  });

  it('reports storage it cannot open, and reads it as empty', () => {
    const error = vi
      .spyOn(console, 'error')
      .mockImplementation(() => undefined);
    expect(readThemePreference(refusingStorage)).toBe('system');
    expect(error).toHaveBeenCalledOnce();
  });
});

describe('writeThemePreference', () => {
  it('stores the preference', () => {
    writeThemePreference(storage, 'dark');
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe('dark');
  });

  it('reports storage it cannot open', () => {
    const error = vi
      .spyOn(console, 'error')
      .mockImplementation(() => undefined);
    writeThemePreference(refusingStorage, 'dark');
    expect(error).toHaveBeenCalledOnce();
  });
});

describe('applyTheme', () => {
  it('sets data-theme and the color scheme native controls follow', () => {
    applyTheme(document.documentElement, 'dark');
    expect(document.documentElement.dataset['theme']).toBe('dark');
    expect(document.documentElement.style.colorScheme).toBe('dark');
  });
});

describe('the first-paint script in index.html', () => {
  const html = readFileSync(
    resolve(import.meta.dirname, '../../index.html'),
    'utf8',
  );
  const scripts = Array.from(
    html.matchAll(/<script>([\s\S]*?)<\/script>/g),
    match => match[1] ?? '',
  );

  function firstPaintTheme(): string | undefined {
    expect(scripts).toHaveLength(1);
    runInThisContext(scripts[0] ?? '');
    return document.documentElement.dataset['theme'];
  }

  const stored: Array<ThemePreference | 'sepia' | null> = [
    ...THEME_PREFERENCES,
    'sepia',
    null,
  ];
  it.each(
    stored.flatMap(value => [true, false].map(dark => [value, dark] as const)),
  )(
    'agrees with ThemeProvider for stored %s, dark system %s',
    (value, dark) => {
      fakeColorScheme(dark);
      vi.spyOn(console, 'error').mockImplementation(() => undefined);
      if (value !== null) {
        window.localStorage.setItem(THEME_STORAGE_KEY, value);
      }
      const theme = firstPaintTheme();
      expect(theme).toBe(resolveTheme(readThemePreference(storage), dark));
      expect(document.documentElement.style.colorScheme).toBe(theme);
    },
  );

  it('follows the system when storage cannot be opened', () => {
    fakeColorScheme(true);
    vi.stubGlobal('localStorage', {getItem: refusingStorage});
    expect(firstPaintTheme()).toBe('dark');
  });
});
