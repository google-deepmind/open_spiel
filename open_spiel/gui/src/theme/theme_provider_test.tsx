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

import type {JSX} from 'react';
import {act, useEffect} from 'react';
import {afterEach, describe, expect, it, vi} from 'vitest';

import {fakeColorScheme} from '../testing/color_scheme.ts';
import type {Rendered} from '../testing/render.tsx';
import {render} from '../testing/render.tsx';
import type {ThemeState} from './theme_provider.tsx';
import {ThemeProvider, useTheme} from './theme_provider.tsx';
import {THEME_STORAGE_KEY} from './theme.ts';

let rendered: Rendered | undefined;
let state: ThemeState | undefined;

/** Reports the theme it sees after every render. */
function Probe({report}: {report: (seen: ThemeState) => void}): JSX.Element {
  const seen = useTheme();
  useEffect(() => {
    report(seen);
  });
  return <></>;
}

function remember(seen: ThemeState): void {
  state = seen;
}

function mount(): ThemeState {
  rendered = render(
    <ThemeProvider>
      <Probe report={remember} />
    </ThemeProvider>,
  );
  if (state === undefined) {
    throw new Error('The probe did not render.');
  }
  return state;
}

function shownTheme(): string | undefined {
  return document.documentElement.dataset['theme'];
}

afterEach(() => {
  rendered?.unmount();
  rendered = undefined;
  state = undefined;
  window.localStorage.clear();
  delete document.documentElement.dataset['theme'];
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('ThemeProvider', () => {
  it('follows the system scheme live while the preference is system', () => {
    const scheme = fakeColorScheme(false);
    expect(mount()).toMatchObject({preference: 'system', theme: 'light'});
    expect(shownTheme()).toBe('light');
    act(() => {
      scheme.setDark(true);
    });
    expect(state?.theme).toBe('dark');
    expect(shownTheme()).toBe('dark');
    expect(document.documentElement.style.colorScheme).toBe('dark');
  });

  it('keeps an explicit choice whatever the system does, and stores it', () => {
    const scheme = fakeColorScheme(false);
    const {setPreference} = mount();
    act(() => {
      setPreference('dark');
    });
    act(() => {
      scheme.setDark(false);
    });
    expect(state).toMatchObject({preference: 'dark', theme: 'dark'});
    expect(shownTheme()).toBe('dark');
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe('dark');
  });

  it('starts from the stored preference', () => {
    fakeColorScheme(true);
    window.localStorage.setItem(THEME_STORAGE_KEY, 'light');
    expect(mount()).toMatchObject({preference: 'light', theme: 'light'});
    expect(shownTheme()).toBe('light');
  });

  it('stores nothing until the user chooses', () => {
    fakeColorScheme(true);
    mount();
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBeNull();
  });

  it('stops listening to the system once unmounted', () => {
    const scheme = fakeColorScheme(false);
    mount();
    expect(scheme.listeners()).toBe(1);
    rendered?.unmount();
    rendered = undefined;
    expect(scheme.listeners()).toBe(0);
  });
});

describe('useTheme', () => {
  it('refuses to run outside a ThemeProvider', () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined);
    expect(() => render(<Probe report={remember} />)).toThrow(
      'outside a ThemeProvider',
    );
  });
});
