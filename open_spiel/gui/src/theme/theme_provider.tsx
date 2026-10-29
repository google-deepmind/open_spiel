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

import type {JSX, ReactNode} from 'react';
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  useSyncExternalStore,
} from 'react';

import type {Theme, ThemePreference} from './theme.ts';
import {
  DARK_QUERY,
  applyTheme,
  readThemePreference,
  resolveTheme,
  writeThemePreference,
} from './theme.ts';

/** The theme as the components below a ThemeProvider see it. */
export interface ThemeState {
  /** What the user asked for; 'system' until they choose. */
  preference: ThemePreference;
  /** What the page shows. */
  theme: Theme;
  /** Records an explicit choice, and keeps it across visits. */
  setPreference: (preference: ThemePreference) => void;
}

const ThemeContext = createContext<ThemeState | null>(null);

function browserStorage(): Storage {
  return window.localStorage;
}

function systemPrefersDarkNow(): boolean {
  return window.matchMedia(DARK_QUERY).matches;
}

function subscribeToSystemScheme(onChange: () => void): () => void {
  const query = window.matchMedia(DARK_QUERY);
  query.addEventListener('change', onChange);
  return () => {
    query.removeEventListener('change', onChange);
  };
}

/**
 * Owns the theme: reads the stored preference, follows the system's colour
 * scheme while the preference is 'system', and shows the result on <html>.
 */
export function ThemeProvider({children}: {children: ReactNode}): JSX.Element {
  const [preference, setPreferenceState] = useState(() =>
    readThemePreference(browserStorage),
  );
  // Read again once subscribed, so a change before then is not lost.
  const systemPrefersDark = useSyncExternalStore(
    subscribeToSystemScheme,
    systemPrefersDarkNow,
  );

  const theme = resolveTheme(preference, systemPrefersDark);
  useEffect(() => {
    applyTheme(document.documentElement, theme);
  }, [theme]);

  // Only an explicit choice is stored, so a user who never chooses keeps
  // following the system instead of whatever it resolved to on the first visit.
  const setPreference = useCallback((next: ThemePreference) => {
    writeThemePreference(browserStorage, next);
    setPreferenceState(next);
  }, []);

  const state = useMemo(
    () => ({preference, theme, setPreference}),
    [preference, theme, setPreference],
  );
  return <ThemeContext value={state}>{children}</ThemeContext>;
}

/** The theme of the nearest ThemeProvider. */
export function useTheme(): ThemeState {
  const state = useContext(ThemeContext);
  if (state === null) {
    throw new Error('useTheme is used outside a ThemeProvider.');
  }
  return state;
}
