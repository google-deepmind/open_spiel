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

import {vi} from 'vitest';

import {DARK_QUERY} from '../theme/theme.ts';

/** The system colour scheme, as a test controls it. */
export interface ColorScheme {
  /** Switches the system scheme and notifies every listener. */
  setDark: (dark: boolean) => void;
  /** How many listeners are attached. */
  listeners: () => number;
}

/**
 * Stands in for window.matchMedia, which jsdom lacks, answering the dark
 * colour scheme query only. Undo it with vi.unstubAllGlobals().
 */
export function fakeColorScheme(dark: boolean): ColorScheme {
  let matches = dark;
  const listeners = new Set<(event: MediaQueryListEvent) => void>();
  vi.stubGlobal('matchMedia', (query: string) => {
    if (query !== DARK_QUERY) {
      throw new Error(`Unexpected media query ${query}.`);
    }
    return {
      media: query,
      get matches() {
        return matches;
      },
      addEventListener: (
        _type: 'change',
        listener: (event: MediaQueryListEvent) => void,
      ) => listeners.add(listener),
      removeEventListener: (
        _type: 'change',
        listener: (event: MediaQueryListEvent) => void,
      ) => listeners.delete(listener),
    };
  });
  return {
    setDark: next => {
      matches = next;
      // jsdom has no MediaQueryListEvent; the theme reads only `matches`.
      const partial: Pick<MediaQueryListEvent, 'matches'> = {matches: next};
      const event = partial as MediaQueryListEvent;
      for (const listener of listeners) {
        listener(event);
      }
    },
    listeners: () => listeners.size,
  };
}
