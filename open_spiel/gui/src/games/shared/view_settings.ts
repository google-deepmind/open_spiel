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

// A board's view settings: one store the Gameplay panel's board and the View
// panel's sections both read, so a change in one shows in the other at once.
// A persisted setting is also kept in localStorage and read back on the next
// load. The one store of the card boards (gin rummy, hearts, euchre)
// and of the trick hold every trick game shares.

import {useCallback, useSyncExternalStore} from 'react';

type Listener = () => void;

const values = new Map<string, unknown>();
const listeners = new Set<Listener>();

function subscribe(listener: Listener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function setValue(key: string, value: unknown): void {
  if (Object.is(values.get(key), value)) return;
  values.set(key, value);
  for (const listener of listeners) listener();
}

/** Where a persisted view setting lives in localStorage. */
export function storedSettingKey(key: string): string {
  return `open_spiel.gui.${key}`;
}

/**
 * A view setting as last chosen on this browser, or `fallback` when none
 * was, or when what is stored is not one of `allowed`.
 */
export function readStoredSetting<T extends string>(
  key: string,
  allowed: readonly T[],
  fallback: T,
): T {
  const stored = window.localStorage.getItem(storedSettingKey(key));
  const match = allowed.find(value => value === stored);
  return match ?? fallback;
}

/**
 * A view setting of the board, `fallback` until one is chosen. The setter
 * takes a value or a function of the current one.
 */
export function useViewSetting<T>(
  key: string,
  fallback: T,
): [T, (next: T | ((current: T) => T)) => void] {
  const stored = useSyncExternalStore(subscribe, () => values.get(key));
  const value = stored === undefined ? fallback : (stored as T);
  const set = useCallback(
    (next: T | ((current: T) => T)) => {
      const held = values.get(key);
      const current = held === undefined ? fallback : (held as T);
      const resolved =
        typeof next === 'function'
          ? (next as (current: T) => T)(current)
          : next;
      setValue(key, resolved);
    },
    [key, fallback],
  );
  return [value, set];
}

/**
 * A view setting, one of `allowed`, that persists across reloads on this
 * browser.
 *
 * @throws Error when set to a value outside `allowed`.
 */
export function usePersistedSetting<T extends string>(
  key: string,
  allowed: readonly T[],
  fallback: T,
): [T, (value: T) => void] {
  const [value, set] = useViewSetting<T>(
    key,
    readStoredSetting(key, allowed, fallback),
  );
  const setPersisted = useCallback(
    (next: T) => {
      if (!allowed.includes(next)) {
        throw new Error(
          `Setting ${key}: '${next}' is not one of ${allowed.join(', ')}.`,
        );
      }
      window.localStorage.setItem(storedSettingKey(key), next);
      set(next);
    },
    [key, allowed, set],
  );
  return [value, setPersisted];
}
