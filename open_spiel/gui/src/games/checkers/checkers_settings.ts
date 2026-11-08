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

// The checkers board's view settings: one store the Gameplay panel's board
// and the View panel's Board section both read, so a change in one shows in
// the other at once. Settings last for the page.

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

/**
 * A view setting of the board, `defaultValue` until one is chosen. The
 * setter takes a value or a function of the current one.
 */
export function useCheckersSetting<T>(
  key: string,
  defaultValue: T,
): [T, (value: T | ((prev: T) => T)) => void] {
  const stored = useSyncExternalStore(subscribe, () => values.get(key));
  const value = stored === undefined ? defaultValue : (stored as T);
  const setValue = useCallback(
    (nextValue: T | ((prevValue: T) => T)) => {
      const held = values.get(key);
      const current = held === undefined ? defaultValue : (held as T);
      const resolved =
        typeof nextValue === 'function'
          ? (nextValue as (prevValue: T) => T)(current)
          : nextValue;
      if (Object.is(held, resolved)) return;
      values.set(key, resolved);
      for (const listener of listeners) listener();
    },
    [key, defaultValue],
  );
  return [value, setValue];
}

/** Forgets every setting; for tests. */
export function resetCheckersSettings(): void {
  values.clear();
  for (const listener of listeners) listener();
}
