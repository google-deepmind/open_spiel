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

// The chess board's view settings: the store behind the Board section of the
// View panel (chess_config_section.tsx) and the board that reads it. A
// setting persists on this browser; the board's orientation also follows
// the viewer's seat, as the board turns itself to it.

import {useCallback, useSyncExternalStore} from 'react';

type Listener = () => void;

const values = new Map<string, string>();
const listeners = new Set<Listener>();

function subscribe(listener: Listener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function storageKey(key: string): string {
  return `open_spiel.gui.${key}`;
}

function read<T extends string>(
  key: string,
  allowed: readonly T[],
  fallback: T,
): T {
  const held = values.get(key) ?? window.localStorage.getItem(storageKey(key));
  return allowed.find(value => value === held) ?? fallback;
}

/**
 * A chess view setting, one of `allowed`, as last chosen on this browser,
 * or `fallback` when none was.
 *
 * @throws Error when set to a value outside `allowed`.
 */
export function useChessSetting<T extends string>(
  key: string,
  allowed: readonly T[],
  fallback: T,
): [T, (value: T) => void] {
  const value = useSyncExternalStore(subscribe, () =>
    read(key, allowed, fallback),
  );
  const set = useCallback(
    (next: T) => {
      if (!allowed.includes(next)) {
        throw new Error(
          `Setting ${key}: '${next}' is not one of ${allowed.join(', ')}.`,
        );
      }
      if (values.get(key) === next) return;
      values.set(key, next);
      window.localStorage.setItem(storageKey(key), next);
      for (const listener of listeners) listener();
    },
    [key, allowed],
  );
  return [value, set];
}

const FLAGS = ['true', 'false'] as const;

/** A yes-or-no chess view setting; `fallback` until one is chosen. */
export function useChessFlag(
  key: string,
  fallback: boolean,
): [boolean, (value: boolean) => void] {
  const [value, set] = useChessSetting(key, FLAGS, fallback ? 'true' : 'false');
  const setFlag = useCallback(
    (next: boolean) => {
      set(next ? 'true' : 'false');
    },
    [set],
  );
  return [value === 'true', setFlag];
}
