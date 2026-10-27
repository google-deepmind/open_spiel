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

// Reading an adapter's action hints. Each hint is the adapter's own JSON; a
// renderer reads the hints it understands with `readHints` and leaves the
// actions it cannot place to the Actions panel's legal list.

import type {Presentation} from '../contract/contract.ts';

/** An action id, and its hint as a renderer reads it. */
export interface Hinted<T> {
  readonly action: number;
  readonly hint: T;
}

/**
 * The hints of `presentation` that `read` accepts, in the server's order.
 * `read` returns null for a hint it does not place; a hint it cannot parse
 * should throw (the struct readers do), since the adapter sent it.
 */
export function readHints<T>(
  presentation: Presentation,
  read: (hint: unknown, path: string) => T | null,
): Array<Hinted<T>> {
  const hinted: Array<Hinted<T>> = [];
  for (const [index, {id, hint}] of (presentation.actions ?? []).entries()) {
    const parsed = read(hint, `actions[${String(index)}].hint`);
    if (parsed !== null) hinted.push({action: id, hint: parsed});
  }
  return hinted;
}

/** The legal-action labels of a frame, by action id. */
export function actionLabels(
  actions: ReadonlyArray<{id: number; label: string}> | null,
): ReadonlyMap<number, string> {
  return new Map((actions ?? []).map(({id, label}) => [id, label]));
}
