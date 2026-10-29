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

// Checked access for the card library's geometry: an index the maths
// guarantees is in range, read so that a broken guarantee throws rather than
// computing with undefined.

/** `items[i]`, which must be there. */
export function itemAt<T>(items: readonly T[], i: number): T {
  const item = items[i];
  if (item === undefined) {
    throw new RangeError(
      `index ${String(i)} is outside a list of ${String(items.length)}`,
    );
  }
  return item;
}

/** `value`, which must not be null or undefined; `what` names it if it is. */
export function present<T>(value: T | null | undefined, what: string): T {
  if (value === null || value === undefined) {
    throw new Error(`${what} is missing`);
  }
  return value;
}
