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

// A value the code knows is there (an index inside its array, a key it has
// just set), checked: a broken invariant throws naming what was missing
// rather than reading on with undefined.

/**
 * `value`, which the caller's invariant says is defined.
 *
 * @throws Error naming `what` when it is undefined.
 */
export function defined<T>(value: T | undefined, what: string): T {
  if (value === undefined) throw new Error(`card library: no ${what}`);
  return value;
}

/**
 * `fields` without the keys whose value is undefined, so that an options
 * object leaves an optional property out rather than setting it to
 * undefined (exactOptionalPropertyTypes).
 */
export function definedFields<T extends Record<string, unknown>>(
  fields: T,
): {[K in keyof T]?: Exclude<T[K], undefined>} {
  const out: {[K in keyof T]?: Exclude<T[K], undefined>} = {};
  for (const key of Object.keys(fields) as Array<keyof T>) {
    const value = fields[key];
    if (value !== undefined) out[key] = value as Exclude<T[keyof T], undefined>;
  }
  return out;
}
