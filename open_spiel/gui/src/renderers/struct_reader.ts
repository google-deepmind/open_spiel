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

// Readers that check a struct's shape as they read it. A frame's structs are
// typed `unknown` on the wire, since each game defines its own; a renderer
// reads the fields it draws through these, so a struct of another shape
// fails with the path of the field at fault rather than drawing nonsense.

/** A struct that is not shaped as its renderer expects. */
export class StructError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'StructError';
  }
}

// The error for a value at `path` that is not what was expected.
function mismatch(path: string, expected: string, value: unknown): StructError {
  let found: string;
  if (value === null) found = 'null';
  else if (Array.isArray(value)) found = 'an array';
  else if (typeof value === 'object') found = 'an object';
  else found = `the ${typeof value} ${JSON.stringify(value)}`;
  return new StructError(`${path} is ${found}, not ${expected}.`);
}

/** `value` as an object of named fields; `path` names it in errors. */
export function record(
  value: unknown,
  path: string,
): Readonly<Record<string, unknown>> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw mismatch(path, 'an object', value);
  }
  // A JSON object, checked above, is a record of its fields.
  return value as Readonly<Record<string, unknown>>;
}

/** `value` as a string. */
export function string(value: unknown, path: string): string {
  if (typeof value !== 'string') throw mismatch(path, 'a string', value);
  return value;
}

/** `value` as an integer. */
export function integer(value: unknown, path: string): number {
  if (typeof value !== 'number' || !Number.isInteger(value)) {
    throw mismatch(path, 'an integer', value);
  }
  return value;
}

/** `value` as a finite number. */
export function number(value: unknown, path: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw mismatch(path, 'a number', value);
  }
  return value;
}

/** `value` as a boolean. */
export function boolean(value: unknown, path: string): boolean {
  if (typeof value !== 'boolean') {
    throw mismatch(path, 'a boolean', value);
  }
  return value;
}

/** `value` as an array, each item read by `item`. */
export function array<T>(
  value: unknown,
  path: string,
  item: (entry: unknown, entryPath: string) => T,
): T[] {
  if (!Array.isArray(value)) throw mismatch(path, 'an array', value);
  return value.map((entry: unknown, index) =>
    item(entry, `${path}[${String(index)}]`),
  );
}

/** `value` as null, or else as `read` reads it. */
export function nullable<T>(
  value: unknown,
  path: string,
  read: (present: unknown, presentPath: string) => T,
): T | null {
  return value === null ? null : read(value, path);
}

/**
 * The field `name` of the object `value`, read by `read`. A missing field is
 * an error: the server sends every field a struct defines.
 */
export function field<T>(
  value: unknown,
  path: string,
  name: string,
  read: (entry: unknown, entryPath: string) => T,
): T {
  const fields = record(value, path);
  const fieldPath = `${path}.${name}`;
  if (!(name in fields)) throw new StructError(`${fieldPath} is missing.`);
  return read(fields[name], fieldPath);
}

/**
 * The optional field `name` of the object `value`, read by `read`, or
 * `fallback` when the field is absent.
 */
export function optionalField<T>(
  value: unknown,
  path: string,
  name: string,
  read: (entry: unknown, entryPath: string) => T,
  fallback: T,
): T {
  return name in record(value, path)
    ? field(value, path, name, read)
    : fallback;
}
