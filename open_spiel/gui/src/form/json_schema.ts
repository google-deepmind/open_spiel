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

// Reads a controller's settings schema, a JSON Schema, as the fields of a
// flat form.

import type {Field, FieldValue, TextConstraints} from './field.ts';
import {admitsText} from './field.ts';
import {nameLabel} from './name_label.ts';

type JsonObject = Readonly<Record<string, unknown>>;

const VALUE_TYPES = {
  boolean: 'boolean',
  integer: 'number',
  number: 'number',
  string: 'string',
} as const;

const SCHEMA_KEYWORDS = new Set([
  'type',
  'properties',
  'additionalProperties',
  'required',
]);

const PROPERTY_KEYWORDS = new Set([
  'type',
  'default',
  'description',
  'minimum',
  'maximum',
  'enum',
  'title',
  'x-presets',
  'minLength',
  'maxLength',
  'pattern',
]);

/**
 * Returns the form fields of a settings schema, in the order of its
 * `properties`, or null when the schema says more than a flat form can show.
 *
 * The schema `{}` takes no settings. Otherwise it is an object schema of
 * `type` and `properties`, `additionalProperties` if it is false, and
 * `required` if it lists properties that must be sent. Every property has a
 * `type` of boolean, integer, number or string and, unless it is required, a
 * `default` of that type. It may give a `title`, which labels it, a
 * `description`, numeric `minimum` and `maximum`, and an `enum` of values of
 * its type, within those bounds, that includes the default. A text property
 * may give `minLength`, `maxLength` and a `pattern` (a regular expression
 * JavaScript reads with the `u` flag), which the default and any `enum`
 * satisfy. A number or text property may give `x-presets`, values within
 * those bounds that the form offers as one-click choices. Any other keyword
 * is a constraint the form cannot enforce, so the whole schema is refused
 * rather than half shown.
 */
export function fieldsOf(schema: JsonObject): Field[] | null {
  if (Object.keys(schema).length === 0) return [];
  const {type, properties, additionalProperties, required} = schema;
  if (
    !Object.keys(schema).every(keyword => SCHEMA_KEYWORDS.has(keyword)) ||
    type !== 'object' ||
    !isObject(properties) ||
    (additionalProperties !== undefined && additionalProperties !== false)
  ) {
    return null;
  }
  const names = requiredNames(required, properties);
  if (names === null) return null;
  const fields: Field[] = [];
  for (const [name, property] of Object.entries(properties)) {
    const field = isObject(property)
      ? fieldOf(name, property, names.has(name))
      : null;
    if (field === null) return null;
    fields.push(field);
  }
  return fields;
}

// The names `required` lists, or null when it is not a list of distinct
// property names. An absent `required` lists none.
function requiredNames(
  required: unknown,
  properties: JsonObject,
): ReadonlySet<string> | null {
  if (required === undefined) return new Set();
  if (
    !Array.isArray(required) ||
    !required.every(
      (name: unknown) =>
        typeof name === 'string' && Object.hasOwn(properties, name),
    )
  ) {
    return null;
  }
  const names = new Set<string>(required);
  return names.size === required.length ? names : null;
}

// The text constraints of a property, or null when they are malformed or
// given for a property that is not text.
function textConstraints(
  type: keyof typeof VALUE_TYPES,
  property: JsonObject,
): TextConstraints | null {
  const {minLength, maxLength, pattern} = property;
  if (
    type !== 'string' &&
    (minLength !== undefined ||
      maxLength !== undefined ||
      pattern !== undefined)
  ) {
    return null;
  }
  if (
    !isOptionalLength(minLength) ||
    !isOptionalLength(maxLength) ||
    !isOptional(pattern, 'string') ||
    (minLength !== undefined &&
      maxLength !== undefined &&
      minLength > maxLength)
  ) {
    return null;
  }
  if (pattern !== undefined) {
    try {
      new RegExp(pattern, 'u');
    } catch (e) {
      if (e instanceof SyntaxError) return null;
      throw e;
    }
  }
  return {
    minLength: minLength ?? null,
    maxLength: maxLength ?? null,
    pattern: pattern ?? null,
  };
}

// Whether a value is of `type`, within the bounds and the text constraints:
// a default or a choice outside them would be offered, then refused.
function admittedBy(
  type: keyof typeof VALUE_TYPES,
  minimum: number | undefined,
  maximum: number | undefined,
  text: TextConstraints,
): (candidate: unknown) => candidate is FieldValue {
  return (candidate: unknown): candidate is FieldValue =>
    admits(type, candidate) &&
    (typeof candidate !== 'number' ||
      ((minimum === undefined || candidate >= minimum) &&
        (maximum === undefined || candidate <= maximum))) &&
    (typeof candidate !== 'string' || admitsText(text, candidate));
}

// A property's default; null for a required property that has none, the
// only kind that may have none; undefined when the default is refused.
function defaultOf(
  value: unknown,
  required: boolean,
  admitted: (candidate: unknown) => candidate is FieldValue,
): FieldValue | null | undefined {
  if (value === undefined) return required ? null : undefined;
  return admitted(value) ? value : undefined;
}

function fieldOf(
  name: string,
  property: JsonObject,
  required: boolean,
): Field | null {
  const {
    type,
    title,
    description,
    minimum,
    maximum,
    default: value,
    enum: choices,
    'x-presets': presets,
  } = property;
  if (
    !Object.keys(property).every(keyword => PROPERTY_KEYWORDS.has(keyword)) ||
    !isValueType(type) ||
    !isOptional(title, 'string') ||
    !isOptional(description, 'string') ||
    !isOptional(minimum, 'number') ||
    !isOptional(maximum, 'number')
  ) {
    return null;
  }
  const text = textConstraints(type, property);
  if (text === null) return null;
  const admitted = admittedBy(type, minimum, maximum, text);
  const fallback = defaultOf(value, required, admitted);
  if (fallback === undefined) return null;
  const base = {
    name,
    label: title === undefined || title === '' ? nameLabel(name) : title,
    description: description ?? null,
    required,
  };
  if (choices !== undefined) {
    if (presets !== undefined) return null;
    if (!Array.isArray(choices) || !choices.every(admitted)) return null;
    const offered: FieldValue[] = choices;
    return fallback === null || offered.includes(fallback)
      ? {...base, kind: 'choice', default: fallback, choices: offered}
      : null;
  }
  if (type === 'boolean') {
    return presets === undefined
      ? {
          ...base,
          kind: 'boolean',
          default: typeof fallback === 'boolean' ? fallback : null,
        }
      : null;
  }
  if (
    presets !== undefined &&
    !(Array.isArray(presets) && presets.every(admitted))
  ) {
    return null;
  }
  const offered: FieldValue[] | undefined = presets;
  if (type === 'string') {
    const strings = offered?.filter(preset => typeof preset === 'string');
    return {
      ...base,
      kind: 'string',
      default: typeof fallback === 'string' ? fallback : null,
      ...text,
      ...(strings === undefined ? {} : {presets: strings}),
    };
  }
  const numbers = offered?.filter(preset => typeof preset === 'number');
  return {
    ...base,
    kind: type === 'integer' ? 'integer' : 'number',
    default: typeof fallback === 'number' ? fallback : null,
    minimum: minimum ?? null,
    maximum: maximum ?? null,
    ...(numbers === undefined ? {} : {presets: numbers}),
  };
}

function isObject(value: unknown): value is JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isValueType(type: unknown): type is keyof typeof VALUE_TYPES {
  return typeof type === 'string' && Object.hasOwn(VALUE_TYPES, type);
}

function admits(
  type: keyof typeof VALUE_TYPES,
  value: unknown,
): value is FieldValue {
  if (typeof value !== VALUE_TYPES[type]) return false;
  return type !== 'integer' || Number.isSafeInteger(value);
}

function isOptionalLength(value: unknown): value is number | undefined {
  return (
    value === undefined ||
    (typeof value === 'number' && Number.isSafeInteger(value) && value >= 0)
  );
}

function isOptional<Name extends 'number' | 'string'>(
  value: unknown,
  type: Name,
): value is (Name extends 'number' ? number : string) | undefined {
  return value === undefined || typeof value === type;
}
