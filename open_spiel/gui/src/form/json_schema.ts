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

import type {Field, FieldValue} from './field.ts';
import {nameLabel} from './name_label.ts';

type JsonObject = Readonly<Record<string, unknown>>;

const VALUE_TYPES = {
  boolean: 'boolean',
  integer: 'number',
  number: 'number',
  string: 'string',
} as const;

const SCHEMA_KEYWORDS = new Set(['type', 'properties', 'additionalProperties']);

const PROPERTY_KEYWORDS = new Set([
  'type',
  'default',
  'description',
  'minimum',
  'maximum',
  'enum',
  'title',
  'x-presets',
]);

/**
 * Returns the form fields of a settings schema, in the order of its
 * `properties`, or null when the schema says more than a flat form can show.
 *
 * The schema `{}` takes no settings. Otherwise it is an object schema of
 * `type` and `properties`, and `additionalProperties` if it is false. Every
 * property has a `type` of boolean, integer, number or string and a
 * `default` of that type, and may give a `title`, which labels it, a
 * `description`, numeric `minimum` and `maximum` that admit the default, and
 * an `enum` of values of its type, within those bounds, that includes the
 * default. A number or text property may give `x-presets`, values within
 * those bounds that the form offers as one-click choices. Any other keyword, such as `required` or a property's `pattern`,
 * is a constraint the form cannot enforce, so the whole schema is refused
 * rather than half shown.
 */
export function fieldsOf(schema: JsonObject): Field[] | null {
  if (Object.keys(schema).length === 0) return [];
  const {type, properties, additionalProperties} = schema;
  if (
    !Object.keys(schema).every(keyword => SCHEMA_KEYWORDS.has(keyword)) ||
    type !== 'object' ||
    !isObject(properties) ||
    (additionalProperties !== undefined && additionalProperties !== false)
  ) {
    return null;
  }
  const fields: Field[] = [];
  for (const [name, property] of Object.entries(properties)) {
    const field = isObject(property) ? fieldOf(name, property) : null;
    if (field === null) return null;
    fields.push(field);
  }
  return fields;
}

function fieldOf(name: string, property: JsonObject): Field | null {
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
  // A default or a choice outside the bounds would be offered, then refused.
  const admitted = (candidate: unknown): candidate is FieldValue =>
    admits(type, candidate) &&
    (typeof candidate !== 'number' ||
      ((minimum === undefined || candidate >= minimum) &&
        (maximum === undefined || candidate <= maximum)));
  if (!admitted(value)) return null;
  const base = {
    name,
    label: title === undefined || title === '' ? nameLabel(name) : title,
    description: description ?? null,
  };
  if (choices !== undefined) {
    if (presets !== undefined) return null;
    if (!Array.isArray(choices) || !choices.every(admitted)) return null;
    const offered: FieldValue[] = choices;
    return offered.includes(value)
      ? {...base, kind: 'choice', default: value, choices: offered}
      : null;
  }
  if (typeof value === 'boolean') {
    return presets === undefined
      ? {...base, kind: 'boolean', default: value}
      : null;
  }
  if (
    presets !== undefined &&
    !(Array.isArray(presets) && presets.every(admitted))
  ) {
    return null;
  }
  const offered: FieldValue[] | undefined = presets;
  if (typeof value === 'string') {
    const strings = offered?.filter(preset => typeof preset === 'string');
    return {
      ...base,
      kind: 'string',
      default: value,
      ...(strings === undefined ? {} : {presets: strings}),
    };
  }
  const numbers = offered?.filter(preset => typeof preset === 'number');
  return {
    ...base,
    kind: type === 'integer' ? 'integer' : 'number',
    default: value,
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

function isOptional<Name extends 'number' | 'string'>(
  value: unknown,
  type: Name,
): value is (Name extends 'number' ? number : string) | undefined {
  return value === undefined || typeof value === type;
}
