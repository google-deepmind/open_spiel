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

// Generates src/contract/contract.ts, the TypeScript types of the wire format,
// from the JSON Schema the Python server publishes. Run it with
// `npm run generate:contract` whenever the schema changes.
//
// The translation covers exactly the schema vocabulary the server emits and
// refuses anything else, so a schema feature it cannot express exactly is a
// loud failure here rather than a silently loosened type.

/// <reference types="node" />

import {readFile, writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';

import * as prettier from 'prettier';

import {isEntryPoint} from './entry_point.ts';

/** The JSON Schema of the wire format, as the Python server publishes it. */
export const SCHEMA_PATH = resolve(
  import.meta.dirname,
  '../../python/gui/contract.schema.json',
);
/** The generated TypeScript types of the wire format, src/contract/contract.ts. */
export const CONTRACT_PATH = resolve(
  import.meta.dirname,
  '../src/contract/contract.ts',
);

/** The name of the type the schema's root, every response body, becomes. */
export const ROOT_NAME = 'ResponseBody';
const ROOT_DESCRIPTION = 'Every response body the server sends.';

// The generated file opens with this file's licence header, then this notice.
const NOTICE = `// Generated from open_spiel/python/gui/contract.schema.json by
// scripts/generate_contract.ts. Do not edit; run npm run generate:contract.`;

/** A schema whose keywords have been checked to have these shapes. */
interface Schema {
  $schema?: string;
  $defs?: Record<string, unknown>;
  title?: string;
  description?: string;
  type?: string | string[];
  enum?: unknown[];
  const?: unknown;
  $ref?: string;
  anyOf?: unknown[];
  oneOf?: unknown[];
  items?: Record<string, unknown>;
  properties?: Record<string, unknown>;
  required?: string[];
  additionalProperties?: boolean | Record<string, unknown>;
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isString(value: unknown): value is string {
  return typeof value === 'string';
}

function isStrings(value: unknown): value is string[] {
  return Array.isArray(value) && value.every(isString);
}

// Every keyword this generator reads, with the shape its value must have.
// Validation keywords such as `minimum` or `pattern` are refused, not
// ignored: the generated types are the client's whole view of the wire
// format, so a constraint they cannot carry needs a decision, not silence.
const SHAPES = new Map<string, [string, (value: unknown) => boolean]>([
  ['$schema', ['a string', isString]],
  ['$defs', ['an object', isObject]],
  ['title', ['a string', isString]],
  ['description', ['a string', isString]],
  [
    'type',
    [
      'a string or a non-empty array of strings',
      value => isString(value) || (isStrings(value) && value.length > 0),
    ],
  ],
  ['enum', ['an array', Array.isArray]],
  ['const', ['any value', () => true]],
  ['$ref', ['a string', isString]],
  ['anyOf', ['an array', Array.isArray]],
  ['oneOf', ['an array', Array.isArray]],
  ['items', ['a schema', isObject]],
  ['properties', ['an object', isObject]],
  ['required', ['an array of strings', isStrings]],
  [
    'additionalProperties',
    [
      'a boolean or a schema',
      value => typeof value === 'boolean' || isObject(value),
    ],
  ],
]);
const ROOT_KEYWORDS = new Set([
  '$schema',
  '$defs',
  'title',
  'description',
  'oneOf',
]);
const KEYWORDS = new Set(
  [...SHAPES.keys()].filter(key => key !== '$schema' && key !== '$defs'),
);
// The keywords that apply to one type only, and that type.
const TYPE_KEYWORDS = new Map([
  ['items', 'array'],
  ['properties', 'object'],
  ['required', 'object'],
  ['additionalProperties', 'object'],
]);
const FORMS = ['$ref', 'const', 'enum', 'anyOf', 'oneOf'] as const;
const PRIMITIVES = new Map([
  ['string', 'string'],
  ['integer', 'number'],
  ['number', 'number'],
  ['boolean', 'boolean'],
  ['null', 'null'],
]);
const REF_PREFIX = '#/$defs/';
const IDENTIFIER = /^[A-Za-z_$][\w$]*$/;

/** `value` as a JSON object; `at` is where it is in the schema. */
function asObject(value: unknown, at: string): Record<string, unknown> {
  if (!isObject(value)) throw new Error(`${at}: expected an object`);
  return value;
}

/**
 * `value` as a schema that uses only `keywords`, each with the shape this
 * generator expects.
 */
function asSchema(value: unknown, at: string, keywords = KEYWORDS): Schema {
  const schema = asObject(value, at);
  const unknown = Object.keys(schema).filter(key => !keywords.has(key));
  if (unknown.length > 0) {
    throw new Error(`${at}: unsupported keyword ${unknown.join(', ')}`);
  }
  for (const [keyword, keywordValue] of Object.entries(schema)) {
    const [shape, test] = SHAPES.get(keyword) ?? ['', () => false];
    if (!test(keywordValue)) {
      throw new Error(`${at}: ${keyword} must be ${shape}`);
    }
  }
  return schema;
}

/** The JSON Schema types of which `value` is an instance. */
function jsonTypes(value: unknown): string[] {
  if (value === null) return ['null'];
  if (Array.isArray(value)) return ['array'];
  if (typeof value === 'number') {
    return Number.isInteger(value) ? ['integer', 'number'] : ['number'];
  }
  return [typeof value];
}

/** The union of `members`, each once. */
function union(members: string[]): string {
  return [...new Set(members)].join(' | ');
}

/**
 * The TypeScript type of `schema`. Exactly one form gives it: `$ref`,
 * `const`, `enum`, `anyOf`, `oneOf` or `type`; `type` may also accompany
 * `const` or `enum`, whose values must then be of that type. A schema with no
 * form admits any value.
 */
function typeOf(schema: Schema, at: string): string {
  const types = schema.type === undefined ? [] : [schema.type].flat();
  for (const [keyword, type] of TYPE_KEYWORDS) {
    if (keyword in schema && !types.includes(type)) {
      throw new Error(`${at}: ${keyword} without type ${type}`);
    }
  }
  const forms: string[] = FORMS.filter(form => form in schema);
  if (
    types.length > 0 &&
    !forms.every(form => form === 'const' || form === 'enum')
  ) {
    forms.push('type');
  }
  if (forms.length > 1) {
    throw new Error(`${at}: ${forms.join(' together with ')}`);
  }
  if (schema.$ref !== undefined) {
    if (!schema.$ref.startsWith(REF_PREFIX)) {
      throw new Error(`${at}: unsupported $ref ${schema.$ref}`);
    }
    return schema.$ref.slice(REF_PREFIX.length);
  }
  const values = 'const' in schema ? [schema.const] : schema.enum;
  if (values !== undefined) {
    const strays = values.filter(
      value =>
        types.length > 0 &&
        !jsonTypes(value).some(jsonType => types.includes(jsonType)),
    );
    if (strays.length > 0) {
      const listed = strays.map(value => JSON.stringify(value)).join(', ');
      throw new Error(`${at}: ${listed} not of type ${types.join(' or ')}`);
    }
    return union(values.map(value => JSON.stringify(value)));
  }
  for (const keyword of ['anyOf', 'oneOf'] as const) {
    const members = schema[keyword];
    if (members !== undefined) {
      return union(
        members.map((member: unknown, index) => {
          const where = `${at}/${keyword}/${String(index)}`;
          return typeOf(asSchema(member, where), where);
        }),
      );
    }
  }
  if (types.length === 0) return 'unknown';
  return union(types.map(type => typeOfType(schema, type, at)));
}

/** The TypeScript type of `schema` restricted to one of its types. */
function typeOfType(schema: Schema, type: string, at: string): string {
  const primitive = PRIMITIVES.get(type);
  if (primitive !== undefined) return primitive;
  if (type === 'array') {
    if (schema.items === undefined) {
      throw new Error(`${at}: an array needs items`);
    }
    const items = typeOf(asSchema(schema.items, `${at}/items`), `${at}/items`);
    return /^[\w.]+$/.test(items) ? `${items}[]` : `Array<${items}>`;
  }
  if (type === 'object') {
    if (schema.properties !== undefined) {
      throw new Error(
        `${at}: an object with properties must be a $defs definition`,
      );
    }
    const values = schema.additionalProperties;
    if (values === undefined || values === false) {
      throw new Error(`${at}: an object needs properties or their values`);
    }
    const where = `${at}/additionalProperties`;
    const value =
      values === true ? 'unknown' : typeOf(asSchema(values, where), where);
    return `Record<string, ${value}>`;
  }
  throw new Error(`${at}: unsupported type ${type}`);
}

/** A doc comment holding `text`, or nothing when there is no text. */
function docComment(text: string | undefined, indent: string): string {
  if (text === undefined) return '';
  const lines = text.replaceAll('*/', '*\\/').split('\n');
  const body = lines.map(line => `${indent} *${line ? ' ' + line : ''}`);
  return `${indent}/**\n${body.join('\n')}\n${indent} */\n`;
}

/** The declaration of the type `name`: an interface or a type alias. */
function declaration(name: string, schema: Schema, at: string): string {
  if (!IDENTIFIER.test(name)) {
    throw new Error(`${at}: not a TypeScript identifier`);
  }
  const comment = docComment(schema.description, '');
  if (schema.properties === undefined) {
    return `${comment}export type ${name} = ${typeOf(schema, at)};\n`;
  }
  if (schema.type !== 'object') {
    throw new Error(`${at}: properties without type object`);
  }
  const forms = FORMS.filter(form => form in schema);
  if (forms.length > 0) {
    throw new Error(`${at}: properties together with ${forms.join(', ')}`);
  }
  if (schema.additionalProperties !== false) {
    throw new Error(`${at}: an interface must refuse extra properties`);
  }
  const {properties} = schema;
  const required = new Set(schema.required);
  const undeclared = [...required].filter(
    key => !Object.hasOwn(properties, key),
  );
  if (undeclared.length > 0) {
    throw new Error(`${at}: requires undeclared ${undeclared.join(', ')}`);
  }
  const members = Object.entries(properties).map(([key, value]) => {
    const where = `${at}/properties/${key}`;
    const property = asSchema(value, where);
    if (!IDENTIFIER.test(key)) {
      throw new Error(`${where}: not a TypeScript identifier`);
    }
    const optional = required.has(key) ? '' : '?';
    const type = typeOf(property, where);
    return `${docComment(property.description, '  ')}  ${key}${optional}: ${type};\n`;
  });
  return `${comment}export interface ${name} {\n${members.join('')}}\n`;
}

/**
 * Translates the contract schema into TypeScript declarations: one exported
 * type per definition, under the definition's name, and `ROOT_NAME` for the
 * root.
 */
export function schemaToTypeScript(root: unknown): string {
  const schema = asSchema(root, '#', ROOT_KEYWORDS);
  if (schema.$defs === undefined || schema.oneOf === undefined) {
    throw new Error('#: the root needs $defs and oneOf');
  }
  const declarations = Object.entries(schema.$defs).map(([name, value]) => {
    const at = `${REF_PREFIX}${name}`;
    return declaration(name, asSchema(value, at), at);
  });
  const responses: Schema = {
    oneOf: schema.oneOf,
    description: ROOT_DESCRIPTION,
  };
  return [declaration(ROOT_NAME, responses, '#'), ...declarations].join('\n');
}

/** The formatted source of src/contract/contract.ts for the current schema. */
export async function generateContract(): Promise<string> {
  const [schemaText, ownText] = await Promise.all([
    readFile(SCHEMA_PATH, 'utf8'),
    readFile(import.meta.filename, 'utf8'),
  ]);
  const header = ownText.slice(0, ownText.indexOf('\n\n'));
  const types = schemaToTypeScript(JSON.parse(schemaText));
  const source = `${header}\n\n${NOTICE}\n\n${types}`;
  const options = await prettier.resolveConfig(CONTRACT_PATH);
  return prettier.format(source, {...options, filepath: CONTRACT_PATH});
}

if (isEntryPoint(import.meta.filename)) {
  await writeFile(CONTRACT_PATH, await generateContract());
}
