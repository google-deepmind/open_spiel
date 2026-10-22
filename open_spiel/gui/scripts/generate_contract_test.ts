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

import {describe, expect, it} from 'vitest';

import {ROOT_NAME, schemaToTypeScript} from './generate_contract.ts';

/** The schema of a contract whose one response is `Body`. */
function contract(body: object, extra: Record<string, object> = {}): object {
  return {
    oneOf: [{$ref: '#/$defs/Body'}],
    $defs: {Body: body, ...extra},
  };
}

/** The declaration of `Body`, on one line. */
function body(schema: object, extra: Record<string, object> = {}): string {
  const source = schemaToTypeScript(contract(schema, extra));
  const start = source.indexOf('export interface Body');
  return source.slice(start, source.indexOf('}\n', start) + 1);
}

function object(properties: Record<string, object>): object {
  return {
    type: 'object',
    properties,
    required: Object.keys(properties),
    additionalProperties: false,
  };
}

describe('schemaToTypeScript translations', () => {
  it('declares the root as the union of the responses', () => {
    expect(schemaToTypeScript(contract(object({})))).toContain(
      `export type ${ROOT_NAME} = Body;`,
    );
  });

  it.each([
    [{type: 'string'}, 'string'],
    [{type: 'integer'}, 'number'],
    [{type: ['integer', 'number', 'null']}, 'number | null'],
    [{type: 'boolean'}, 'boolean'],
    [{const: 1}, '1'],
    [{enum: ['a', 'b']}, '"a" | "b"'],
    [{type: 'string', enum: ['a', 'b']}, '"a" | "b"'],
    [{type: ['string', 'null'], enum: ['a', null]}, '"a" | null'],
    [{type: 'number', const: 1.5}, '1.5'],
    [{$ref: '#/$defs/Other'}, 'Other'],
    [{type: 'array', items: {type: 'integer'}}, 'number[]'],
    [
      {type: 'array', items: {type: ['string', 'null']}},
      'Array<string | null>',
    ],
    [
      {anyOf: [{type: 'array', items: {type: 'integer'}}, {type: 'null'}]},
      'number[] | null',
    ],
    [
      {type: 'object', additionalProperties: {type: 'boolean'}},
      'Record<string, boolean>',
    ],
    [{type: 'object', additionalProperties: {}}, 'Record<string, unknown>'],
    [{description: 'Any JSON value.'}, 'unknown'],
  ])('translates %j to %s', (schema, type) => {
    const other = {Other: object({})};
    expect(body(object({field: schema}), other)).toContain(`field: ${type};`);
  });

  it('marks a property that is not required as optional', () => {
    const schema = {...object({field: {type: 'string'}}), required: []};
    expect(body(schema)).toContain('field?: string;');
  });

  it('declares an enum or a union definition as a type alias', () => {
    const source = schemaToTypeScript(
      contract(object({}), {Kind: {type: 'string', enum: ['x', 'y']}}),
    );
    expect(source).toContain('export type Kind = "x" | "y";');
  });

  it('turns descriptions into doc comments, escaping their end', () => {
    const schema = {
      ...object({field: {type: 'string', description: 'A */ B'}}),
      description: 'First.\n\nSecond.',
    };
    const source = schemaToTypeScript(contract(schema));
    expect(source).toContain('/**\n * First.\n *\n * Second.\n */\n');
    expect(source).toContain('  /**\n   * A *\\/ B\n   */\n  field: string;');
  });
});

describe('schemaToTypeScript refusals', () => {
  it.each([
    [{type: 'string', format: 'date'}, 'unsupported keyword format'],
    [{type: 'object', $defs: {}}, 'unsupported keyword $defs'],
    [{$schema: 'x'}, 'unsupported keyword $schema'],
    [{$ref: 'other.json#/Thing'}, 'unsupported $ref other.json#/Thing'],
    [{type: 'tuple'}, 'unsupported type tuple'],
    [{type: 'object'}, 'an object needs properties or their values'],
    [
      {type: 'object', properties: {}, additionalProperties: false},
      'an object with properties must be a $defs definition',
    ],
    [{type: 'array'}, 'an array needs items'],
    [{items: {type: 'string'}}, 'items without type array'],
    [{type: 'string', required: []}, 'required without type object'],
    [{$ref: '#/$defs/Other', type: 'object'}, '$ref together with type'],
    [{enum: ['a'], const: 'a'}, 'const together with enum'],
    [{type: 'string', enum: [1, null]}, '1, null not of type string'],
    [{type: 'integer', const: 1.5}, '1.5 not of type integer'],
    [{type: 1}, 'type must be a string or a non-empty array of strings'],
    [{type: []}, 'type must be a string or a non-empty array of strings'],
    [{type: 'string', description: 1}, 'description must be a string'],
    [{$ref: 1}, '$ref must be a string'],
    [{type: 'string', enum: 'x'}, 'enum must be an array'],
    [{anyOf: {type: 'string'}}, 'anyOf must be an array'],
    [{oneOf: {type: 'string'}}, 'oneOf must be an array'],
    [{type: 'array', items: true}, 'items must be a schema'],
    [
      {type: 'object', additionalProperties: 'x'},
      'additionalProperties must be a boolean or a schema',
    ],
  ])('refuses the property %j', (schema, message) => {
    expect(() => schemaToTypeScript(contract(object({field: schema})))).toThrow(
      `#/$defs/Body/properties/field: ${message}`,
    );
  });

  it('gives the pointer into the values of a record', () => {
    const field = {type: 'object', additionalProperties: {format: 'x'}};
    expect(() => schemaToTypeScript(contract(object({field})))).toThrow(
      '#/$defs/Body/properties/field/additionalProperties: unsupported keyword format',
    );
  });

  it.each([
    [{required: 'field'}, 'required must be an array of strings'],
    [{required: [1]}, 'required must be an array of strings'],
    [{required: ['toString']}, 'requires undeclared toString'],
    [{additionalProperties: true}, 'an interface must refuse extra properties'],
    [{type: 'array'}, 'properties without type object'],
    [{enum: [{}]}, 'properties together with enum'],
  ])('refuses the definition change %j', (change, message) => {
    const schema = {...object({field: {type: 'string'}}), ...change};
    expect(() => schemaToTypeScript(contract(schema))).toThrow(
      `#/$defs/Body: ${message}`,
    );
  });

  it('refuses a property name that is not an identifier', () => {
    expect(() =>
      schemaToTypeScript(contract(object({'two words': {type: 'string'}}))),
    ).toThrow('#/$defs/Body/properties/two words: not a TypeScript identifier');
  });

  it('refuses a definition name that is not an identifier', () => {
    expect(() =>
      schemaToTypeScript(contract(object({}), {'two words': object({})})),
    ).toThrow('#/$defs/two words: not a TypeScript identifier');
  });

  it.each([
    [{$defs: {}}, '#: the root needs $defs and oneOf'],
    [{oneOf: []}, '#: the root needs $defs and oneOf'],
    [{oneOf: [], $defs: {}, type: 'object'}, '#: unsupported keyword type'],
  ])('refuses the root %j', (root, message) => {
    expect(() => schemaToTypeScript(root)).toThrow(message);
  });
});
