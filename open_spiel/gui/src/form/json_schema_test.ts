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

import {fieldsOf} from './json_schema.ts';

// The schema the server gives its MCTS controller.
const MCTS_SCHEMA = {
  type: 'object',
  properties: {
    max_simulations: {
      type: 'integer',
      default: 1000,
      description: 'Simulations per move, each ending in a random rollout.',
      minimum: 1,
      maximum: 10000,
    },
    uct_c: {
      type: 'number',
      default: 2,
      description: 'The exploration constant of the UCT selection rule.',
      minimum: 0,
      maximum: 100,
    },
    solve: {
      type: 'boolean',
      default: true,
      description:
        'Whether proven wins and losses are backed up the search tree.',
    },
  },
  additionalProperties: false,
};

describe('fieldsOf', () => {
  it('takes the empty schema for no settings', () => {
    expect(fieldsOf({})).toEqual([]);
  });

  it('reads every property, in order', () => {
    expect(fieldsOf(MCTS_SCHEMA)).toEqual([
      {
        name: 'max_simulations',
        label: 'Max Simulations',
        description: 'Simulations per move, each ending in a random rollout.',
        kind: 'integer',
        default: 1000,
        minimum: 1,
        maximum: 10000,
      },
      {
        name: 'uct_c',
        label: 'Uct C',
        description: 'The exploration constant of the UCT selection rule.',
        kind: 'number',
        default: 2,
        minimum: 0,
        maximum: 100,
      },
      {
        name: 'solve',
        label: 'Solve',
        description:
          'Whether proven wins and losses are backed up the search tree.',
        kind: 'boolean',
        default: true,
      },
    ]);
  });

  it('takes additionalProperties false, or none', () => {
    const properties = {x: {type: 'boolean', default: false}};
    expect(fieldsOf({type: 'object', properties})).toHaveLength(1);
    expect(
      fieldsOf({type: 'object', properties, additionalProperties: false}),
    ).toHaveLength(1);
    expect(
      fieldsOf({type: 'object', properties, additionalProperties: true}),
    ).toBeNull();
  });
});

describe('fieldsOf labels and presets', () => {
  it('labels a field by its name, for people, when its title is empty', () => {
    const schema = {
      type: 'object',
      properties: {max_depth: {type: 'boolean', default: false, title: ''}},
    };
    expect(fieldsOf(schema)?.[0]?.label).toBe('Max Depth');
  });

  it('reads the presets of a number or text field', () => {
    const schema = {
      type: 'object',
      properties: {
        n: {
          type: 'integer',
          default: 10,
          maximum: 100,
          'x-presets': [1, 10, 100],
        },
        model: {type: 'string', default: 'a', 'x-presets': ['a', 'b']},
      },
    };
    expect(
      fieldsOf(schema)?.map(field => 'presets' in field && field.presets),
    ).toEqual([
      [1, 10, 100],
      ['a', 'b'],
    ]);
  });

  it('labels a field by its title, and reads text and choices', () => {
    const schema = {
      type: 'object',
      properties: {
        model: {type: 'string', default: 'small', title: 'Model'},
        size: {type: 'integer', default: 9, enum: [9, 13, 19], maximum: 19},
      },
    };
    expect(fieldsOf(schema)).toEqual([
      {
        name: 'model',
        label: 'Model',
        description: null,
        kind: 'string',
        default: 'small',
      },
      {
        name: 'size',
        label: 'Size',
        description: null,
        kind: 'choice',
        default: 9,
        choices: [9, 13, 19],
      },
    ]);
  });
});

describe('fieldsOf refusals', () => {
  it.each([
    ['a schema that is not an object schema', {type: 'array'}],
    ['an object schema without properties', {type: 'object'}],
    ['a property that is not an object', {x: 1}],
    ['a type the form cannot show', {x: {type: 'array', default: []}}],
    ['a missing default', {x: {type: 'integer'}}],
    ['a default of another type', {x: {type: 'boolean', default: 0}}],
    ['a fractional integer default', {x: {type: 'integer', default: 1.5}}],
    [
      'a keyword the form cannot enforce',
      {x: {type: 'string', default: '', pattern: 'a'}},
    ],
    [
      'a bound that is not a number',
      {x: {type: 'number', default: 1, minimum: '0'}},
    ],
    [
      'a choice of another type',
      {x: {type: 'string', default: 'a', enum: ['a', 1]}},
    ],
    [
      'a default that is not a choice',
      {x: {type: 'string', default: 'b', enum: ['a']}},
    ],
    [
      'a default under the minimum',
      {x: {type: 'integer', default: 0, minimum: 1}},
    ],
    [
      'a default over the maximum',
      {x: {type: 'number', default: 2, maximum: 1}},
    ],
    [
      'a choice over the maximum',
      {x: {type: 'integer', default: 2, enum: [1, 2, 50], maximum: 10}},
    ],
    [
      'a choice under the minimum',
      {x: {type: 'number', default: 2, enum: [-1, 2], minimum: 0}},
    ],
    [
      'presets of another type',
      {x: {type: 'integer', default: 1, 'x-presets': ['a']}},
    ],
    [
      'a preset over the maximum',
      {x: {type: 'integer', default: 1, maximum: 5, 'x-presets': [9]}},
    ],
    [
      'presets on a checkbox',
      {x: {type: 'boolean', default: true, 'x-presets': [true]}},
    ],
    [
      'presets on a list of choices',
      {x: {type: 'integer', default: 1, enum: [1, 2], 'x-presets': [1]}},
    ],
    [
      'bounds that cross',
      {x: {type: 'number', default: 1, minimum: 2, maximum: 0}},
    ],
    [
      'a schema keyword the form cannot enforce',
      {type: 'object', properties: {}, required: ['x']},
    ],
    [
      'extra properties a schema allows',
      {type: 'object', properties: {}, additionalProperties: {type: 'string'}},
    ],
  ])('refuses %s', (_, properties) => {
    const schema =
      'type' in properties ? properties : {type: 'object', properties};
    expect(fieldsOf(schema)).toBeNull();
  });
});
