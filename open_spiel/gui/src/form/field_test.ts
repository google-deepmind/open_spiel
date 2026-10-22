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

import type {ChoiceField, Field, NumberField} from './field.ts';
import {admittedNumbers, draftOf, readDraft} from './field.ts';

const SIMULATIONS: NumberField = {
  name: 'max_simulations',
  label: 'max_simulations',
  description: null,
  kind: 'integer',
  default: 1000,
  minimum: 1,
  maximum: 10000,
};
const UCT_C: NumberField = {
  ...SIMULATIONS,
  name: 'uct_c',
  kind: 'number',
  default: 2,
  minimum: 0,
  maximum: 100,
};
const SOLVE: Field = {
  name: 'solve',
  label: 'solve',
  description: null,
  kind: 'boolean',
  default: true,
};
const MODEL: Field = {
  name: 'model',
  label: 'model',
  description: null,
  kind: 'string',
  default: 'small',
};
const SIZE: ChoiceField = {
  name: 'size',
  label: 'size',
  description: null,
  kind: 'choice',
  default: 9,
  choices: [9, 13, 19],
};

describe('draftOf', () => {
  it('shows the default when there is no value', () => {
    expect(draftOf(SIMULATIONS)).toBe('1000');
    expect(draftOf(SOLVE)).toBe(true);
    expect(draftOf(MODEL)).toBe('small');
    expect(draftOf(SIZE)).toBe('0');
  });

  it('shows a value given', () => {
    expect(draftOf(UCT_C, 1.5)).toBe('1.5');
    expect(draftOf(SOLVE, false)).toBe(false);
    expect(draftOf(SIZE, 19)).toBe('2');
  });

  it('shows a choice that is not one of the choices as no choice', () => {
    expect(draftOf(SIZE, 7)).toBe('');
  });
});

describe('readDraft', () => {
  it.each([
    [SIMULATIONS, '1', 1],
    [SIMULATIONS, ' 10000 ', 10000],
    [UCT_C, '.5', 0.5],
    [UCT_C, '0', 0],
    [UCT_C, '2.5', 2.5],
    [UCT_C, '1e1', 10],
  ])('reads %o from %j', (field, draft, value) => {
    expect(readDraft(field, draft)).toEqual({ok: true, value});
  });

  it.each([
    [SIMULATIONS, ''],
    [SIMULATIONS, '0'],
    [SIMULATIONS, '10001'],
    [SIMULATIONS, '0x10'],
    [SIMULATIONS, '0b11'],
    [SIMULATIONS, '1.5'],
    [SIMULATIONS, '1e20'],
    [UCT_C, '-0.1'],
    [UCT_C, 'two'],
    [UCT_C, 'Infinity'],
  ])('refuses %o from %j', (field, draft) => {
    expect(readDraft(field, draft)).toEqual({
      ok: false,
      error: admittedNumbers(field),
    });
  });

  it('reads a box and text as they are', () => {
    expect(readDraft(SOLVE, false)).toEqual({ok: true, value: false});
    expect(readDraft(MODEL, '')).toEqual({ok: true, value: ''});
  });

  it('reads a choice by its index', () => {
    expect(readDraft(SIZE, '1')).toEqual({ok: true, value: 13});
    expect(readDraft(SIZE, '')).toEqual({
      ok: false,
      error: 'Choose one of the options.',
    });
    expect(readDraft(SIZE, '3').ok).toBe(false);
  });

  it('refuses a draft of another kind as a defect', () => {
    expect(() => readDraft(SOLVE, 'true')).toThrow(
      'The box solve has a text draft.',
    );
    expect(() => readDraft(MODEL, true)).toThrow(
      "The field model has a box's draft.",
    );
  });
});

describe('admittedNumbers', () => {
  it('says which numbers a field admits', () => {
    const bounds = (minimum: number | null, maximum: number | null): string =>
      admittedNumbers({...SIMULATIONS, minimum, maximum});
    expect(bounds(1, 9)).toBe('A whole number from 1 to 9.');
    expect(bounds(1, null)).toBe('A whole number, at least 1.');
    expect(bounds(null, 9)).toBe('A whole number, at most 9.');
    expect(bounds(null, null)).toBe('A whole number.');
    expect(admittedNumbers(UCT_C)).toBe('A number from 0 to 100.');
  });
});
