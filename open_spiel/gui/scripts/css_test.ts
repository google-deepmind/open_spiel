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

import {readDeclarations} from './css.ts';

describe('readDeclarations', () => {
  it('reads declarations with their selectors and lines', () => {
    const css = 'a,\nb {\n  color: red;\n  margin: 0\n}\n';
    expect(readDeclarations(css)).toEqual([
      {selectors: ['a,\nb'], property: 'color', value: 'red', line: 3},
      {selectors: ['a,\nb'], property: 'margin', value: '0', line: 4},
    ]);
  });

  it('reads nested at-rules and rules', () => {
    const css = '@media (min-width: 1px) {\n  a:hover { opacity: 1; }\n}';
    expect(readDeclarations(css)).toEqual([
      {
        selectors: ['@media (min-width: 1px)', 'a:hover'],
        property: 'opacity',
        value: '1',
        line: 2,
      },
    ]);
  });

  it('skips comments and keeps the lines after them', () => {
    const css = '/* a { color: red; }\n*/\nb { color: blue; }';
    expect(readDeclarations(css)).toEqual([
      {selectors: ['b'], property: 'color', value: 'blue', line: 3},
    ]);
  });

  it('reads braces and semicolons in strings and parentheses as text', () => {
    const css = "a { content: '{;}'; background: url(data:x;base64,AA==); }";
    expect(readDeclarations(css).map(({value}) => value)).toEqual([
      "'{;}'",
      'url(data:x;base64,AA==)',
    ]);
  });

  it('lower-cases properties and keeps custom property values as written', () => {
    const css = ':root { COLOR: Red; --Tab-Size: 13PX; }';
    expect(
      readDeclarations(css).map(({property, value}) => [property, value]),
    ).toEqual([
      ['color', 'Red'],
      ['--tab-size', '13PX'],
    ]);
  });

  it.each([
    ['a { color: red;', 'Unclosed {'],
    ['a { color: red; } }', 'Unmatched } on line 1'],
    ["a { content: 'open; }", 'Unterminated string'],
    ['a { color red; }', 'Not a declaration on line 1: color red'],
  ])('refuses %j', (css, message) => {
    expect(() => readDeclarations(css)).toThrow(message);
  });
});
