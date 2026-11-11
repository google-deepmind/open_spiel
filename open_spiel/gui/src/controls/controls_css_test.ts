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

/// <reference types="node" />

import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';

import {describe, expect, it} from 'vitest';

import {readDeclarations} from '../../scripts/css.ts';
import {pixels} from '../styles/pixels.ts';

// jsdom neither matches :hover nor cascades by specificity, so the rules are
// read as written.
const declarations = readDeclarations(
  readFileSync(resolve(import.meta.dirname, 'controls.css'), 'utf8'),
);

/**
 * The selectors of the rules that set a property `matches`, without the
 * spaces Prettier puts inside the parentheses of a long one.
 */
function selectorsSetting(matches: string | RegExp): string[] {
  return declarations
    .filter(({property}) =>
      typeof matches === 'string'
        ? property === matches
        : matches.test(property),
    )
    .flatMap(({selectors}) => selectors)
    .map(selector => selector.replace(/\(\s+/g, '(').replace(/\s+\)/g, ')'));
}

describe('controls.css', () => {
  it('keeps a pressed option and a selected tab on the accent under the pointer', () => {
    const hovers = selectorsSetting(/^background|color$/).filter(selector =>
      selector.includes(':hover'),
    );
    expect(hovers).not.toEqual([]);
    for (const selector of hovers) {
      expect(selector).toContain(":not([aria-pressed='true'])");
      expect(selector).toContain(":not([aria-selected='true'])");
    }
  });

  it('keeps an aria-disabled button looking disabled under the pointer', () => {
    const hovers = selectorsSetting('background').filter(selector =>
      selector.includes(':hover'),
    );
    expect(hovers).not.toEqual([]);
    for (const selector of hovers) {
      expect(selector).toContain(":not([aria-disabled='true'])");
    }
  });

  it('leaves the cursor and edge of a disabled button to base.css', () => {
    // :enabled alone would outrank base.css's [aria-disabled='true'].
    const enabled = ".button:enabled:not([aria-disabled='true'])";
    expect(selectorsSetting('cursor')).toEqual([enabled]);
    expect(selectorsSetting('border-style')).toEqual([enabled]);
    const buttonBorders = selectorsSetting('border').filter(selector =>
      selector.includes('.button'),
    );
    expect(buttonBorders).toEqual([]);
  });

  it('keeps the copy button in a gutter beside the text, outside its scroller', () => {
    // A long line scrolls inside the text's own box, so padding on that box
    // cannot keep it from under the button; the card's gutter does.
    const value = (selector: string, property: string): string => {
      const found = declarations.filter(
        declaration =>
          declaration.property === property &&
          declaration.selectors.includes(selector),
      );
      const [only] = found;
      if (found.length !== 1 || only === undefined) {
        throw new Error(
          `${selector} sets ${property} ${String(found.length)} times.`,
        );
      }
      return only.value;
    };
    const gutter = pixels(value('.copyable', 'padding-right'));
    const inset = pixels(value('.copy-button', 'right'));
    const width = pixels(value('.button--icon', 'width'));
    expect(gutter).toBeGreaterThanOrEqual(inset + width + 8);
    expect(value('.copyable__text', 'overflow')).toBe('auto');
    expect(value('.copyable__text', 'padding').split(' ')[1]).toBe('0');
  });
});
