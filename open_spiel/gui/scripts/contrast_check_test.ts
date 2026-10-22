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

import type {Rgb} from './contrast_check.ts';
import {
  PAIRS,
  STYLESHEETS,
  contrastRatio,
  measureContrast,
  resolveColour,
  themeVariables,
} from './contrast_check.ts';

const stylesheets = STYLESHEETS.map(path =>
  readFileSync(resolve(import.meta.dirname, '..', path), 'utf8'),
);

describe('the client stylesheets', () => {
  const measurements = measureContrast(stylesheets);

  it('measure every pair in both themes', () => {
    expect(measurements).toHaveLength(2 * PAIRS.length);
  });

  it.each(['light', 'dark'])('reach every minimum in the %s theme', theme => {
    const failing = measurements.filter(
      measurement =>
        measurement.theme === theme && measurement.ratio < measurement.minimum,
    );
    expect(failing).toEqual([]);
  });
});

describe('contrastRatio', () => {
  it.each<[Rgb, Rgb, number]>([
    [[0, 0, 0], [255, 255, 255], 21],
    [[255, 255, 255], [0, 0, 0], 21],
    [[118, 118, 118], [255, 255, 255], 4.54],
    [[136, 136, 136], [255, 255, 255], 3.54],
    [[90, 90, 90], [90, 90, 90], 1],
  ])('measures %j on %j as %f', (first, second, ratio) => {
    expect(contrastRatio(first, second)).toBeCloseTo(ratio, 2);
  });
});

describe('resolveColour', () => {
  const variables = new Map([
    ['--grey', '#888'],
    ['--text', 'var(--grey)'],
    ['--translucent', 'rgb(0 0 0 / 0.5)'],
    ['--missing', 'var(--nowhere)'],
    ['--loop', 'var(--back)'],
    ['--back', 'var(--loop)'],
  ]);

  it('follows references to a hex colour', () => {
    expect(resolveColour(variables, '--text')).toEqual([136, 136, 136]);
  });

  it.each([
    [
      '--translucent',
      '--translucent is rgb(0 0 0 / 0.5), not an opaque hex colour.',
    ],
    ['--missing', '--nowhere is not defined.'],
    ['--loop', '--loop -> --back -> --loop refer to each other.'],
  ])('refuses %s', (name, message) => {
    expect(() => resolveColour(variables, name)).toThrow(message);
  });
});

describe('themeVariables', () => {
  it('layers the dark theme over the light one', () => {
    const {light, dark} = themeVariables([
      ":root { --a: #fff; --b: #000; }\n:root[data-theme='dark'] { --a: #111; }",
      '.dockview-theme-open-spiel { --c: var(--a); }',
    ]);
    expect([...light]).toEqual([
      ['--a', '#fff'],
      ['--b', '#000'],
      ['--c', 'var(--a)'],
    ]);
    expect([...dark]).toEqual([
      ['--a', '#111'],
      ['--b', '#000'],
      ['--c', 'var(--a)'],
    ]);
  });

  it.each([
    ['a { --a: #fff; }', 'is under a, not one of'],
    ['@media print { :root { --a: #fff; } }', 'is under @media print :root'],
    [
      ":root[data-theme='dark'] { --typo: #fff; }",
      'the dark theme sets --typo',
    ],
  ])('refuses %j', (css, message) => {
    expect(() => themeVariables([css])).toThrow(message);
  });
});

describe('a seeded low-contrast token', () => {
  it('fails its pairs in the theme it is in, and only there', () => {
    const seeded = stylesheets.map(css =>
      css.replace(
        /(:root \{[^}]*--color-text-muted: )#[0-9a-f]{6}/,
        '$1#888888',
      ),
    );
    const failing = measureContrast(seeded).filter(
      ({ratio, minimum}) => ratio < minimum,
    );
    expect(
      failing.map(({theme, foreground, background}) => [
        theme,
        foreground,
        background,
      ]),
    ).toEqual([
      ['light', '--color-text-muted', '--color-surface'],
      ['light', '--color-text-muted', '--color-surface-raised'],
      ['light', '--color-text-muted', '--color-surface-sunken'],
      ['light', '--color-text-muted', '--color-accent-soft'],
    ]);
  });
});
