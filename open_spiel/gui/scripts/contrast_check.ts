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

// The contrast gate, `npm run contrast`. In both themes it resolves the
// design tokens and the dock's variables to colours and measures each pair
// that is drawn one on the other against WCAG 2.2's minimum contrast ratio:
// 4.5:1 for text (success criterion 1.4.3) and 3:1 for the focus ring and a
// control's edge (1.4.11).
//
// It measures the tokens, not how rules combine: a rule that changes a
// background must leave alone any state whose text colour assumes another
// (controls_css_test.ts checks this for the shared controls).

/// <reference types="node" />

import {readFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import process from 'node:process';

import type {Theme} from '../src/theme/theme.ts';
import {readDeclarations} from './css.ts';
import {isEntryPoint} from './entry_point.ts';

/** A colour as red, green and blue, each from 0 to 255. */
export type Rgb = readonly [number, number, number];

/** A colour drawn on another, and the least contrast it needs. */
export interface Pair {
  foreground: string;
  background: string;
  minimum: number;
}

/** A pair as measured in a theme. */
export interface Measurement extends Pair {
  theme: Theme;
  ratio: number;
}

/** The stylesheets that define the variables, relative to the client root. */
export const STYLESHEETS = ['src/styles/tokens.css', 'src/styles/dock.css'];

// The selectors those stylesheets may use, and the themes each applies to.
const SCOPES: Record<string, readonly Theme[]> = {
  ':root': ['light', 'dark'],
  ":root[data-theme='dark']": ['dark'],
  '.dockview-theme-open-spiel': ['light', 'dark'],
};

const TEXT_MINIMUM = 4.5;
const MARK_MINIMUM = 3;
const SURFACES = [
  '--color-surface',
  '--color-surface-raised',
  '--color-surface-sunken',
];
const TEXTS = [
  '--color-text',
  '--color-text-secondary',
  '--color-text-muted',
  '--color-accent',
  '--color-success',
  '--color-warning',
  '--color-danger',
];
const MARKS = ['--color-focus', '--color-border-strong'];
const TABS = ['activegroup', 'inactivegroup'].flatMap(group =>
  ['visiblepanel', 'hiddenpanel'].map(panel => `--dv-${group}-${panel}-tab`),
);

function onSurfaces(colours: string[], minimum: number): Pair[] {
  return colours.flatMap(foreground =>
    SURFACES.map(background => ({foreground, background, minimum})),
  );
}

/** Every pair the client draws, one colour on the other. */
export const PAIRS: readonly Pair[] = [
  ...onSurfaces(TEXTS, TEXT_MINIMUM),
  ...onSurfaces(MARKS, MARK_MINIMUM),
  {
    foreground: '--color-on-accent',
    background: '--color-accent',
    minimum: TEXT_MINIMUM,
  },
  ...['--color-accent', '--color-text', '--color-text-muted'].map(
    foreground => ({
      foreground,
      background: '--color-accent-soft',
      minimum: TEXT_MINIMUM,
    }),
  ),
  ...TABS.map(tab => ({
    foreground: `${tab}-color`,
    background: `${tab}-background-color`,
    minimum: TEXT_MINIMUM,
  })),
];

/** Each theme's custom properties, as the stylesheets declare them. */
export function themeVariables(
  stylesheets: readonly string[],
): Record<Theme, Map<string, string>> {
  const variables: Record<Theme, Map<string, string>> = {
    light: new Map(),
    dark: new Map(),
  };
  for (const css of stylesheets) {
    for (const {selectors, property, value, line} of readDeclarations(css)) {
      const themes =
        selectors.length === 1 ? SCOPES[selectors[0] ?? ''] : undefined;
      if (themes === undefined) {
        throw new Error(
          `Line ${String(line)}: ${property} is under ${selectors.join(' ')},` +
            ` not one of ${Object.keys(SCOPES).join(', ')}.`,
        );
      }
      if (themes.length === 1 && !variables.light.has(property)) {
        throw new Error(
          `Line ${String(line)}: the dark theme sets ${property},` +
            ' which the light theme does not define.',
        );
      }
      for (const theme of themes) {
        variables[theme].set(property, value);
      }
    }
  }
  return variables;
}

/** The colour `name` resolves to, as red, green and blue from 0 to 255. */
export function resolveColour(
  variables: ReadonlyMap<string, string>,
  name: string,
  seen: readonly string[] = [],
): Rgb {
  const value = variables.get(name);
  if (value === undefined) {
    throw new Error(`${name} is not defined.`);
  }
  if (seen.includes(name)) {
    throw new Error(`${[...seen, name].join(' -> ')} refer to each other.`);
  }
  const reference = /^var\(\s*(--[\w-]+)\s*\)$/.exec(value);
  if (reference?.[1] !== undefined) {
    return resolveColour(variables, reference[1], [...seen, name]);
  }
  const hex = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(value)?.[1];
  if (hex === undefined) {
    throw new Error(`${name} is ${value}, not an opaque hex colour.`);
  }
  const digits = hex.length === 3 ? hex.replace(/./g, '$&$&') : hex;
  const channel = (start: number): number =>
    Number(`0x${digits.slice(start, start + 2)}`);
  return [channel(0), channel(2), channel(4)];
}

// WCAG 2.2's relative luminance of an sRGB colour.
function luminance([red, green, blue]: Rgb): number {
  const linear = (channel: number): number => {
    const value = channel / 255;
    return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * linear(red) + 0.7152 * linear(green) + 0.0722 * linear(blue);
}

/** WCAG 2.2's contrast ratio between two colours, from 1 to 21. */
export function contrastRatio(first: Rgb, second: Rgb): number {
  const lighter = Math.max(luminance(first), luminance(second));
  const darker = Math.min(luminance(first), luminance(second));
  return (lighter + 0.05) / (darker + 0.05);
}

/** Every pair, measured in every theme. */
export function measureContrast(stylesheets: readonly string[]): Measurement[] {
  const variables = themeVariables(stylesheets);
  return (['light', 'dark'] as const).flatMap(theme =>
    PAIRS.map(pair => ({
      ...pair,
      theme,
      ratio: contrastRatio(
        resolveColour(variables[theme], pair.foreground),
        resolveColour(variables[theme], pair.background),
      ),
    })),
  );
}

function formatMeasurement({
  theme,
  foreground,
  background,
  ratio,
  minimum,
}: Measurement): string {
  return (
    `${theme}: ${foreground} on ${background} is ${ratio.toFixed(2)}:1,` +
    ` at least ${String(minimum)}:1 needed`
  );
}

if (isEntryPoint(import.meta.filename)) {
  const root = resolve(import.meta.dirname, '..');
  const stylesheets = await Promise.all(
    STYLESHEETS.map(path => readFile(resolve(root, path), 'utf8')),
  );
  const measurements = measureContrast(stylesheets);
  const failed = measurements.filter(({ratio, minimum}) => ratio < minimum);
  for (const measurement of failed) {
    console.error(`FAIL ${formatMeasurement(measurement)}`);
  }
  const closest = measurements.reduce((best, next) =>
    next.ratio / next.minimum < best.ratio / best.minimum ? next : best,
  );
  console.log(
    `contrast: ${String(measurements.length - failed.length)} of` +
      ` ${String(measurements.length)} pairs pass; the closest is` +
      ` ${formatMeasurement(closest)}.`,
  );
  if (failed.length > 0) {
    process.exitCode = 1;
  }
}
