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

import {mkdir, mkdtemp, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {dirname, join, resolve} from 'node:path';

import {afterEach, describe, expect, it} from 'vitest';

import {themeVariables} from './contrast_check.ts';
import {lintClient, lintCss, lintHtml, lintScript} from './style_lint.ts';

const TOKENS = themeVariables([
  `:root { --font-size: 14px; --text-small: 10px; --text-loop: var(--text-loop);
     --text-themed: 14px; }
   :root[data-theme='dark'] { --text-themed: 10px; }`,
]);

function cssMessages(css: string, file = 'src/panel.css'): string[] {
  return lintCss(file, css, TOKENS).map(({message}) => message);
}

function scriptMessages(source: string): string[] {
  return lintScript('src/a.tsx', source, TOKENS).map(({message}) => message);
}

describe('lintClient', () => {
  let root: string | undefined;

  /** A client root holding `files`, keyed by their path from the root. */
  async function client(files: Record<string, string>): Promise<string> {
    root = await mkdtemp(join(tmpdir(), 'style-lint-'));
    for (const [path, text] of Object.entries(files)) {
      await mkdir(dirname(join(root, path)), {recursive: true});
      await writeFile(join(root, path), text);
    }
    return root;
  }

  afterEach(async () => {
    if (root !== undefined) {
      await rm(root, {recursive: true});
      root = undefined;
    }
  });

  it('finds no problem in the client', async () => {
    expect(await lintClient(resolve(import.meta.dirname, '..'))).toEqual([]);
  });

  it('reads the page, the stylesheets and the scripts, and nothing else', async () => {
    const problems = await lintClient(
      await client({
        'index.html': '<p>\n</p><p style="opacity: 0.5"></p>',
        'src/styles/tokens.css': ':root { --text-small: 10px; }',
        'src/styles/dock.css': '',
        'src/panel/small.css': 'a { font-size: var(--text-small); }',
        'src/panel/panel.css':
          'a {\n  color: var(--color-text);\n}\nb { color: red; }',
        'src/panel/panel.tsx': '\n\nconst row = <p style={{fontSize: 10}} />;',
        'src/panel/notes.md': 'color: red; opacity: 0;',
        'src/panel/clean.ts': 'export const ok = 1;',
      }),
    );
    expect(
      problems.map(({file, line}) => `${file}:${String(line)}`).sort(),
    ).toEqual([
      'index.html:2',
      'src/panel/panel.css:4',
      'src/panel/panel.tsx:3',
      'src/panel/small.css:1',
    ]);
  });

  it('rejects a stylesheet it cannot read', async () => {
    const unbalanced = await client({
      'index.html': '',
      'src/styles/tokens.css': '',
      'src/styles/dock.css': '',
      'src/broken.css': 'a { color: var(--color-text);',
    });
    await expect(lintClient(unbalanced)).rejects.toThrow('Unclosed {');
  });
});

/** Declarations the gate refuses, each with part of its message. */
const REFUSED: Array<[string, string]> = [
  ['opacity: .5', 'opacity .5 fades content'],
  ['opacity: 0', 'opacity 0 fades content'],
  ['filter: grayscale(1)', 'filter tints or blurs'],
  ['backdrop-filter: blur(2px)', 'backdrop-filter tints or blurs'],
  ['color: #888', 'the colour in "#888" belongs in src/styles/tokens.css'],
  ['border: 1px solid #ABCDEF', 'belongs in src/styles/tokens.css'],
  ['color: grey', 'belongs in src/styles/tokens.css'],
  ['color: transparent', 'belongs in src/styles/tokens.css'],
  ['background: transparent', 'belongs in src/styles/tokens.css'],
  ['outline-color: CanvasText', 'belongs in src/styles/tokens.css'],
  ['background: rgb(0 0 0 / 0.5)', 'belongs in src/styles/tokens.css'],
  ['color: hsl(0 0% 50%)', 'belongs in src/styles/tokens.css'],
  [
    'color: color-mix(in srgb, var(--a), white)',
    'belongs in src/styles/tokens.css',
  ],
  ['--shade: #000', 'belongs in src/styles/tokens.css'],
  ['font-size: 11px', 'text at 11px is under 12 px'],
  ['font-size: 0.7rem', 'text at 0.7rem is under 12 px'],
  ['font-size: 70%', 'text at 70% is under 12 px'],
  ['font-size: x-small', 'text at x-small is under 12 px'],
  ['font-size: 8pt', 'text at 8pt is under 12 px'],
  ['font-size: 3mm', 'text at 3mm is under 12 px'],
  ['font-size: var(--font-size, 10px)', 'text at 10px is under 12 px'],
  ['font-size: var(--text-small)', 'text at 10px is under 12 px'],
  ['font-size: var(--text-themed)', 'text at 10px is under 12 px'],
  [
    'font-size: var(--text-unknown, 14px)',
    '--text-unknown is not defined in both themes',
  ],
  ['font-size: var(--text-loop)', 'refers to itself through --text-loop'],
  ['font-size: 10px !important', 'text at 10px !important cannot be checked'],
  ['font-size: 0.6vw', 'text at 0.6vw cannot be checked'],
  ['font-size: 1ex', 'text at 1ex cannot be checked'],
  ['font-size: calc(0.5rem + 2px)', 'cannot be checked'],
  ['font-size: clamp(10px, 1vw, 16px)', 'cannot be checked'],
  ['font-size: 1constructor', 'cannot be checked'],
  ['opacity: var(--faded)', 'opacity var(--faded) fades content'],
  ['opacity: calc(1 - 0.5)', 'fades content'],
  ['fill-opacity: 0.5', 'fill-opacity 0.5 fades content'],
  ['stroke-opacity: 0', 'stroke-opacity 0 fades content'],
  ['stop-opacity: 0.4', 'stop-opacity 0.4 fades content'],
  ['flood-opacity: 0.4', 'flood-opacity 0.4 fades content'],
  ['color: var(--color-text, #888)', 'belongs in src/styles/tokens.css'],
  [
    'background: var(--shade, rgb(0 0 0 / 0.5))',
    'belongs in src/styles/tokens.css',
  ],
  ['color: var(--a, var(--b, grey))', 'belongs in src/styles/tokens.css'],
  ['--dv-tab-font-size: 10px', 'text at 10px is under 12 px'],
  ['font: 10px sans-serif', 'set font-size and font-family apart'],
];

describe('lintCss', () => {
  it.each([
    'opacity: 1',
    'opacity: 1; transition: opacity 1s',
    'filter: none',
    'color: var(--color-text)',
    'border: 1px solid var(--color-border-strong)',
    'fill-opacity: 1',
    'color: currentcolor',
    'font: inherit',
    'font-size: 12px',
    'font-size: 0.75rem',
    'font-size: 75%',
    'font-size: var(--font-size)',
    'font-size: var(--font-size, 14px)',
    'font-size: 9pt',
    'font-size: small',
    'font-size: inherit',
    'font-family: system-ui, sans-serif',
    "content: '#888 red'",
    '--dv-tab-font-size: 13px',
  ])('passes %s', declaration => {
    expect(cssMessages(`a { ${declaration}; }`)).toEqual([]);
  });

  it.each(REFUSED)('refuses %s', (declaration, message) => {
    const messages = cssMessages(`a { ${declaration}; }`);
    expect(messages).toHaveLength(1);
    expect(messages[0]).toContain(message);
  });

  it('allows colours in the token file only, and nothing else there', () => {
    expect(cssMessages(':root { --c: #888; }', 'src/tokens.css')).toHaveLength(
      1,
    );
    expect(
      cssMessages(':root { --c: #888; }', 'src/styles/tokens.css'),
    ).toEqual([]);
    expect(
      cssMessages(':root { opacity: 0.5; }', 'src/styles/tokens.css'),
    ).toHaveLength(1);
  });

  it('gives the line of each problem', () => {
    const problems = lintCss(
      'src/a.css',
      'a {\n  color: red;\n}\n\nb {\n  opacity: 0;\n}',
      TOKENS,
    );
    expect(problems.map(({file, line}) => `${file}:${String(line)}`)).toEqual([
      'src/a.css:2',
      'src/a.css:6',
    ]);
  });
});

describe('lintCss on a disabled control', () => {
  it.each([
    'cursor: not-allowed',
    'border-style: dashed',
    'outline-style: dashed',
  ])('lets a disabled control set %s', declaration => {
    expect(cssMessages(`button:disabled { ${declaration}; }`)).toEqual([]);
  });

  it.each([
    'button:disabled { color: var(--color-text-muted); }',
    '.button[disabled] { background: var(--color-surface-sunken); }',
    "[aria-disabled='true'] { cursor: default; }",
    '@media (hover) { .row { a:disabled { color: var(--color-text); } } }',
  ])('refuses a greyed-out disabled control: %s', css => {
    const messages = cssMessages(css);
    expect(messages).toHaveLength(1);
    expect(messages[0]).toContain('a disabled control sets');
  });
});

describe('lintCss on a live control', () => {
  it.each([
    ".button:enabled:not([aria-disabled='true']) { cursor: pointer; }",
    '.row:hover:not([aria-disabled="true"]) { background: var(--c); }',
    ".row:not([aria-disabled='true']) { &:hover { background: var(--c); } }",
    "@media (hover) { a:not([aria-disabled='true']) { cursor: pointer; } }",
    '.row { cursor: default; }',
  ])('passes %s', css => {
    expect(cssMessages(css)).toEqual([]);
  });

  it.each([
    '.button:enabled { border-style: solid; }',
    "button.row:hover:not([aria-pressed='true']) { background: var(--c); }",
    'button.row { cursor: pointer; }',
    "a:not([aria-disabled='true']), b:hover { background: var(--c); }",
    '@media (hover) { .row:hover { background: var(--c); } }',
  ])('refuses a live style a disabled control would take: %s', css => {
    const messages = cssMessages(css);
    expect(messages).toHaveLength(1);
    expect(messages[0]).toContain("add :not([aria-disabled='true'])");
  });
});

describe('lintScript', () => {
  it.each([
    ['<p style={{opacity: 0.5}} />', 'opacity 0.5 fades content'],
    ['<p style={{filter: "blur(1px)"}} />', 'filter tints or blurs'],
    ['<p style={{color: "#888"}} />', 'belongs in src/styles/tokens.css'],
    [
      "<p style={{'backgroundColor': 'white'}} />",
      'belongs in src/styles/tokens.css',
    ],
    ['<p style={{fontSize: 10}} />', 'text at 10px is under 12 px'],
    ['element.style.opacity = "0.5";', 'opacity 0.5 fades content'],
    ['element.style.borderColor = `red`;', 'belongs in src/styles/tokens.css'],
    ['<circle fill="#888" />', 'belongs in src/styles/tokens.css'],
    ["<path stroke={'black'} />", 'belongs in src/styles/tokens.css'],
    ['context.fillStyle = "#123456";', 'belongs in src/styles/tokens.css'],
    ["context.strokeStyle = 'white';", 'belongs in src/styles/tokens.css'],
    [
      '<button style={{opacity: disabled ? 0.5 : 1}} />',
      'opacity 0.5 fades content',
    ],
    [
      "<rect fill={on ? '#888' : 'var(--color-accent)'} />",
      'belongs in src/styles/tokens.css',
    ],
    ["<rect fill={colour ?? ('grey')} />", 'belongs in src/styles/tokens.css'],
    ["<rect stroke={colour || 'black'} />", 'belongs in src/styles/tokens.css'],
    ['<p style={{opacity: faded && 0}} />', 'opacity 0 fades content'],
    ['<circle fillOpacity={0.5} />', 'fill-opacity 0.5 fades content'],
    ['<circle stroke-opacity="0.5" />', 'stroke-opacity 0.5 fades content'],
    ['<stop stopOpacity={0} />', 'stop-opacity 0 fades content'],
    ['<g opacity={0.3} />', 'opacity 0.3 fades content'],
    ['<p style={{fontSize: "var(--text-small)"}} />', 'text at 10px'],
  ])('refuses %s', (source, message) => {
    const problems = lintScript('src/a.tsx', `\n${source}`, TOKENS);
    expect(problems).toHaveLength(1);
    expect(problems[0]).toMatchObject({line: 2});
    expect(problems[0]?.message).toContain(message);
  });

  it.each([
    '<p style={{opacity: 1, fontSize: 14, color: "var(--color-text)"}} />',
    '<p style={{color: computed}} />',
    'element.style.colorScheme = theme;',
    'const label = "#123 red";',
    'const options = {filter: "red", opacity: 0};',
    '<circle fill="var(--color-accent)" stroke="none" />',
    '<path fill={colour} />',
    'context.fillStyle = palette.accent;',
    '<p style={{fontSize: large ? 16 : 14, opacity: 1}} />',
    "<rect fill={on ? 'var(--color-accent)' : 'var(--color-surface)'} />",
    '<g opacity={1} />',
  ])('passes %s', source => {
    expect(scriptMessages(source)).toEqual([]);
  });
});

describe('lintHtml', () => {
  it('reads style elements and attributes, on their own lines', () => {
    const html =
      '<html>\n<style>\n  a { color: red; }\n</style>\n<p style="opacity: 0.5"></p>';
    expect(lintHtml('index.html', html, TOKENS).map(({line}) => line)).toEqual([
      3, 5,
    ]);
  });

  it('leaves attributes that only end in style alone', () => {
    expect(
      lintHtml('index.html', '<p data-style="color: red"></p>', TOKENS),
    ).toEqual([]);
  });
});
