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

// The style gate, `npm run lint:style`. It refuses the styling that makes
// text hard to read or a colour impossible to theme:
//
// - a colour written anywhere but src/styles/tokens.css, the one place both
//   themes define and the contrast check measures, `transparent` included;
// - text under 12 px, and any text size it cannot evaluate, a token's
//   included: a size given by a token is checked as each theme defines it.
//   A size computed by calc(), min(), max() or clamp() scales with the
//   board (`max(12px, calc(var(--card-w) * 0.2))`) and is left to review,
//   as written;
// - a live control's style, its :enabled and :hover rules and its pointer
//   cursor, on a selector that does not exclude [aria-disabled='true']: a
//   control refusing presses while a request is in flight is enabled, and
//   such a selector outranks base.css's disabled rule.
//
// Opacity, filters and how a disabled control looks are not refused: a
// game board keeps its own faded controls, washed-out ghost cards and
// hover filters. The lesson the gates keep is the one about text: the
// contrast check (`npm run contrast`) measures every text colour.
//
// It reads every stylesheet under src/, the inline styles of index.html, and
// in TypeScript the literal values of JSX `style` attributes, of the JSX
// paint attributes (`fill`, `stroke` and the like), of assignments
// to `element.style` and of a canvas context's colour properties; a value
// chosen by `?:`, `??`, `||` or `&&` is read branch by branch. It judges each
// declaration alone, not how rules combine, so a hover rule that changes the
// background behind text another rule colours is left to review and tests.

/// <reference types="node" />

import {readFile, readdir} from 'node:fs/promises';
import {join, relative, resolve} from 'node:path';
import process from 'node:process';

import ts from 'typescript';

import type {Theme} from '../src/theme/theme.ts';
import {STYLESHEETS, themeVariables} from './contrast_check.ts';
import {readDeclarations} from './css.ts';
import {isEntryPoint} from './entry_point.ts';

/** The one stylesheet allowed to write colours, from the client root. */
export const TOKENS_FILE = 'src/styles/tokens.css';

/**
 * The smallest text size, in px. rem, em and % are taken against 16 px, the
 * size of the root and of any parent the client does not resize.
 */
export const MIN_FONT_PX = 12;

/** Each theme's custom properties, as the token stylesheets define them. */
export type Tokens = Readonly<Record<Theme, ReadonlyMap<string, string>>>;

/** A rule broken at a place in a file. */
export interface Problem {
  file: string;
  line: number;
  message: string;
}

/** What one declaration says, wherever it was written. */
export interface Style {
  property: string;
  value: string;
  selectors: readonly string[];
}

const NAMED_COLOURS = new Set(
  (
    'aliceblue antiquewhite aqua aquamarine azure beige bisque black ' +
    'blanchedalmond blue blueviolet brown burlywood cadetblue chartreuse ' +
    'chocolate coral cornflowerblue cornsilk crimson cyan darkblue darkcyan ' +
    'darkgoldenrod darkgray darkgreen darkgrey darkkhaki darkmagenta ' +
    'darkolivegreen darkorange darkorchid darkred darksalmon darkseagreen ' +
    'darkslateblue darkslategray darkslategrey darkturquoise darkviolet ' +
    'deeppink deepskyblue dimgray dimgrey dodgerblue firebrick floralwhite ' +
    'forestgreen fuchsia gainsboro ghostwhite gold goldenrod gray green ' +
    'greenyellow grey honeydew hotpink indianred indigo ivory khaki lavender ' +
    'lavenderblush lawngreen lemonchiffon lightblue lightcoral lightcyan ' +
    'lightgoldenrodyellow lightgray lightgreen lightgrey lightpink ' +
    'lightsalmon lightseagreen lightskyblue lightslategray lightslategrey ' +
    'lightsteelblue lightyellow lime limegreen linen magenta maroon ' +
    'mediumaquamarine mediumblue mediumorchid mediumpurple mediumseagreen ' +
    'mediumslateblue mediumspringgreen mediumturquoise mediumvioletred ' +
    'midnightblue mintcream mistyrose moccasin navajowhite navy oldlace ' +
    'olive olivedrab orange orangered orchid palegoldenrod palegreen ' +
    'paleturquoise palevioletred papayawhip peachpuff peru pink plum ' +
    'powderblue purple rebeccapurple red rosybrown royalblue saddlebrown ' +
    'salmon sandybrown seagreen seashell sienna silver skyblue slateblue ' +
    'slategray slategrey snow springgreen steelblue tan teal thistle tomato ' +
    'turquoise violet wheat white whitesmoke yellow yellowgreen ' +
    // No colour at all, which fades text as surely as opacity 0.
    'transparent ' +
    // The system colours: just as fixed, and just as unthemed.
    'accentcolor accentcolortext activetext buttonborder buttonface ' +
    'buttontext canvas canvastext field fieldtext graytext highlight ' +
    'highlighttext linktext mark marktext selecteditem selecteditemtext ' +
    'visitedtext'
  ).split(' '),
);
const HEX_COLOUR = /#[0-9a-f]{3,8}\b/i;
const COLOUR_FUNCTION =
  /\b(?:rgba?|hsla?|hwb|lab|lch|oklab|oklch|color|color-mix|light-dark)\(/i;
const LIVE_SELECTOR = /:enabled|:hover/;
const NOT_ARIA_DISABLED = /:not\(\[aria-disabled=(['"])true\1\]\)/;
const SMALL_FONT_KEYWORDS = new Set(['xx-small', 'x-small', 'smaller']);
const FONT_KEYWORDS = new Set([
  'small',
  'medium',
  'large',
  'x-large',
  'xx-large',
  'xxx-large',
  'larger',
  'inherit',
  'initial',
  'unset',
]);
// CSS's absolute units are fixed multiples of the px (CSS Values 4, 6.2).
const PX_PER_UNIT = new Map([
  ['px', 1],
  ['rem', 16],
  ['em', 16],
  ['%', 0.16],
  ['pt', 4 / 3],
  ['pc', 16],
  ['in', 96],
  ['cm', 96 / 2.54],
  ['mm', 96 / 25.4],
  ['q', 96 / 101.6],
]);
const FONT_SIZE = /^(\d*\.?\d+)([a-z]+|%)$/;
const TOKEN_REFERENCE = /^var\(\s*(--[\w-]+)\s*(?:,\s*(.*))?\)$/s;
// The unitless opacity properties, to which React appends no px.
const OPACITY = /^(?:fill-|stroke-|stop-|flood-)?opacity$/;
const MATH_FUNCTION = /^(?:calc|min|max|clamp)\(.*\)$/s;

/**
 * `value` with strings and url() arguments blanked, and each var() reduced to
 * its fallback, which is a value like any other.
 */
function bareValue(value: string): string {
  return value
    .replace(/"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'/g, '""')
    .replace(/\burl\([^)]*\)/gi, 'url()')
    .replace(/\bvar\(\s*--[\w-]+\s*,?/gi, 'var(');
}

function colourProblem(value: string): string | null {
  const bare = bareValue(value);
  const words = bare.toLowerCase().match(/(?<![\w-])[a-z]+(?![\w-])/g) ?? [];
  if (
    HEX_COLOUR.test(bare) ||
    COLOUR_FUNCTION.test(bare) ||
    words.some(word => NAMED_COLOURS.has(word))
  ) {
    return (
      `the colour in ${JSON.stringify(value)} belongs in ${TOKENS_FILE};` +
      ' refer to a token with var().'
    );
  }
  return null;
}

// A size passes once it is read and found large enough: a length in a
// known unit, a size keyword, or a token reference, read as each theme
// defines the token and, if it has one, as its fallback. A size computed by
// calc(), min(), max() or clamp() passes unread. Anything else,
// !important and an undefined token among them, is refused.
function fontSizeProblem(
  value: string,
  tokens: Tokens,
  seen: readonly string[] = [],
): string | null {
  const size = value.trim().toLowerCase();
  const reference = TOKEN_REFERENCE.exec(size);
  if (reference !== null) {
    const [, name = '', fallback] = reference;
    if (seen.includes(name)) {
      return `text at ${value} refers to itself through ${seen.join(', ')}.`;
    }
    const defined = Object.values(tokens).map(theme => theme.get(name));
    if (defined.includes(undefined)) {
      return (
        `text at ${value} cannot be checked; ${name} is not defined in` +
        ` both themes by ${STYLESHEETS.join(' and ')}.`
      );
    }
    const sizes = [...defined, fallback].filter(size => size !== undefined);
    return (
      sizes
        .map(size => fontSizeProblem(size, tokens, [...seen, name]))
        .find(problem => problem !== null) ?? null
    );
  }
  if (MATH_FUNCTION.test(size)) {
    return null;
  }
  const length = FONT_SIZE.exec(size);
  const pxPerUnit = PX_PER_UNIT.get(length?.[2] ?? '');
  const pixels =
    length === null || pxPerUnit === undefined
      ? undefined
      : Number(length[1]) * pxPerUnit;
  if (
    SMALL_FONT_KEYWORDS.has(size) ||
    (pixels !== undefined && pixels < MIN_FONT_PX)
  ) {
    return `text at ${value} is under ${String(MIN_FONT_PX)} px.`;
  }
  if (pixels === undefined && !FONT_KEYWORDS.has(size)) {
    return (
      `text at ${value} cannot be checked; give a length in px, rem, em, %` +
      ' or an absolute unit, a size keyword, or a token with var().'
    );
  }
  return null;
}

/** The selectors of a comma-separated list, split outside parentheses. */
function selectorList(prelude: string): string[] {
  const selectors: string[] = [];
  let depth = 0;
  let start = 0;
  for (let offset = 0; offset < prelude.length; offset++) {
    const character = prelude[offset];
    if (character === '(') depth++;
    else if (character === ')') depth--;
    else if (character === ',' && depth === 0) {
      selectors.push(prelude.slice(start, offset));
      start = offset + 1;
    }
  }
  return [...selectors, prelude.slice(start)].map(part => part.trim());
}

// A control marked aria-disabled refuses presses but stays enabled, so its
// live style must exclude it by name; base.css's disabled rule is weaker
// than any selector with a class and a pseudo-class.
function liveProblem({property, value, selectors}: Style): string | null {
  const rules = selectors.filter(selector => !selector.startsWith('@'));
  const rule = rules.at(-1);
  const enclosing = rules.slice(0, -1);
  if (
    rule === undefined ||
    enclosing.some(selector => NOT_ARIA_DISABLED.test(selector))
  ) {
    return null;
  }
  const pointer = property === 'cursor' && value === 'pointer';
  const live = selectorList(rule).find(
    selector =>
      (pointer || LIVE_SELECTOR.test(selector)) &&
      !NOT_ARIA_DISABLED.test(selector),
  );
  if (live === undefined) {
    return null;
  }
  return (
    `${live} styles a live control without excluding a disabled one;` +
    " add :not([aria-disabled='true'])."
  );
}

/** What is wrong with one declaration, if anything. */
export function styleProblems(
  style: Style,
  allowColours: boolean,
  tokens: Tokens,
): string[] {
  const {property, value} = style;
  const problems: Array<string | null> = [liveProblem(style)];
  if (property === 'font' && value !== 'inherit') {
    problems.push(
      'set font-size and font-family apart, so the size is checked.',
    );
  }
  if (property.includes('font-size')) {
    problems.push(fontSizeProblem(value, tokens));
  }
  if (!allowColours) {
    problems.push(colourProblem(value));
  }
  return problems.filter(problem => problem !== null);
}

/** Every problem in the stylesheet `css`, read from `file`. */
export function lintCss(file: string, css: string, tokens: Tokens): Problem[] {
  const allowColours = file === TOKENS_FILE;
  return readDeclarations(css).flatMap(declaration =>
    styleProblems(declaration, allowColours, tokens).map(message => ({
      file,
      line: declaration.line,
      message,
    })),
  );
}

/** Every problem in the inline styles of the page `html`, read from `file`. */
export function lintHtml(
  file: string,
  html: string,
  tokens: Tokens,
): Problem[] {
  const lineAt = (offset: number): number =>
    html.slice(0, offset).split('\n').length;
  const blocks = [
    ...html.matchAll(/<style\b[^>]*>([\s\S]*?)<\/style>/gi),
    ...html.matchAll(/(?<![\w-])style\s*=\s*"([^"]*)"/gi),
    ...html.matchAll(/(?<![\w-])style\s*=\s*'([^']*)'/gi),
  ];
  return blocks.flatMap(match => {
    const line = lineAt(match.index) - 1;
    return lintCss(file, match[1] ?? '', tokens).map(problem => ({
      ...problem,
      line: problem.line + line,
    }));
  });
}

// The JSX attributes that paint SVG, by their CSS names, and the canvas
// context's colours.
const PAINT_ATTRIBUTES = new Set([
  'fill',
  'stroke',
  'color',
  'stop-color',
  'flood-color',
  'lighting-color',
]);
const CANVAS_COLOURS = new Set(['fillStyle', 'strokeStyle', 'shadowColor']);

function kebabCase(name: string): string {
  return name.replace(/[A-Z]/g, letter => `-${letter.toLowerCase()}`);
}

const CHOICES = new Set([
  ts.SyntaxKind.QuestionQuestionToken,
  ts.SyntaxKind.BarBarToken,
  ts.SyntaxKind.AmpersandAmpersandToken,
]);

/**
 * The literal values `node` can take, as CSS would read them: itself if it
 * is a literal, each branch's if it chooses between values, and none if it
 * is computed.
 */
function literalValues(property: string, node: ts.Expression): string[] {
  if (ts.isParenthesizedExpression(node)) {
    return literalValues(property, node.expression);
  }
  if (ts.isConditionalExpression(node)) {
    return [node.whenTrue, node.whenFalse].flatMap(branch =>
      literalValues(property, branch),
    );
  }
  if (ts.isBinaryExpression(node) && CHOICES.has(node.operatorToken.kind)) {
    return [node.left, node.right].flatMap(branch =>
      literalValues(property, branch),
    );
  }
  if (ts.isStringLiteralLike(node)) {
    return [node.text];
  }
  if (ts.isNumericLiteral(node)) {
    // React appends px to a bare number, except for unitless properties.
    return [OPACITY.test(property) ? node.text : `${node.text}px`];
  }
  return [];
}

function stylesOf(property: string, node: ts.Expression): Style[] {
  return literalValues(property, node).map(value => ({
    property,
    value,
    selectors: [],
  }));
}

/** The styles a TypeScript node writes, if it is a style it can read. */
function writtenStyles(node: ts.Node): Style[] {
  // <circle fill="..." />
  if (
    ts.isJsxAttribute(node) &&
    PAINT_ATTRIBUTES.has(kebabCase(node.name.getText())) &&
    node.initializer !== undefined
  ) {
    const initializer = ts.isJsxExpression(node.initializer)
      ? node.initializer.expression
      : node.initializer;
    const property = kebabCase(node.name.getText());
    return initializer === undefined ? [] : stylesOf(property, initializer);
  }
  // <element style={{property: value}} />
  if (
    ts.isJsxAttribute(node) &&
    node.name.getText() === 'style' &&
    node.initializer !== undefined &&
    ts.isJsxExpression(node.initializer) &&
    node.initializer.expression !== undefined &&
    ts.isObjectLiteralExpression(node.initializer.expression)
  ) {
    return node.initializer.expression.properties.flatMap(member => {
      if (!ts.isPropertyAssignment(member)) {
        return [];
      }
      const property = kebabCase(member.name.getText().replace(/['"]/g, ''));
      return stylesOf(property, member.initializer);
    });
  }
  // element.style.property = value
  if (
    ts.isBinaryExpression(node) &&
    node.operatorToken.kind === ts.SyntaxKind.EqualsToken &&
    ts.isPropertyAccessExpression(node.left) &&
    ts.isPropertyAccessExpression(node.left.expression) &&
    node.left.expression.name.text === 'style'
  ) {
    return stylesOf(kebabCase(node.left.name.text), node.right);
  }
  // context.fillStyle = value
  if (
    ts.isBinaryExpression(node) &&
    node.operatorToken.kind === ts.SyntaxKind.EqualsToken &&
    ts.isPropertyAccessExpression(node.left) &&
    CANVAS_COLOURS.has(node.left.name.text)
  ) {
    return stylesOf(kebabCase(node.left.name.text), node.right);
  }
  return [];
}

/** Every problem in the literal styles of the TypeScript `source`. */
export function lintScript(
  file: string,
  source: string,
  tokens: Tokens,
): Problem[] {
  const tree = ts.createSourceFile(
    file,
    source,
    ts.ScriptTarget.Latest,
    true,
    file.endsWith('.tsx') ? ts.ScriptKind.TSX : ts.ScriptKind.TS,
  );
  const problems: Problem[] = [];
  const visit = (node: ts.Node): void => {
    for (const style of writtenStyles(node)) {
      const start = tree.getLineAndCharacterOfPosition(node.getStart(tree));
      for (const message of styleProblems(style, false, tokens)) {
        problems.push({file, line: start.line + 1, message});
      }
    }
    ts.forEachChild(node, visit);
  };
  visit(tree);
  return problems;
}

const LINTERS: Array<
  [RegExp, (file: string, text: string, tokens: Tokens) => Problem[]]
> = [
  [/\.css$/, lintCss],
  [/\.html$/, lintHtml],
  [/\.tsx?$/, lintScript],
];

/** Every problem in the client rooted at `root`: src/ and index.html. */
export async function lintClient(root: string): Promise<Problem[]> {
  const tokens = themeVariables(
    await Promise.all(
      STYLESHEETS.map(path => readFile(join(root, path), 'utf8')),
    ),
  );
  const sources = await readdir(join(root, 'src'), {recursive: true});
  const files = ['index.html', ...sources.map(path => join('src', path))];
  const problems = await Promise.all(
    files.map(async file => {
      const linter = LINTERS.find(([pattern]) => pattern.test(file))?.[1];
      return linter === undefined
        ? []
        : linter(file, await readFile(join(root, file), 'utf8'), tokens);
    }),
  );
  return problems.flat();
}

if (isEntryPoint(import.meta.filename)) {
  const root = resolve(import.meta.dirname, '..');
  const problems = await lintClient(root);
  for (const {file, line, message} of problems) {
    console.error(
      `${relative(process.cwd(), join(root, file))}:${String(line)}: ${message}`,
    );
  }
  if (problems.length > 0) {
    process.exitCode = 1;
  } else {
    console.log('lint:style: no problems.');
  }
}
