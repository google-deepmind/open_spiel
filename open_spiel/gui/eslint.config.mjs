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

// Google TypeScript Style: the gts rule set on top of typescript-eslint's
// strict, type-checked configuration. Formatting belongs to Prettier.

/// <reference types="node" />

import eslint from '@eslint/js';
import {defineConfig} from 'eslint/config';
import prettier from 'eslint-config-prettier';
import jsdoc from 'eslint-plugin-jsdoc';
import reactHooks from 'eslint-plugin-react-hooks';
import tseslint from 'typescript-eslint';

// The client is self-contained: no import may leave this directory or name an
// absolute path. A relative import names its file with the extension, which
// both Vite and Node resolve. ESLint cannot resolve a path, so each directory depth, up to
// MAX_DEPTH, gets the number of parent steps that would leave it.
const MAX_DEPTH = 8;
const SOURCES = '*.{ts,tsx,js,jsx,mjs,cjs}';

/**
 * @param {number} depth directories between this one and the file
 * @returns {import('eslint').Linter.Config}
 */
function staysInClient(depth) {
  return {
    files: ['*/'.repeat(depth) + SOURCES],
    rules: {
      'no-restricted-imports': [
        'error',
        {
          patterns: [
            {
              regex: `^(\\.\\./){${String(depth)}}\\.\\.(/|$)`,
              message: 'Imports stay inside open_spiel/gui.',
            },
            {regex: '^/', message: 'No absolute imports.'},
            {
              regex: '^\\.\\.?(/.*)?(?<!\\.(ts|tsx|css|json|svg\\?raw))$',
              message: 'A relative import names the file with its extension.',
            },
          ],
        },
      ],
    },
  };
}

const NUMBER_PARSING =
  'Use Number() to parse numbers; parseInt only with a radix other than 10.';
const ARRAY_CONSTRUCTOR = 'Use an array literal or Array.from({length}).';
const NUMBER_PARSING_PROPERTIES = [
  {object: 'Number', property: 'parseFloat', message: NUMBER_PARSING},
  {object: 'Number', property: 'parseInt', message: NUMBER_PARSING},
];

// Every restriction but the one on default exports, which Playwright's
// configuration files need.
const RESTRICTED_SYNTAX = [
  {
    selector:
      "CallExpression[callee.name='parseInt'][arguments.length<2], CallExpression[callee.name='parseInt'][arguments.1.value=10]",
    message: NUMBER_PARSING,
  },
  {
    selector:
      'Program > VariableDeclaration > VariableDeclarator > ArrowFunctionExpression, Program > ExportNamedDeclaration > VariableDeclaration > VariableDeclarator > ArrowFunctionExpression',
    message: 'Use a function declaration for top-level functions.',
  },
  {selector: "CallExpression[callee.name='Array']", message: ARRAY_CONSTRUCTOR},
  {selector: "NewExpression[callee.name='Array']", message: ARRAY_CONSTRUCTOR},
];

// The Google TypeScript Style Guide rules that neither the gts rule set nor
// typescript-eslint's presets enforce. They come after Prettier's config,
// which turns `curly` off.
/** @type {import('eslint').Linter.Config[]} */
const googleStyle = [
  {
    files: ['**/*.{ts,tsx}'],
    plugins: {jsdoc},
    rules: {
      'jsdoc/require-jsdoc': [
        'error',
        {
          // The fixer only inserts empty comments; exports need real ones.
          enableFixer: false,
          publicOnly: true,
          require: {
            FunctionDeclaration: true,
            ClassDeclaration: true,
            ArrowFunctionExpression: true,
          },
          contexts: [
            'ExportNamedDeclaration > VariableDeclaration',
            'ExportNamedDeclaration > TSTypeAliasDeclaration',
            'ExportNamedDeclaration > TSInterfaceDeclaration',
          ],
        },
      ],
      'default-case': 'error',
      'no-restricted-globals': [
        'error',
        {name: 'parseFloat', message: NUMBER_PARSING},
      ],
      'no-restricted-properties': ['error', ...NUMBER_PARSING_PROPERTIES],
      'no-restricted-syntax': [
        'error',
        ...RESTRICTED_SYNTAX,
        {selector: 'ExportDefaultDeclaration', message: 'Use named exports.'},
      ],
      // A body without braces sits on the line of its condition; one-line
      // `if (x) return;` stays allowed.
      curly: ['error', 'multi-line'],
      'nonblock-statement-body-position': ['error', 'beside'],
    },
  },
  {
    // DOM code reads textContent. Playwright's locator.innerText() in e2e/ is
    // a different API: the rendered text the privacy tests read.
    files: ['src/**/*.{ts,tsx}'],
    rules: {
      'no-restricted-properties': [
        'error',
        ...NUMBER_PARSING_PROPERTIES,
        {property: 'innerText', message: 'Use textContent.'},
      ],
    },
  },
  {
    // Production code types an object literal by its declaration,
    // `const x: T = {...}`. Tests and their fakes may cast partial objects
    // (events jsdom does not implement, for one).
    files: ['src/**/*.{ts,tsx}'],
    ignores: ['src/**/testing/**', 'src/**/*_test.{ts,tsx}'],
    rules: {
      '@typescript-eslint/consistent-type-assertions': [
        'error',
        {
          assertionStyle: 'as',
          objectLiteralTypeAssertions: 'allow-as-parameter',
        },
      ],
    },
  },
  {
    // Playwright requires a default export from these files.
    files: ['playwright.config.ts', 'e2e/global_setup.ts'],
    rules: {'no-restricted-syntax': ['error', ...RESTRICTED_SYNTAX]},
  },
];

const boundaries = Array.from({length: MAX_DEPTH + 1}, (_, depth) =>
  staysInClient(depth),
);

export default defineConfig(
  // addons/ and piece_sets/ hold the user's own add-ons and fetched files;
  // npm run typecheck still checks the add-ons.
  {
    ignores: [
      'dist/',
      'node_modules/',
      'playwright-report/',
      'test-results/',
      'addons/',
      'piece_sets/',
    ],
  },
  eslint.configs.recommended,
  tseslint.configs.strictTypeChecked,
  tseslint.configs.stylisticTypeChecked,
  reactHooks.configs.flat.recommended,
  {
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
    linterOptions: {reportUnusedDisableDirectives: 'error'},
    rules: {
      'block-scoped-var': 'error',
      eqeqeq: 'error',
      'no-var': 'error',
      'prefer-arrow-callback': 'error',
      'prefer-const': 'error',
      '@typescript-eslint/consistent-type-imports': 'error',
      '@typescript-eslint/explicit-module-boundary-types': 'error',
      // T[] for simple types, Array<T> otherwise: Google TypeScript Style Guide.
      '@typescript-eslint/array-type': [
        'error',
        {default: 'array-simple', readonly: 'array-simple'},
      ],
      'max-lines': [
        'error',
        {max: 500, skipBlankLines: true, skipComments: true},
      ],
      'max-lines-per-function': [
        'error',
        {max: 80, skipBlankLines: true, skipComments: true},
      ],
    },
  },
  ...boundaries,
  prettier,
  ...googleStyle,
);
