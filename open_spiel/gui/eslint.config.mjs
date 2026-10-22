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
              regex: '^\\.\\.?(/.*)?(?<!\\.(ts|tsx|css|json))$',
              message: 'A relative import names the file with its extension.',
            },
          ],
        },
      ],
    },
  };
}

const boundaries = Array.from({length: MAX_DEPTH + 1}, (_, depth) =>
  staysInClient(depth),
);

export default defineConfig(
  {ignores: ['dist/', 'node_modules/']},
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
);
