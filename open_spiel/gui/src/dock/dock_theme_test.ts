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

import {readFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {describe, expect, it} from 'vitest';

import {DOCK_TOKENS} from './dock_theme.ts';

describe('dock.css', () => {
  it('reads exactly the theme tokens the dock declares', async () => {
    const css = await readFile(
      resolve(import.meta.dirname, 'dock.css'),
      'utf8',
    );
    const read = new Set(
      [...css.matchAll(/var\((--[\w-]+)/g)].map(([, token]) => token),
    );
    expect([...read].sort()).toEqual([...DOCK_TOKENS].sort());
  });
});
