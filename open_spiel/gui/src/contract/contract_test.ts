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
import {describe, expect, it} from 'vitest';

import {
  CONTRACT_PATH,
  SCHEMA_PATH,
  generateContract,
} from '../../scripts/generate_contract.ts';

describe('contract.ts', () => {
  it('is what the generator writes for the current schema', async () => {
    const [checkedIn, generated] = await Promise.all([
      readFile(CONTRACT_PATH, 'utf8'),
      generateContract(),
    ]);
    expect(
      checkedIn,
      'stale contract types: run npm run generate:contract',
    ).toBe(generated);
  });

  it('exports a type for every schema definition, under its name', async () => {
    const [schemaText, source] = await Promise.all([
      readFile(SCHEMA_PATH, 'utf8'),
      readFile(CONTRACT_PATH, 'utf8'),
    ]);
    const schema = JSON.parse(schemaText) as {$defs: Record<string, unknown>};
    const exported = new Set(
      Array.from(
        source.matchAll(/^export (?:interface|type) (\w+)/gm),
        match => match[1],
      ),
    );
    const names = Object.keys(schema.$defs);
    expect(names.length).toBeGreaterThan(0);
    expect(names.filter(name => !exported.has(name))).toEqual([]);
  });
});
