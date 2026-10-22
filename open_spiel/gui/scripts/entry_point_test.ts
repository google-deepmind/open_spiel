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

import {realpathSync} from 'node:fs';
import process from 'node:process';

import {afterEach, describe, expect, it} from 'vitest';

import {isEntryPoint} from './entry_point.ts';

const argv = process.argv;

afterEach(() => {
  process.argv = argv;
});

describe('isEntryPoint', () => {
  it('is false for an imported module', () => {
    expect(isEntryPoint(import.meta.filename)).toBe(false);
  });

  it('is true for the script Node runs, however its path is spelled', () => {
    process.argv = [
      process.execPath,
      `${import.meta.dirname}/../scripts/./entry_point_test.ts`,
    ];
    expect(isEntryPoint(realpathSync(import.meta.filename))).toBe(true);
  });

  it('is false with no script', () => {
    process.argv = [process.execPath];
    expect(isEntryPoint(import.meta.filename)).toBe(false);
  });
});
