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

import {beforeEach, describe, expect, it} from 'vitest';

import {
  GROUPS_STORAGE_KEY,
  readActiveSections,
  writeActiveSection,
} from './group_store.ts';

function storage(): Storage {
  return window.localStorage;
}
function refusing(): Storage {
  throw new DOMException('Storage is disabled.', 'SecurityError');
}

beforeEach(() => {
  window.localStorage.clear();
});

describe('the open sections', () => {
  it('are kept per game, over what storage holds', () => {
    expect(readActiveSections(storage, 'chess')).toEqual({});
    expect(writeActiveSection(storage, 'chess', 'actions', 'legal')).toBe(true);
    expect(writeActiveSection(storage, 'go', 'setup', 'players')).toBe(true);
    expect(writeActiveSection(storage, 'chess', 'setup', 'position')).toBe(
      true,
    );
    expect(readActiveSections(storage, 'chess')).toEqual({
      actions: 'legal',
      setup: 'position',
    });
    expect(readActiveSections(storage, 'go')).toEqual({setup: 'players'});
  });

  it.each(['{', '{"version": 2, "games": {}}', '{"version": 1, "games": 1}'])(
    'leave storage holding %s alone',
    text => {
      localStorage.setItem(GROUPS_STORAGE_KEY, text);
      expect(readActiveSections(storage, 'chess')).toEqual({});
      expect(writeActiveSection(storage, 'chess', 'actions', 'legal')).toBe(
        false,
      );
      expect(localStorage.getItem(GROUPS_STORAGE_KEY)).toBe(text);
    },
  );

  it('are forgotten, not refused, where the browser keeps nothing', () => {
    expect(readActiveSections(refusing, 'chess')).toEqual({});
    expect(writeActiveSection(refusing, 'chess', 'actions', 'legal')).toBe(
      false,
    );
  });
});
