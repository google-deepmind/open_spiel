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

import type {IDockviewPanelProps} from 'dockview-core';
import {act, createElement} from 'react';
import {afterEach, describe, expect, it} from 'vitest';

import {type Rendered, render} from '../testing/render.tsx';
import {
  COLUMN_RESERVE,
  groupMinimum,
  useGroupMinimumHeight,
} from './group_minimum.ts';

describe('groupMinimum', () => {
  it('is the floor when nothing is asked', () => {
    expect(groupMinimum(300, null, 200, 900)).toBe(200);
  });

  it('grows the group by what its content wants', () => {
    expect(groupMinimum(300, 40.5, 200, 900)).toBe(341);
  });

  it('never goes under the floor', () => {
    expect(groupMinimum(300, -250, 200, 900)).toBe(200);
  });

  it('leaves the rest of the column its reserve', () => {
    expect(groupMinimum(300, 900, 200, 900)).toBe(900 - COLUMN_RESERVE);
    expect(groupMinimum(300, 900, 200, 300)).toBe(200);
  });
});

// A dock group as the hook sees it: its height, which a size request sets,
// and every constraint and size asked of it.
interface FakeGroup {
  height: number;
  readonly minimums: number[];
  readonly sizes: number[];
}

let rendered: Rendered | undefined;

afterEach(() => {
  rendered?.unmount();
  rendered = undefined;
});

// Renders the hook for a visible panel in a group `height` pixels tall, in
// a column 1000 pixels tall; returns the group and the hook's callback.
function ask(height: number): {
  fake: FakeGroup;
  request: (overflow: number | null) => void;
} {
  const fake: FakeGroup = {height, minimums: [], sizes: []};
  const groupApi = {
    get height(): number {
      return fake.height;
    },
    setConstraints: ({minimumHeight}: {minimumHeight: number}): void => {
      fake.minimums.push(minimumHeight);
    },
    setSize: (size: {height: number}): void => {
      fake.sizes.push(size.height);
      fake.height = size.height;
    },
  };
  const none = {dispose: (): void => undefined};
  const props = {
    api: {
      group: {api: groupApi},
      isVisible: true,
      onDidVisibilityChange: () => none,
      onDidGroupChange: () => none,
    },
    containerApi: {height: 1000},
  } as unknown as IDockviewPanelProps;
  let callback: ((overflow: number | null) => void) | undefined;
  function Probe(): null {
    callback = useGroupMinimumHeight(props, 200);
    return null;
  }
  rendered = render(createElement(Probe));
  if (callback === undefined) throw new Error('The hook did not run.');
  return {fake, request: callback};
}

describe('useGroupMinimumHeight', () => {
  it('grows the group at once, then gives the room back', () => {
    const {fake, request} = ask(500);
    act(() => {
      request(280);
    });
    expect(fake.minimums.at(-1)).toBe(760);
    expect(fake.sizes).toEqual([760]);
    // Fitted, the content asks for no more: the group holds its height.
    act(() => {
      request(0);
    });
    expect(fake.sizes).toEqual([760]);
    act(() => {
      request(null);
    });
    expect(fake.minimums.at(-1)).toBe(200);
    expect(fake.sizes).toEqual([760, 500]);
  });

  it('keeps a height a person set since', () => {
    const {fake, request} = ask(500);
    act(() => {
      request(100);
    });
    fake.height = 700;
    act(() => {
      request(null);
    });
    expect(fake.sizes).toEqual([600]);
    expect(fake.height).toBe(700);
  });

  it('resizes nothing for content that fits', () => {
    const {fake, request} = ask(500);
    act(() => {
      request(-40);
    });
    act(() => {
      request(null);
    });
    expect(fake.sizes).toEqual([]);
  });
});
