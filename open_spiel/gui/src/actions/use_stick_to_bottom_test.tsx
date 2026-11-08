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

import {act, type JSX} from 'react';
import {createRoot, type Root} from 'react-dom/client';
import {afterEach, beforeEach, describe, expect, it} from 'vitest';

import {layOut, restoreLayout, scrollTo} from '../testing/dom.ts';
import {only} from '../testing/render.tsx';
import {BOTTOM_SLACK_PX, useStickToBottom} from './use_stick_to_bottom.ts';

// 600 px of content in a 200 px view: the bottom is at scrollTop 400.
const BOTTOM = 400;

let container: HTMLElement;
let root: Root;

function List({count}: {count: number}): JSX.Element {
  const {ref, onScroll} = useStickToBottom<HTMLDivElement>(count);
  return <div ref={ref} className="list" onScroll={onScroll} />;
}

function show(count: number): HTMLElement {
  act(() => {
    root.render(<List count={count} />);
  });
  return only(container, '.list');
}

beforeEach(() => {
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
  layOut(HTMLDivElement.prototype, {clientHeight: 200, scrollHeight: 600});
});

afterEach(() => {
  act(() => {
    root.unmount();
  });
  container.remove();
  restoreLayout();
});

describe('useStickToBottom', () => {
  it('starts at the bottom and follows a growing count there', () => {
    const list = show(3);
    expect(list.scrollTop).toBe(600);
    scrollTo(list, BOTTOM);
    show(4);
    expect(list.scrollTop).toBe(600);
  });

  it('counts a list within the slack of its bottom as at the bottom', () => {
    const list = show(3);
    scrollTo(list, BOTTOM - BOTTOM_SLACK_PX);
    show(4);
    expect(list.scrollTop).toBe(600);
  });

  it('stays where the reader scrolled up to', () => {
    const list = show(3);
    scrollTo(list, BOTTOM - BOTTOM_SLACK_PX - 1);
    show(4);
    expect(list.scrollTop).toBe(BOTTOM - BOTTOM_SLACK_PX - 1);
  });

  it('moves only when the count changes', () => {
    const list = show(3);
    list.scrollTop = 250;
    show(3);
    expect(list.scrollTop).toBe(250);
  });
});
