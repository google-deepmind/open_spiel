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

import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import {
  CLIPPED_TAB_CLASS,
  markClippedTabs,
  resetDockScroll,
  watchTabStrips,
} from './tab_strip.ts';

const TAB_WIDTH = 80;

/** `element`, made to keep its scrollLeft as a laid-out element does. */
function scrollable<T extends HTMLElement>(element: T): T {
  let scrollLeft = 0;
  Object.defineProperty(element, 'scrollLeft', {
    get: () => scrollLeft,
    set: (value: number) => {
      scrollLeft = value;
    },
  });
  return element;
}

/**
 * A strip `width` pixels wide holding `count` tabs of TAB_WIDTH side by side,
 * laid out as a browser would: each tab's box moves with the strip's scroll.
 */
function strip(
  width: number,
  count: number,
  active: number,
): {root: HTMLElement; strip: HTMLElement; tabs: HTMLElement[]} {
  const root = document.createElement('div');
  const container = scrollable(document.createElement('div'));
  container.className = 'dv-tabs-container';
  container.getBoundingClientRect = () => new DOMRect(0, 0, width, 30);
  const tabs = Array.from({length: count}, (_, index) => {
    const tab = document.createElement('div');
    tab.className = index === active ? 'dv-tab dv-active-tab' : 'dv-tab';
    tab.getBoundingClientRect = () =>
      new DOMRect(index * TAB_WIDTH - container.scrollLeft, 0, TAB_WIDTH, 30);
    return tab;
  });
  container.append(...tabs);
  root.append(container);
  document.body.append(root);
  return {root, strip: container, tabs};
}

function clipped(tabs: readonly HTMLElement[]): boolean[] {
  return tabs.map(tab => tab.classList.contains(CLIPPED_TAB_CLASS));
}

afterEach(() => {
  document.body.replaceChildren();
});

describe('markClippedTabs', () => {
  it('marks the tabs partly outside the strip', () => {
    const {root, strip: container, tabs} = strip(200, 3, 0);
    markClippedTabs(root, true);
    expect(clipped(tabs)).toEqual([false, false, true]);
    expect(container.scrollLeft).toBe(0);
  });

  it('scrolls a cut active tab whole into view, starting on a whole tab', () => {
    const {root, strip: container, tabs} = strip(200, 3, 2);
    markClippedTabs(root, true);
    expect(container.scrollLeft).toBe(TAB_WIDTH);
    expect(clipped(tabs)).toEqual([true, false, false]);
  });

  it('scrolls back to an active tab cut on the left', () => {
    const {root, strip: container, tabs} = strip(200, 4, 0);
    container.scrollLeft = 100;
    markClippedTabs(root, true);
    expect(container.scrollLeft).toBe(0);
    expect(clipped(tabs)).toEqual([false, false, true, true]);
  });

  it('leaves a scrolled strip alone when not revealing', () => {
    const {root, strip: container, tabs} = strip(200, 3, 2);
    markClippedTabs(root, false);
    expect(container.scrollLeft).toBe(0);
    expect(clipped(tabs)).toEqual([false, false, false]);
  });

  it('always reveals a tab that was hidden until it became active', () => {
    const {root, strip: container, tabs} = strip(200, 3, 2);
    tabs[2]?.classList.add(CLIPPED_TAB_CLASS);
    markClippedTabs(root, false);
    expect(container.scrollLeft).toBe(TAB_WIDTH);
    expect(clipped(tabs)).toEqual([true, false, false]);
  });

  it('leaves a strip that is not laid out alone', () => {
    const {root, tabs} = strip(0, 3, 0);
    markClippedTabs(root, true);
    expect(clipped(tabs)).toEqual([false, false, false]);
  });
});

describe('resetDockScroll', () => {
  it('scrolls a dock container back to its left edge', () => {
    const element = scrollable(document.createElement('div'));
    element.scrollLeft = 96;
    expect(element.scrollLeft).toBe(96);
    resetDockScroll(element);
    expect(element.scrollLeft).toBe(0);
  });

  it('leaves a tab strip scrolled', () => {
    const {strip: container} = strip(200, 3, 0);
    container.scrollLeft = 80;
    resetDockScroll(container);
    expect(container.scrollLeft).toBe(80);
  });
});

describe('watchTabStrips', () => {
  beforeEach(() => {
    vi.useFakeTimers({
      toFake: ['requestAnimationFrame', 'cancelAnimationFrame'],
    });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('marks the tabs once a frame after the dock changes', async () => {
    const {root, tabs} = strip(200, 3, 0);
    const stop = watchTabStrips(root);
    tabs[1]?.setAttribute('style', 'width: 80px');
    await Promise.resolve();
    expect(clipped(tabs)).toEqual([false, false, false]);
    vi.advanceTimersToNextFrame();
    expect(clipped(tabs)).toEqual([false, false, true]);
    stop();
  });

  it('ignores changes inside panel content', async () => {
    const {root, tabs} = strip(200, 3, 0);
    const content = document.createElement('div');
    content.className = 'dv-content-container';
    root.append(content);
    const stop = watchTabStrips(root);
    content.append(document.createElement('p'));
    await Promise.resolve();
    vi.advanceTimersToNextFrame();
    expect(clipped(tabs)).toEqual([false, false, false]);
    stop();
  });

  it('undoes a sideways scroll of the dock frame', () => {
    const {root} = strip(200, 3, 0);
    const frame = scrollable(document.createElement('div'));
    root.append(frame);
    const stop = watchTabStrips(root);
    frame.scrollLeft = 96;
    expect(frame.scrollLeft).toBe(96);
    frame.dispatchEvent(new Event('scroll'));
    expect(frame.scrollLeft).toBe(0);
    stop();
  });

  it('stops watching when stopped', async () => {
    const {root, tabs} = strip(200, 3, 0);
    watchTabStrips(root)();
    tabs[1]?.setAttribute('style', 'width: 80px');
    await Promise.resolve();
    vi.advanceTimersToNextFrame();
    expect(clipped(tabs)).toEqual([false, false, false]);
  });
});
