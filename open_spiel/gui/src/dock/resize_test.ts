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

import {afterEach, describe, expect, it, vi} from 'vitest';

import {SETTLE_MS, groupsOutside, settled} from './resize.ts';

/** A 1000 x 600 px dock holding groups at the given boxes. */
function dockWithGroups(...groups: DOMRect[]): Element {
  const container = document.createElement('div');
  container.getBoundingClientRect = () => new DOMRect(0, 0, 1000, 600);
  for (const box of groups) {
    const group = document.createElement('div');
    group.className = 'dv-groupview';
    group.getBoundingClientRect = () => box;
    container.append(group);
  }
  return container;
}

describe('groupsOutside', () => {
  it('is false while every group is inside the dock', () => {
    const left = new DOMRect(0, 0, 600, 600);
    const right = new DOMRect(600, 0, 400, 600);
    expect(groupsOutside(dockWithGroups(left, right))).toBe(false);
  });

  it('is true when a group runs past the right edge', () => {
    const left = new DOMRect(0, 0, 700, 600);
    const right = new DOMRect(700, 0, 400, 600);
    expect(groupsOutside(dockWithGroups(left, right))).toBe(true);
  });

  it('is true when a group runs past the bottom edge', () => {
    const top = new DOMRect(0, 0, 1000, 400);
    const bottom = new DOMRect(0, 400, 1000, 240);
    expect(groupsOutside(dockWithGroups(top, bottom))).toBe(true);
  });

  it('allows a pixel of rounding', () => {
    const top = new DOMRect(0, 0, 1000, 300.5);
    const bottom = new DOMRect(0, 300.5, 1000, 300.5);
    expect(groupsOutside(dockWithGroups(top, bottom))).toBe(false);
  });

  it('ignores a group that is not laid out', () => {
    const shown = new DOMRect(0, 0, 1000, 600);
    const unsized = new DOMRect(1200, 0, 0, 0);
    expect(groupsOutside(dockWithGroups(shown, unsized))).toBe(false);
  });
});

describe('settled', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it('runs once, SETTLE_MS after the last call', () => {
    vi.useFakeTimers();
    const callback = vi.fn();
    const {schedule} = settled(callback);
    schedule();
    vi.advanceTimersByTime(SETTLE_MS - 1);
    schedule();
    vi.advanceTimersByTime(SETTLE_MS - 1);
    expect(callback).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);
    expect(callback).toHaveBeenCalledOnce();
  });

  it('does not run once cancelled', () => {
    vi.useFakeTimers();
    const callback = vi.fn();
    const {schedule, cancel} = settled(callback);
    schedule();
    cancel();
    vi.advanceTimersByTime(SETTLE_MS);
    expect(callback).not.toHaveBeenCalled();
  });
});
