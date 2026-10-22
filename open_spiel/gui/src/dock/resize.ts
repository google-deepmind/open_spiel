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

/**
 * Milliseconds a dock resize must settle before the dock is rebuilt. A
 * rebuild remounts every panel, so it runs once per resize, not per frame.
 */
export const SETTLE_MS = 250;

// Layout rounding can leave a group up to a pixel past the edge.
const ROUNDING = 1;

/**
 * Whether a dock group lies partly outside `container`. Dockview honours
 * minimum sizes that no longer fit by laying groups out past the edge.
 */
export function groupsOutside(container: Element): boolean {
  const box = container.getBoundingClientRect();
  return [...container.querySelectorAll('.dv-groupview')].some(group => {
    const rect = group.getBoundingClientRect();
    return (
      rect.width > 0 &&
      rect.height > 0 &&
      (rect.left < box.left - ROUNDING ||
        rect.right > box.right + ROUNDING ||
        rect.top < box.top - ROUNDING ||
        rect.bottom > box.bottom + ROUNDING)
    );
  });
}

/** A callback that runs once calls to `schedule` have stopped for a while. */
export interface Settled {
  /** Runs the callback `SETTLE_MS` from now, unless called again first. */
  readonly schedule: () => void;
  readonly cancel: () => void;
}

/** `callback`, run once calls to `schedule` have settled. */
export function settled(callback: () => void): Settled {
  let timer: ReturnType<typeof setTimeout> | undefined;
  return {
    schedule(): void {
      clearTimeout(timer);
      timer = setTimeout(callback, SETTLE_MS);
    },
    cancel(): void {
      clearTimeout(timer);
    },
  };
}
