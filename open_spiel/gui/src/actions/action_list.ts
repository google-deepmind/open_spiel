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

// The arithmetic of a legal action list of any size. Labels are the game's
// own text; nothing here parses one.

import type {Action} from '../contract/contract.ts';

/** Above this many rows, only the rows in view are rendered. */
export const WINDOW_ABOVE = 200;

/**
 * The height of one windowed row and the gap below it, in pixels; every row
 * is exactly this.
 */
export const ROW_STRIDE_PX = 34;

/** The gap below each windowed row, in pixels, within ROW_STRIDE_PX. */
export const ROW_GAP_PX = 2;

/** Rows rendered beyond each edge of the view, so a scroll shows no gap. */
export const OVERSCAN_ROWS = 12;

/** The actions in id order, which is the order every list shows them in. */
export function sortById<Row extends Action>(actions: readonly Row[]): Row[] {
  return [...actions].sort((a, b) => a.id - b.id);
}

/** The ids of the actions whose label another action also has. */
export function duplicateLabelIds(
  actions: readonly Action[],
): ReadonlySet<number> {
  const idsByLabel = new Map<string, number[]>();
  for (const {id, label} of actions) {
    const ids = idsByLabel.get(label);
    if (ids === undefined) {
      idsByLabel.set(label, [id]);
    } else {
      ids.push(id);
    }
  }
  return new Set([...idsByLabel.values()].filter(ids => ids.length > 1).flat());
}

/** The rows of a windowed list to render: from `start` up to `end`. */
export interface WindowRange {
  readonly start: number;
  readonly end: number;
}

/**
 * The rows to render for a list of `count` rows scrolled to `scrollTop` in a
 * view `viewportPx` tall: those in view, plus OVERSCAN_ROWS on each side. A
 * `scrollTop` past the end, which the browser keeps for a moment after the
 * list shrinks, counts as the last full view.
 */
export function windowRange(
  count: number,
  scrollTop: number,
  viewportPx: number,
): WindowRange {
  if (!Number.isInteger(count) || count < 0) {
    throw new Error(`A row count is a whole number, not ${String(count)}.`);
  }
  if (
    !Number.isFinite(scrollTop) ||
    Number.isNaN(viewportPx) ||
    viewportPx < 0
  ) {
    throw new Error(
      `No window at scrollTop ${String(scrollTop)} in ${String(viewportPx)} px.`,
    );
  }
  const top = clampScrollTop(count, scrollTop, viewportPx);
  const start = Math.max(0, Math.floor(top / ROW_STRIDE_PX) - OVERSCAN_ROWS);
  const end = Math.min(
    count,
    Math.ceil((top + viewportPx) / ROW_STRIDE_PX) + OVERSCAN_ROWS,
  );
  return {start, end};
}

/** The nearest `scrollTop` that a list of `count` rows can reach. */
export function clampScrollTop(
  count: number,
  scrollTop: number,
  viewportPx: number,
): number {
  const bottom = Math.max(0, count * ROW_STRIDE_PX - viewportPx);
  return Math.min(Math.max(0, scrollTop), bottom);
}

/**
 * The nearest `scrollTop` to `scrollTop` at which row `index` is wholly in a
 * view `viewportPx` tall.
 */
export function scrollTopShowing(
  index: number,
  scrollTop: number,
  viewportPx: number,
): number {
  const rowTop = index * ROW_STRIDE_PX;
  if (rowTop < scrollTop) return rowTop;
  const rowBottom = rowTop + ROW_STRIDE_PX;
  return rowBottom > scrollTop + viewportPx
    ? rowBottom - viewportPx
    : scrollTop;
}
