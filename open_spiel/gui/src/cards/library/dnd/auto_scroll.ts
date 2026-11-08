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

// Auto-scroll during a pointer drag: the scroll containers of the tables, and
// the window, scroll while the pointer is near their edges.
import type {Point} from '../core/identity.ts';
import type {SceneRuntime} from '../react/runtime.ts';

const EDGE_PX = 40;
const MAX_SCROLL_PX_PER_FRAME = 18;

/** Scrollable ancestors of an element, nearest first. */
function scrollParents(el: Element | null): HTMLElement[] {
  const out: HTMLElement[] = [];
  for (
    let n = el?.parentElement ?? null;
    n && n !== document.body && n !== document.documentElement;
    n = n.parentElement
  ) {
    const cs = getComputedStyle(n);
    const y =
      /(auto|scroll|overlay)/.test(cs.overflowY) &&
      n.scrollHeight > n.clientHeight;
    const x =
      /(auto|scroll|overlay)/.test(cs.overflowX) &&
      n.scrollWidth > n.clientWidth;
    if (x || y) out.push(n);
  }
  return out;
}

function edgeSpeed(p: number, lo: number, hi: number): number {
  if (p < lo + EDGE_PX && p >= lo - EDGE_PX) {
    return -MAX_SCROLL_PX_PER_FRAME * Math.min(1, (lo + EDGE_PX - p) / EDGE_PX);
  }
  if (p > hi - EDGE_PX && p <= hi + EDGE_PX) {
    return (
      MAX_SCROLL_PX_PER_FRAME * Math.min(1, (p - (hi - EDGE_PX)) / EDGE_PX)
    );
  }
  return 0;
}

// Scrolls `el` toward the pointer `p` when it is near the element's edges;
// whether it scrolled.
function scrollElement(el: HTMLElement, p: Point, k: number): boolean {
  const r = el.getBoundingClientRect();
  if (
    p.x < r.left - EDGE_PX ||
    p.x > r.right + EDGE_PX ||
    p.y < r.top - EDGE_PX ||
    p.y > r.bottom + EDGE_PX
  ) {
    return false;
  }
  let scrolled = false;
  const vy = edgeSpeed(p.y, r.top, r.bottom) * k;
  const vx = edgeSpeed(p.x, r.left, r.right) * k;
  if (
    (vy < 0 && el.scrollTop > 0) ||
    (vy > 0 && el.scrollTop + el.clientHeight < el.scrollHeight)
  ) {
    el.scrollTop = el.scrollTop + vy;
    scrolled = true;
  }
  if (
    (vx < 0 && el.scrollLeft > 0) ||
    (vx > 0 && el.scrollLeft + el.clientWidth < el.scrollWidth)
  ) {
    el.scrollLeft = el.scrollLeft + vx;
    scrolled = true;
  }
  return scrolled;
}

// Scrolls the window toward the pointer near its edges; whether it scrolled.
function scrollWindow(p: Point, k: number): boolean {
  const doc = document.scrollingElement ?? document.documentElement;
  const vy = edgeSpeed(p.y, 0, window.innerHeight) * k;
  const vx = edgeSpeed(p.x, 0, window.innerWidth) * k;
  const dy =
    (vy < 0 && window.scrollY > 0) ||
    (vy > 0 && window.scrollY + window.innerHeight < doc.scrollHeight)
      ? vy
      : 0;
  const dx =
    (vx < 0 && window.scrollX > 0) ||
    (vx > 0 && window.scrollX + window.innerWidth < doc.scrollWidth)
      ? vx
      : 0;
  if (!dx && !dy) return false;
  window.scrollBy(dx, dy);
  return true;
}

/** The auto-scroll of one drag controller. */
export class AutoScroll {
  private handle: number | null = null;
  private last = 0;
  private scrollers: HTMLElement[] = [];
  private readonly rt: SceneRuntime;
  /** The pointer while a pointer drag is live; null otherwise. */
  private readonly pointer: () => Point | null;
  /** Called after a frame that scrolled, to resolve the target again. */
  private readonly scrolled: () => void;

  constructor(
    rt: SceneRuntime,
    pointer: () => Point | null,
    scrolled: () => void,
  ) {
    this.rt = rt;
    this.pointer = pointer;
    this.scrolled = scrolled;
  }

  /** Starts scrolling near edges, finding the scroll containers once. */
  start(): void {
    if (this.handle !== null) return;
    this.last = 0;
    // The scroll containers of every table, found once per drag.
    const found = new Set<HTMLElement>();
    for (const tb of this.rt.tables()) {
      for (const el of scrollParents(tb.root)) found.add(el);
    }
    this.scrollers = [...found];
    this.handle = this.rt.ticker().request(this.frame);
  }

  /** Stops scrolling. */
  stop(): void {
    if (this.handle !== null) this.rt.ticker().cancel(this.handle);
    this.handle = null;
  }

  private readonly frame = (t: number): void => {
    this.handle = null;
    const p = this.pointer();
    if (!p) return;
    const dt = this.last ? Math.min(50, Math.max(0, t - this.last)) : 16;
    this.last = t;
    const k = dt / 16;
    let scrolled = false;
    for (const el of this.scrollers) {
      if (scrollElement(el, p, k)) scrolled = true;
    }
    if (scrollWindow(p, k)) scrolled = true;
    if (scrolled) this.scrolled();
    this.handle = this.rt.ticker().request(this.frame);
  };
}
