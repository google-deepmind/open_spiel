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

// Shrinks the text of a fixed box until it fits the box, never under the
// platform's text floor (textFloor.ts).
//
// A card table sizes its regions in card widths (a gin board's stats beside
// the stock); the text a game puts in one (a turn line, its notes, a block
// reason) does not scale with the cards. Wrapping alone does not keep it
// inside: a long turn line would be drawn over the stock. So the box's font is a CSS multiplier this hook lowers, by
// bisection, until every line box of its text lies inside the box, or the
// font reaches its floor: TEXT_FLOOR_PX unless the caller names a lower one.
import {useLayoutEffect, type RefObject} from 'react';
import {pixels} from './pixels.ts';
import {TEXT_FLOOR_PX} from './text_floor.ts';

/** The CSS custom property the multiplier is written to (0 < scale <= 1). */
export const FIT_SCALE_VAR = '--fit-scale';
const STEPS = 7;
/** Sub-pixel rounding of text boxes against the box. */
const EPS = 0.5;

/**
 * Whether every line box of text inside `el` lies inside `el`'s own box.
 * Without a layout engine (jsdom: no Range.getClientRects) there is nothing to
 * measure and it reports true.
 */
export function textFits(el: HTMLElement): boolean {
  const range = el.ownerDocument.createRange();
  if (typeof range.getClientRects !== 'function') return true;
  const box = el.getBoundingClientRect();
  range.selectNodeContents(el);
  for (const r of Array.from(range.getClientRects())) {
    if (r.width === 0 && r.height === 0) continue;
    if (
      r.left < box.left - EPS ||
      r.right > box.right + EPS ||
      r.top < box.top - EPS ||
      r.bottom > box.bottom + EPS
    ) {
      return false;
    }
  }
  return true;
}

/**
 * The largest multiplier in [floor, 1] at which `el`'s text fits it, written
 * to FIT_SCALE_VAR; `floor` is the multiplier at which its font is
 * `minFontPx` (1 when it is already at or under that: this never enlarges
 * text, and never shrinks it under `minFontPx`).
 */
export function fitFontScale(
  el: HTMLElement,
  minFontPx: number = TEXT_FLOOR_PX,
): number {
  if (!(minFontPx > 0)) {
    throw new Error(
      `fitFontScale: minFontPx must be positive, got ${String(minFontPx)}`,
    );
  }
  el.style.setProperty(FIT_SCALE_VAR, '1');
  if (textFits(el)) return 1;
  const base = pixels(getComputedStyle(el).fontSize);
  if (!(base > 0)) {
    throw new Error(
      `fitFontScale: font-size '${getComputedStyle(el).fontSize}' is not a positive length`,
    );
  }
  let lo = Math.min(1, minFontPx / base);
  let hi = 1;
  for (let i = 0; i < STEPS && hi - lo > 1e-3; i++) {
    const mid = (lo + hi) / 2;
    el.style.setProperty(FIT_SCALE_VAR, String(mid));
    if (textFits(el)) lo = mid;
    else hi = mid;
  }
  el.style.setProperty(FIT_SCALE_VAR, String(lo));
  return lo;
}

/**
 * Keeps FIT_SCALE_VAR on `ref`'s element fitted (fitFontScale) as the element
 * resizes, its text changes, or a web font loads. The element's CSS sizes its
 * font as `calc(<base> * var(--fit-scale, 1))` and has a fixed box (its text
 * overflows it rather than growing it). At the floor the text may still
 * overflow: the caller's layout owns that case.
 */
export function useFitFontScale(
  ref: RefObject<HTMLElement | null>,
  minFontPx: number = TEXT_FLOOR_PX,
): void {
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) {
      throw new Error('useFitFontScale: the ref is not attached to an element');
    }
    const fit = () => {
      fitFontScale(el, minFontPx);
    };
    fit();
    let live = true;
    // A web font that loads after the first fit changes every line's width.
    void el.ownerDocument.fonts.ready.then(() => {
      if (live) fit();
    });
    // New text refits. The fit writes only el's own style attribute,
    // which is not observed.
    const mo =
      typeof MutationObserver === 'undefined'
        ? null
        : new MutationObserver(fit);
    mo?.observe(el, {childList: true, characterData: true, subtree: true});
    // So does a new box size (a new layout, card size or panel size); the
    // fit itself never resizes the box.
    let last = '';
    const ro =
      typeof ResizeObserver === 'undefined'
        ? null
        : new ResizeObserver(([entry]) => {
            if (entry === undefined) return;
            const size = `${String(entry.contentRect.width)}x${String(entry.contentRect.height)}`;
            if (size === last) return;
            last = size;
            fit();
          });
    ro?.observe(el);
    return () => {
      live = false;
      mo?.disconnect();
      ro?.disconnect();
    };
  }, [ref, minFontPx]);
}
