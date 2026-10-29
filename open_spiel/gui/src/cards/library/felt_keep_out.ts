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

// The felt's keep-out: the info box's corner of the felt, as the cards see
// it (felt_table.tsx). The box takes only its own height in the felt's
// top-left corner, so the felt fills everything (a strip the box's width down
// the whole felt would leave an empty column under it). The cards lay out
// over the WHOLE felt,
// and only what the box actually overlaps yields room: a <Table> inside a
// FeltTable reads its keep-out here and keeps every card out of it (the
// scene's keepOut), and a game that lays its rows out itself asks for a
// row's keep-out with useFeltKeepOut(rowRef) or useFeltInset(rowRef).
//
// The keep-out is the box's visible rectangle grown by FELT_KEEP_GAP on its
// right and bottom (the felt between the box and a card), in the asking
// element's own px. When the box does not overlap the element at all (a
// phone, where the box stacks over the cards; a row under the box) it is
// null: nothing yields.
import {useLayoutEffect, useState, type RefObject} from 'react';
import {pixels} from '../../styles/pixels.ts';

/** Felt between the info box and the nearest card, px. */
export const FELT_KEEP_GAP = 8;

/** A keep-out rectangle in an element's own px (its top-left is 0, 0). */
export interface KeepOut {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** The info box of the felt `el` lies on (FeltTable's data-felt-info), or
 *  null off a felt. Found in the DOM, not through React context, so a game
 *  that draws its own FeltTable can ask from the component that renders it. */
export function feltInfoOf(el: Element): HTMLElement | null {
  const felt = el.closest('[data-felt-table]');
  return felt
    ? felt.querySelector<HTMLElement>(
        ':scope > .felt-table__grid > [data-felt-info]',
      )
    : null;
}

/** The keep-out of the box `info` over `el`, in el's px; null when the box
 *  does not overlap el. Rounded outward to whole px, so a sub-px layout
 *  change never re-lays anything out. */
export function keepOutOver(el: Element, info: Element): KeepOut | null {
  const a = el.getBoundingClientRect();
  const b = info.getBoundingClientRect();
  if (!(b.width > 0 && b.height > 0 && a.width > 0 && a.height > 0)) {
    return null;
  }
  // Only an element the box itself overlaps yields; the gap only widens
  // the room it yields (a phone's cards area starts just under the box).
  if (!(
    b.left < a.right &&
    a.left < b.right &&
    b.top < a.bottom &&
    a.top < b.bottom
  )) {
    return null;
  }
  const x = Math.floor(b.left - a.left);
  const y = Math.floor(b.top - a.top);
  return {
    x,
    y,
    width: Math.ceil(b.right - a.left) + FELT_KEEP_GAP - x,
    height: Math.ceil(b.bottom - a.top) + FELT_KEEP_GAP - y,
  };
}

function same(p: KeepOut | null, q: KeepOut | null): boolean {
  return (
    p === q ||
    (!!p &&
      !!q &&
      p.x === q.x &&
      p.y === q.y &&
      p.width === q.width &&
      p.height === q.height)
  );
}

/** The felt's info box over the element `ref` (in its px), or null: off a
 *  felt, on a phone, or where the box does not reach the element. Re-read
 *  when the box, the felt or the element changes size, and after every
 *  render (a row moves without changing size when a row above it grows). */
export function useFeltKeepOut(
  ref: RefObject<HTMLElement | null>,
): KeepOut | null {
  const [keep, setKeep] = useState<KeepOut | null>(null);
  const read = () => {
    const el = ref.current;
    const info = el ? feltInfoOf(el) : null;
    const next = el && info ? keepOutOver(el, info) : null;
    setKeep(k => (same(k, next) ? k : next));
  };
  useLayoutEffect(read);
  useLayoutEffect(() => {
    const el = ref.current;
    const info = el ? feltInfoOf(el) : null;
    if (!el || !info || typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(read);
    ro.observe(el);
    ro.observe(info);
    const felt = el.closest('[data-felt-table]');
    if (felt === null) throw new Error('useFeltKeepOut: no felt table');
    ro.observe(felt);
    return () => {
      ro.disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ref]);
  return keep;
}

/** How far a row laid out by a game itself (a flex row across the felt)
 *  stands in from each side to keep clear of the box: the box's right edge
 *  with its gap, px, on BOTH sides, so what the row centres stays on the
 *  felt's midline (the table's axis); 0 where the box does not reach the
 *  row (useFeltKeepOut), and the row uses the whole width. */
export function useFeltInset(ref: RefObject<HTMLElement | null>): number {
  const k = useFeltKeepOut(ref);
  return k ? Math.max(0, k.x + k.width) : 0;
}

/**
 * Whether a group a row lays out LEFT of a centred anchor (a trick, with a
 * stock or a talon beside it) would reach into the info box: the anchor's
 * left edge, less the row's column gap and the group's width (`widthOf`, as
 * it lies left of the anchor), against the row's inset (useFeltInset).
 * True: the row puts the group on the anchor's other side instead, so the
 * anchor keeps the table's axis and nothing goes under the box. Measured
 * after every render; the group's width must not depend on which side it
 * is on (or `widthOf` must give its width on the left), or the row would
 * swap back and forth.
 */
export function useSideYield(
  rowRef: RefObject<HTMLElement | null>,
  anchorRef: RefObject<HTMLElement | null>,
  widthOf: () => number,
  inset: number,
): boolean {
  const [swap, setSwap] = useState(false);
  // After every render, on purpose (the row's layout is the input): setSwap
  // with an unchanged value renders nothing, so this settles.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useLayoutEffect(() => {
    const row = rowRef.current;
    const anchor = anchorRef.current;
    if (!row || !anchor || !(inset > 0)) {
      setSwap(false);
      return;
    }
    const columnGap = pixels(getComputedStyle(row).columnGap);
    const gap = Number.isFinite(columnGap) ? columnGap : 0;
    const left =
      anchor.getBoundingClientRect().left -
      row.getBoundingClientRect().left -
      gap -
      widthOf();
    setSwap(left < inset - 0.5);
  });
  return inset > 0 && swap;
}
