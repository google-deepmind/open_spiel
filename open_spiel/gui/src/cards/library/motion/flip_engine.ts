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

// The FLIP engine: one laid-out card animated from where it was to where it
// is, by transform writes in the card's own slot frame. The element stays in
// its place in the layout; the transform inverts the move and plays it back.
import type {CardId} from '../core/identity.ts';
import {cardSideElement} from './sides.ts';
import {
  FLIGHT_Z,
  angleDelta,
  inlinePixels,
  styleFor,
  type AnimateRequest,
  type Flight,
  type MotionHost,
} from './motion_util.ts';

// The transform that draws the card at progress `e` of the request, and the
// card's pose there.
function transformAt(
  req: AnimateRequest,
  e: number,
  base: string,
  innerW: number,
): {tf: string; pose: Flight['pose']} {
  const {from, to} = req;
  const rad = (-req.slotRotate * Math.PI) / 180;
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);
  const cx = from.cx + (to.cx - from.cx) * e;
  const cy = from.cy + (to.cy - from.cy) * e;
  const rotate = from.rotate + angleDelta(from.rotate, to.rotate) * e;
  const width = from.width + (to.width - from.width) * e;
  const dx = cx - to.cx;
  const dy = cy - to.cy;
  const lx = dx * cos - dy * sin;
  const ly = dx * sin + dy * cos;
  const dr = angleDelta(to.rotate, rotate);
  const ds = to.width > 0 ? width / to.width : 1;
  let tf = `translate(${String(lx)}px, ${String(ly)}px) rotate(${String(dr)}deg) scale(${String(ds)})`;
  if (base) tf += ` ${base}`;
  if (req.oldSide) {
    // First half: the old side turns away (0 -> 90deg); second half:
    // the new side turns in (-90 -> 0deg).
    const theta = e < 0.5 ? e * 180 : (e - 1) * 180;
    tf += ` perspective(${String(Math.round(4 * innerW))}px) rotateY(${String(theta)}deg)`;
  }
  return {tf, pose: {cx, cy, rotate, width}};
}

// The element's own transform and z-index, taking over from a flight in
// progress on it; stops that flight.
function takeOver(
  flights: Map<CardId, Flight>,
  req: AnimateRequest,
): {base: string; zPrev: string; prev: Flight | undefined} {
  const {el} = req;
  const prev = flights.get(req.id);
  let base = el.style.transform;
  let zPrev = el.style.zIndex;
  if (prev) {
    if (prev.el === el) {
      // React re-rendered the element's own transform iff it differs from ours.
      base =
        el.style.transform === prev.lastWritten
          ? prev.base
          : el.style.transform;
      zPrev = el.style.zIndex === FLIGHT_Z ? prev.zPrev : el.style.zIndex;
    }
    prev.stop(false);
  }
  return {base, zPrev, prev};
}

// The old side, drawn over the card for the first half of a flip.
function flipOverlay(
  host: MotionHost,
  req: AnimateRequest,
  innerW: number,
): HTMLElement | null {
  if (!req.oldSide) return null;
  if (typeof document === 'undefined') {
    throw new Error('motion needs a DOM document');
  }
  const overlay = cardSideElement(
    document,
    styleFor(host, req.el),
    req.oldSide.card,
    req.oldSide.faceUp,
    innerW,
  );
  overlay.setAttribute('data-flip-old', '');
  overlay.style.zIndex = '2';
  req.el.appendChild(overlay);
  return overlay;
}

/** Plays one FLIP on the host's ticker; resolves when it ends or is stopped.
 *  A flight already on the card is taken over from its current pose. */
export function animate(
  host: MotionHost,
  flights: Map<CardId, Flight>,
  req: AnimateRequest,
): Promise<void> {
  const {id, el} = req;
  const {base, zPrev, prev} = takeOver(flights, req);
  if (req.durationMs <= 0) {
    if (prev?.el === el) {
      el.style.transform = base;
      el.style.zIndex = zPrev;
      el.removeAttribute('data-motion');
    }
    return Promise.resolve();
  }
  const ticker = host.ticker();
  const start = ticker.now();
  const innerW = inlinePixels(el.style.width) || req.to.width;
  let overlay = flipOverlay(host, req, innerW);
  let handle = 0;
  let resolve: () => void = () => {
    // Replaced by the promise's resolver below.
  };
  const done = new Promise<void>(r => {
    resolve = r;
  });
  const f: Flight = {
    id,
    el,
    tableId: req.tableId,
    base,
    lastWritten: '',
    zPrev,
    pose: {...req.from},
    stop: restore => {
      ticker.cancel(handle);
      overlay?.remove();
      overlay = null;
      if (restore) {
        if (el.style.transform === f.lastWritten) el.style.transform = f.base;
        if (el.style.zIndex === FLIGHT_Z) el.style.zIndex = f.zPrev;
        el.removeAttribute('data-motion');
      }
      if (flights.get(id) === f) flights.delete(id);
      resolve();
    },
  };
  flights.set(id, f);
  el.setAttribute('data-motion', 'flip');
  const write = (p: number): void => {
    const e = req.easing(p);
    const at = transformAt(req, e, base, innerW);
    f.pose = at.pose;
    if (req.oldSide && e >= 0.5 && overlay) {
      overlay.remove();
      overlay = null;
    }
    el.style.transform = at.tf;
    f.lastWritten = el.style.transform;
  };
  // Invert before the browser paints: the card starts where it was.
  write(0);
  if (req.raise) el.style.zIndex = FLIGHT_Z;
  const tick = (now: number): void => {
    if (flights.get(id) !== f) return;
    const p = (now - start - req.delayMs) / req.durationMs;
    if (p >= 1) {
      f.stop(true);
      return;
    }
    write(Math.max(0, p));
    handle = ticker.request(tick);
  };
  handle = ticker.request(tick);
  return done;
}
