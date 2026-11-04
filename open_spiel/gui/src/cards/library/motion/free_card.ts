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

// A card lifted out of the layout into a free layer (MotionService.free): a
// snapshot node the game poses directly, which can flip to its other side
// and fly home when it is disposed.
import type {Box, CardId, TableCard} from '../core/identity.ts';
import {cardSideElement} from './sides.ts';
import {
  inlinePixels,
  stripA11y,
  styleFor,
  tableScale,
  type FlipSide,
  type MotionHost,
} from './motion_util.ts';

/** 'table': px relative to the Table root; 'viewport': CSS px of the window. */
export type Frame = 'table' | 'viewport';

/** Where a free card is drawn, in its frame. */
export interface Pose {
  /** Top-left of the card, in the FreeCard's frame. */
  x: number;
  y: number;
  /** Degrees. */
  rotate?: number;
  /** 1 = launch size. */
  scale?: number;
  /** 0..1: 0 shows the launch side, 1 the other side (rotateY; the face swaps at 0.5). */
  flip?: number;
  z?: number;
  /** 0..1 (default 1): the card fades out toward 0 (a trick collect). */
  opacity?: number;
}

/** A card in the free layer, posed by the game. */
export interface FreeCard {
  readonly id: CardId;
  /** Snapshot at the moment of freeing. */
  readonly item: TableCard;
  readonly frame: Frame;
  /** MEASURED box at the moment of freeing, in `frame`. */
  readonly launch: Box;
  /** Direct DOM transform write. No React render. Safe to call every frame. */
  set(p: Pose): void;
  /** Removes the free node. If the id is still in `items`, it reappears in the
   *  layout; with { home: true } it FLIPs from its current pose to its slot. */
  dispose(o?: {home?: boolean}): void;
}

/** What a free card needs of the motion service that made it. */
export interface FreeCardHost {
  host: MotionHost;
  free: Map<CardId, FreeCard>;
  pushHeld(): void;
  flyFrom(
    id: CardId,
    from: {cx: number; cy: number; rotate: number; width: number},
    o?: {oldSide?: FlipSide},
  ): Promise<void>;
}

/** Where and how a free card is made: its element and its launch box. */
export interface FreeCardSpec {
  id: CardId;
  el: HTMLElement;
  tableId: string;
  item: TableCard;
  frame: Frame;
  launch: Box;
  /** Viewport px per table px when the frame is the viewport, else 1. */
  scale: number;
  layer: HTMLElement;
}

interface Side {
  outer: HTMLElement;
  inner: HTMLElement;
}

interface Inner {
  width: number;
  height: number;
  /** The scale from the card's own px to the launch box. */
  k: number;
}

interface FreePose {
  x: number;
  y: number;
  rotate: number;
  scale: number;
  flip: number;
}

function sideBox(d: Document, inner: Inner, mirror: boolean): Side {
  const outer = d.createElement('div');
  Object.assign(outer.style, {
    position: 'absolute',
    left: '0',
    top: '0',
    width: '100%',
    height: '100%',
  });
  if (mirror) outer.style.transform = 'scaleX(-1)';
  const box = d.createElement('div');
  Object.assign(box.style, {
    position: 'absolute',
    left: '0',
    top: '0',
    width: `${String(inner.width)}px`,
    height: `${String(inner.height)}px`,
    transformOrigin: '0 0',
  });
  if (inner.k !== 1) box.style.transform = `scale(${String(inner.k)})`;
  outer.appendChild(box);
  return {outer, inner: box};
}

function freeNode(d: Document, id: CardId, launch: Box): HTMLElement {
  const node = d.createElement('div');
  node.setAttribute('data-free-card', id);
  node.setAttribute('aria-hidden', 'true');
  Object.assign(node.style, {
    position: 'absolute',
    left: '0',
    top: '0',
    width: `${String(launch.width)}px`,
    height: `${String(launch.height)}px`,
    transformOrigin: '50% 50%',
    willChange: 'transform',
    pointerEvents: 'none',
  });
  return node;
}

// Side A: what the layout showed, cloned (a snapshot: the free node is
// not React's, so no render can reparent or unmount it).
function launchSide(d: Document, el: HTMLElement, inner: Inner): Side {
  const a = sideBox(d, inner, false);
  a.outer.setAttribute('data-free-side', 'launch');
  for (const child of [...el.children]) {
    if (child.hasAttribute('hidden') || child.hasAttribute('data-flip-old')) {
      continue;
    }
    const c = child.cloneNode(true) as Element;
    stripA11y(c);
    a.inner.appendChild(c);
  }
  return a;
}

// A pose a game passes to set(), checked and clamped.
function checkedPose(id: CardId, launch: Box, p: Pose): FreePose {
  const given: unknown = p;
  if (!given || !Number.isFinite(p.x) || !Number.isFinite(p.y)) {
    throw new Error(
      `FreeCard '${id}': pose x and y must be finite, got ${JSON.stringify(p)}`,
    );
  }
  const rotate = p.rotate ?? launch.rotate;
  const scale = p.scale ?? 1;
  const flip = p.flip ?? 0;
  if (
    !Number.isFinite(rotate) ||
    !Number.isFinite(scale) ||
    !Number.isFinite(flip)
  ) {
    throw new Error(
      `FreeCard '${id}': pose rotate, scale and flip must be finite, got ${JSON.stringify(p)}`,
    );
  }
  return {x: p.x, y: p.y, rotate, scale, flip: Math.min(1, Math.max(0, flip))};
}

// The pieces of one free card that its methods share.
interface FreeState {
  spec: FreeCardSpec;
  node: HTMLElement;
  a: Side;
  b: HTMLElement | null;
  inner: Inner;
  launchFace: boolean;
  pose: FreePose;
  disposed: boolean;
}

// The other side, made the first time the card turns past half way: a back
// behind a face; the face behind a back when it is known.
function ensureOther(h: FreeCardHost, s: FreeState): HTMLElement {
  if (s.b) return s.b;
  const {item, el} = s.spec;
  const otherUp = !s.launchFace && item.card !== null;
  const sb = sideBox(document, s.inner, true);
  sb.outer.setAttribute('data-free-side', 'other');
  sb.inner.appendChild(
    cardSideElement(
      document,
      styleFor(h.host, el),
      otherUp ? item.card : null,
      otherUp,
      s.inner.width,
    ),
  );
  sb.outer.style.visibility = 'hidden';
  s.node.appendChild(sb.outer);
  s.b = sb.outer;
  return s.b;
}

function writePose(h: FreeCardHost, s: FreeState): void {
  const {pose, node} = s;
  let tf = `translate(${String(pose.x)}px, ${String(pose.y)}px) rotate(${String(pose.rotate)}deg) scale(${String(pose.scale)})`;
  if (pose.flip > 0) {
    tf += ` perspective(${String(Math.round(4 * s.spec.launch.width))}px) rotateY(${String(pose.flip * 180)}deg)`;
  }
  node.style.transform = tf;
  const other = pose.flip >= 0.5;
  if (other) ensureOther(h, s);
  s.a.outer.style.visibility = other ? 'hidden' : '';
  if (s.b) s.b.style.visibility = other ? '' : 'hidden';
}

function setPose(h: FreeCardHost, s: FreeState, p: Pose): void {
  const {id, launch} = s.spec;
  if (s.disposed) throw new Error(`FreeCard '${id}': set() after dispose()`);
  s.pose = checkedPose(id, launch, p);
  if (p.z !== undefined) s.node.style.zIndex = String(p.z);
  if (p.opacity !== undefined) {
    if (!Number.isFinite(p.opacity)) {
      throw new Error(
        `FreeCard '${id}': pose opacity must be finite, got ${String(p.opacity)}`,
      );
    }
    s.node.style.opacity = String(Math.min(1, Math.max(0, p.opacity)));
  }
  writePose(h, s);
}

function dispose(h: FreeCardHost, s: FreeState, home: boolean): void {
  if (s.disposed) return;
  const {id, frame, tableId, launch, item} = s.spec;
  s.disposed = true;
  s.node.remove();
  h.free.delete(id);
  h.pushHeld();
  if (!home) return;
  const {pose} = s;
  // Current visual pose -> viewport.
  let cx = pose.x + launch.width / 2;
  let cy = pose.y + launch.height / 2;
  let width = launch.width * pose.scale;
  if (frame === 'table') {
    const t = h.host.tables().find(x => x.tableId === tableId);
    if (!t) return;
    const c = h.host.tableToClient(tableId, {x: cx, y: cy});
    cx = c.x;
    cy = c.y;
    width *= tableScale(t);
  }
  const known = item.card !== null;
  const shown: FlipSide =
    pose.flip >= 0.5
      ? {
          card: !s.launchFace && known ? item.card : null,
          faceUp: !s.launchFace && known,
        }
      : {card: s.launchFace ? item.card : null, faceUp: s.launchFace};
  const targetUp = h.host.element(id)?.getAttribute('data-face-up') === 'true';
  void h.flyFrom(
    id,
    {cx, cy, rotate: pose.rotate, width},
    shown.faceUp !== targetUp ? {oldSide: shown} : {},
  );
}

/** Makes a free card over `spec.el`, adds it to the layer and to `h.free`. */
export function makeFreeCard(h: FreeCardHost, spec: FreeCardSpec): FreeCard {
  if (typeof document === 'undefined') {
    throw new Error('motion needs a DOM document');
  }
  const {id, el, launch, scale} = spec;
  const width = inlinePixels(el.style.width) || launch.width / scale;
  const height = inlinePixels(el.style.height) || launch.height / scale;
  const inner: Inner = {
    width,
    height,
    k: width > 0 ? launch.width / width : 1,
  };
  const node = freeNode(document, id, launch);
  const a = launchSide(document, el, inner);
  node.appendChild(a.outer);
  spec.layer.appendChild(node);
  const s: FreeState = {
    spec,
    node,
    a,
    b: null,
    inner,
    launchFace: el.getAttribute('data-face-up') === 'true',
    pose: {x: launch.x, y: launch.y, rotate: launch.rotate, scale: 1, flip: 0},
    disposed: false,
  };
  writePose(h, s);
  const fc: FreeCard = {
    id,
    item: spec.item,
    frame: spec.frame,
    launch,
    set: p => {
      setPose(h, s, p);
    },
    dispose: o => {
      dispose(h, s, o?.home === true);
    },
  };
  h.free.set(id, fc);
  return fc;
}
