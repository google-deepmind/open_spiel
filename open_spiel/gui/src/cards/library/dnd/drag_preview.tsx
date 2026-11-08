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

// The drag preview, owned by the dnd module.
//
// The dragged cards are hidden in place (tag 'dragging', layout space kept)
// and a preview follows the pointer in a viewport-fixed layer. The preview is
// a DOM clone of each dragged card's library wrapper, so whatever drew the
// card (the library renderer, a renderCard override, a deck's own SVG) is what
// moves. It is built once at drag start, and every later update is a direct
// `transform` write on one node: the per-frame path makes no React render.
//
// Geometry is the computed Scene: each clone is centred on its
// placement's box mapped to the viewport by the table's affine map, and turned
// by its slot's rotation, so the preview starts exactly where the card was
// drawn and keeps the source fan's offsets.
import type {Box, CardId, Point} from '../core/identity.ts';
import type {Placement, Scene} from '../core/scene.ts';
import {parseEasing} from '../motion/easing.ts';
import {motionDuration} from '../motion/service.ts';
import type {Ticker} from '../motion/ticker.ts';
import type {SceneRuntime} from '../react/runtime.ts';
import {pixels} from '../../../styles/pixels.ts';

// ---------------------------------------------------------------------------
// Styles the dnd module needs on the page: hide the dragged and held cards in
// place (visibility, so the layout keeps their space), the preview layer, and
// the drop marks. Injected once per document.

const STYLE_ID = 'cards-dnd-styles';

/** The dnd module's styles, injected once per document. Colours are the
 *  theme's tokens; the preview's lift is a drop shadow under its card. */
export const DND_CSS = [
  // The wrapper stays visible (a focused card must stay focusable during a
  // keyboard drag); what it draws is hidden.
  '[data-card-id].pcard--t-dragging>*,[data-card-id].pcard--t-held>*{visibility:hidden}',
  '[data-dnd-preview]{position:fixed;left:0;top:0;width:0;height:0;overflow:visible;pointer-events:none;z-index:2147483000}',
  '[data-dnd-preview] [data-preview-card]{filter:drop-shadow(0 6px 10px var(--card-drag-shadow))}',
  '[data-drop-marks]{position:absolute;left:0;top:0;width:0;height:0;overflow:visible;pointer-events:none;z-index:900}',
  '[data-drop-mark]{position:absolute;box-sizing:border-box;pointer-events:none;border:2px dashed var(--dnd-legal,var(--color-success));border-radius:6px}',
  '[data-drop-mark][data-drop-over]{border-style:solid;border-color:var(--dnd-over,var(--color-success));box-shadow:0 0 12px 2px var(--dnd-over,var(--color-success))}',
  '[data-drop-mark][data-insert]{border:0;border-radius:2px;background:var(--dnd-legal,var(--color-success))}',
  '[data-drop-mark][data-insert][data-drop-over]{background:var(--dnd-over,var(--color-success));box-shadow:0 0 10px 2px var(--dnd-over,var(--color-success))}',
].join('\n');

/** Adds the dnd styles to `doc` once. */
export function ensureDndStyles(doc: Document = document): void {
  if (doc.getElementById(STYLE_ID)) return;
  const el = doc.createElement('style');
  el.id = STYLE_ID;
  el.textContent = DND_CSS;
  doc.head.appendChild(el);
}

// ---------------------------------------------------------------------------
// Easing and tweens on the scene's ticker (so tests drive them with
// manualTicker, and nothing here touches React).

/** A CSS easing as a function of 0..1: the motion module's parser, so dnd and
 *  motion ease identically and an invalid CardStyle.motion.easing throws. */
export function easingFn(css: string): (k: number) => number {
  return parseEasing(css);
}

/** A running tween on the scene's ticker. */
export interface Tween {
  /** Stop where it is; `onEnd` does not run. */
  cancel(): void;
  /** Jump to the end and run `onEnd` now. */
  finish(): void;
  readonly done: Promise<void>;
}

/** Runs step(eased 0..1) on the ticker over durationMs, then onEnd. A duration
 *  of 0 (or less) runs step(1) and onEnd synchronously. */
export function tween(
  ticker: Ticker,
  durationMs: number,
  easing: string,
  step: (k: number) => void,
  onEnd: () => void,
): Tween {
  let resolve: () => void = () => {
    // Replaced by the promise's resolver below.
  };
  const done = new Promise<void>(r => {
    resolve = r;
  });
  let over = false;
  const end = (): void => {
    if (over) return;
    over = true;
    step(1);
    onEnd();
    resolve();
  };
  if (!(durationMs > 0)) {
    end();
    const ended = (): void => {
      // The tween has already ended.
    };
    return {cancel: ended, finish: ended, done};
  }
  const ease = easingFn(easing);
  const t0 = ticker.now();
  let handle: number | null = null;
  const frame = (t: number): void => {
    handle = null;
    if (over) return;
    const k = Math.min(1, Math.max(0, (t - t0) / durationMs));
    if (k >= 1) {
      end();
      return;
    }
    step(ease(k));
    handle = ticker.request(frame);
  };
  handle = ticker.request(frame);
  return {
    cancel: () => {
      if (over) return;
      over = true;
      if (handle !== null) ticker.cancel(handle);
      resolve();
    },
    finish: () => {
      if (handle !== null) ticker.cancel(handle);
      handle = null;
      end();
    },
    done,
  };
}

/** The motion module's FLIP duration: 0 under reduced motion when respected. */
export {motionDuration};

// ---------------------------------------------------------------------------
// Table px <-> viewport, through the runtime's affine map.

/** A table's map to the viewport: its origin and scale. */
export interface Affine {
  left: number;
  top: number;
  scale: number;
}

/** The affine map of table `tableId` now. */
export function affineOf(rt: SceneRuntime, tableId: string): Affine {
  const o = rt.tableToClient(tableId, {x: 0, y: 0});
  const u = rt.tableToClient(tableId, {x: 1, y: 0});
  const scale = Math.hypot(u.x - o.x, u.y - o.y) || 1;
  return {left: o.x, top: o.y, scale};
}

/** A table point in viewport px. */
export function toClient(a: Affine, p: Point): Point {
  return {x: a.left + p.x * a.scale, y: a.top + p.y * a.scale};
}

/** A viewport point in table px. */
export function toTable(a: Affine, p: Point): Point {
  return {x: (p.x - a.left) / a.scale, y: (p.y - a.top) / a.scale};
}

/** The corners of a box, clockwise from its unrotated top-left. */
export function quad(b: Box): Point[] {
  const cx = b.x + b.width / 2;
  const cy = b.y + b.height / 2;
  const a = ((b.rotate || 0) * Math.PI) / 180;
  const cos = Math.cos(a);
  const sin = Math.sin(a);
  const corners: ReadonlyArray<readonly [number, number]> = [
    [b.x, b.y],
    [b.x + b.width, b.y],
    [b.x + b.width, b.y + b.height],
    [b.x, b.y + b.height],
  ];
  return corners.map(([x, y]) => ({
    x: cx + (x - cx) * cos - (y - cy) * sin,
    y: cy + (x - cx) * sin + (y - cy) * cos,
  }));
}

/** An axis-aligned box by its edges. */
export interface Bounds {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

/** The bounds round `pts`. */
export function boundsOf(pts: readonly Point[]): Bounds {
  let left = Infinity,
    top = Infinity,
    right = -Infinity,
    bottom = -Infinity;
  for (const p of pts) {
    left = Math.min(left, p.x);
    top = Math.min(top, p.y);
    right = Math.max(right, p.x);
    bottom = Math.max(bottom, p.y);
  }
  return {left, top, right, bottom};
}

/** The area two bounds share. */
export function overlapArea(a: Bounds, b: Bounds): number {
  const w = Math.min(a.right, b.right) - Math.max(a.left, b.left);
  const h = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
  return w > 0 && h > 0 ? w * h : 0;
}

/** A box's client-space bounds. */
export function clientBounds(a: Affine, b: Box): Bounds {
  return boundsOf(quad(b).map(p => toClient(a, p)));
}

/** The rotation of the slot an element is drawn in (data-rotate on its slot). */
export function slotRotation(el: Element): number {
  const r = el.closest('[data-slot-id]')?.getAttribute('data-rotate');
  const n = r ? Number(r) : 0;
  return Number.isFinite(n) ? n : 0;
}

// ---------------------------------------------------------------------------
// The preview.

/** One card of a drag preview. */
export interface PreviewCard {
  readonly id: CardId;
  /** Positioned, turned and scaled holder of the clone. */
  readonly holder: HTMLElement;
  /** Viewport centre at creation (offset 0). */
  readonly centre: Point;
  /** Unscaled size of the card wrapper, px. */
  readonly width: number;
  readonly height: number;
  readonly slotRotate: number;
  readonly scale: number;
}

/** The preview that follows the pointer during a drag. */
export interface DragPreview {
  readonly root: HTMLElement;
  readonly cards: readonly PreviewCard[];
  /** The lead card's preview (null when the lead is not drawn). */
  readonly lead: PreviewCard | null;
  /** Current translation of the whole preview from its start, viewport px. */
  offset(): Point;
  /** Direct transform write. No React render. */
  setOffset(p: Point): void;
  setHeld(on: boolean): void;
  /** Viewport centre of a card's preview now. */
  centreOf(id: CardId): Point | null;
  remove(): void;
}

const STRIP_ATTRS = [
  'id',
  'tabindex',
  'role',
  'title',
  'aria-label',
  'aria-pressed',
  'aria-disabled',
  'aria-describedby',
  'aria-hidden',
  'data-slot',
  'data-position',
];
const TRANSIENT = [
  'pcard--t-hover',
  'pcard--t-focus',
  'pcard--t-drop-legal',
  'pcard--t-drop-over',
  'pcard--t-dragging',
  'pcard--t-held',
];

function scrubClone(root: HTMLElement): void {
  const all = [root, ...root.querySelectorAll<HTMLElement>('*')];
  for (const el of all) {
    for (const a of STRIP_ATTRS) el.removeAttribute(a);
    const id = el.getAttribute('data-card-id');
    if (id !== null) {
      el.removeAttribute('data-card-id');
      el.setAttribute('data-preview-card', id);
    }
    for (const c of TRANSIENT) el.classList.remove(c);
  }
  root.setAttribute('aria-hidden', 'true');
}

/** Copies the CSS custom properties in effect on `from` (deck and style
 *  variables set by providers) onto `to`, so the clone outside the table draws
 *  the same. Browsers enumerate custom properties on computed styles. */
function copyCustomProperties(from: Element | null, to: HTMLElement): void {
  if (!from || typeof getComputedStyle !== 'function') return;
  const cs = getComputedStyle(from);
  for (const n of Array.from(cs)) {
    if (n.startsWith('--')) to.style.setProperty(n, cs.getPropertyValue(n));
  }
}

function px(v: string, fallback: number): number {
  const n = pixels(v);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

// The preview's root and the group that moves, appended to nothing yet.
function previewRoot(
  doc: Document,
  tableRoot: HTMLElement,
): {root: HTMLElement; group: HTMLElement} {
  const root = doc.createElement('div');
  root.setAttribute('data-dnd-preview', '');
  root.setAttribute('aria-hidden', 'true');
  const scope = tableRoot.getAttribute('data-cs');
  if (scope) root.setAttribute('data-cs', scope);
  copyCustomProperties(tableRoot, root);
  const cw = tableRoot.style.getPropertyValue('--card-w');
  if (cw) root.style.setProperty('--card-w', cw);
  const group = doc.createElement('div');
  group.style.position = 'absolute';
  group.style.left = '0';
  group.style.top = '0';
  group.style.willChange = 'transform';
  root.appendChild(group);
  return {root, group};
}

// One card's preview: a scrubbed clone of its wrapper in a holder placed
// over where the card is drawn.
function previewCard(
  doc: Document,
  a: Affine,
  id: CardId,
  p: Placement,
  el: HTMLElement,
): PreviewCard {
  const width = px(el.style.width, p.box.width);
  const height = px(el.style.height, p.box.height);
  const slotRotate = slotRotation(el);
  const centre = toClient(a, {
    x: p.box.x + p.box.width / 2,
    y: p.box.y + p.box.height / 2,
  });
  const holder = doc.createElement('div');
  holder.setAttribute('data-preview-holder', id);
  Object.assign(holder.style, {
    position: 'absolute',
    left: `${String(centre.x - width / 2)}px`,
    top: `${String(centre.y - height / 2)}px`,
    width: `${String(width)}px`,
    height: `${String(height)}px`,
    transformOrigin: '50% 50%',
    transform:
      `${slotRotate ? `rotate(${String(slotRotate)}deg) ` : ''}${a.scale !== 1 ? `scale(${String(a.scale)})` : ''}`.trim() ||
      '',
  });
  copyCustomProperties(el.parentElement, holder);
  const clone = el.cloneNode(true) as HTMLElement;
  scrubClone(clone);
  clone.style.left = '0px';
  clone.style.top = '0px';
  clone.style.visibility = 'visible';
  clone.style.zIndex = '';
  holder.appendChild(clone);
  return {id, holder, centre, width, height, slotRotate, scale: a.scale};
}

/** Builds the preview of the dragged cards `ids` and adds it to the page. */
export function createDragPreview(o: {
  rt: SceneRuntime;
  tableId: string;
  scene: Scene;
  ids: readonly CardId[];
  lead: CardId;
  doc?: Document;
}): DragPreview {
  const doc = o.doc ?? document;
  ensureDndStyles(doc);
  const a = affineOf(o.rt, o.tableId);
  const {root, group} = previewRoot(doc, o.rt.table(o.tableId).root);
  const placed = o.ids
    .map(id => ({id, p: o.scene.byId(id), el: o.rt.element(id)}))
    .filter(
      (x): x is {id: CardId; p: Placement; el: HTMLElement} => !!x.p && !!x.el,
    )
    .sort((x, y) => x.p.z - y.p.z);
  const cards = placed.map(({id, p, el}) => {
    const card = previewCard(doc, a, id, p, el);
    group.appendChild(card.holder);
    return card;
  });
  doc.body.appendChild(root);

  let off: Point = {x: 0, y: 0};
  return {
    root,
    cards,
    lead: cards.find(c => c.id === o.lead) ?? null,
    offset: () => off,
    setOffset: p => {
      off = {x: p.x, y: p.y};
      group.style.transform = `translate3d(${String(p.x)}px, ${String(p.y)}px, 0)`;
    },
    setHeld: on => {
      if (on) root.setAttribute('data-held', 'true');
      else root.removeAttribute('data-held');
      for (const c of root.querySelectorAll('[data-preview-card]')) {
        c.classList.toggle('pcard--t-held', on);
      }
    },
    centreOf: id => {
      const c = cards.find(x => x.id === id);
      return c ? {x: c.centre.x + off.x, y: c.centre.y + off.y} : null;
    },
    remove: () => {
      root.remove();
    },
  };
}
