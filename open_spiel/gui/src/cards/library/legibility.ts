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

// DOM legibility check: measure what the browser actually drew and report any
// face-up card whose index box is covered. The pure maths lives in
// pileGeometry.ts (findCoveredIndexes); this reads rendered <Pile> cards, so a
// layout bug outside the geometry — two columns overlapping, a badge on an
// index — is caught too.
//
// Each card says which index it keeps visible (data-index-corner):
// fan-left/fan-up cards the bottom-right twin, which cornerIndex refuses for a
// deck without one, so a twinless deck in such a pile fails loud here. A
// quarter-turned card (data-rotated) is measured on its turned box; any
// other rotation (an arc hand, a rotated seat) by its CSS transform.
import {getDeck} from './registry.ts';
import {itemAt} from './checked.ts';
import {cornerIndex, quarterTurnIndex} from './index_scale.ts';
import {
  findCoveredIndexes,
  indexQuad,
  indexRect,
  type PlacedCard,
  type Point,
  type Rect,
} from './pile_geometry.ts';

/** A card's index box as drawn, for the legibility check. */
export interface LegibilityBox {
  /** Index box relative to the container, px, before its card's rotation. */
  rect: Rect;
  /** The card's rotation (degrees clockwise) and centre: draw `rect` rotated
   *  by `rotate` about `pivot`. Absent for unrotated cards. */
  rotate?: number | undefined;
  pivot?: Point | undefined;
  covered: boolean;
  deck: string;
  /** Covered, inside a <Table> slot that declares coveredByDesign. */
  excused?: boolean | undefined;
}

/** Rotation (degrees clockwise) and scale of one element's own CSS transform.
 *  Translations do not matter (the bounding box already reflects them); a
 *  skew would make the card a parallelogram this check cannot model, so it
 *  fails loud rather than measuring the wrong shape. */
export function parseTransform(t: string): {deg: number; scale: number} {
  if (!t || t === 'none') return {deg: 0, scale: 1};
  const m = /^matrix(3d)?\(([^)]*)\)$/.exec(t.trim());
  if (m) {
    const v = (m[2] ?? '').split(',').map(Number);
    const [a, b, c, d] = (m[1] ? [0, 1, 4, 5] : [0, 1, 2, 3]).map(i =>
      itemAt(v, i),
    ) as [number, number, number, number];
    const sx = Math.hypot(a, b);
    const sy = Math.hypot(c, d);
    if (
      Math.abs(a * c + b * d) > 1e-6 * sx * sy ||
      Math.abs(sx - sy) > 1e-6 * sx
    ) {
      throw new Error(
        `legibility check cannot model a skewed or non-uniform transform: ${t}`,
      );
    }
    return {deg: (Math.atan2(b, a) * 180) / Math.PI, scale: sx};
  }
  // Environments that do not resolve transforms to a matrix (jsdom) return
  // the authored functions.
  let deg = 0;
  for (const [, fn = '', arg = ''] of t.matchAll(
    /([a-zA-Z0-9]+)\(([^)]*)\)/g,
  )) {
    if (fn.startsWith('translate')) continue;
    if (fn !== 'rotate' && fn !== 'rotateZ') {
      throw new Error(`legibility check cannot model transform ${fn}(): ${t}`);
    }
    const a = /^([-+]?[\d.]+(?:e[-+]?\d+)?)(deg|rad|turn)$/i.exec(arg.trim());
    if (!a) {
      throw new Error(`legibility check cannot read rotation '${arg}'`);
    }
    const unit = (a[2] ?? '').toLowerCase();
    const k = unit === 'deg' ? 1 : unit === 'rad' ? 180 / Math.PI : 360;
    deg += Number(a[1]) * k;
  }
  return {deg, scale: 1};
}

function inside(inner: Rect, outer: Rect): boolean {
  const eps = 0.5;
  return (
    inner.x >= outer.x - eps &&
    inner.y >= outer.y - eps &&
    inner.x + inner.width <= outer.x + outer.width + eps &&
    inner.y + inner.height <= outer.y + outer.height + eps
  );
}

/** What a legibility check found. */
export interface LegibilityReport {
  /** Face-up cards whose index was checked. */
  checked: number;
  covered: number;
  /** Face-up cards whose index is scrolled out of an overflowing pile or row:
   *  reachable by scrolling, so neither checked nor covered. */
  scrolledOut: number;
  /** Face-up cards whose index is covered in a slot that declares it is
   *  covered by design (a card under two others in the row below): not checked. */
  excused: number;
  boxes: LegibilityBox[];
}

const CARD = '[data-pile-card]';
/** Rotations this close to a multiple of 360° are unrotated. */
const ANGLE_EPS = 1e-6;
/** A positional hole is opaque: it covers what it is painted over. */
const HOLE = '[data-pile-hole]';
/** Painted above cards (z-index 1, or a <Table> caption or cue drawn after
 *  earlier slots), so they can cover an index too. */
const OVERLAY =
  '[data-pile-count],[data-pile-overflow],[data-slot-caption],[data-slot-cue],[data-region-id][data-occludes="true"]';

const ZERO = {top: 0, left: 0, width: 0, height: 0};

// Measures cards inside one container: boxes relative to it, the clip rects
// of scrolling ancestors and cumulative transforms, each computed once per
// check (getComputedStyle per card per ancestor dominated the cost).
class Probe {
  private readonly container: HTMLElement;
  private readonly origin: DOMRect;
  private readonly clipOf = new Map<Element, Rect | null>();
  private readonly ownTransform = new Map<
    Element,
    {deg: number; scale: number}
  >();

  constructor(container: HTMLElement) {
    this.container = container;
    this.origin = container.getBoundingClientRect();
  }

  rel(el: Element): Rect {
    const r = el.getBoundingClientRect();
    return {
      x: r.left - this.origin.left,
      y: r.top - this.origin.top,
      width: r.width,
      height: r.height,
    };
  }

  private clipRect(a: Element): Rect | null {
    let r = this.clipOf.get(a);
    if (r === undefined) {
      const cs = getComputedStyle(a);
      const clips = [cs.overflowX, cs.overflowY].some(
        o => o === 'auto' || o === 'scroll' || o === 'hidden',
      );
      r = clips ? this.rel(a) : null;
      this.clipOf.set(a, r);
    }
    return r;
  }

  /** An index outside the viewport of a scrolling ancestor (an overflowing
   *  pile or row) is reachable by scrolling: neither checked nor covered. */
  clippedByScroll(el: Element, box: Rect): boolean {
    for (
      let a = el.parentElement;
      a && a !== this.container;
      a = a.parentElement
    ) {
      const clip = this.clipRect(a);
      if (clip && !inside(box, clip)) return true;
    }
    return false;
  }

  private transformOf(el: Element): {deg: number; scale: number} {
    let t = this.ownTransform.get(el);
    if (!t) {
      t = parseTransform(getComputedStyle(el).transform);
      this.ownTransform.set(el, t);
    }
    return t;
  }

  /** Cumulative rotation and scale of an element within the container: a
   *  rotated hand card or a rotated seat turns the index box with it, and a
   *  bounding box would both mislocate the index and over-report cover. */
  cumulative(el: Element): {deg: number; scale: number} {
    let deg = 0;
    let scale = 1;
    for (
      let a: Element | null = el;
      a && a !== this.container;
      a = a.parentElement
    ) {
      const t = this.transformOf(a);
      deg += t.deg;
      scale *= t.scale;
    }
    return {deg, scale};
  }

  /** A card or hole element, placed as it paints. */
  place(el: Element): PlacedCard {
    if (el.matches(HOLE)) {
      return {rect: this.rel(el), faceUp: false, index: ZERO};
    }
    const deckId = el.getAttribute('data-deck');
    if (!deckId) throw new Error('pile card without data-deck');
    // The index box the card was drawn with: the deck's, at the style's
    // indexScale, in the corner its pile leaves visible.
    const corner = el.getAttribute('data-index-corner') ?? 'top-left';
    if (corner !== 'top-left' && corner !== 'bottom-right') {
      throw new Error(`pile card with data-index-corner=${corner}`);
    }
    const cornerBox = cornerIndex(
      getDeck(deckId),
      Number(el.getAttribute('data-index-scale') ?? '1'),
      corner,
    );
    const rotated = el.getAttribute('data-rotated') === 'true';
    if (rotated && corner !== 'top-left') {
      throw new Error('a rotated pile card keeps its top-left index');
    }
    const faceUp = el.getAttribute('data-face-up') === 'true';
    // Measure the rendered card, not its slot: a transform on the card (a
    // selection lift, an animation) moves it, and that is what paints. A
    // quarter-turned base card (data-rotated) is the exception: its wrapper
    // is the turned box, so it is measured there with the index
    // quarter-turned, and its inner rotate(-90deg) is not counted again.
    // Transforms above the wrapper (a rotated seat) still are.
    const drawn = rotated ? el : (el.firstElementChild ?? el);
    const index = rotated ? quarterTurnIndex(cornerBox) : cornerBox;
    const box = this.rel(drawn);
    const {deg, scale} = this.cumulative(drawn);
    const turn = ((deg % 360) + 360) % 360;
    if (turn < ANGLE_EPS || 360 - turn < ANGLE_EPS) {
      return {rect: box, faceUp, index};
    }
    // A rotated rectangle's bounding box shares its centre; its size is the
    // layout size times the accumulated scale.
    const w = (drawn as HTMLElement).offsetWidth * scale;
    const h = (drawn as HTMLElement).offsetHeight * scale;
    const cx = box.x + box.width / 2;
    const cy = box.y + box.height / 2;
    return {
      rect: {x: cx - w / 2, y: cy - h / 2, width: w, height: h},
      rotate: deg,
      faceUp,
      index,
    };
  }
}

// The bounds of a placed card's index quad.
function quadBounds(p: PlacedCard): Rect {
  const quad = indexQuad(p);
  const xs = quad.map(q => q.x);
  const ys = quad.map(q => q.y);
  return {
    x: Math.min(...xs),
    y: Math.min(...ys),
    width: Math.max(...xs) - Math.min(...xs),
    height: Math.max(...ys) - Math.min(...ys),
  };
}

/** Checks that every face-up card's index in `container` is in view. */
export function checkLegibility(container: HTMLElement): LegibilityReport {
  const probe = new Probe(container);
  const cardEls = [...container.querySelectorAll(`${CARD},${HOLE}`)];
  const placed: PlacedCard[] = cardEls.map(el => probe.place(el));
  // Overlays paint after every card; model them as face-down cards at the end.
  for (const el of container.querySelectorAll(OVERLAY)) {
    placed.push({rect: probe.rel(el), faceUp: false, index: ZERO});
  }
  const coveredSet = new Set(findCoveredIndexes(placed));
  const boxes: LegibilityBox[] = [];
  let scrolledOut = 0;
  const excusedBoxes: LegibilityBox[] = [];
  placed.forEach((p, i) => {
    const el = cardEls[i];
    if (el === undefined || !p.faceUp) return;
    if (probe.clippedByScroll(el, quadBounds(p))) {
      scrolledOut++;
      return;
    }
    const box = indexRect(p.rect, p.index);
    const deck = el.getAttribute('data-deck') ?? '';
    const covered = coveredSet.has(i);
    const turn = p.rotate
      ? {
          rotate: p.rotate,
          pivot: {
            x: p.rect.x + p.rect.width / 2,
            y: p.rect.y + p.rect.height / 2,
          },
        }
      : {};
    const byDesign = el.closest('[data-covered-by-design="true"]');
    if (covered && byDesign && container.contains(byDesign)) {
      excusedBoxes.push({
        rect: box,
        ...turn,
        covered: false,
        excused: true,
        deck,
      });
      return;
    }
    boxes.push({rect: box, ...turn, covered, deck});
  });
  return {
    checked: boxes.length,
    covered: boxes.filter(b => b.covered).length,
    scrolledOut,
    excused: excusedBoxes.length,
    boxes: [...boxes, ...excusedBoxes],
  };
}
