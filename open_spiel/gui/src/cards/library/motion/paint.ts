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

// Painting cards to canvas and bitmaps. Owned by the motion module.
//
// Two paths, one result:
// - Deck.draw (bitmap/sprite decks): one synchronous drawImage after
//   Deck.preload has loaded the sheet.
// - The fallback, for SVG decks: renderFace/renderBack -> static markup ->
//   an SVG blob -> an Image, cached per side; drawCard then clips it to the
//   style's rounded rect and strokes the style's edge line, as the DOM
//   container does. The markup's CSS custom properties (the colour tokens
//   a deck paints with) are resolved first, since an SVG image cannot see
//   the page's stylesheets.
// A deck that can do neither fails loud, naming the deck.
import {renderToStaticMarkup} from 'react-dom/server';
import type {ReactNode} from 'react';
import type {Rect, TableCard} from '../core/identity.ts';
import {cardKey} from '../model.ts';
import {getDeck} from '../registry.ts';
import {
  backVariant,
  cardEdgeWidth,
  DEFAULT_CARD_STYLE,
  type CardStyle,
} from '../style.ts';
import type {Card, Deck} from '../types.ts';
import {definedFields} from '../core/defined.ts';

interface Raster {
  img: CanvasImageSource;
  width: number;
}

/** Keyed by deck + side (+ scheme and index scale for faces, variant for backs). */
const rasters = new Map<string, Raster>();
const loading = new Map<string, Promise<Raster>>();
/** Decks with Deck.draw whose preload resolved. */
const drawReady = new Set<string>();

/** Test-only: forget every cached raster and preload. */
export function resetPaintCacheForTest(): void {
  rasters.clear();
  loading.clear();
  drawReady.clear();
}

function styleOf(o?: {style?: CardStyle}): CardStyle {
  return o?.style ?? DEFAULT_CARD_STYLE;
}

/** The back variant to draw, as <PlayingCard> resolves it. */
function backOf(deck: Deck, style: CardStyle): string | undefined {
  const v = backVariant(style.back);
  return v !== undefined && deck.backs?.includes(v) ? v : undefined;
}

/** The side a TableCard shows: its card when known and face up, else null (a back). */
function sideOf(item: TableCard): Card | null {
  const given: unknown = item;
  if (
    given === null ||
    typeof given !== 'object' ||
    (given as {card?: unknown}).card === undefined
  ) {
    throw new Error(`paint: not a TableCard: ${JSON.stringify(item)}`);
  }
  const faceUp = item.faceUp ?? item.card !== null;
  if (faceUp && item.card === null) {
    throw new Error(`paint: card '${item.id}' is faceUp with card: null`);
  }
  return faceUp ? item.card : null;
}

function sideKey(deck: Deck, style: CardStyle, side: Card | null): string {
  return side === null
    ? `${deck.id}|back|${backOf(deck, style) ?? 'default'}`
    : `${deck.id}|${cardKey(side)}|${style.scheme}|${String(style.indexScale)}|${style.suitGlyphs}`;
}

function sideLabel(side: Card | null): string {
  return side === null ? 'back' : cardKey(side);
}

function dpr(): number {
  return typeof window !== 'undefined' && window.devicePixelRatio > 0
    ? window.devicePixelRatio
    : 1;
}

const SVG_NS = 'http://www.w3.org/2000/svg';

// A CSS var() reference: its name and an optional fallback, which may hold
// one level of parentheses.
const CSS_VAR = /var\(\s*(--[\w-]+)\s*(?:,[^()]*(?:\([^()]*\)[^()]*)*)?\)/g;

/** `text` with each `var(--name)` (or `var(--name, fallback)`) replaced by
 *  the value the document's root gives `--name`. The card tokens do not
 *  change with the theme, so a raster made with them stays right.
 *  Throws, naming `what` and the token, when a token has no value. */
export function resolveCssVars(text: string, what: string): string {
  if (!text.includes('var(')) return text;
  const root = getComputedStyle(document.documentElement);
  return text.replace(CSS_VAR, (_match, name: string) => {
    const value = root.getPropertyValue(name).trim();
    if (value === '') {
      throw new Error(`${what}: the colour token ${name} has no value`);
    }
    return value;
  });
}

/** The deck's art for one side as a standalone SVG document `w`x`h` px.
 *  Throws, naming the deck, when the art cannot be drawn as an SVG image:
 *  its root is not <svg>, or it references external resources or HTML
 *  (an SVG image loads neither). Exported for tests. */
export function standaloneSvg(
  deck: Deck,
  side: Card | null,
  style: CardStyle,
  w: number,
  h: number,
): string {
  const node: ReactNode =
    side === null
      ? deck.renderBack(definedFields({variant: backOf(deck, style)}))
      : deck.renderFace(side, {
          scheme: style.scheme,
          indexScale: style.indexScale,
          suitGlyphs: style.suitGlyphs,
        });
  const markup = resolveCssVars(
    renderToStaticMarkup(node).trim(),
    `deck '${deck.id}' (${sideLabel(side)})`,
  );
  const refuse = (why: string): Error =>
    new Error(
      `deck '${deck.id}' cannot be rasterised (${sideLabel(side)}): ${why}; implement Deck.draw for it`,
    );
  const open = /^<svg\b[^>]*>/.exec(markup);
  if (!open) throw refuse('its art is not a single <svg> element');
  if (/<foreignObject\b/i.test(markup)) {
    throw refuse('its SVG embeds HTML (a foreignObject element)');
  }
  const ext = /\s(?:xlink:)?href="(?!#|data:)([^"]*)"/.exec(markup);
  if (ext) {
    throw refuse(
      `its SVG references an external resource (${(ext[1] ?? '').slice(0, 80)})`,
    );
  }
  if (/url\((?!\s*['"]?#)/.test(markup)) {
    throw refuse('its SVG references an external url()');
  }
  let root = open[0]
    .replace(/\swidth="[^"]*"/, '')
    .replace(/\sheight="[^"]*"/, '');
  root = root.replace(
    /^<svg\b/,
    `<svg width="${String(w)}" height="${String(h)}"`,
  );
  if (!/\sxmlns="/.test(root)) {
    root = root.replace(/^<svg\b/, `<svg xmlns="${SVG_NS}"`);
  }
  if (markup.includes('xlink:') && !/\sxmlns:xlink="/.test(root)) {
    root = root.replace(
      /^<svg\b/,
      '<svg xmlns:xlink="http://www.w3.org/1999/xlink"',
    );
  }
  return root + markup.slice(open[0].length);
}

function loadSvgImage(
  svg: string,
  deckId: string,
  label: string,
): Promise<HTMLImageElement> {
  if (
    typeof Image === 'undefined' ||
    typeof URL === 'undefined' ||
    typeof URL.createObjectURL !== 'function'
  ) {
    return Promise.reject(
      new Error(
        `deck '${deckId}' cannot be rasterised here: no Image/URL.createObjectURL (not a browser)`,
      ),
    );
  }
  const url = URL.createObjectURL(new Blob([svg], {type: 'image/svg+xml'}));
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      if (!(img.naturalWidth > 0 || img.width > 0)) {
        reject(
          new Error(
            `deck '${deckId}' cannot be rasterised (${label}): its SVG decoded to an empty image`,
          ),
        );
        return;
      }
      resolve(img);
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(
        new Error(
          `deck '${deckId}' cannot be rasterised (${label}): the browser failed to decode its SVG`,
        ),
      );
    };
    img.src = url;
  });
}

function rasterise(
  deck: Deck,
  style: CardStyle,
  side: Card | null,
  pxWidth: number,
): Promise<Raster> {
  const key = sideKey(deck, style, side);
  const have = rasters.get(key);
  if (have && have.width >= pxWidth) return Promise.resolve(have);
  const pending = loading.get(key);
  const width = Math.max(pxWidth, have?.width ?? 0);
  const p = (pending ?? Promise.resolve(null as Raster | null)).then(prev => {
    if (prev && prev.width >= pxWidth) return prev;
    const w = Math.ceil(width);
    const h = Math.ceil(width / deck.aspect);
    const svg = standaloneSvg(deck, side, style, w, h);
    return loadSvgImage(svg, deck.id, sideLabel(side)).then(img => {
      const r: Raster = {img, width: w};
      const cur = rasters.get(key);
      if (!cur || cur.width < r.width) {
        rasters.set(key, r);
        return r;
      }
      return cur;
    });
  });
  loading.set(key, p);
  const clear = (): void => {
    if (loading.get(key) === p) loading.delete(key);
  };
  p.then(clear, clear);
  return p;
}

/** Load the faces and backs needed to draw these items synchronously. Throws
 *  (rejects) naming the deck if it cannot be rasterised. `width` is the CSS
 *  width the cards will be drawn at; rasters are made at width x devicePixelRatio. */
export async function preloadCards(
  items: readonly TableCard[],
  o?: {deck?: string; style?: CardStyle; width: number},
): Promise<void> {
  const given: unknown = items;
  if (!Array.isArray(given)) {
    throw new Error('preloadCards: items must be an array of TableCards');
  }
  if (!o || !(o.width > 0) || !Number.isFinite(o.width)) {
    throw new Error(
      `preloadCards: o.width must be a CSS width > 0, got ${JSON.stringify(o?.width)}`,
    );
  }
  const style = styleOf(o);
  const deck = getDeck(o.deck ?? style.deck);
  const sides = new Map<string, Card | null>();
  for (const it of items) {
    const side = sideOf(it);
    sides.set(sideKey(deck, style, side), side);
  }
  if (deck.draw) {
    if (!drawReady.has(deck.id)) {
      if (deck.preload) await deck.preload();
      drawReady.add(deck.id);
    }
    return;
  }
  const px = o.width * dpr();
  await Promise.all(
    [...sides.values()].map(side => rasterise(deck, style, side, px)),
  );
}

function roundRectPath(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
): void {
  const rr = Math.max(0, Math.min(r, w / 2, h / 2));
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.lineTo(x + w - rr, y);
  ctx.arcTo(x + w, y, x + w, y + rr, rr);
  ctx.lineTo(x + w, y + h - rr);
  ctx.arcTo(x + w, y + h, x + w - rr, y + h, rr);
  ctx.lineTo(x + rr, y + h);
  ctx.arcTo(x, y + h, x, y + h - rr, rr);
  ctx.lineTo(x, y + rr);
  ctx.arcTo(x, y, x + rr, y, rr);
  ctx.closePath();
}

/** Synchronous after preloadCards. Uses Deck.draw when present, otherwise the
 *  cached raster of renderFace/renderBack (renderToStaticMarkup -> SVG blob ->
 *  Image), clipped to the style's corner radius with its edge line. Throws if
 *  not preloaded. r is in the ctx's current units; `rotate` is degrees
 *  clockwise about the rect's centre. */
export function drawCard(
  ctx: CanvasRenderingContext2D,
  item: TableCard,
  r: Rect & {rotate?: number},
  o?: {deck?: string; style?: CardStyle},
): void {
  const givenCtx: unknown = ctx;
  if (!givenCtx) throw new Error('drawCard: no canvas context');
  for (const k of ['x', 'y', 'width', 'height'] as const) {
    if (!Number.isFinite(r[k])) {
      throw new Error(
        `drawCard: r.${k} must be finite, got ${JSON.stringify(r)}`,
      );
    }
  }
  const style = styleOf(o);
  const deck = getDeck(o?.deck ?? style.deck);
  const side = sideOf(item);
  const rot = r.rotate ?? 0;
  ctx.save();
  try {
    if (rot) {
      const cx = r.x + r.width / 2;
      const cy = r.y + r.height / 2;
      ctx.translate(cx, cy);
      ctx.rotate((rot * Math.PI) / 180);
      ctx.translate(-cx, -cy);
    }
    if (deck.draw) {
      if (!drawReady.has(deck.id)) {
        throw new Error(
          `drawCard: deck '${deck.id}' is not preloaded; await preloadCards(...) first`,
        );
      }
      deck.draw(
        ctx,
        side,
        {x: r.x, y: r.y, width: r.width, height: r.height},
        {
          scheme: style.scheme,
          indexScale: style.indexScale,
          ...definedFields({back: backOf(deck, style)}),
        },
      );
      return;
    }
    const raster = rasters.get(sideKey(deck, style, side));
    if (!raster) {
      throw new Error(
        `drawCard: '${sideLabel(side)}' of deck '${deck.id}' is not preloaded; await preloadCards(...) first`,
      );
    }
    const radius = Math.max(0, style.radius) * r.width;
    ctx.save();
    roundRectPath(ctx, r.x, r.y, r.width, r.height, radius);
    ctx.clip();
    ctx.drawImage(raster.img, r.x, r.y, r.width, r.height);
    ctx.restore();
    const edge = deck.drawsOwnOutline ? 0 : cardEdgeWidth(style, r.width);
    if (edge > 0) {
      ctx.lineWidth = edge;
      ctx.strokeStyle = resolveCssVars(style.outline.color, 'the card outline');
      roundRectPath(
        ctx,
        r.x + edge / 2,
        r.y + edge / 2,
        r.width - edge,
        r.height - edge,
        Math.max(0, radius - edge / 2),
      );
      ctx.stroke();
    }
  } finally {
    ctx.restore();
  }
}

/** A standalone bitmap (drag images, SVG <image>, foreign scenes). */
export async function cardImage(
  item: TableCard,
  width: number,
  o?: {deck?: string; style?: CardStyle; dpr?: number},
): Promise<ImageBitmap> {
  if (!(width > 0) || !Number.isFinite(width)) {
    throw new Error(`cardImage: width must be > 0, got ${String(width)}`);
  }
  const ratio = o?.dpr ?? dpr();
  if (!(ratio > 0) || !Number.isFinite(ratio)) {
    throw new Error(`cardImage: dpr must be > 0, got ${String(ratio)}`);
  }
  const style = styleOf(o);
  const deck = getDeck(o?.deck ?? style.deck);
  if (typeof createImageBitmap !== 'function') {
    throw new Error('cardImage: createImageBitmap is not available here');
  }
  await preloadCards([item], {
    deck: deck.id,
    style,
    width: (width * ratio) / dpr(),
  });
  const height = width / deck.aspect;
  const pw = Math.max(1, Math.round(width * ratio));
  const ph = Math.max(1, Math.round(height * ratio));
  const canvas: HTMLCanvasElement | OffscreenCanvas =
    typeof OffscreenCanvas === 'function'
      ? new OffscreenCanvas(pw, ph)
      : Object.assign(document.createElement('canvas'), {
          width: pw,
          height: ph,
        });
  const ctx = canvas.getContext('2d') as CanvasRenderingContext2D | null;
  if (!ctx) throw new Error('cardImage: no 2D canvas context here');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  drawCard(ctx, item, {x: 0, y: 0, width, height}, {deck: deck.id, style});
  return createImageBitmap(canvas);
}

/** DPR-correct canvas sizing: sets the backing store to css size x devicePixelRatio,
 *  sets the CSS size, and returns a ctx scaled so 1 unit = 1 CSS px. Call again
 *  on resize (resizing the backing store clears it). */
export function sizeCanvas(
  c: HTMLCanvasElement,
  box: {width: number; height: number},
): CanvasRenderingContext2D {
  const given: unknown = c;
  if (!given) throw new Error('sizeCanvas: no canvas');
  if (
    !(box.width >= 0) ||
    !(box.height >= 0) ||
    !Number.isFinite(box.width) ||
    !Number.isFinite(box.height)
  ) {
    throw new Error(
      `sizeCanvas: box must have finite width and height >= 0, got ${JSON.stringify(box)}`,
    );
  }
  const ratio = dpr();
  c.width = Math.max(1, Math.round(box.width * ratio));
  c.height = Math.max(1, Math.round(box.height * ratio));
  c.style.width = `${String(box.width)}px`;
  c.style.height = `${String(box.height)}px`;
  const ctx = c.getContext('2d');
  if (!ctx) throw new Error('sizeCanvas: the canvas has no 2D context');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return ctx;
}
