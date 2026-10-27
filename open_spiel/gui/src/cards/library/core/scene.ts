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

// layoutScene: the computed geometry of a table, in table px. It is
// authoritative for layout, hit-testing, keyboard navigation, drop targets
// and legibility; the React <Table> renders exactly this. Pure: no React, no
// DOM.
//
// It computes what the table, pile and hand components each need (slot
// boxes, pile and hand geometry, the fitted card width, the variant), with
// the same inputs, so a table drawn from a Scene matches what those
// components draw.
import type {IndexGeometry} from '../types.ts';
import {tagStyles} from '../style.ts';
import {
  minSlotScale,
  selectTableVariant,
  widenTableLayout,
  type TableFit,
  type TableSource,
  type TableSpec,
} from '../table_spec.ts';
import {deckIndex} from '../index_scale.ts';
import {MIN_CARD_WIDTH} from '../card_widths.ts';
import {assertItems, isFaceUp, isToken, type Rect} from './identity.ts';
import type {TagStyle} from './tags.ts';
import {defined} from './defined.ts';
import {
  boxBounds,
  boxQuad,
  norm,
  rectsMeet,
  slotBottomOf,
  toTableBox,
} from './scene_geometry.ts';
import {layoutSlot} from './scene_slot.ts';
import {placeRegion} from './scene_region.ts';
import {paintOrder, sceneOf} from './scene_build.ts';
import {fitLayout, fitTableCardWidth, resolveSceneSpec} from './scene_fit.ts';
import {blockShifts, leadHoles, packBlocks} from './scene_blocks.ts';
import {
  VERTICAL,
  sceneInternals,
  setSceneInternals,
  type Items,
  type Layout,
  type Scene,
  type SceneEnv,
  type SlotItems,
  type SlotLayout,
  type WidthBounds,
} from './scene_types.ts';

export {
  OVERFLOW_MARKER_PX,
  sceneInternals,
  type Placement,
  type RegionPlacement,
  type Scene,
  type SceneEnv,
  type SceneInternals,
  type SlotBox,
  type SlotEntry,
  type SlotLayout,
} from './scene_types.ts';
export {
  boxBounds,
  boxQuad,
  localCardGeometry,
  turnedIndex,
} from './scene_geometry.ts';
export {handWrapOf, slotBox} from './scene_slot_box.ts';
export {fitLayout, fitTableCardWidth, resolveSceneSpec} from './scene_fit.ts';

// What laying out one table shares: the spec, its layouts and the fit.
interface Plan {
  spec: TableSpec;
  layouts: readonly Layout[];
  items: Items;
  env: SceneEnv;
  fit: TableFit;
  bounds: WidthBounds;
}

function layoutAt(plan: Plan, i: number): Layout {
  return defined(plan.layouts[i], `table layout ${String(i)}`);
}

function variantOf(
  spec: TableSpec,
  i: number,
): NonNullable<TableSpec['variants']>[number] {
  return defined(spec.variants?.[i - 1], `table variant ${String(i - 1)}`);
}

function minDrawnOf(spec: TableSpec, i: number): number | undefined {
  return i === 0
    ? spec.minDrawnCardWidth
    : variantOf(spec, i).minDrawnCardWidth;
}

// layoutScene's choice of layout: the one selectTableVariant picks, again
// without it when it draws under its minDrawnCardWidth.
function chooseAndFit(plan: Plan): Scene {
  const {spec, env, fit, bounds} = plan;
  // A 'contain' table in a container with no height (0: it gives none)
  // fits the width alone, and picks its layout as 'width' does.
  const select = (exclude?: ReadonlySet<number>) =>
    selectTableVariant(
      spec,
      env.box.width,
      l => fitLayout(l, env.box, fit, bounds),
      fit === 'contain' && env.box.height > 0 ? 'contain' : 'width',
      env.cardScale ?? 1,
      exclude,
    );
  const chosen = select();
  const scene = fitChosen(plan, chosen.layout, chosen.index);
  // The choice predicts each layout's card width from its spec; drawn, it
  // can come out smaller (the shrink below). A layout drawn under its
  // minDrawnCardWidth is chosen again without it, where another qualifies.
  const minDrawn = minDrawnOf(spec, chosen.index);
  if (minDrawn === undefined || scene.cardWidth >= minDrawn) return scene;
  const again = select(new Set([chosen.index]));
  if (again.index === chosen.index) return scene;
  const retry = fitChosen(plan, again.layout, again.index);
  const retryMin = minDrawnOf(spec, again.index);
  return retryMin === undefined || retry.cardWidth >= retryMin ? retry : scene;
}

/** Pure. The geometry of a table: every slot's box and every item's
 *  placement. The React <Table> renders exactly this. */
export function layoutScene(
  source: TableSource,
  items: Items,
  env: SceneEnv,
): Scene {
  const {deck} = env;
  if (!(deck.aspect > 0)) {
    throw new Error(
      `layoutScene: deck aspect must be > 0, got ${String(deck.aspect)}`,
    );
  }
  const spec = resolveSceneSpec(source, env);
  const layouts: Layout[] = [spec, ...(spec.variants ?? []).map(v => v.spec)];
  const plan: Plan = {
    spec,
    layouts,
    items,
    env,
    fit: env.fit ?? 'width',
    bounds: {min: env.minCardWidth, max: env.maxCardWidth},
  };
  const hold =
    env.cardWidth === undefined &&
    env.hold &&
    env.hold.variant >= 0 &&
    env.hold.variant < layouts.length
      ? env.hold
      : undefined;
  // An explicit card width pins the base spec: variants answer to the
  // container.
  if (env.cardWidth !== undefined) {
    return drawLayout(plan, layoutAt(plan, 0), 0, env.cardWidth, false);
  }
  if (hold) {
    return drawLayout(
      plan,
      layoutAt(plan, hold.variant),
      hold.variant,
      hold.cardWidth,
      hold.overflow,
    );
  }
  return keepClear(plan, chooseAndFit(plan));
}

/** SceneEnv.keepOut in table px: the container's rectangle less where
 *  <Table> puts the surface in it (centred across unless it scrolls; with
 *  'contain', centred down too). */
function keepInTable(
  env: SceneEnv,
  width: number,
  height: number,
  overflow: boolean,
): Rect | null {
  const k = env.keepOut;
  if (!k || !(env.box.width > 0) || !(k.width > 0 && k.height > 0)) return null;
  const ox = overflow ? 0 : (env.box.width - width) / 2;
  const oy =
    !overflow && env.fit === 'contain' && env.box.height > 0
      ? (env.box.height - height) / 2
      : 0;
  return {x: k.x - ox, y: k.y - oy, width: k.width, height: k.height};
}

/** No card of the scene lies in the keep-out (SceneEnv.keepOut), nor a
 *  region on the table that names or counts something (a seat's name, a
 *  badge): one that occludes and is narrower than half the table. A band
 *  across the table (a plate line whose text is centred on the axis) is not
 *  held to it. */
export function sceneKeepsClear(scene: Scene, env: SceneEnv): boolean {
  const k = keepInTable(
    env,
    scene.width,
    scene.height,
    sceneInternals(scene).overflow,
  );
  if (!k) return true;
  if (scene.placements.some(p => rectsMeet(boxBounds(p.box), k))) return false;
  return !scene.regions.some(
    r =>
      r.occludes &&
      r.box.width < scene.width / 2 &&
      rectsMeet(boxBounds(r.box), k),
  );
}

function floorOf(plan: Plan, i: number): number {
  return Math.ceil(
    (plan.bounds.min ?? MIN_CARD_WIDTH) / minSlotScale(layoutAt(plan, i)),
  );
}

/** Layout i at the largest card width, at most its fitted one, that keeps
 *  clear (null: none). */
function clearAt(plan: Plan, i: number, fitted: Scene): Scene | null {
  const {env} = plan;
  if (sceneInternals(fitted).overflow) return null;
  if (sceneKeepsClear(fitted, env)) return fitted;
  const floor = floorOf(plan, i);
  if (fitted.cardWidth <= floor) return null;
  const atFloor = drawLayout(plan, layoutAt(plan, i), i, floor, false);
  if (!sceneKeepsClear(atFloor, env)) return null;
  // Clear at the floor, not at the fitted width: the largest clear width between.
  let lo = atFloor;
  let hi = fitted.cardWidth;
  while (hi - lo.cardWidth > 1) {
    const mid = Math.floor((lo.cardWidth + hi) / 2);
    const sc = drawLayout(plan, layoutAt(plan, i), i, mid, false);
    if (sceneKeepsClear(sc, env)) lo = sc;
    else hi = mid;
  }
  return lo;
}

// Of the layouts' fitted scenes, the largest card for its cost that keeps
// clear ('contain'), or null.
function bestClear(
  plan: Plan,
  fitted: ReadonlyArray<Scene | null>,
): Scene | null {
  const {spec} = plan;
  const costOf = (i: number): number =>
    i === 0 ? 1 : (variantOf(spec, i).cost ?? 1);
  const order = fitted
    .map((f, i) => ({f, i}))
    .filter((c): c is {f: Scene; i: number} => c.f !== null)
    .sort(
      (a, b) =>
        b.f.cardWidth / costOf(b.i) - a.f.cardWidth / costOf(a.i) || a.i - b.i,
    );
  let best: Scene | null = null;
  let bestScore = -Infinity;
  for (const {f, i} of order) {
    if (f.cardWidth / costOf(i) <= bestScore) break;
    const sc = clearAt(plan, i, f);
    if (sc && sc.cardWidth / costOf(i) > bestScore) {
      best = sc;
      bestScore = sc.cardWidth / costOf(i);
    }
  }
  return best;
}

/** The fitted scene, or, when a card of it lies in the keep-out, the layout
 *  and card width that keep every card clear (SceneEnv.keepOut): each
 *  layout at the largest card width, at most its fitted one, at which it
 *  keeps clear, and of those the largest card for its cost ('contain'; for
 *  'width', the first layout in the spec's order that keeps clear). A
 *  layout whose fitted card, for its cost, is no larger than the best found
 *  is not drawn. None keeps clear even at the smallest card: the fitted
 *  scene drawn at the smallest. */
function keepClear(plan: Plan, scene: Scene): Scene {
  const {spec, env, layouts} = plan;
  if (!env.keepOut || !(env.box.width > 0) || sceneKeepsClear(scene, env)) {
    return scene;
  }
  const gated = (i: number): boolean => {
    const minWidth = i === 0 ? spec.minWidth : variantOf(spec, i).minWidth;
    return minWidth !== undefined && env.box.width < minWidth;
  };
  const index = scene.variant + 1;
  const fitted = layouts.map((l, i) =>
    gated(i) ? null : i === index ? scene : fitChosen(plan, l, i),
  );
  const atFloor = (): Scene =>
    drawLayout(plan, layoutAt(plan, index), index, floorOf(plan, index), false);
  if (plan.fit !== 'contain') {
    for (const f of fitted) {
      if (f && !sceneInternals(f).overflow && sceneKeepsClear(f, env)) return f;
    }
    return clearAt(plan, index, scene) ?? atFloor();
  }
  return bestClear(plan, fitted) ?? atFloor();
}

/** A chosen layout fitted to the box, then drawn at env.cardScale
 *  (layoutScene). */
function fitChosen(plan: Plan, layout: Layout, index: number): Scene {
  const {env, fit, bounds} = plan;
  const fitted = fitTableCardWidth(layout, env.box, fit, bounds);
  let scene = drawLayout(
    plan,
    layout,
    index,
    fitted.cardWidth,
    fitted.overflow,
  );
  // The fit is taken from the spec's size; what is drawn can reach past it
  // (a pile that grows, a region below the last slot, px paddings that do
  // not scale). A table that fits must lie inside its box, so it is drawn
  // smaller until it does, or scrolls at the minimum card width.
  if (!fitted.overflow && env.box.width > 0) {
    const floor = Math.ceil(
      (bounds.min ?? MIN_CARD_WIDTH) / minSlotScale(layout),
    );
    const limitH = fit === 'contain' && env.box.height > 0;
    const over = (sc: Scene): number =>
      Math.max(
        sc.width / env.box.width,
        limitH ? sc.height / env.box.height : 0,
      );
    // Each step draws at least 1 px smaller, so this ends by the floor.
    while (over(scene) > 1 + 1e-9) {
      const next = Math.min(
        scene.cardWidth - 1,
        Math.floor(scene.cardWidth / over(scene)),
      );
      if (next < floor) {
        scene = drawLayout(plan, layout, index, floor, true);
        break;
      }
      scene = drawLayout(plan, layout, index, next, false);
    }
  }
  return scaleScene(plan, scene, layout, index);
}

/** The fitted scene drawn at env.cardScale x its card width (layoutScene):
 *  it overflows (scrolls) when the larger table no longer lies in its box. */
function scaleScene(
  plan: Plan,
  fitted: Scene,
  layout: Layout,
  index: number,
): Scene {
  const {env} = plan;
  const k = env.cardScale ?? 1;
  if (!(k > 0 && Number.isFinite(k))) {
    throw new Error(
      `layoutScene: cardScale must be a finite number > 0, got ${String(k)}`,
    );
  }
  if (k === 1) return fitted;
  // Smaller never draws the smallest slot below minCardWidth: the table
  // card is floored as the fit loop floors it (min / the smallest slot
  // scale), and never above the width it fitted at.
  const floor =
    k < 1
      ? Math.min(
          fitted.cardWidth,
          Math.ceil((plan.bounds.min ?? MIN_CARD_WIDTH) / minSlotScale(layout)),
        )
      : 0;
  const cardWidth = Math.max(floor, Math.round(fitted.cardWidth * k));
  if (cardWidth === fitted.cardWidth) return fitted;
  const drawn = drawLayout(plan, layout, index, cardWidth, false);
  const limitH = env.fit === 'contain' && env.box.height > 0;
  const over =
    env.box.width > 0 &&
    (drawn.width > env.box.width + 1e-6 ||
      (limitH && drawn.height > env.box.height + 1e-6));
  return over || sceneInternals(fitted).overflow
    ? drawLayout(plan, layout, index, cardWidth, true)
    : drawn;
}

// What laying out the slots of one layout shares.
interface SlotPass {
  env: SceneEnv;
  cardWidth: number;
  tags: Record<string, TagStyle>;
  baseIndex: IndexGeometry;
  drawn: Items;
}

// Every slot of `layout` laid out, blocks moved and the keep-out yielded.
function laySlots(
  pass: SlotPass,
  layout: Layout,
  keepFor: (slots: readonly SlotLayout[]) => Rect | null,
): SlotLayout[] {
  const {env, cardWidth, tags, baseIndex, drawn} = pass;
  const shifts = blockShifts(layout.slots, cardWidth, s =>
    layoutSlot(s, drawn[s.id] ?? [], env, cardWidth, tags, baseIndex),
  );
  const slots = layout.slots.map(s0 => {
    const d = shifts.get(s0.id);
    const s = d ? {...s0, x: s0.x + d.x, y: s0.y + d.y} : s0;
    const list = drawn[s.id] ?? [];
    if (s.faceDown) {
      const up = list.find(it => it !== null && !isToken(it) && isFaceUp(it));
      if (up) {
        throw new Error(
          `layoutScene: slot '${s.id}' is faceDown but holds the face-up card '${up.id}'`,
        );
      }
    }
    return layoutSlot(s, list, env, cardWidth, tags, baseIndex);
  });
  // The keep-out (SceneEnv.keepOut): a centred hand or fan that reaches
  // into it is compressed about its own centre until it clears.
  // Under a hold too (a drag in progress keeps the layout and card width it
  // drew): the yield depends only on the box and the keep-out, so a hand
  // squeezed clear of the box stays squeezed while a card is dragged.
  const keep = keepFor(slots);
  if (!keep) return slots;
  return slots.map(l => yieldSlot(pass, l, keep, drawn[l.spec.id] ?? []));
}

/** A layout drawn at one card width: the Scene layoutScene returns. */
function drawLayout(
  plan: Plan,
  chosen: Layout,
  index: number,
  cardWidth: number,
  overflow: boolean,
): Scene {
  const {env, spec, items} = plan;
  if (!(cardWidth > 0 && Number.isFinite(cardWidth))) {
    throw new Error(
      `layoutScene: card width must be > 0, got ${String(cardWidth)}`,
    );
  }
  // TableSpec.fillWidth: a 'contain' table that fits takes its box's width.
  const extra =
    chosen.fillWidth && env.fit === 'contain' && !overflow && env.box.width > 0
      ? env.box.width / cardWidth - chosen.width
      : 0;
  const layout = extra > 1e-9 ? widenTableLayout(chosen, extra) : chosen;
  assertItems(
    items,
    layout.slots.map(s => s.id),
  );
  // Widened, exactly the box (no float drift past it).
  const width = layout === chosen ? layout.width * cardWidth : env.box.width;
  const bottom = (slots: readonly SlotLayout[]): number =>
    Math.max(layout.height * cardWidth, ...slots.map(l => slotBottomOf(l.box)));
  // Blocks (SlotSpec.block): packed, then each moved as one so that what
  // it draws is centred where the spec says.
  const pass: SlotPass = {
    env,
    cardWidth,
    tags: tagStyles(env.style),
    baseIndex: deckIndex(env.deck, env.style.indexScale),
    drawn: leadHoles(layout.slots, packBlocks(layout.slots, items)),
  };
  const slots = laySlots(pass, layout, laid =>
    env.cardWidth === undefined
      ? keepInTable(env, width, bottom(laid), overflow)
      : null,
  );
  const placements = paintOrder(slots);
  // Piles that grow past the spec's height extend the table instead of
  // spilling over whatever follows it.
  let height = bottom(slots);
  const slotById = new Map(slots.map(l => [l.spec.id, l] as const));
  const regionSpecs = layout.regions ?? [];
  const regions = regionSpecs.map(r =>
    placeRegion(r, slotById, cardWidth, width),
  );
  for (const r of regions) {
    height = Math.max(height, ...boxQuad(r.box).map(q => q.y));
  }
  const scene = sceneOf(
    {cardWidth, width, height, variant: index - 1},
    slots,
    placements,
    regions,
    slotById,
  );
  setSceneInternals(scene, {
    spec,
    layout,
    overflow,
    slots,
    slotById,
    regionSpecs,
  });
  return scene;
}

/** A slot that reaches into the keep-out `k` (table px), compressed about
 *  its own centre along its fan until it clears: a centred hand or fan
 *  only (a squared pile or a free slot cannot give room), and never past
 *  the index floor (a compression that would scroll is not taken). Along
 *  the screen's x it clears the keep-out's right edge, along y its bottom
 *  edge. Otherwise the slot as laid out. */
function yieldSlot(
  pass: SlotPass,
  l: SlotLayout,
  k: Rect,
  list: SlotItems,
): SlotLayout {
  const s = l.spec;
  if (
    s.anchor !== 'center' ||
    l.kind === 'free' ||
    s.pile === 'squared' ||
    l.entries.length === 0
  ) {
    return l;
  }
  const b = l.box;
  const bounds = boxBounds(
    toTableBox(b, {x: 0, y: 0, width: b.width, height: b.height}, 0),
  );
  if (!rectsMeet(bounds, k)) return l;
  // The fan runs along the box's width (hands, fan-right) or its height
  // (fan-down, fan-up), turned with the slot.
  const alongWidth = !VERTICAL.includes(s.pile);
  const quarter = norm(b.rotate) === 90 || norm(b.rotate) === 270;
  const acrossScreen = alongWidth !== quarter;
  const along = alongWidth ? b.width : b.height;
  const room = acrossScreen
    ? 2 * (bounds.x + bounds.width / 2 - (k.x + k.width))
    : 2 * (bounds.y + bounds.height / 2 - (k.y + k.height));
  if (!(room > 0) || room >= along - 0.5) return l;
  const maxExtent = Math.min(
    s.maxExtent ?? Infinity,
    Math.floor(room) / pass.cardWidth,
  );
  const next = layoutSlot(
    {...s, maxExtent},
    list,
    pass.env,
    pass.cardWidth,
    pass.tags,
    pass.baseIndex,
  );
  return next.overflow ? l : next;
}
