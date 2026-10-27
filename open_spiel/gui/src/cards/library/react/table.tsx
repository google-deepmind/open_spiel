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

// <Table>: renders a TableSpec from
// the computed Scene (core/scene.ts layoutScene), so what is drawn, what is
// hit-tested and what the legibility check verified are one geometry.
//
// The spec says where each slot sits in card widths; the table picks ONE card
// width for the whole layout — given, or fitted to its container, regions
// included — so the table scales as a unit. Below the readable minimum it
// stops shrinking and scrolls. A spec with variants is drawn as the first
// variant that fits; a TableLayout is resolved for the deck and style it is
// drawn with.
//
// Interaction is one InteractionPolicy (absent: an inert display table — no
// buttons, no tab stops). Items are TableCards with required ids.
import {
  useContext,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
  type ComponentType,
  type JSX,
  type ReactNode,
  type RefObject,
} from 'react';
import {getDeck} from '../registry.ts';
import {useCardStyle, type CardStyle} from '../style.ts';
import type {Deck} from '../types.ts';
import {
  type FreePlacement,
  type TableFit,
  type TableSource,
} from '../table_spec.ts';
import type {Point, Rect, TableCard, TableItem} from '../core/identity.ts';
import {isToken} from '../core/identity.ts';
import {assertTagName, isLibraryTag} from '../core/tags.ts';
import type {InteractionPolicy} from '../core/policy.ts';
import type {ItemRef} from '../core/targets.ts';
import {targetKey} from '../core/targets.ts';
import {
  layoutScene,
  sceneInternals,
  type Scene,
  type SceneEnv,
} from '../core/scene.ts';
import {a11yModel, type A11yNode} from '../core/a11y.ts';
import {DEV} from '../core/dev.ts';
import {planTransitions} from '../core/motion_plan.ts';
import {runFlip} from '../motion/flip.ts';
import {useTicker} from '../motion/ticker.ts';
import {
  describedById,
  descriptionStore,
  type DescriptionStore,
} from '../a11y/describe.ts';
import {checkMeasured, useFrameBox} from './table_frame.ts';
import {type Renderer} from './table_slot.tsx';
import {TableFrame} from './table_surface.tsx';
import {
  DrawnHold,
  useHold,
  useOverflowReport,
  useTableCollect,
} from './table_hooks.ts';
import {SceneRuntimeContext} from './runtime.ts';
import {SceneRuntimeImpl} from './scene_runtime.ts';
import {attachInput} from './input_listeners.ts';
import {TableIdContext} from './card_scene.tsx';
import {useFeltKeepOut} from '../felt_keep_out.ts';

export type {TableFit} from '../table_spec.ts';
export {fitTableCardWidth, slotBox, type SlotBox} from '../core/scene.ts';

/** What a custom card renderer is given. */
export interface CardRenderProps {
  ref: ItemRef;
  card: TableCard;
  width: number;
  faceUp: boolean;
  tags: readonly string[];
  /** Covered by design (a squared pile's depth layer, drawn as a back). */
  covered: boolean;
}

/** A table: its spec, cards, policy and how it draws them. */
export interface TableProps {
  /** A fixed spec, or a TableLayout resolved for the active deck and style. */
  spec: TableSource;
  /** Content for spec.regions ids. Each region receives the CSS variable
   *  --card-w: <px>; inside it, useCardWidth() returns the same number. */
  regions?: Readonly<Record<string, ReactNode>> | undefined;
  /** Game-owned drawing for tokens. Required when items contain a token (throws otherwise). */
  renderToken?:
    | ((ref: ItemRef, px: {width: number; height: number}) => ReactNode)
    | undefined;
  /** A layer inside the card box, above the face (badges, counters, pips).
   *  geom.badgeAnchor is the index-free corner in card-local px. */
  renderOverlay?:
    | ((
        ref: ItemRef,
        geom: {
          width: number;
          height: number;
          badgeAnchor: Point;
          indexBox: Rect | null;
        },
      ) => ReactNode)
    | undefined;
  /** Total override of a card's drawing. Default is the library renderer, so
   *  wrapping is possible. Plumbing (events, focus, motion, data-*) stays on
   *  the library's wrapper element. */
  renderCard?:
    | ((
        p: CardRenderProps,
        Default: ComponentType<CardRenderProps>,
      ) => ReactNode)
    | undefined;
  /** Run-time caption text per slot id (a hand score), replacing the
   *  spec's. Only slots whose spec declares a caption may take one. */
  captions?: Readonly<Record<string, string>> | undefined;
  /** 'width': fill the container's width. 'contain': also fit its height. */
  fit?: TableFit | undefined;
  /** Explicit card width, px; omitted, the table fits its container. */
  cardWidth?: number | undefined;
  minCardWidth?: number | undefined;
  maxCardWidth?: number | undefined;
  /** A viewer's card size, x the fitted card width (default 1). The table
   *  is fitted as usual, then drawn this much larger or smaller (smaller
   *  never draws its smallest slot below minCardWidth); a table grown past
   *  its box scrolls. */
  cardScale?: number | undefined;
  /** Called with whether the table scrolls in its box (it is drawn past
   *  it: under the minimum card width, or grown by cardScale), on the
   *  first measured draw and whenever that changes, before paint. A game
   *  keeps a control in view with it (it can pin the control outside a
   *  scrolling table). An unmeasured box is not known to scroll. */
  onOverflowChange?: ((overflow: boolean) => void) | undefined;
  /** Content held in view at the foot of a table that scrolls: in the
   *  scrolling box after the surface, so the table scrolls far enough to
   *  bring every card clear of it, and sticky at the box's bottom and left
   *  edges, inside its scrollbars, so it never scrolls out of view. It
   *  takes no room from the fit: the table is fitted to its whole box. A
   *  game pins a control there that its layout places where a scrolled
   *  table can hide it, and passes it only while the table scrolls
   *  (onOverflowChange); it is drawn whenever given, so a control that
   *  moves there and back stays mounted until the game moves it. */
  foot?: ReactNode | undefined;
  /** Mounted-table id for events and CardScene (default: useId()). */
  tableId?: string | undefined;
  testId?: string | undefined;
  /** Items per slot id, bottom first. A slot with no entry is empty; an id
   *  the spec does not name throws, as do duplicate or missing item ids.
   *  In a fan slot an entry may be null: a positional hole that keeps its
   *  room (later cards never move), draws nothing and is no target;
   *  positions count it. A hole in a squared, hand or free slot throws. */
  items: Readonly<
    Record<string, ReadonlyArray<(TableItem & Partial<FreePlacement>) | null>>
  >;
  /** Absent: an inert display table (no buttons, no tab stops). */
  policy?: InteractionPolicy | undefined;
  /** Game tags per slot id for this render, merged after the spec's own
   *  SlotSpec.tags and styled by CardStyle.tags like card tags. The look
   *  draws on the slot's EMPTY outline (a slot with cards shows its cards'
   *  own tags): e.g. 'highlighted' on an empty column that is a legal
   *  target of a selection; TagStyle.empty 'hidden' (the built-in
   *  'hide-empty') draws no outline. An id the spec does not name throws. */
  slotTags?: Readonly<Record<string, readonly string[]>> | undefined;
  /** A trick collect (trick_collect.ts, motion/collect.ts): these cards
   *  slide together toward the slot or region `toward` and fade out over
   *  durationMs, in the free layer. It starts when `key` changes and what
   *  it lifted is removed when the prop changes or goes; the game drops the
   *  cards from `items` at that point. Ids not laid out here are skipped;
   *  under reduced motion (CardStyle.motion) nothing moves. */
  collect?: TableCollect | null | undefined;
}

/** A finished trick to fly to its winner's seat. */
export interface TableCollect {
  key: string;
  ids: readonly string[];
  /** A slot id, else a region id, of this table: the winner's seat. */
  toward: string;
  durationMs: number;
}

function noShared(): () => void {
  return () => undefined;
}

interface SceneInputs {
  props: TableProps;
  rt: SceneRuntimeImpl;
  ctx: SceneRuntimeImpl | null;
  deck: Deck;
  style: CardStyle;
  policy: InteractionPolicy | undefined;
  box: {width: number; height: number};
  keepOut: ReturnType<typeof useFeltKeepOut>;
  held: DrawnHold;
}

function envOf(i: SceneInputs, hold: SceneEnv['hold']): SceneEnv {
  const {props, deck, style, policy, box, keepOut} = i;
  const env: SceneEnv = {
    deck,
    style,
    box,
    fit: props.fit ?? 'width',
    cardWidth: props.cardWidth,
    minCardWidth: props.minCardWidth,
    maxCardWidth: props.maxCardWidth,
    cardScale: props.cardScale,
    hold,
    keepOut,
  };
  if (policy?.slotInteractive) {
    env.slotInteractive = s => policy.slotInteractive?.(s) ?? false;
  }
  return env;
}

// The scene the table draws: laid out for its box, deck and style, at the
// card width a shared-scale scene settles on, held still through a drag.
function useTableScene(i: SceneInputs): {
  scene: Scene;
  sharedMode: boolean;
  ownFit: number | undefined;
} {
  const {props, rt, ctx, deck, style, policy, box, keepOut} = i;

  const {spec: source, items, fit = 'width', cardWidth: cardWidthProp} = props;
  const {minCardWidth, maxCardWidth, cardScale} = props;
  // One responsive scale across a CardScene with scale='shared'.
  const shared = useSyncExternalStore(
    ctx ? l => ctx.onFit(l) : noShared,
    () => (ctx ? ctx.sharedWidth() : undefined),
    () => undefined,
  );
  // While a drag is in progress the table keeps the layout and card width
  // it drew: a container that changes size under a drag (or a frame whose
  // measure moves with the drag's own marks) never re-lays it out, which
  // would move the target, rebuild the cards under the pointer or the
  // keyboard, and drop focus. The fit catches up when the drag ends.
  const hold = useHold(rt, i.held);
  const baseEnv = envOf(i, hold);
  const sharedMode =
    !!ctx && ctx.scale === 'shared' && cardWidthProp === undefined;
  const ownFit = useMemo(
    () =>
      sharedMode ? layoutScene(source, items, baseEnv).cardWidth : undefined,
    // eslint-disable-next-line react-hooks/exhaustive-deps -- baseEnv is rebuilt each render; these are its parts.
    [
      sharedMode,
      source,
      items,
      deck,
      style,
      box.width,
      box.height,
      fit,
      minCardWidth,
      maxCardWidth,
      cardScale,
      hold,
      keepOut,
    ],
  );
  const scene = useMemo(
    () =>
      layoutScene(
        source,
        items,
        sharedMode && shared !== undefined
          ? {...baseEnv, cardWidth: Math.min(shared, ownFit ?? shared)}
          : baseEnv,
      ),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- baseEnv is rebuilt each render; these are its parts.
    [
      source,
      items,
      deck,
      style,
      box.width,
      box.height,
      fit,
      cardWidthProp,
      minCardWidth,
      maxCardWidth,
      cardScale,
      policy,
      shared,
      ownFit,
      sharedMode,
      hold,
      keepOut,
    ],
  );
  return {scene, sharedMode, ownFit};
}

// Throws on props the scene drawn does not declare: captions, region
// content, slot tags, and tokens without renderToken.
function checkTableProps(props: TableProps, scene: Scene): void {
  const {layout} = sceneInternals(scene);
  for (const id of Object.keys(props.captions ?? {})) {
    if (layout.slots.find(s => s.id === id)?.caption === undefined) {
      throw new Error(
        `<Table> got a caption for '${id}', which declares none in the spec drawn`,
      );
    }
  }
  for (const id of Object.keys(props.regions ?? {})) {
    if (!scene.regions.some(r => r.id === id)) {
      throw new Error(
        `<Table> got content for region '${id}', which the spec drawn does not declare`,
      );
    }
  }
  for (const [id, tags] of Object.entries(props.slotTags ?? {})) {
    if (!layout.slots.some(s => s.id === id)) {
      throw new Error(
        `<Table> got slotTags for '${id}', which the spec drawn does not declare`,
      );
    }
    for (const t of tags) {
      assertTagName(t, `<Table> slotTags['${id}']`);
      if (isLibraryTag(t)) {
        throw new Error(
          `<Table> slotTags['${id}']: '${t}' is a library tag; games only style it`,
        );
      }
    }
  }
  const hasToken = scene.placements.some(p => isToken(p.item));
  if (hasToken && !props.renderToken) {
    throw new Error(
      '<Table> items contain a token; renderToken is required to draw it',
    );
  }
}

// The a11y model of the scene, and its descriptions published in the
// scene's one hidden description node.
function useTableA11y(
  scene: Scene,
  policy: InteractionPolicy | undefined,
  rt: SceneRuntimeImpl,
  tableId: string,
): {a11y: Map<string, A11yNode>; descStore: DescriptionStore} {
  const a11y = useMemo(
    () =>
      new Map(
        a11yModel(scene, policy, rt.focus.current()).map(
          n => [n.key, n] as const,
        ),
      ),
    [scene, policy, rt],
  );
  const descStore = descriptionStore(rt);
  const descriptions = useMemo(() => {
    const m = new Map<string, string>();
    for (const p of scene.placements) {
      const n = a11y.get(
        targetKey({
          kind: 'card',
          ref: {id: p.id, slot: p.slot, position: p.position, item: p.item},
        }),
      );
      if (n?.description !== undefined) {
        m.set(describedById(descStore, tableId, p.id), n.description);
      }
    }
    return m;
  }, [a11y, scene, descStore, tableId]);
  useLayoutEffect(() => {
    descStore.publish(tableId, descriptions);
  }, [descStore, tableId, descriptions]);
  useLayoutEffect(
    () => () => {
      descStore.remove(tableId);
    },
    [descStore, tableId],
  );
  return {a11y, descStore};
}

interface CommitInputs {
  rt: SceneRuntimeImpl;
  ctx: SceneRuntimeImpl | null;
  tableId: string;
  scene: Scene;
  policy: InteractionPolicy | undefined;
  surfaceRef: RefObject<HTMLDivElement | null>;
  held: DrawnHold;
  sharedMode: boolean;
  ownFit: number | undefined;
  style: CardStyle;
  overflow: boolean;
}

// Mount, commit (onCommit subscribers: dnd cancels on items change, motion
// plans FLIP) and the check of computed against measured.
function useTableCommit(c: CommitInputs): void {
  const {rt, ctx, tableId, scene, policy, surfaceRef, held} = c;
  const prevScene = useRef<Scene | null>(null);
  useLayoutEffect(() => {
    const root = surfaceRef.current;
    if (!root) return;
    rt.commit({tableId, scene, root, policy});
    held.set(scene, c.overflow);
    if (c.sharedMode && ctx && c.ownFit !== undefined) {
      ctx.reportFit(tableId, c.ownFit);
    }
    const prev = prevScene.current;
    prevScene.current = scene;
    if (prev && prev !== scene) {
      void runFlip(rt, planTransitions(prev, scene), c.style.motion);
    }
    if (DEV) checkMeasured(rt, tableId, scene);
  });
  useLayoutEffect(
    () => () => {
      rt.unmount(tableId);
    },
    [rt, tableId],
  );
}

// The table's scene runtime: its CardScene's, or (a lone <Table>) its own.
function useTableRuntime(): {
  ctx: SceneRuntimeImpl | null;
  own: SceneRuntimeImpl | null;
  rt: SceneRuntimeImpl;
  style: CardStyle;
} {
  const ctx = useContext(SceneRuntimeContext) as SceneRuntimeImpl | null;
  const [own] = useState(() => (ctx ? null : new SceneRuntimeImpl()));
  const rt = ctx ?? own;
  if (!rt) throw new Error('<Table> lost its card scene between renders');
  const style = useCardStyle();
  const ticker = useTicker();
  if (own) {
    own.setStyle(style);
    own.setTicker(ticker);
  }
  return {ctx, own, rt, style};
}

// An implicit scene listens on the table's own root, and announces there.
function useOwnScene(
  own: SceneRuntimeImpl | null,
  ref: RefObject<HTMLDivElement | null>,
): string {
  useLayoutEffect(() => {
    if (!own || !ref.current) return;
    return attachInput(own, ref.current);
  }, [own, ref]);
  const [live, setLive] = useState('');
  useLayoutEffect(() => (own ? own.onLive(setLive) : undefined), [own]);
  return live;
}

/** A card table: its spec's slots and regions, drawn and made interactive. */
export function Table(props: TableProps): JSX.Element {
  const {ctx, own, rt, style} = useTableRuntime();
  const generated = useId();
  const tableId = props.tableId ?? generated;
  // A Table without its own policy answers with its CardScene's (the same
  // policy rt.policy() hands events to), so what it renders — buttons, tab
  // stops, labels — matches what its events do.
  const policy = props.policy ?? ctx?.scenePolicy;
  const items: unknown = props.items;
  if (items === undefined || items === null) {
    throw new Error('<Table> needs items');
  }
  const deck = getDeck(style.deck);
  const ref = useRef<HTMLDivElement | null>(null);
  const surfaceRef = useRef<HTMLDivElement | null>(null);
  const box = useFrameBox(ref, surfaceRef, props.fit === 'contain');
  // On a felt, the info box's corner: no card is laid out under it.
  const keepOut = useFeltKeepOut(ref);
  const [held] = useState(() => new DrawnHold());
  const inputs = {props, rt, ctx, deck, style, policy, box, keepOut, held};
  const {scene, sharedMode, ownFit} = useTableScene(inputs);
  checkTableProps(props, scene);
  const {overflow} = sceneInternals(scene);
  const {a11y, descStore} = useTableA11y(scene, policy, rt, tableId);
  useTableCommit({
    ...inputs,
    tableId,
    scene,
    surfaceRef,
    sharedMode,
    ownFit,
    overflow,
  });
  useOverflowReport(box.width > 0 ? overflow : null, props.onOverflowChange);
  useTableCollect(props.collect, rt, tableId, surfaceRef, style);
  const live = useOwnScene(own, ref);
  const renderer: Renderer = {
    rt,
    scene,
    policy,
    a11y,
    deck,
    style,
    cardWidth: scene.cardWidth,
    tableId,
    descStore,
    renderCard: props.renderCard,
    renderOverlay: props.renderOverlay,
    renderToken: props.renderToken,
    slotTags: props.slotTags,
  };
  const body = (
    <TableFrame
      props={props}
      renderer={renderer}
      frameRef={ref}
      surfaceRef={surfaceRef}
      boxWidth={box.width}
      own={own}
      live={live}
    />
  );
  const wrapped = (
    <TableIdContext.Provider value={tableId}>{body}</TableIdContext.Provider>
  );
  return own ? (
    <SceneRuntimeContext.Provider value={own}>
      {wrapped}
    </SceneRuntimeContext.Provider>
  ) : (
    wrapped
  );
}

export type {Scene};
