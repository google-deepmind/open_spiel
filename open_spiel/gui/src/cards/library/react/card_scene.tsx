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

// <CardScene> and its hooks: one registry, focus ring, drag
// controller, motion service and live region shared by the <Table>s in a
// subtree. A lone <Table> creates an implicit scene.
import {
  createContext,
  useContext,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type JSX,
  type ReactNode,
} from 'react';
import {
  NO_MODIFIERS,
  type DragSource,
  type EventResult,
  type Modifiers,
  type TableEvent,
} from '../core/events.ts';
import type {CardId, Point} from '../core/identity.ts';
import type {InteractionPolicy} from '../core/policy.ts';
import type {Scene} from '../core/scene.ts';
import type {Target} from '../core/targets.ts';
import {targetKey} from '../core/targets.ts';
import type {MotionService} from '../motion/service.ts';
import {useTicker} from '../motion/ticker.ts';
import {useCardStyle} from '../style.ts';
import {LiveRegion} from '../a11y/live_region.tsx';
import {Descriptions} from '../a11y/describe.ts';
import {SceneRuntimeContext} from './runtime.ts';
import {SceneRuntimeImpl} from './scene_runtime.ts';
import {attachInput} from './input_listeners.ts';

/** The nearest mounted Table's id (inside a Table: regions, overlays, renderCard). */
export const TableIdContext = createContext<string | null>(null);
/** The card width in px, inside regions and overlays. */
export const CardWidthContext = createContext<number | null>(null);

/** The enclosing scene's runtime, or null outside one. */
export function useSceneRuntime(): SceneRuntimeImpl | null {
  return useContext(SceneRuntimeContext) as SceneRuntimeImpl | null;
}

/** A scene: the policy and scaling its tables share. */
export interface CardSceneProps {
  policy?: InteractionPolicy | undefined;
  scale?: 'shared' | 'independent' | undefined;
  children: ReactNode;
}

/** Shares the registry, focus ring, drag controller (drag across Tables),
 *  motion and live region across several <Table>s in one subtree. scale
 *  'shared': every child Table uses min(each child's max fitting cardWidth);
 *  the child boxes must not depend on cardWidth (a dev build throws after 3
 *  oscillating resize passes). */
export function CardScene({
  policy,
  scale = 'independent',
  children,
}: CardSceneProps): JSX.Element {
  const [rt] = useState(() => new SceneRuntimeImpl());
  const style = useCardStyle();
  const ticker = useTicker();
  rt.setScene(policy, scale);
  rt.setStyle(style);
  rt.setTicker(ticker);
  const root = useRef<HTMLDivElement | null>(null);
  const [live, setLive] = useState('');
  useEffect(() => rt.onLive(setLive), [rt]);
  useLayoutEffect(() => {
    const el = root.current;
    if (!el) return;
    return attachInput(rt, el);
  }, [rt]);
  return (
    <SceneRuntimeContext.Provider value={rt}>
      <div ref={root} data-card-scene style={{display: 'contents'}}>
        {children}
        <LiveRegion text={live} />
        <Descriptions scene={rt} />
      </div>
    </SceneRuntimeContext.Provider>
  );
}

/** What a scene offers its children: widths, scenes and motion. */
export interface CardSceneApi {
  /** The nearest Table's (or the shared) width. */
  cardWidth: number;
  /** The computed scene currently rendered. */
  scene(tableId?: string): Scene;
  element(id: CardId): HTMLElement | null;
  /** Synthesise an event (bots, tests, custom renderers). via='synthetic', native=null.
   *  Goes through the same validation as pointer input (drop requires canDrop). */
  dispatch(e: SyntheticTableEvent): EventResult;
  drag: Readonly<{source: DragSource | null; over: Target | null}>;
  /** Move the roving focus programmatically. */
  focus(target: Target | null): void;
  /** aria-live polite. */
  announce(text: string): void;
  motion: MotionService;
}

/** A TableEvent as a caller synthesises it: the library fills in via
 *  ('synthetic'), native (null), client (null) and, by default, tableId,
 *  modifiers (none held) and point (null). */
export type SyntheticTableEvent = TableEvent extends infer E
  ? E extends TableEvent
    ? Omit<
        E,
        'native' | 'via' | 'client' | 'tableId' | 'modifiers' | 'point'
      > & {tableId?: string; modifiers?: Modifiers; point?: Point | null}
    : never
  : never;

/** Activations need an interactive card; passive events an interactive or observed one. */
const ACTIVATIONS = new Set(['activate', 'doubleActivate']);
const PASSIVE = new Set(['contextMenu', 'longPress', 'hoverStart', 'hoverEnd']);

/** The scene API. Throws outside a Table/CardScene. */
export function useCardScene(): CardSceneApi {
  const rt = useSceneRuntime();
  const tableId = useContext(TableIdContext);
  if (!rt) {
    throw new Error(
      'useCardScene() must be used inside a <Table> or <CardScene>',
    );
  }
  return useMemo(() => makeApi(rt, tableId), [rt, tableId]);
}

// Throws unless `e` may be dispatched on `scene`: its card, slot or region
// exists and takes the event, and a drop is legal.
function checkDispatch(
  e: Parameters<CardSceneApi['dispatch']>[0],
  tableId: string,
  scene: Scene,
  policy: InteractionPolicy | undefined,
): void {
  const t = e.target;
  if (t.kind === 'card') {
    const p = scene.byId(t.ref.id);
    if (!p) {
      throw new Error(
        `dispatch(${e.type}): no card '${t.ref.id}' in table '${tableId}'`,
      );
    }
    if (ACTIVATIONS.has(e.type) && !p.interactive) {
      throw new Error(
        `dispatch(${e.type}): card '${t.ref.id}' is not interactive`,
      );
    }
    if (PASSIVE.has(e.type) && !p.interactive && !policy?.observe?.(t.ref)) {
      throw new Error(
        `dispatch(${e.type}): card '${t.ref.id}' is neither interactive nor observed (policy.observe)`,
      );
    }
  } else if (t.kind === 'slot' || t.kind === 'insert') {
    scene.slotBox(t.slot); // throws on an unknown slot
    if (
      t.kind === 'slot' &&
      (ACTIVATIONS.has(e.type) ||
        e.type === 'contextMenu' ||
        e.type === 'longPress') &&
      !policy?.slotInteractive?.(t.slot)
    ) {
      throw new Error(
        `dispatch(${e.type}): slot '${t.slot}' is not interactive (slotInteractive)`,
      );
    }
  } else if (
    t.kind === 'region' &&
    !scene.regions.some(r => r.id === t.region)
  ) {
    throw new Error(
      `dispatch(${e.type}): no region '${t.region}' in table '${tableId}'`,
    );
  }
  if (e.type === 'drop' && !policy?.canDrop?.(e.source, t)) {
    throw new Error(
      `dispatch(drop): ${targetKey(t)} is not a legal drop for [${e.source.ids.join(', ')}] (canDrop)`,
    );
  }
}

// Moves the keyboard focus to `target`'s element.
function focusTarget(rt: SceneRuntimeImpl, target: Target | null): void {
  rt.focus.set(target);
  if (!target) return;
  const key = target.kind === 'card' ? target.ref.id : targetKey(target);
  const el =
    target.kind === 'card'
      ? rt.element(key)
      : (rt
          .tables()
          .map(tb =>
            tb.root.querySelector<HTMLElement>(
              `[data-slot-id="${target.kind === 'slot' ? target.slot : ''}"]`,
            ),
          )
          .find(Boolean) ?? null);
  el?.focus();
}

function makeApi(rt: SceneRuntimeImpl, nearest: string | null): CardSceneApi {
  const pick = (tableId?: string): string => {
    if (tableId !== undefined) return tableId;
    if (nearest !== null && rt.hasTable(nearest)) return nearest;
    const first = rt.tables()[0];
    if (first === undefined) {
      throw new Error('card scene has no mounted table yet');
    }
    return first.tableId;
  };
  return {
    get cardWidth() {
      const shared = rt.sharedWidth();
      if (shared !== undefined) return shared;
      return rt.table(pick()).scene.cardWidth;
    },
    scene: tableId => rt.table(pick(tableId)).scene,
    element: id => rt.element(id),
    dispatch: e => {
      const tableId = pick(e.tableId);
      checkDispatch(e, tableId, rt.table(tableId).scene, rt.policy(tableId));
      const full = Object.assign({modifiers: NO_MODIFIERS, point: null}, e, {
        tableId,
        via: 'synthetic' as const,
        client: null,
        native: null,
      }) as TableEvent;
      return rt.emit(full);
    },
    get drag() {
      const s = rt.drag.state();
      return {source: s.source, over: s.over};
    },
    focus: target => {
      focusTarget(rt, target);
    },
    announce: text => {
      rt.announce(text);
    },
    motion: rt.motion,
  };
}

/** The card width in px inside regions and overlays. Throws elsewhere. */
export function useCardWidth(): number {
  const w = useContext(CardWidthContext);
  if (w === null) {
    throw new Error(
      'useCardWidth() must be used inside a <Table> region or overlay',
    );
  }
  return w;
}
