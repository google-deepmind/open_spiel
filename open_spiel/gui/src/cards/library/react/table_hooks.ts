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

// The table's effects and the state it holds across renders: what it drew
// (kept through a drag), the overflow report and the trick collect.
import {
  useLayoutEffect,
  useRef,
  useSyncExternalStore,
  type RefObject,
} from 'react';
import type {Scene, SceneEnv} from '../core/scene.ts';
import {dropMarksStore} from '../dnd/controller.ts';
import {collectTableCards} from '../motion/collect.ts';
import {prefersReducedMotion} from '../motion/service.ts';
import type {CardStyle} from '../style.ts';
import type {SceneRuntimeImpl} from './scene_runtime.ts';
import type {TableProps} from './table.tsx';

// A box rather than a ref: the render reads it.
/** What the table drew last, held through a drag so the layout stays put. */
export class DrawnHold {
  private value: SceneEnv['hold'] = undefined;

  /** What was drawn last. */
  get(): SceneEnv['hold'] {
    return this.value;
  }

  /** Records what was drawn; true when it changed. */
  set(scene: Scene, overflow: boolean): void {
    const d = this.value;
    if (
      d?.variant !== scene.variant + 1 ||
      d.cardWidth !== scene.cardWidth ||
      d.overflow !== overflow
    ) {
      this.value = {
        variant: scene.variant + 1,
        cardWidth: scene.cardWidth,
        overflow,
      };
    }
  }
}

/** What the table drew, while a drag is in progress; else undefined. */
export function useHold(
  rt: SceneRuntimeImpl,
  held: DrawnHold,
): SceneEnv['hold'] {
  const marks = dropMarksStore(rt.drag);
  const dragging = useSyncExternalStore(
    marks.subscribe,
    () => rt.drag.state().source !== null,
    () => false,
  );
  return dragging ? held.get() : undefined;
}

// Whether the table scrolls, for onOverflowChange: only once the box is
// measured, and only on a change (the latest callback, not a stale one).
/** Reports whether the table scrolls, on a change. */
export function useOverflowReport(
  scrolls: boolean | null,
  onOverflowChange: TableProps['onOverflowChange'],
): void {
  const onOverflow = useRef(onOverflowChange);
  useLayoutEffect(() => {
    onOverflow.current = onOverflowChange;
  });
  useLayoutEffect(() => {
    if (scrolls !== null) onOverflow.current?.(scrolls);
  }, [scrolls]);
}

// The trick collect, after the commit laid its cards out.
/** Runs the trick collect once per key, after the commit. */
export function useTableCollect(
  collect: TableProps['collect'],
  rt: SceneRuntimeImpl,
  tableId: string,
  surfaceRef: RefObject<HTMLDivElement | null>,
  style: CardStyle,
): void {
  const collectProp = useRef(collect);
  useLayoutEffect(() => {
    collectProp.current = collect;
  });
  const collectKey = collect?.key ?? null;
  useLayoutEffect(() => {
    const c = collectProp.current;
    if (!c || prefersReducedMotion(style.motion)) return;
    const root = surfaceRef.current;
    if (!root || root.getBoundingClientRect().width <= 0) return;
    const laid = rt.table(tableId).scene;
    const ids = c.ids.filter(
      id => laid.byId(id) !== undefined && rt.element(id) !== null,
    );
    if (ids.length === 0) return;
    const seat =
      rt.motion.measureSlot(c.toward, 'table', tableId) ??
      rt.motion.measureRegion(c.toward, 'table', tableId);
    if (!seat) {
      throw new Error(
        `<Table collect>: no slot or region '${c.toward}' in this table`,
      );
    }
    const run = collectTableCards(
      rt.motion,
      ids,
      {x: seat.x + seat.width / 2, y: seat.y + seat.height / 2},
      {durationMs: c.durationMs},
    );
    return () => {
      run.dispose();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- the collect runs once per key.
  }, [collectKey, rt, tableId]);
}
