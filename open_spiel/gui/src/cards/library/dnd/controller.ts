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

// DragController. Owned by the dnd module.
//
// Pointer events drive it (native HTML5 drag is not used). The DOM input layer
// (react/input.ts) calls press/move/release/cancel and the keyboard entry
// points; while a pointer is down the controller also listens on the window,
// so a drag keeps following the pointer after it leaves the table, and Escape
// cancels wherever focus is.
//
// Normative behaviour:
//  1. press: canDrag(ref, e) on pointerdown; false means the press is only a
//     potential activate.
//  2. start: once the pointer moves past dragThresholdPx (touchDragThresholdPx
//     for touch). legal = every candidate of each Table's dropKinds that the
//     Table's canDrop accepts, computed ONCE across the CardScene; the legal
//     cards are tagged 'drop-legal', legal slots/inserts/regions marked
//     (DropMarks draws empty-slot outlines and insert carets); then dragStart.
//  3. preview: dragged ids tagged 'dragging' (hidden in place), a DOM preview
//     follows the pointer by direct transform writes.
//  4. over: 'overlap' (largest overlap of the lead card's preview box with a
//     legal target) or 'pointer' (first legal target under the pointer);
//     'drop-over' tag/mark; dragOver only when `over` changes.
//  5. release: drop on `over` ({hold:true} keeps the preview, tagged 'held',
//     until items change, then the new layout FLIPs from the preview's pose;
//     after holdTimeoutMs it animates home and logs); otherwise the preview
//     returns home. No `over`: dragCancel and return home.
//  6. keyboard: keyPick (Space) / keyCycle (arrows, Tab) / keyDrop (Enter,
//     Space) / cancel('escape'); every step is announced.
//  7. an items change mid-drag cancels ('items-changed').
//
// Plus: the scroll container under the pointer auto-scrolls near its edges,
// the click that ends a real drag is swallowed (it is not an activate), and
// cards on tables whose policy can drag get touch-action: none.
import type {Point} from '../core/identity.ts';
import type {DragSource, Modifiers} from '../core/events.ts';
import {NO_MODIFIERS} from '../core/events.ts';
import type {Scene} from '../core/scene.ts';
import type {ItemRef, Target} from '../core/targets.ts';
import type {SceneRuntime} from '../react/runtime.ts';
import {DEV} from '../core/dev.ts';
import {completeHold} from './drag_pending.ts';
import {DragEngine} from './drag_engine.ts';
import {
  currentEvent,
  cycleFrom,
  itemsSignature,
  keyNative,
  nearestFrom,
  snapPreview,
  type Located,
  type Press,
} from './drag_util.ts';

/** The drag-and-drop of one card scene, driven by the input layer. */
export interface DragController {
  /** Called by the DOM input layer on pointerdown over an interactive card or token. */
  press(ref: ItemRef, tableId: string, input: PointerInput): void;
  move(input: PointerInput): void;
  release(input: PointerInput): void;
  cancel(reason: 'escape' | 'blur' | 'pointercancel' | 'items-changed'): void;
  /** Keyboard path: Space on a focused card. false: canDrag said no. */
  keyPick(ref: ItemRef, tableId: string): boolean;
  keyCycle(dir: 'next' | 'prev' | 'up' | 'down' | 'left' | 'right'): void;
  keyDrop(): void;
  state(): Readonly<{
    source: DragSource | null;
    over: Target | null;
    legal: readonly Target[];
  }>;
}

/** A pointer or keyboard input, as the input layer passes it. */
export interface PointerInput {
  client: Point;
  modifiers: Modifiers;
  pointerType: 'mouse' | 'touch' | 'pen';
  native: PointerEvent | KeyboardEvent;
}

/** The legal-target store DropMarks renders from (internal to dnd/ + Table). */
export interface DropMarksStore {
  subscribe: (l: () => void) => () => void;
  /** Legal targets on one table; a stable array between changes. */
  legal: (tableId: string) => readonly Target[];
}

const STORES = new WeakMap<DragController, DropMarksStore>();
const EMPTY: readonly Target[] = Object.freeze([]);
const IDLE = Object.freeze({source: null, over: null, legal: EMPTY});

/** The drop-marks store of a controller made by createDragController. */
export function dropMarksStore(drag: DragController): DropMarksStore {
  const s = STORES.get(drag);
  if (!s) {
    throw new Error(
      'dropMarksStore: not a DragController made by createDragController',
    );
  }
  return s;
}

type CancelReason = 'escape' | 'blur' | 'pointercancel' | 'items-changed';

function cancelDrag(d: DragEngine, reason: CancelReason, log: boolean): void {
  const a = d.active;
  if (!a) {
    d.press = null;
    d.detachWindow?.();
    return;
  }
  if (reason === 'items-changed' && log && DEV) {
    console.warn(
      `[cards] items changed during a drag of [${a.source.ids.join(', ')}]; the drag is cancelled`,
    );
  }
  const native =
    currentEvent() ?? new Event(reason === 'escape' ? 'keydown' : reason);
  d.finish(native, a.via === 'keyboard' ? null : a.last, a.modifiers, 'cancel');
}

function inputOf(e: PointerEvent): PointerInput {
  return {
    client: {x: e.clientX, y: e.clientY},
    modifiers: {
      shift: e.shiftKey,
      ctrl: e.ctrlKey,
      alt: e.altKey,
      meta: e.metaKey,
    },
    pointerType:
      e.pointerType === 'touch' || e.pointerType === 'pen'
        ? e.pointerType
        : 'mouse',
    native: e,
  };
}

// Window listeners while a pointer is down.
function attachWindow(d: DragEngine, api: DragController): void {
  if (d.detachWindow || typeof window === 'undefined') return;
  const onMove = (e: PointerEvent): void => {
    api.move(inputOf(e));
  };
  const onUp = (e: PointerEvent): void => {
    api.release(inputOf(e));
  };
  const onCancel = (e: PointerEvent): void => {
    if (e === d.lastNative) return;
    d.lastNative = e;
    api.cancel('pointercancel');
  };
  // A face drawn as an image element would otherwise start a native image
  // drag, which ends the pointer stream with a pointercancel.
  const onNativeDrag = (e: DragEvent): void => {
    if (d.press || d.active) e.preventDefault();
  };
  const onKey = (e: KeyboardEvent): void => {
    if (e.key !== 'Escape' || e.defaultPrevented) return;
    if (!d.active && !d.press) return;
    e.preventDefault();
    api.cancel('escape');
  };
  window.addEventListener('pointermove', onMove, true);
  window.addEventListener('pointerup', onUp, true);
  window.addEventListener('pointercancel', onCancel, true);
  window.addEventListener('keydown', onKey, true);
  window.addEventListener('dragstart', onNativeDrag, true);
  d.detachWindow = () => {
    window.removeEventListener('pointermove', onMove, true);
    window.removeEventListener('pointerup', onUp, true);
    window.removeEventListener('pointercancel', onCancel, true);
    window.removeEventListener('keydown', onKey, true);
    window.removeEventListener('dragstart', onNativeDrag, true);
    d.detachWindow = null;
  };
}

/** The click that ends a real drag is not an activate. */
function swallowNextClick(): void {
  if (typeof window === 'undefined') return;
  const onClick = (e: MouseEvent): void => {
    e.stopPropagation();
    e.preventDefault();
    done();
  };
  const done = (): void => {
    window.removeEventListener('click', onClick, true);
    clearTimeout(timer);
  };
  const timer = setTimeout(done, 0);
  window.addEventListener('click', onClick, true);
}

// Draggable cards get touch-action: none (the browser must not pan), and
// data-draggable for game CSS (a grab cursor) and tests.
function markDraggable(
  d: DragEngine,
  prev: ReadonlyMap<string, Scene>,
  next: ReadonlyMap<string, Scene>,
): void {
  for (const [tableId, scene] of next) {
    if (prev.get(tableId) === scene) continue;
    const can = !!d.policyOf(tableId)?.canDrag;
    for (const p of scene.placements) {
      const el = d.rt.element(p.id);
      if (!el) continue;
      if (can && p.interactive) {
        el.style.setProperty('touch-action', 'none');
        el.setAttribute('data-draggable', 'true');
      } else if (el.hasAttribute('data-draggable')) {
        el.style.removeProperty('touch-action');
        el.removeAttribute('data-draggable');
      }
    }
  }
}

// Items changes (every Table commit): a drag whose items changed, or whose
// table unmounted, ends; a held drop completes.
function onCommit(
  d: DragEngine,
  prev: ReadonlyMap<string, Scene>,
  next: ReadonlyMap<string, Scene>,
): void {
  markDraggable(d, prev, next);
  let changed = false;
  let unmounted = false;
  for (const [tableId, scene] of prev) {
    const n = next.get(tableId);
    if (n === undefined) {
      const a = d.active;
      if (
        a &&
        (a.tableId === tableId || a.legal.some(l => l.tableId === tableId))
      ) {
        unmounted = true;
      }
      if (d.press?.tableId === tableId) unmounted = true;
      continue;
    }
    if (n !== scene && itemsSignature(n) !== itemsSignature(scene)) {
      changed = true;
    }
  }
  if (!changed && !unmounted) return;
  // A Table unmounting under a drag (navigation, a scene torn down) ends it quietly.
  if (d.active) {
    cancelDrag(d, 'items-changed', changed);
  } else if (d.press) {
    d.press = null;
    d.detachWindow?.();
  }
  const p = d.pending;
  if (p?.kind === 'hold') {
    // Every Table in the commit has registered its nodes and scene by
    // the microtask; FLIP before paint.
    queueMicrotask(() => {
      completeHold(d, p);
    });
  }
}

function seen(d: DragEngine, e: Event): boolean {
  if (e === d.lastNative) return true;
  d.lastNative = e;
  return false;
}

function pointerPress(
  d: DragEngine,
  api: DragController,
  ref: ItemRef,
  tableId: string,
  input: PointerInput,
): void {
  if (d.active) return; // a second pointer during a drag
  const policy = d.policyOf(tableId);
  d.press = null;
  if (!policy?.canDrag) return;
  const ev = d.pressEvent(
    ref,
    tableId,
    input.pointerType,
    input.native,
    input.client,
    input.modifiers,
  );
  const ids = policy.canDrag(ref, ev);
  if (ids === false) return;
  d.press = {
    ref,
    tableId,
    ids: d.validateIds(ids, ref, tableId),
    start: input.client,
    pointerType: input.pointerType,
  };
  attachWindow(d, api);
}

function pointerMove(d: DragEngine, input: PointerInput): void {
  if (seen(d, input.native)) return;
  const a = d.active;
  if (a) {
    if (a.via === 'keyboard') return;
    a.last = input.client;
    a.modifiers = input.modifiers;
    a.native = input.native;
    d.resolveOver(input.native);
    a.preview.setOffset({
      x: input.client.x - a.start.x,
      y: input.client.y - a.start.y,
    });
    return;
  }
  const pr = d.press;
  if (!pr) return;
  const st = d.rt.style().interaction;
  const limit =
    pr.pointerType === 'touch' ? st.touchDragThresholdPx : st.dragThresholdPx;
  if (
    Math.hypot(input.client.x - pr.start.x, input.client.y - pr.start.y) <=
    limit
  ) {
    return;
  }
  d.startDrag(pr, pr.pointerType, input.native, input.client, input.modifiers);
}

function pointerRelease(d: DragEngine, input: PointerInput): void {
  if (seen(d, input.native)) return;
  const a = d.active;
  if (!a) {
    d.press = null;
    d.detachWindow?.();
    return;
  }
  if (a.via === 'keyboard') return;
  a.last = input.client;
  a.modifiers = input.modifiers;
  d.resolveOver(input.native);
  swallowNextClick();
  d.finish(input.native, input.client, input.modifiers, 'release');
}

function keyPick(d: DragEngine, ref: ItemRef, tableId: string): boolean {
  if (d.active) return false;
  const policy = d.policyOf(tableId);
  if (!policy?.canDrag) return false;
  const native = keyNative();
  const mods =
    native instanceof KeyboardEvent
      ? {
          shift: native.shiftKey,
          ctrl: native.ctrlKey,
          alt: native.altKey,
          meta: native.metaKey,
        }
      : NO_MODIFIERS;
  const ids = policy.canDrag(
    ref,
    d.pressEvent(ref, tableId, 'keyboard', native, null, mods),
  );
  if (ids === false) return false;
  d.press = null;
  const pr: Press = {
    ref,
    tableId,
    ids: d.validateIds(ids, ref, tableId),
    start: {x: 0, y: 0},
    pointerType: 'mouse',
  };
  d.startDrag(pr, 'keyboard', native, null, mods);
  return true;
}

function keyCycle(
  d: DragEngine,
  dir: 'next' | 'prev' | 'up' | 'down' | 'left' | 'right',
): void {
  const a = d.active;
  if (a?.via !== 'keyboard' || a.legal.length === 0) {
    if (a?.via === 'keyboard') d.rt.announce('No legal targets');
    return;
  }
  const next: Located | null =
    dir === 'next' || dir === 'prev'
      ? cycleFrom(a, dir)
      : nearestFrom(d.rt, a, dir);
  if (!next) return;
  const native = keyNative();
  a.native = native;
  d.setOver(a, next, native, null);
  snapPreview(d.rt, a);
}

/** The drag controller of card scene `rt`. */
export function createDragController(rt: SceneRuntime): DragController {
  const d = new DragEngine(rt);
  const api: DragController = {
    press: (ref, tableId, input) => {
      pointerPress(d, api, ref, tableId, input);
    },
    move: input => {
      pointerMove(d, input);
    },
    release: input => {
      pointerRelease(d, input);
    },
    cancel: reason => {
      cancelDrag(d, reason, true);
    },
    keyPick: (ref, tableId) => keyPick(d, ref, tableId),
    keyCycle: dir => {
      keyCycle(d, dir);
    },
    keyDrop: () => {
      const a = d.active;
      if (a?.via !== 'keyboard') return;
      d.finish(keyNative(), null, a.modifiers, 'release');
    },
    state: () => {
      const a = d.active;
      if (!a) return IDLE;
      return {
        source: a.source,
        over: a.over?.target ?? null,
        legal: a.legal.map(l => l.target),
      };
    },
  };
  rt.onCommit((prev, next) => {
    onCommit(d, prev, next);
  });
  STORES.set(api, {
    subscribe: l => {
      d.listeners.add(l);
      return () => {
        d.listeners.delete(l);
      };
    },
    legal: tableId => d.legalByTable.get(tableId) ?? EMPTY,
  });
  return api;
}
