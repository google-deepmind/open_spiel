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

// The state behind one DragController (controller.ts): the press, the drag
// in progress, the pending preview, the legal targets, and the drag's start,
// over-resolution and end.
import type {CardId, Point} from '../core/identity.ts';
import type {
  DragSource,
  EventResult,
  Modifiers,
  TableEvent,
} from '../core/events.ts';
import type {InteractionPolicy} from '../core/policy.ts';
import type {Scene} from '../core/scene.ts';
import type {ItemRef, Target} from '../core/targets.ts';
import {targetKey} from '../core/targets.ts';
import {allTargets} from '../core/hit.ts';
import {defined} from '../core/defined.ts';
import type {SceneRuntime} from '../react/runtime.ts';
import {
  affineOf,
  boundsOf,
  createDragPreview,
  quad,
  toClient,
} from './drag_preview.tsx';
import {AutoScroll} from './auto_scroll.ts';
import {
  completeHold,
  holdDrop,
  returnHome,
  type PendingHost,
} from './drag_pending.ts';
import {
  describeTarget,
  itemsSignature,
  refOf,
  resolveOverlapMode,
  resolvePointerMode,
  type Active,
  type Located,
  type Pending,
  type Press,
  type Via,
} from './drag_util.ts';

type DnDTags = Map<string, readonly string[]>;

/** The fields of a table event a drag step fills in. */
interface EventBaseFields {
  target: Target;
  via: Via;
  modifiers: Modifiers;
  point: Point | null;
  client: Point | null;
  native: Event;
  tableId: string;
}

/** A drag controller's state and its drag steps. */
export class DragEngine implements PendingHost {
  press: Press | null = null;
  active: Active | null = null;
  pending: Pending | null = null;
  lastNative: Event | null = null;
  legalByTable = new Map<string, readonly Target[]>();
  readonly listeners = new Set<() => void>();
  detachWindow: (() => void) | null = null;
  private restoreSelect: (() => void) | null = null;
  readonly scroll: AutoScroll;
  readonly rt: SceneRuntime;

  constructor(rt: SceneRuntime) {
    this.rt = rt;
    this.scroll = new AutoScroll(
      rt,
      () => {
        const a = this.active;
        return a && a.via !== 'keyboard' ? a.last : null;
      },
      () => {
        if (this.active) this.resolveOver(this.active.native);
      },
    );
  }

  // ---- store ----
  /** Publishes the legal targets by table, for DropMarks. */
  setLegal(legal: readonly Located[]): void {
    const m = new Map<string, Target[]>();
    for (const l of legal) {
      const list = m.get(l.tableId) ?? [];
      list.push(l.target);
      m.set(l.tableId, list);
    }
    this.legalByTable = m;
    for (const l of [...this.listeners]) l();
  }

  // ---- tags and marks ----
  /** Writes the dnd library tags and the 'drop-over' mark. */
  readonly writeTags = (): void => {
    const {pending, active} = this;
    const tags: DnDTags = new Map();
    const add = (k: string, t: string): void => {
      tags.set(k, [...(tags.get(k) ?? []), t]);
    };
    if (pending) {
      for (const id of pending.ids) {
        add(id, pending.kind === 'hold' ? 'held' : 'dragging');
      }
    }
    if (active) {
      for (const id of active.source.ids) add(id, 'dragging');
      for (const l of active.legal) {
        add(l.target.kind === 'card' ? l.target.ref.id : l.key, 'drop-legal');
      }
      if (active.over) {
        const o = active.over;
        add(o.target.kind === 'card' ? o.target.ref.id : o.key, 'drop-over');
      }
    }
    this.rt.setLibraryTags('dnd', tags);
    const overKey = active?.over?.key ?? null;
    for (const t of this.rt.tables()) {
      for (const el of t.root.querySelectorAll<HTMLElement>(
        '[data-drop-mark]',
      )) {
        const on =
          el.getAttribute('data-drop-mark') === overKey &&
          active?.over?.tableId === t.tableId;
        if (on) el.setAttribute('data-drop-over', 'true');
        else el.removeAttribute('data-drop-over');
      }
    }
  };

  // ---- helpers ----
  /** The policy of table `tableId`. */
  policyOf(tableId: string): InteractionPolicy | undefined {
    return this.rt.policy(tableId);
  }

  private sceneOf(tableId: string): Scene {
    return this.rt.table(tableId).scene;
  }

  /** The activate event a press would be, for canDrag. */
  pressEvent(
    ref: ItemRef,
    tableId: string,
    via: Via,
    native: Event,
    client: Point | null,
    modifiers: Modifiers,
  ): TableEvent {
    return {
      type: 'activate',
      target: {kind: 'card', ref},
      via,
      modifiers,
      point: client ? this.rt.clientToTable(tableId, client) : null,
      client,
      native,
      tableId,
    };
  }

  /** canDrag's ids, checked: non-empty, the pressed card's, one slot. */
  validateIds(
    ids: readonly CardId[],
    ref: ItemRef,
    tableId: string,
  ): readonly CardId[] {
    const given: unknown = ids;
    if (!Array.isArray(given) || ids.length === 0) {
      throw new Error(
        `canDrag('${ref.id}') must return a non-empty id list or false, got ${JSON.stringify(given)}`,
      );
    }
    if (!ids.includes(ref.id)) {
      throw new Error(
        `canDrag('${ref.id}') returned [${ids.join(', ')}], which does not include the pressed card '${ref.id}'`,
      );
    }
    if (new Set(ids).size !== ids.length) {
      throw new Error(
        `canDrag('${ref.id}') returned a duplicate id: [${ids.join(', ')}]`,
      );
    }
    const scene = this.sceneOf(tableId);
    for (const id of ids) {
      const p = scene.byId(id);
      if (!p) {
        throw new Error(
          `canDrag('${ref.id}') returned '${id}', which is not on table '${tableId}'`,
        );
      }
      if (p.slot !== ref.slot) {
        throw new Error(
          `canDrag('${ref.id}') returned '${id}' from slot '${p.slot}'; dragged ids must all be in slot '${ref.slot}'`,
        );
      }
    }
    return [...ids];
  }

  private sourceOf(
    ids: readonly CardId[],
    ref: ItemRef,
    tableId: string,
  ): DragSource {
    const scene = this.sceneOf(tableId);
    return Object.freeze({
      ids: Object.freeze([...ids]),
      from: ref.slot,
      refs: Object.freeze(
        ids.map(id => refOf(defined(scene.byId(id), `dragged card '${id}'`))),
      ),
    });
  }

  /** Once per drag: every candidate of each table's dropKinds its canDrop accepts. */
  private computeLegal(src: DragSource): Located[] {
    const out: Located[] = [];
    const own = new Set(src.ids);
    for (const t of this.rt.tables()) {
      const policy = this.policyOf(t.tableId);
      if (!policy?.canDrop) continue;
      const kinds = policy.dropKinds ?? ['slot'];
      for (const target of allTargets(t.scene, kinds)) {
        // A dragged card is never a target for itself.
        if (target.kind === 'card' && own.has(target.ref.id)) continue;
        if (policy.canDrop(src, target)) {
          out.push({tableId: t.tableId, target, key: targetKey(target)});
        }
      }
    }
    return out;
  }

  /** Delivers a table event to the game. */
  emit(e: TableEvent): EventResult {
    return this.rt.emit(e);
  }

  /** The common fields of a drag event on `tableId`. */
  baseFor(
    tableId: string,
    target: Target,
    via: Via,
    native: Event,
    client: Point | null,
    modifiers: Modifiers,
  ): EventBaseFields {
    return {
      target,
      via,
      modifiers,
      point:
        client && this.rt.tables().some(t => t.tableId === tableId)
          ? this.rt.clientToTable(tableId, client)
          : null,
      client,
      native,
      tableId,
    };
  }

  // ---- over resolution ----
  /** Sets the target the drag is over; dragOver when it changes. */
  setOver(
    a: Active,
    next: Located | null,
    native: Event,
    client: Point | null,
  ): void {
    if (
      (a.over?.tableId ?? null) === (next?.tableId ?? null) &&
      (a.over?.key ?? null) === (next?.key ?? null)
    ) {
      return;
    }
    a.over = next;
    this.writeTags();
    const tableId = next?.tableId ?? a.tableId;
    const target: Target = next?.target ?? {kind: 'card', ref: a.lead};
    const e: TableEvent = {
      ...this.baseFor(tableId, target, a.via, native, client, a.modifiers),
      type: 'dragOver',
      source: a.source,
      over: next?.target ?? null,
      legal: next !== null,
    };
    this.emit(e);
    if (a.via !== 'keyboard') return;
    const policy = this.policyOf(tableId);
    if (policy?.announce?.(e) === undefined) {
      this.rt.announce(
        next
          ? `Over ${describeTarget(policy, this.sceneOf(next.tableId), next.target)}`
          : 'No target',
      );
    }
  }

  /** Resolves the target under the drag now. */
  resolveOver(native: Event): void {
    const a = this.active;
    if (!a) return;
    const policy = this.policyOf(a.tableId);
    const next =
      (policy?.dropResolve ?? 'overlap') === 'pointer'
        ? resolvePointerMode(this.rt, a)
        : resolveOverlapMode(this.rt, a);
    this.setOver(a, next, native, a.last);
  }

  // ---- lifecycle ----
  /** No text selection while a pointer drag sweeps across the page. */
  private blockSelection(): void {
    if (this.restoreSelect || typeof document === 'undefined') return;
    const s = document.body.style;
    const prev = s.getPropertyValue('user-select');
    const prevWebkit = s.getPropertyValue('-webkit-user-select');
    s.setProperty('user-select', 'none');
    s.setProperty('-webkit-user-select', 'none');
    window.getSelection()?.removeAllRanges();
    this.restoreSelect = () => {
      if (prev) s.setProperty('user-select', prev);
      else s.removeProperty('user-select');
      if (prevWebkit) s.setProperty('-webkit-user-select', prevWebkit);
      else s.removeProperty('-webkit-user-select');
      this.restoreSelect = null;
    };
  }

  private clearDrag(): void {
    this.active = null;
    this.press = null;
    this.restoreSelect?.();
    this.scroll.stop();
    this.detachWindow?.();
    this.setLegal([]);
  }

  /** Starts a drag from a press (pointer past its threshold, or Space). */
  startDrag(
    pr: Press,
    via: Via,
    native: Event,
    client: Point | null,
    modifiers: Modifiers,
  ): void {
    this.pending?.finish();
    const scene = this.sceneOf(pr.tableId);
    const source = this.sourceOf(pr.ids, pr.ref, pr.tableId);
    const legal = this.computeLegal(source);
    const leadP = defined(scene.byId(pr.ref.id), `dragged card '${pr.ref.id}'`);
    const af = affineOf(this.rt, pr.tableId);
    const leadBounds = boundsOf(quad(leadP.box).map(q => toClient(af, q)));
    const preview = createDragPreview({
      rt: this.rt,
      tableId: pr.tableId,
      scene,
      ids: source.ids,
      lead: pr.ref.id,
    });
    const start = client ?? {
      x: (leadBounds.left + leadBounds.right) / 2,
      y: (leadBounds.top + leadBounds.bottom) / 2,
    };
    const active: Active = {
      source,
      tableId: pr.tableId,
      lead: refOf(leadP),
      via,
      legal,
      over: null,
      start: via === 'keyboard' ? start : pr.start,
      last: start,
      modifiers,
      native,
      preview,
      leadBounds,
    };
    this.active = active;
    if (via !== 'keyboard') {
      preview.setOffset({x: start.x - pr.start.x, y: start.y - pr.start.y});
    }
    this.writeTags();
    this.setLegal(legal);
    this.emit({
      ...this.baseFor(
        pr.tableId,
        {kind: 'card', ref: active.lead},
        via,
        native,
        client,
        modifiers,
      ),
      type: 'dragStart',
      source,
      legal: legal.map(l => l.target),
    });
    // The game's dragStart handler may have ended the drag.
    if (this.active === active && via !== 'keyboard') {
      this.blockSelection();
      this.resolveOver(native);
      this.scroll.start();
    }
  }

  // A drop on `over`: held, or the preview returns home.
  private drop(
    a: Active,
    over: Located,
    native: Event,
    client: Point | null,
    modifiers: Modifiers,
  ): void {
    const ids = a.source.ids;
    const before = new Map(
      this.rt.tables().map(t => [t.tableId, itemsSignature(t.scene)] as const),
    );
    let res: EventResult;
    try {
      res = this.emit({
        ...this.baseFor(
          over.tableId,
          over.target,
          a.via,
          native,
          client,
          modifiers,
        ),
        type: 'drop',
        source: a.source,
      });
    } catch (err) {
      returnHome(this, ids, a.preview, a.lead.id);
      throw err;
    }
    if ((res as {hold?: unknown} | undefined)?.hold !== true) {
      returnHome(this, ids, a.preview, a.lead.id);
      return;
    }
    const p = holdDrop(this, ids, a.preview, a.lead.id);
    // The game committed new items synchronously (flushSync) inside onEvent.
    const changed = this.rt
      .tables()
      .some(
        t =>
          before.has(t.tableId) &&
          before.get(t.tableId) !== itemsSignature(t.scene),
      );
    if (changed) {
      queueMicrotask(() => {
        completeHold(this, p);
      });
    }
  }

  /** End the drag: drop on `over`, or cancel. */
  finish(
    native: Event,
    client: Point | null,
    modifiers: Modifiers,
    reason: 'release' | 'cancel',
  ): void {
    const a = this.active;
    if (!a) return;
    this.clearDrag();
    if (reason === 'release' && a.over) {
      this.drop(a, a.over, native, client, modifiers);
      return;
    }
    returnHome(this, a.source.ids, a.preview, a.lead.id);
    this.emit({
      ...this.baseFor(
        a.tableId,
        {kind: 'card', ref: a.lead},
        a.via,
        native,
        client,
        modifiers,
      ),
      type: 'dragCancel',
      source: a.source,
    });
  }
}
