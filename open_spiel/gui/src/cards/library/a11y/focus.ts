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

// FocusModel: roving focus across a CardScene.
//
// - One tab stop per CardScene: the last-focused target, or else the first
//   interactive target in slot order (tables in mount order). The DOM is
//   normalised after every commit and every focus move, so several <Table>s in
//   one scene still expose exactly one tabindex="0".
// - Arrow keys move to nearestInDirection among the interactive targets of the
//   same table (computed geometry); when the table has nothing that way,
//   the other tables of the scene are searched by their on-screen centres.
//   Home and End go to the first and last target in slot order. In a hand
//   wrapped onto several rows, left and right read on across the wrap (row-
//   major, like text): past a row's last card, right goes to the next row's
//   first, and left from a row's first card to the previous row's last.
// - Activation keys (Enter, Shift+Enter, Space, Shift+F10, ContextMenu) and
//   the keyboard drag are the input layer's: onKey returns 'unhandled' for them.
// - The focus ring is the library tag 'focus', present only while focus is
//   VISIBLE (focus-visible): focus that follows a keyboard interaction, or a
//   programmatic move after one. A mouse or touch press focuses without the
//   ring. Its look is CardStyle.tags.focus (an offset outline; cards.css).
// - When an items change removes the focused id, focus moves to its slot's new
//   top card, then to the slot, then to the first target. DOM focus follows
//   only when it was on the removed node (focus is never stolen).
import {a11yModel} from '../core/a11y.ts';
import type {Point} from '../core/identity.ts';
import type {InteractionPolicy} from '../core/policy.ts';
import {sceneInternals, type Scene} from '../core/scene.ts';
import {nearestInDirection, targetBox} from '../core/hit.ts';
import type {ItemRef, Target} from '../core/targets.ts';
import {targetKey} from '../core/targets.ts';
import type {SceneRuntime} from '../react/runtime.ts';

/** Roving focus across a card scene: one tab stop, arrow keys between targets. */
export interface FocusModel {
  /** targetKey. */
  current(): string | null;
  set(t: Target | null): void;
  onKey(
    e: KeyboardEvent,
    scene: Scene,
    policy: InteractionPolicy | undefined,
  ): 'handled' | 'unhandled';
}

// ---- input modality (the focus-visible heuristic) ----
// Keyboard until the first pointer press, as browsers and the focus-visible
// polyfill do: focus moved by script at load is visible. One pair of capture
// listeners per document, installed once; they hold no scene state.
let modality: 'keyboard' | 'pointer' = 'keyboard';
const watched = new WeakSet<Document>();

function watchModality(doc: Document | undefined): void {
  if (!doc || watched.has(doc)) return;
  watched.add(doc);
  doc.addEventListener(
    'keydown',
    e => {
      if (e.metaKey || e.altKey || e.ctrlKey) return; // shortcuts do not make focus visible
      modality = 'keyboard';
    },
    true,
  );
  const pointer = () => {
    modality = 'pointer';
  };
  doc.addEventListener('pointerdown', pointer, true);
  doc.addEventListener('mousedown', pointer, true);
  doc.addEventListener('touchstart', pointer, true);
}

/** The current input modality (for tests). */
export function inputModality(): 'keyboard' | 'pointer' {
  return modality;
}

// ---- focusable targets ----
interface Entry {
  tableId: string;
  key: string;
  target: Target;
  slot: string;
}

function refOf(scene: Scene, id: string): ItemRef | null {
  const p = scene.byId(id);
  return p
    ? {id: p.id, slot: p.slot, position: p.position, item: p.item}
    : null;
}

/** The target a targetKey names in a scene ('card:<id>' or 'slot:<id>'). */
function targetOf(scene: Scene, key: string): Target | null {
  if (key.startsWith('card:')) {
    const ref = refOf(scene, key.slice(5));
    return ref ? {kind: 'card', ref} : null;
  }
  if (key.startsWith('slot:')) return {kind: 'slot', slot: key.slice(5)};
  return null;
}

const ARROWS = {
  ArrowUp: 'up',
  ArrowDown: 'down',
  ArrowLeft: 'left',
  ArrowRight: 'right',
} as const;
type Dir = (typeof ARROWS)[keyof typeof ARROWS];
/** Clockwise, so a slot turned by r quarter turns maps screen directions to
 *  its own frame by stepping back r places. */
const CLOCKWISE: readonly Dir[] = ['up', 'right', 'down', 'left'];

/**
 * The reading-order step in a wrapped hand (rows > 1): the nearest card of
 * the same slot that way, else, for left/right in the hand's own frame, the
 * previous or next card by position (across the wrap). Null when `from` is
 * not a card of a wrapped hand, or the hand has nothing that way.
 */
export function readingStep(
  scene: Scene,
  from: Target,
  dir: Dir,
  among: readonly Target[],
): Target | null {
  if (from.kind !== 'card') return null;
  const l = sceneInternals(scene).slotById.get(from.ref.slot);
  if (l?.kind !== 'hand' || !l.hand || l.hand.rows < 2) return null;
  const inSlot = among.filter(
    (t): t is Extract<Target, {kind: 'card'}> =>
      t.kind === 'card' && t.ref.slot === from.ref.slot,
  );
  const near = nearestInDirection(scene, from, dir, inSlot);
  if (near) return near;
  const quarter = Math.round(((((l.spec.rotate ?? 0) % 360) + 360) % 360) / 90);
  const local = CLOCKWISE[(CLOCKWISE.indexOf(dir) - quarter + 4) % 4];
  if (local !== 'left' && local !== 'right') return null;
  const pos = from.ref.position;
  const ahead = inSlot.filter(t =>
    local === 'right' ? t.ref.position > pos : t.ref.position < pos,
  );
  if (ahead.length === 0) return null;
  return ahead.reduce((a, b) =>
    local === 'right'
      ? b.ref.position < a.ref.position
        ? b
        : a
      : b.ref.position > a.ref.position
        ? b
        : a,
  );
}

// The focused target: its key, table and slot.
interface Current {
  key: string;
  tableId: string;
  slot: string;
}

// Whether `el` holds DOM focus.
function hasDomFocus(el: HTMLElement | null): boolean {
  return (
    el !== null &&
    typeof document !== 'undefined' &&
    document.activeElement === el
  );
}

// The best entry of another table in a direction, by on-screen centres.
function acrossTables(
  from: Entry,
  dir: Dir,
  among: readonly Entry[],
  centreOf: (e: Entry) => Point | null,
): Entry | null {
  const c0 = centreOf(from);
  if (!c0) return null;
  let best: Entry | null = null;
  let bestScore = Infinity;
  for (const e of among) {
    if (e.tableId === from.tableId) continue;
    const c = centreOf(e);
    if (!c) continue;
    const dx = c.x - c0.x;
    const dy = c.y - c0.y;
    const along =
      dir === 'right' ? dx : dir === 'left' ? -dx : dir === 'down' ? dy : -dy;
    const across =
      dir === 'right' || dir === 'left' ? Math.abs(dy) : Math.abs(dx);
    if (along <= 1e-6 || across > along) continue;
    const score = along + 2 * across;
    if (score < bestScore) {
      bestScore = score;
      best = e;
    }
  }
  return best;
}

/** The library-tag key for a target key: the card id, or the slot's target key. */
function tagKey(key: string): string {
  return key.startsWith('card:') ? key.slice(5) : key;
}

// The roving focus of one card scene (FocusModel).
class RovingFocus implements FocusModel {
  private cur: Current | null = null;
  private visible = false;
  private lastEl: HTMLElement | null = null;
  /** DOM focus is on lastEl. A blur while lastEl is still connected clears
   *  it (the user went elsewhere); a blur caused by its removal does not,
   *  so the repair can bring focus back to the replacement. */
  private hadDom = false;
  private untrack: (() => void) | null = null;
  private readonly rt: SceneRuntime;

  constructor(rt: SceneRuntime) {
    this.rt = rt;
    // Focus repair and the one-tab-stop invariant after every commit.
    rt.onCommit((_prev, next) => {
      this.repair(next);
    });
  }

  readonly current = (): string | null => this.cur?.key ?? null;

  readonly set = (t: Target | null): void => {
    if (t === null) {
      this.clear();
      return;
    }
    const e = this.locate(t);
    if (!e) {
      throw new Error(
        `focus(${targetKey(t)}): not a focusable target in this card scene (focus goes only to interactive cards, tokens and slotInteractive slots of a Table with a policy)`,
      );
    }
    this.visible = modality === 'keyboard';
    this.moveTo(e, false, true);
  };

  readonly onKey = (
    e: KeyboardEvent,
    _scene: Scene,
    policy: InteractionPolicy | undefined,
  ): 'handled' | 'unhandled' => {
    if (!policy) return 'unhandled';
    // A key on a focused target makes its focus visible (focus-visible).
    if (!(e.metaKey || e.altKey || e.ctrlKey) && this.cur && !this.visible) {
      this.visible = true;
      this.applyTag(true);
    }
    const dir: Dir | undefined = (ARROWS as Partial<Record<string, Dir>>)[
      e.key
    ];
    const edge = e.key === 'Home' ? 'first' : e.key === 'End' ? 'last' : null;
    if (!dir && !edge) return 'unhandled';
    if (e.metaKey || e.altKey || e.ctrlKey || e.shiftKey) return 'unhandled'; // modified arrows are the game's ('key')
    const entries = this.allEntries();
    if (entries.length === 0) return 'unhandled';
    const next = dir
      ? this.stepFrom(e, entries, dir)
      : edge === 'first'
        ? entries[0]
        : entries[entries.length - 1];
    // A bound key: the page must not scroll even when there is nowhere to go.
    e.preventDefault();
    if (next) {
      this.visible = true;
      this.moveTo(next, true);
    }
    return 'handled';
  };

  private clear(): void {
    this.cur = null;
    this.track(null);
    this.applyTag();
    this.normalise();
  }

  // The entry an arrow key moves to from the focused one.
  private stepFrom(
    e: KeyboardEvent,
    entries: readonly Entry[],
    dir: Dir,
  ): Entry | null | undefined {
    const from = this.fromEvent(e, entries);
    if (!from) return entries[0];
    const same = entries.filter(x => x.tableId === from.tableId);
    const tableScene = this.rt.table(from.tableId).scene;
    const targets = same.map(x => x.target);
    const t =
      readingStep(tableScene, from.target, dir, targets) ??
      nearestInDirection(tableScene, from.target, dir, targets);
    return t
      ? (same.find(x => x.key === targetKey(t)) ?? null)
      : acrossTables(from, dir, entries, x => this.clientCentre(x));
  }

  private track(el: HTMLElement | null): void {
    if (el === this.lastEl && this.untrack) return;
    this.untrack?.();
    this.untrack = null;
    this.lastEl = el;
    this.hadDom = hasDomFocus(el);
    if (!el) return;
    const onFocus = (): void => {
      this.hadDom = true;
    };
    const onBlur = (): void => {
      queueMicrotask(() => {
        if (el.isConnected && this.lastEl === el) this.hadDom = false;
      });
    };
    el.addEventListener('focus', onFocus);
    el.addEventListener('blur', onBlur);
    this.untrack = () => {
      el.removeEventListener('focus', onFocus);
      el.removeEventListener('blur', onBlur);
    };
  }

  /** Focusable targets of one table, in slot order (exactly a11yModel's buttons). */
  private tableEntries(tableId: string, scene: Scene): Entry[] {
    const policy = this.rt.policy(tableId);
    if (!policy) return [];
    const out: Entry[] = [];
    for (const n of a11yModel(scene, policy, null)) {
      if (n.role !== 'button' || n.hidden) continue;
      const target = targetOf(scene, n.key);
      if (!target) continue;
      out.push({
        tableId,
        key: n.key,
        target,
        slot:
          target.kind === 'card'
            ? target.ref.slot
            : target.kind === 'slot'
              ? target.slot
              : '',
      });
    }
    return out;
  }

  private allEntries(): Entry[] {
    return this.rt.tables().flatMap(t => this.tableEntries(t.tableId, t.scene));
  }

  private elementOf(tableId: string, key: string): HTMLElement | null {
    if (key.startsWith('card:')) {
      const el = this.rt.element(key.slice(5));
      return el?.getAttribute('role') === 'button' ? el : null;
    }
    if (key.startsWith('slot:')) {
      const slot = key.slice(5);
      for (const el of this.rt
        .table(tableId)
        .root.querySelectorAll<HTMLElement>('[data-slot-id]')) {
        if (
          el.getAttribute('data-slot-id') === slot &&
          el.getAttribute('role') === 'button'
        ) {
          return el;
        }
      }
    }
    return null;
  }

  private focusInside(): boolean {
    const a = typeof document === 'undefined' ? null : document.activeElement;
    return (
      !!a &&
      a !== document.body &&
      this.rt.tables().some(t => t.root.contains(a))
    );
  }

  /** The ring is drawn only while focus is visible and in the scene. */
  private applyTag(inScene = this.focusInside()): void {
    const {cur} = this;
    this.rt.setLibraryTags(
      'focus',
      cur && this.visible && inScene
        ? new Map([[tagKey(cur.key), ['focus']]])
        : new Map(),
    );
  }

  // The entry of the focused target among `entries`, if it is there.
  private currentIn(entries: readonly Entry[]): Entry | undefined {
    const {cur} = this;
    if (!cur) return undefined;
    return entries.find(e => e.key === cur.key && e.tableId === cur.tableId);
  }

  /** Exactly one tabindex="0" across the scene's tables. Only nodes the a11y
   *  model made buttons are touched. */
  private normalise(): void {
    const entries = this.allEntries();
    const stopEntry = this.currentIn(entries) ?? entries[0];
    const stop = stopEntry
      ? this.elementOf(stopEntry.tableId, stopEntry.key)
      : null;
    for (const t of this.rt.tables()) {
      for (const el of t.root.querySelectorAll<HTMLElement>(
        '[data-card-id][role="button"][tabindex],[data-slot-id][role="button"][tabindex]',
      )) {
        const want = el === stop ? '0' : '-1';
        if (el.getAttribute('tabindex') !== want) {
          el.setAttribute('tabindex', want);
        }
      }
    }
  }

  /** Which mounted table a target lives in; prefers the one holding DOM focus. */
  private locate(t: Target): Entry | null {
    const key = targetKey(t);
    const hits = this.allEntries().filter(e => e.key === key);
    if (hits.length <= 1) return hits[0] ?? null;
    const active =
      typeof document === 'undefined' ? null : document.activeElement;
    return (
      hits.find(
        e => active && this.rt.table(e.tableId).root.contains(active),
      ) ??
      hits[0] ??
      null
    );
  }

  /** domFocus: also move DOM focus there (keys, repair). inScene: draw the
   *  ring even though DOM focus is not (yet) inside (set() is called from
   *  focusin, and by CardSceneApi.focus just before it focuses the node). */
  private moveTo(e: Entry, domFocus: boolean, inScene = false): void {
    this.cur = {key: e.key, tableId: e.tableId, slot: e.slot};
    const el = this.elementOf(e.tableId, e.key);
    this.track(el);
    this.normalise();
    if (domFocus && el && typeof document !== 'undefined' && !hasDomFocus(el)) {
      el.focus();
    }
    if (hasDomFocus(el)) this.hadDom = true;
    this.applyTag(inScene || domFocus || this.focusInside());
  }

  /** On-screen centre of an entry (cross-table navigation only). */
  private clientCentre(e: Entry): Point | null {
    const b = targetBox(this.rt.table(e.tableId).scene, e.target);
    if (!b) return null;
    return this.rt.tableToClient(e.tableId, {
      x: b.x + b.width / 2,
      y: b.y + b.height / 2,
    });
  }

  /** The focused entry for a key event: from the event's element, else cur. */
  private fromEvent(e: KeyboardEvent, entries: readonly Entry[]): Entry | null {
    const el =
      e.target instanceof Element
        ? e.target.closest('[data-card-id],[data-slot-id]')
        : null;
    const tableEl = el?.closest('[data-table-id]');
    const tableId = tableEl?.getAttribute('data-table-id');
    if (el && tableId !== null && tableId !== undefined) {
      const key = el.hasAttribute('data-card-id')
        ? `card:${String(el.getAttribute('data-card-id'))}`
        : `slot:${String(el.getAttribute('data-slot-id'))}`;
      const hit = entries.find(x => x.tableId === tableId && x.key === key);
      if (hit) return hit;
    }
    return this.currentIn(entries) ?? null;
  }

  // After a commit: keep focus on its target, or move it to the slot's new
  // top card, the slot, or the first target.
  private repair(next: ReadonlyMap<string, Scene>): void {
    const {cur} = this;
    if (!cur) {
      this.normalise();
      return;
    }
    const removed = this.lastEl;
    // DOM focus was on the node this commit unmounted (the focused card was
    // removed, or moved to another slot and re-mounted there).
    const lostDom =
      this.hadDom &&
      !!removed &&
      !removed.isConnected &&
      typeof document !== 'undefined' &&
      (document.activeElement === null ||
        document.activeElement === document.body);
    const scene = next.get(cur.tableId);
    const entries = scene ? this.tableEntries(cur.tableId, scene) : [];
    const still = entries.find(x => x.key === cur.key);
    if (still) {
      if (lostDom) {
        this.moveTo(still, true);
        return;
      }
      this.cur = {key: still.key, tableId: still.tableId, slot: still.slot};
      const el = this.elementOf(still.tableId, still.key);
      if (el && el !== this.lastEl) this.track(el);
      this.normalise();
      return;
    }
    const repl = this.replacement(entries, cur.slot);
    if (repl) {
      this.moveTo(repl, lostDom);
      return;
    }
    this.clear();
  }

  // Where focus goes when its target is gone: the slot's top card, the
  // slot, or the first target.
  private replacement(
    entries: readonly Entry[],
    slot: string,
  ): Entry | undefined {
    const inSlot = entries.filter(
      x => x.target.kind === 'card' && x.slot === slot,
    );
    const pos = (x: Entry): number =>
      x.target.kind === 'card' ? x.target.ref.position : -1;
    const top = inSlot.length
      ? inSlot.reduce((a, b) => (pos(a) >= pos(b) ? a : b))
      : undefined;
    return (
      top ??
      entries.find(x => x.target.kind === 'slot' && x.slot === slot) ??
      this.allEntries()[0]
    );
  }
}

/** The roving focus of the card scene `rt`. */
export function createFocusModel(rt: SceneRuntime): FocusModel {
  watchModality(typeof document === 'undefined' ? undefined : document);
  return new RovingFocus(rt);
}
