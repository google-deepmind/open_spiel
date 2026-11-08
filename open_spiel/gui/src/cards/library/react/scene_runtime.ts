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

// The SceneRuntime implementation: registry, mounted tables, library
// tags written straight to the DOM, the event emitter and announcements.
// Owned by core; dnd/, motion/ and a11y/ see it only through SceneRuntime.
import type {CardId, Point} from '../core/identity.ts';
import type {EventResult, TableEvent} from '../core/events.ts';
import type {InteractionPolicy} from '../core/policy.ts';
import {activePolicy, announcementFor} from '../core/policy.ts';
import type {Target} from '../core/targets.ts';
import type {Scene} from '../core/scene.ts';
import {DEV} from '../core/dev.ts';
import {hitTest} from '../core/hit.ts';
import {isLibraryTag, LIBRARY_TAGS} from '../core/tags.ts';
import {DEFAULT_CARD_STYLE, type CardStyle} from '../style.ts';
import {rafTicker, type Ticker} from '../motion/ticker.ts';
import {createMotionService} from '../motion/service.ts';
import type {MotionService} from '../motion/service.ts';
import {createDragController, type DragController} from '../dnd/controller.ts';
import {createFocusModel, type FocusModel} from '../a11y/focus.ts';
import type {LibraryTagOwner, MountedTable, SceneRuntime} from './runtime.ts';

type CommitCb = (
  prev: ReadonlyMap<string, Scene>,
  next: ReadonlyMap<string, Scene>,
) => void;

/** The scene runtime: mounted tables, card elements and library tags. */
export class SceneRuntimeImpl implements SceneRuntime {
  private mounted = new Map<string, MountedTable>();
  private registry = new Map<CardId, HTMLElement>();
  private owners = new Map<
    LibraryTagOwner,
    ReadonlyMap<string, readonly string[]>
  >();
  private applied = new Map<string, readonly string[]>();
  private commitCbs = new Set<CommitCb>();
  private liveText = '';
  private liveListeners = new Set<(text: string) => void>();
  private fits = new Map<string, number[]>();
  private fitListeners = new Set<() => void>();
  private sharedCache: number | undefined;
  scenePolicy: InteractionPolicy | undefined;
  scale: 'shared' | 'independent' = 'independent';

  /** Sets the scene's own policy and how its tables scale. */
  setScene(
    policy: InteractionPolicy | undefined,
    scale: 'shared' | 'independent',
  ): void {
    this.scenePolicy = policy;
    this.scale = scale;
  }
  private styleRef: CardStyle = DEFAULT_CARD_STYLE;
  private tickerRef: Ticker = rafTicker;
  readonly motion: MotionService;
  readonly drag: DragController;
  readonly focus: FocusModel;

  constructor() {
    this.motion = createMotionService(this);
    this.drag = createDragController(this);
    this.focus = createFocusModel(this);
  }

  // ---- configuration from the React layer ----
  setStyle(s: CardStyle): void {
    this.styleRef = s;
  }
  setTicker(t: Ticker): void {
    this.tickerRef = t;
  }

  // ---- SceneRuntime ----
  tables(): readonly MountedTable[] {
    return [...this.mounted.values()];
  }

  table(tableId: string): MountedTable {
    const t = this.mounted.get(tableId);
    if (!t) {
      throw new Error(
        `card scene has no mounted table '${tableId}' (mounted: ${[...this.mounted.keys()].join(', ') || 'none'})`,
      );
    }
    return t;
  }

  hasTable(tableId: string): boolean {
    return this.mounted.has(tableId);
  }

  private scaleOf(t: MountedTable): {r: DOMRect; s: number} {
    const r = t.root.getBoundingClientRect();
    const s = r.width > 0 && t.scene.width > 0 ? r.width / t.scene.width : 1;
    return {r, s};
  }

  clientToTable(tableId: string, p: Point): Point {
    const {r, s} = this.scaleOf(this.table(tableId));
    return {x: (p.x - r.left) / s, y: (p.y - r.top) / s};
  }

  tableToClient(tableId: string, p: Point): Point {
    const {r, s} = this.scaleOf(this.table(tableId));
    return {x: r.left + p.x * s, y: r.top + p.y * s};
  }

  /** The target under a viewport point. `passive`: resolve for hover, context
   *  menu and long press (observed non-interactive cards count). */
  hit(
    client: Point,
    passive = false,
  ): {tableId: string; target: Target} | null {
    const tables = [...this.mounted.values()].reverse();
    for (const t of tables) {
      const p = this.clientToTable(t.tableId, client);
      if (p.x < 0 || p.y < 0 || p.x > t.scene.width || p.y > t.scene.height) {
        continue;
      }
      const policy = this.policy(t.tableId);
      const o: Parameters<typeof hitTest>[2] = {};
      if (policy?.slotInteractive) {
        o.slotInteractive = s => policy.slotInteractive?.(s) ?? false;
      }
      if (passive && policy?.observe) {
        o.observe = r => policy.observe?.(r) ?? false;
      }
      const target = hitTest(t.scene, p, o);
      if (target) return {tableId: t.tableId, target};
    }
    return null;
  }

  element(id: CardId): HTMLElement | null {
    return this.registry.get(id) ?? null;
  }

  register(id: CardId, el: HTMLElement): void {
    const prev = this.registry.get(id);
    if (prev && prev !== el && prev.isConnected) {
      throw new Error(
        `card id '${id}' is mounted twice in one card scene (ids are unique within a CardScene)`,
      );
    }
    this.registry.set(id, el);
    // Sync every library class, not only add the current ones: the same
    // node can be unregistered and registered again in one commit (a ref
    // detach and attach), and a tag cleared in between (a trick collect's
    // cleanup clearing 'held') reached no element.
    this.applyTo(id, el, this.libraryTagsFor(id), LIBRARY_TAGS);
  }

  unregister(id: CardId, el: HTMLElement): void {
    if (this.registry.get(id) === el) this.registry.delete(id);
  }

  setLibraryTags(
    owner: LibraryTagOwner,
    tags: ReadonlyMap<CardId, readonly string[]>,
  ): void {
    for (const [k, list] of tags) {
      for (const t of list) {
        if (!isLibraryTag(t)) {
          throw new Error(
            `setLibraryTags(${owner}): '${t}' on '${k}' is not a library tag (${LIBRARY_TAGS.join(', ')})`,
          );
        }
      }
    }
    const before = new Set<string>([...this.applied.keys()]);
    this.owners.set(owner, new Map(tags));
    const keys = new Set<string>(before);
    for (const m of this.owners.values()) for (const k of m.keys()) keys.add(k);
    for (const k of keys) {
      const next = this.libraryTagsFor(k);
      const prev = this.applied.get(k) ?? [];
      if (next.length === prev.length && next.every((t, i) => t === prev[i])) {
        continue;
      }
      if (next.length) this.applied.set(k, next);
      else this.applied.delete(k);
      for (const el of this.elementsFor(k)) this.applyTo(k, el, next, prev);
    }
  }

  /** The library tags on a key now (card id or target key), in LIBRARY_TAGS order. */
  libraryTagsFor(key: string): string[] {
    const all = new Set<string>();
    for (const m of this.owners.values()) {
      for (const t of m.get(key) ?? []) all.add(t);
    }
    return LIBRARY_TAGS.filter(t => all.has(t));
  }

  private elementsFor(key: string): HTMLElement[] {
    const card = this.registry.get(key);
    if (card) return [card];
    const out: HTMLElement[] = [];
    const m = /^(slot|region|insert):(.*)$/.exec(key);
    if (!m) return out;
    const attr =
      m[1] === 'slot'
        ? 'data-slot-id'
        : m[1] === 'region'
          ? 'data-region-id'
          : 'data-insert';
    for (const t of this.mounted.values()) {
      for (const el of t.root.querySelectorAll<HTMLElement>(
        `[${attr}="${(m[2] ?? '').replace(/["\\]/g, '\\$&')}"]`,
      )) {
        out.push(el);
      }
    }
    return out;
  }

  private applyTo(
    key: string,
    el: HTMLElement,
    next: readonly string[],
    prev: readonly string[] = [],
  ): void {
    for (const t of prev) {
      if (!next.includes(t)) el.classList.remove(`pcard--t-${t}`);
    }
    for (const t of next) el.classList.add(`pcard--t-${t}`);
    const game = (el.getAttribute('data-game-tags') ?? '')
      .split(' ')
      .filter(Boolean);
    const all = [...game, ...next];
    if (all.length) el.setAttribute('data-tags', all.join(' '));
    else el.removeAttribute('data-tags');
    if (!key.startsWith('card:') && !this.registry.has(key)) {
      if (next.includes('drop-legal')) {
        el.setAttribute('data-drop-legal', 'true');
      } else el.removeAttribute('data-drop-legal');
    }
  }

  emit(e: TableEvent): EventResult {
    const policy = this.policy(e.tableId);
    if (!policy) return undefined;
    const res = policy.onEvent(e);
    const text = announcementFor(policy, e);
    if (text !== undefined && text !== '') this.announce(text);
    return res;
  }

  /** The policy that answers this table's events, drags and focus: none
   *  while it is inert (policy.inert keeps only the words). */
  policy(tableId: string): InteractionPolicy | undefined {
    return activePolicy(this.mounted.get(tableId)?.policy ?? this.scenePolicy);
  }

  style(): CardStyle {
    return this.styleRef;
  }
  ticker(): Ticker {
    return this.tickerRef;
  }

  onCommit(cb: CommitCb): () => void {
    this.commitCbs.add(cb);
    return () => {
      this.commitCbs.delete(cb);
    };
  }

  // ---- mounting (called by <Table> in layout effects) ----
  commit(t: MountedTable): void {
    const prev = this.sceneMap();
    this.mounted.set(t.tableId, t);
    const next = this.sceneMap();
    for (const cb of [...this.commitCbs]) cb(prev, next);
  }

  unmount(tableId: string): void {
    if (!this.mounted.has(tableId)) return;
    const prev = this.sceneMap();
    this.mounted.delete(tableId);
    this.fits.delete(tableId);
    this.sharedCache = undefined;
    const next = this.sceneMap();
    for (const cb of [...this.commitCbs]) cb(prev, next);
  }

  private sceneMap(): Map<string, Scene> {
    return new Map([...this.mounted].map(([k, v]) => [k, v.scene]));
  }

  // ---- live region ----
  announce(text: string): void {
    // A repeat of the same words must still be spoken: alternate a trailing no-break space.
    this.liveText =
      text === this.liveText.replace(/\u00a0$/, '') &&
      !this.liveText.endsWith('\u00a0')
        ? `${text}\u00a0`
        : text;
    for (const l of this.liveListeners) l(this.liveText);
  }

  onLive(l: (text: string) => void): () => void {
    this.liveListeners.add(l);
    return () => {
      this.liveListeners.delete(l);
    };
  }

  // ---- one responsive scale across Tables (CardScene scale='shared') ----
  reportFit(tableId: string, width: number): void {
    const hist = this.fits.get(tableId) ?? [];
    if (hist[hist.length - 1] === width) return;
    hist.push(width);
    if (hist.length > 8) hist.shift();
    this.fits.set(tableId, hist);
    // a, b, a, b: the table's box depends on the card width it is drawn at.
    if (DEV && hist.length >= 4) {
      const [a, b, c, d] = hist.slice(-4);
      if (a === c && b === d && a !== b) {
        throw new Error(
          `<CardScene scale="shared">: table '${tableId}' oscillates between card widths ${String(a)} and ${String(b)}; a child Table's box must not depend on its card width`,
        );
      }
    }
    const shared = this.sharedWidth();
    if (shared !== this.sharedCache) {
      this.sharedCache = shared;
      for (const l of [...this.fitListeners]) l();
    }
  }

  sharedWidth(): number | undefined {
    if (this.scale !== 'shared' || this.fits.size === 0) return undefined;
    return Math.min(...[...this.fits.values()].map(h => h.at(-1) ?? Infinity));
  }

  onFit(l: () => void): () => void {
    this.fitListeners.add(l);
    return () => {
      this.fitListeners.delete(l);
    };
  }
}
