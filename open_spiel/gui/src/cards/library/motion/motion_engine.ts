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

// The state behind one MotionService: the flights in progress, the free
// cards, the motion in flight for settled(), and where each card was last
// laid out, for cards that fly between the Tables of one card scene.
import type {CardId, TableItem} from '../core/identity.ts';
import {isToken} from '../core/identity.ts';
import type {Transition} from '../core/motion_plan.ts';
import type {Scene} from '../core/scene.ts';
import type {MountedTable} from '../react/runtime.ts';
import type {CardStyle} from '../style.ts';
import type {Card} from '../types.ts';
import {parseEasing, type Easing} from './easing.ts';
import {animate} from './flip_engine.ts';
import {MotionMeasure, doc} from './motion_measure.ts';
import {
  makeFreeCard,
  type FreeCard,
  type FreeCardHost,
  type Frame,
} from './free_card.ts';
import {
  motionDuration,
  motionStagger,
  tableLayer,
  viewportLayer,
} from './motion_timing.ts';
import {
  drawnFace,
  hasLayout,
  poseOf,
  slotRotation,
  tableScale,
  type AnimateRequest,
  type CrossFrom,
  type FlipSide,
  type MotionHost,
  type Pose2,
  type Seen,
} from './motion_util.ts';

/** A card that left one Table this recently (ms) may fly into another. */
const CROSS_TABLE_WINDOW_MS = 100;

// What one plan's cards share while their jobs are made.
interface PlanPass {
  skip: (id: CardId) => boolean;
  duration: number;
  easing: Easing;
  style: CardStyle['motion'];
  isLaidOut: (t: MountedTable) => boolean;
}

/** The motion engine of one card scene (MotionService's state). */
export class MotionEngine extends MotionMeasure implements FreeCardHost {
  readonly free = new Map<CardId, FreeCard>();
  private readonly inflight = new Set<Promise<void>>();
  private waiters: Array<() => void> = [];
  /** The scene each table showed before its latest change. */
  private readonly prevScenes = new Map<string, Scene | null>();
  /** Where each id was last laid out, for cards that change Table. */
  private readonly lastSeen = new Map<CardId, Seen>();
  private readonly crossFrom = new Map<CardId, CrossFrom>();
  private suppress: {all: boolean; ids: Set<CardId>} | null = null;
  private suppressConsumed = false;
  constructor(host: MotionHost) {
    super(host);
    // Scene changes: remember where cards were, for cross-table flights.
    host.onCommit((prev, next) => {
      this.onCommit(prev, next);
    });
  }

  /** Counts `p` as motion in flight for settled(). */
  readonly track = (p: Promise<void>): Promise<void> => {
    this.inflight.add(p);
    const done = (): void => {
      this.inflight.delete(p);
      if (this.inflight.size === 0) {
        const w = this.waiters;
        this.waiters = [];
        for (const f of w) f();
      }
    };
    p.then(done, done);
    return p;
  };

  /** Resolves when no motion is in flight. */
  settled(): Promise<void> {
    return this.inflight.size === 0
      ? Promise.resolve()
      : new Promise<void>(r => {
          this.waiters.push(r);
        });
  }

  /** Tags the free cards 'held' in the layout. */
  readonly pushHeld = (): void => {
    this.host.setLibraryTags(
      'motion',
      new Map([...this.free.keys()].map(id => [id, ['held']] as const)),
    );
  };

  private stopFlight(id: CardId, restore: boolean): void {
    this.flights.get(id)?.stop(restore);
  }

  // ---- the FLIP engine ----

  private prevCardOf(
    tableId: string,
    id: CardId,
    was: CardId | undefined,
  ): Card | null {
    const ps = this.prevScenes.get(tableId);
    const q = ps?.byId(id) ?? (was !== undefined ? ps?.byId(was) : undefined);
    return q && !isToken(q.item) ? q.item.card : null;
  }

  private consumeSuppression(): (id: CardId) => boolean {
    const s = this.suppress;
    if (!s) return () => false;
    if (!this.suppressConsumed) {
      this.suppressConsumed = true;
      // Every Table committing in this flush sees it; the next change does not.
      queueMicrotask(() => {
        if (this.suppress === s) this.suppress = null;
        this.suppressConsumed = false;
      });
    }
    return id => s.all || s.ids.has(id);
  }

  /** Skips FLIP for the next items change (only for `ids` when given). */
  readonly suppressNext = (ids?: readonly CardId[]): void => {
    if (ids === undefined) {
      this.suppress = {all: true, ids: new Set()};
      return;
    }
    const given: unknown = ids;
    if (!Array.isArray(given)) {
      throw new Error('motion.suppressNext: ids must be an array of card ids');
    }
    for (const id of given as unknown[]) {
      if (typeof id !== 'string' || id === '') {
        throw new Error(
          `motion.suppressNext: bad card id ${JSON.stringify(id)}`,
        );
      }
    }
    if (!this.suppress || this.suppressConsumed) {
      this.suppress = {all: false, ids: new Set()};
    }
    for (const id of ids) this.suppress.ids.add(id);
  };

  // Where a transition's card flies from, and the side it shows there; null
  // when it does not fly (an entering card that came from no other table).
  private startOf(
    tr: Transition,
    t: MountedTable,
    was: CardId | undefined,
  ): {from: Pose2; fromFace: FlipSide} | null {
    if (tr.kind === 'enter') {
      const c = this.crossFrom.get(tr.id);
      this.crossFrom.delete(tr.id);
      if (!c) return null;
      const p = this.host.clientToTable(t.tableId, {x: c.cx, y: c.cy});
      return {
        from: {
          cx: p.x,
          cy: p.y,
          rotate: c.rotate,
          width: c.width / tableScale(t),
        },
        fromFace: {card: c.card, faceUp: c.faceUp},
      };
    }
    let from = poseOf(tr.from);
    const inFlight =
      this.flights.get(tr.id) ??
      (was !== undefined ? this.flights.get(was) : undefined);
    if (inFlight?.tableId === t.tableId) from = {...inFlight.pose};
    const card = this.prevCardOf(t.tableId, tr.id, was);
    return {
      from,
      fromFace: {
        card: tr.from.faceUp ? card : null,
        faceUp: tr.from.faceUp && card !== null,
      },
    };
  }

  // The FLIP request of one transition, or null when the card does not move.
  private requestFor(
    pass: PlanPass,
    tr: Transition,
    delayMs: number,
  ): AnimateRequest | null {
    const el = this.host.element(tr.id);
    if (!el || this.free.has(tr.id)) return null;
    const t = this.tableOf(tr.id);
    const placed = t?.scene.byId(tr.id);
    if (!t || !placed) return null;
    const item = placed.item;
    const was = isToken(item) ? undefined : item.was;
    if (was !== undefined && this.flights.get(was)?.el !== el) {
      this.stopFlight(was, true);
    }
    if (
      pass.skip(tr.id) ||
      (was !== undefined && pass.skip(was)) ||
      pass.duration <= 0 ||
      !pass.isLaidOut(t)
    ) {
      this.stopFlight(tr.id, true);
      this.crossFrom.delete(tr.id);
      return null;
    }
    const ps = this.prevScenes.get(t.tableId);
    if (ps && ps.cardWidth !== t.scene.cardWidth) {
      // A resize, not a move: the whole table rescaled.
      this.stopFlight(tr.id, true);
      return null;
    }
    const start = this.startOf(tr, t, was);
    if (!start) return null;
    const to = poseOf(tr.to);
    const flips =
      pass.style.flip &&
      (start.fromFace.faceUp !== tr.to.faceUp ||
        tr.kind === 'flip' ||
        tr.kind === 'move+flip');
    const moves = tr.kind !== 'flip';
    if (!moves && !flips) return null;
    return {
      id: tr.id,
      el,
      tableId: t.tableId,
      from: moves ? start.from : to,
      to,
      slotRotate: slotRotation(el),
      oldSide: flips ? start.fromFace : null,
      durationMs: pass.duration,
      delayMs,
      easing: pass.easing,
      raise: tr.kind === 'enter' || !tr.sameSlot,
    };
  }

  /** Plays a plan of transitions (runFlip's body). */
  readonly runPlan = (
    plan: readonly Transition[],
    style: CardStyle['motion'],
  ): Promise<void> => {
    if (plan.length === 0) return Promise.resolve();
    const laidOut = new Map<string, boolean>();
    const pass: PlanPass = {
      skip: this.consumeSuppression(),
      duration: motionDuration(style),
      easing: parseEasing(style.easing),
      style,
      isLaidOut: t => {
        let v = laidOut.get(t.tableId);
        if (v === undefined) laidOut.set(t.tableId, (v = hasLayout(t)));
        return v;
      },
    };
    const stagger = motionStagger(style);
    const jobs: Array<Promise<void>> = [];
    for (const tr of plan) {
      if (tr.kind === 'exit') continue;
      const req = this.requestFor(pass, tr, jobs.length * stagger);
      if (req) jobs.push(animate(this.host, this.flights, req));
    }
    if (jobs.length === 0) return Promise.resolve();
    return this.track(Promise.all(jobs).then(() => undefined));
  };

  /** FLIPs a laid-out card from a viewport pose to its place. */
  readonly flyFrom = (
    id: CardId,
    from: {cx: number; cy: number; rotate: number; width: number},
    o?: {oldSide?: FlipSide},
  ): Promise<void> => {
    const el = this.host.element(id);
    const t = this.tableOf(id);
    const placed = t?.scene.byId(id);
    if (!el || !t || !placed || this.free.has(id) || !hasLayout(t)) {
      return Promise.resolve();
    }
    const style = this.host.style().motion;
    const duration = motionDuration(style);
    if (duration <= 0) {
      this.stopFlight(id, true);
      return Promise.resolve();
    }
    const p = this.host.clientToTable(t.tableId, {x: from.cx, y: from.cy});
    const s = tableScale(t);
    return this.track(
      animate(this.host, this.flights, {
        id,
        el,
        tableId: t.tableId,
        from: {cx: p.x, cy: p.y, rotate: from.rotate, width: from.width / s},
        to: poseOf(placed.box),
        slotRotate: slotRotation(el),
        oldSide: o?.oldSide ?? null,
        durationMs: duration,
        delayMs: 0,
        easing: parseEasing(style.easing),
        raise: true,
      }),
    );
  };

  // ---- scene changes ----

  private markExit(tid: string, ids: readonly CardId[], now: number): void {
    for (const id of ids) {
      const ls = this.lastSeen.get(id);
      if (ls?.tableId === tid) ls.exitAt = now;
    }
  }

  // A card that arrived in table `tid` from another Table just now flies in
  // from where it was there.
  private noteArrival(
    tid: string,
    q: {id: CardId; item: TableItem},
    now: number,
  ): void {
    const was = isToken(q.item) ? undefined : q.item.was;
    const src =
      this.lastSeen.get(q.id) ??
      (was !== undefined ? this.lastSeen.get(was) : undefined);
    if (!src || src.tableId === tid) return;
    if (src.exitAt !== undefined && now - src.exitAt > CROSS_TABLE_WINDOW_MS) {
      return;
    }
    const st = this.host.tables().find(x => x.tableId === src.tableId);
    if (!st || !hasLayout(st)) return;
    const c = poseOf(src.box);
    const pt = this.host.tableToClient(st.tableId, {x: c.cx, y: c.cy});
    this.crossFrom.set(q.id, {
      cx: pt.x,
      cy: pt.y,
      rotate: c.rotate,
      width: c.width * tableScale(st),
      ...drawnFace(src.item),
    });
  }

  private onCommit(
    prev: ReadonlyMap<string, Scene>,
    next: ReadonlyMap<string, Scene>,
  ): void {
    const now = this.host.ticker().now();
    for (const [tid, p] of prev) {
      if (next.has(tid)) continue;
      this.prevScenes.delete(tid);
      this.markExit(
        tid,
        p.placements.map(q => q.id),
        now,
      );
    }
    for (const [tid, scene] of next) {
      const p = prev.get(tid);
      if (p === scene) continue;
      this.prevScenes.set(tid, p ?? null);
      if (p) {
        const gone = p.placements.filter(q => !scene.byId(q.id));
        this.markExit(
          tid,
          gone.map(q => q.id),
          now,
        );
      }
      for (const q of scene.placements) {
        if (!p?.byId(q.id)) this.noteArrival(tid, q, now);
      }
      for (const q of scene.placements) {
        this.lastSeen.set(q.id, {tableId: tid, box: q.box, item: q.item});
      }
    }
    if (this.crossFrom.size) {
      queueMicrotask(() => {
        this.crossFrom.clear();
      });
    }
  }

  // ---- free ----

  /** Lifts cards out of the layout into the free layer. */
  readonly freeCards = (
    ids: readonly CardId[],
    o?: {frame?: Frame},
  ): Promise<FreeCard[]> => {
    try {
      return Promise.resolve(this.freeNow(ids, o?.frame ?? 'viewport'));
    } catch (e) {
      return Promise.reject(e instanceof Error ? e : new Error(String(e)));
    }
  };

  private freeNow(ids: readonly CardId[], frame: Frame): FreeCard[] {
    const given: unknown = frame;
    if (given !== 'viewport' && given !== 'table') {
      throw new Error(`motion.free: unknown frame ${JSON.stringify(given)}`);
    }
    const list: unknown = ids;
    if (!Array.isArray(list)) {
      throw new Error('motion.free: ids must be an array');
    }
    const seen = new Set<CardId>();
    const plans = ids.map(id => {
      if (seen.has(id)) {
        throw new Error(`motion.free: card '${id}' is listed twice`);
      }
      seen.add(id);
      if (this.free.has(id)) {
        throw new Error(
          `motion.free: card '${id}' is already free (dispose it first)`,
        );
      }
      const el = this.host.element(id);
      const t = this.tableOf(id);
      const placed = t?.scene.byId(id);
      if (!el || !t || !placed) {
        throw new Error(
          `motion.free: no card '${id}' is laid out in this card scene`,
        );
      }
      if (isToken(placed.item)) {
        throw new Error(
          `motion.free: '${id}' is a token; only cards can be freed`,
        );
      }
      return {id, el, t, item: placed.item};
    });
    const out = plans.map(({id, el, t, item}) => {
      const launch = this.measure(id, frame);
      if (!launch) {
        throw new Error(`motion.free: card '${id}' cannot be measured`);
      }
      this.stopFlight(id, true);
      return makeFreeCard(this, {
        id,
        el,
        tableId: t.tableId,
        item,
        frame,
        launch,
        scale: frame === 'viewport' ? tableScale(t) : 1,
        layer: frame === 'viewport' ? viewportLayer(doc()) : tableLayer(t.root),
      });
    });
    this.pushHeld();
    return out;
  }
}
