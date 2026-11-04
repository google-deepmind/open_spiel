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

// MotionService: measurement, the free layer, the frame loop, and the
// FLIP engine behind runFlip. Public via useCardScene().motion. Owned by the
// motion module; it sees the scene only through the SceneRuntime interface.
//
// Geometry. FLIP deltas come from the computed scenes: a card's
// from/to boxes are table px, and the delta is written as a transform in the
// card's own slot frame, so it is exact under any ancestor transform (the
// same ancestors apply to both ends) and works without a layout engine.
// Measured geometry (getBoundingClientRect) is what leaves the layout:
// measure*, free() launch boxes, FreeCard.dispose({home}) and cards that fly
// between two Tables of one CardScene.
//
// Performance: every per-frame path writes `transform` straight to a DOM
// node. Nothing here causes a React render.
import type {Box, CardId, Rect} from '../core/identity.ts';
import type {Transition} from '../core/motion_plan.ts';
import type {CardStyle} from '../style.ts';
import type {FreeCard, Frame} from './free_card.ts';
import {MotionEngine} from './motion_engine.ts';
import type {FlipSide, MotionHost} from './motion_util.ts';

export type {FreeCard, Frame, Pose} from './free_card.ts';
export {
  FLIGHT_Z,
  type AnimateRequest,
  type FlipSide,
  type MotionHost,
} from './motion_util.ts';
export {
  VIEWPORT_LAYER_Z,
  motionDuration,
  motionStagger,
  prefersReducedMotion,
  tableLayer,
  viewportLayer,
} from './motion_timing.ts';

/** A running frame loop: when it ends, and how to stop it. */
export interface Run {
  done: Promise<void>;
  cancel(): void;
}

/** A card scene's motion: measurement, free cards, the frame loop. */
export interface MotionService {
  /** Measured on-screen box (getBoundingClientRect, after all transforms). Default frame 'viewport'. */
  measure(id: CardId, frame?: Frame): Box | null;
  measureSlot(slot: string, frame?: Frame, tableId?: string): Box | null;
  measureRegion(region: string, frame?: Frame, tableId?: string): Box | null;
  /** The frame's visible box. */
  viewport(frame?: Frame, tableId?: string): Rect;
  /** Lift cards out of the layout into the free layer, at their measured boxes.
   *  The layout keeps their space and hides them (tag 'held') until dispose.
   *  The nodes are pinned: a React re-render cannot reparent or unmount them.
   *  Default frame 'viewport'. */
  free(ids: readonly CardId[], o?: {frame?: Frame}): Promise<FreeCard[]>;
  /** Frame loop on the scene's ticker. step returns true to continue. dt is capped at 50 ms. */
  run(step: (dtMs: number, tMs: number) => boolean): Run;
  /** Skip FLIP for the next items change (optionally only for these ids). */
  suppressNext(ids?: readonly CardId[]): void;
  /** Resolves when no FLIP, flip or held drop is in flight (tests and bots). */
  settled(): Promise<void>;
}

/** Frame-loop dt cap. */
export const MAX_DT_MS = 50;

/** What runFlip, dnd and tests reach through motionInternals. */
export interface MotionInternals {
  /** runFlip's body. */
  runPlan(
    plan: readonly Transition[],
    style: CardStyle['motion'],
  ): Promise<void>;
  /** FLIP a laid-out card from a viewport pose (centre, rotation, width in
   *  viewport px) to its place: dispose({home}), and dnd's held drops. */
  flyFrom(
    id: CardId,
    from: {cx: number; cy: number; rotate: number; width: number},
    o?: {oldSide?: FlipSide},
  ): Promise<void>;
  /** Count a promise as motion in flight for settled(). */
  track(p: Promise<void>): Promise<void>;
  /** Ids currently in the free layer. */
  freeIds(): ReadonlySet<CardId>;
  /** Ids with a FLIP in flight. */
  flying(): ReadonlySet<CardId>;
}

const INTERNALS = new WeakMap<MotionService, MotionInternals>();

/** The internals of a MotionService made by createMotionService. */
export function motionInternals(m: MotionService): MotionInternals {
  const i = INTERNALS.get(m);
  if (!i) {
    throw new Error(
      'motionInternals: not a MotionService made by createMotionService',
    );
  }
  return i;
}

// A frame loop on `host`'s ticker; dt is capped at MAX_DT_MS.
function runLoop(
  host: MotionHost,
  step: (dtMs: number, tMs: number) => boolean,
): Run {
  const given: unknown = step;
  if (typeof given !== 'function') {
    throw new Error('motion.run: step must be a function');
  }
  const ticker = host.ticker();
  const start = ticker.now();
  let last = start;
  let handle = 0;
  let finished = false;
  let resolve: () => void = () => {
    // Replaced by the promise's resolver below.
  };
  let reject: (e: unknown) => void = () => {
    // Replaced by the promise's rejecter below.
  };
  const done = new Promise<void>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  const frame = (now: number): void => {
    if (finished) return;
    const dt = Math.min(MAX_DT_MS, Math.max(0, now - last));
    last = now;
    let more: boolean;
    try {
      more = step(dt, now - start);
    } catch (e) {
      finished = true;
      reject(e);
      return;
    }
    if (more) {
      handle = ticker.request(frame);
    } else {
      finished = true;
      resolve();
    }
  };
  handle = ticker.request(frame);
  return {
    done,
    cancel: () => {
      if (finished) return;
      finished = true;
      ticker.cancel(handle);
      resolve();
    },
  };
}

/** The motion service of a card scene. */
export function createMotionService(host: MotionHost): MotionService {
  const engine = new MotionEngine(host);
  const service: MotionService = {
    measure: engine.measure,
    measureSlot: (slot, frame = 'viewport', tableId) =>
      engine.measureMarked('slot', slot, frame, tableId),
    measureRegion: (region, frame = 'viewport', tableId) =>
      engine.measureMarked('region', region, frame, tableId),
    viewport: engine.viewport,
    free: engine.freeCards,
    run: step => runLoop(host, step),
    suppressNext: engine.suppressNext,
    settled: () => engine.settled(),
  };
  INTERNALS.set(service, {
    runPlan: engine.runPlan,
    flyFrom: engine.flyFrom,
    track: engine.track,
    freeIds: () => new Set(engine.free.keys()),
    flying: () => new Set(engine.flights.keys()),
  });
  return service;
}
