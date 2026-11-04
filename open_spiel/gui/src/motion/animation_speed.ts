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

// The viewer's Animation speed: one setting, for every game, that scales
// every move animation (a bot's draw and discard, a dealt card, a piece's
// slide, a trick being collected, a solitaire's autoplay flight).
//
// It is the one source of truth for motion timing: TypeScript timers pass their base duration through
// scaleMs(), the card library's motionDuration() applies it to every card
// flight, and stylesheets multiply their durations by the CSS variable
// `--anim-scale`, which installAnimationScaleRoot() keeps on <html>.
//
// Stored on this browser under `open_spiel.gui.anim.speed` (the key scheme
// of the View panel's persisted settings), for every game: the field says
// so. A write in another tab arrives as a 'storage' event.
//
// prefers-reduced-motion wins over any speed: the scale is then 0 and only
// the static end state (and a game's static cues) shows.

/** The viewer's Animation speed. */
export type AnimationSpeed = 'slow' | 'normal' | 'fast' | 'instant';
/** Every speed, slowest first. */
export const ANIMATION_SPEEDS: readonly AnimationSpeed[] = [
  'slow',
  'normal',
  'fast',
  'instant',
];
/** The speed until the viewer picks one. */
export const DEFAULT_ANIMATION_SPEED: AnimationSpeed = 'normal';
/** Each speed's name in the View panel. */
export const ANIMATION_SPEED_LABELS: Readonly<Record<AnimationSpeed, string>> =
  {
    slow: 'Slow',
    normal: 'Normal',
    fast: 'Fast',
    instant: 'Instant',
  };
/** Each speed's multiplier on a base duration. Slow makes a card library
 *  flight (240 ms) last 600 ms; Instant shows the end state at once. */
export const ANIMATION_SPEED_SCALE: Readonly<Record<AnimationSpeed, number>> = {
  slow: 2.5,
  normal: 1,
  fast: 0.5,
  instant: 0,
};

/** localStorage key of the speed (the View panel's persisted-setting scheme). */
export const ANIMATION_SPEED_STORAGE_KEY = 'open_spiel.gui.anim.speed';
/** The CSS variable on <html>: the scale in effect (reduced motion applied). */
export const ANIMATION_SCALE_VAR = '--anim-scale';

const REDUCE_QUERY = '(prefers-reduced-motion: reduce)';

const listeners = new Set<() => void>();
function notify(): void {
  for (const l of [...listeners]) l();
}

function storage(): Storage | null {
  return typeof window !== 'undefined' ? window.localStorage : null;
}

function isSpeed(v: unknown): v is AnimationSpeed {
  return (
    typeof v === 'string' && (ANIMATION_SPEEDS as readonly string[]).includes(v)
  );
}

/** The speed chosen on this browser, or the default when none was (or what
 *  is stored is not a speed). */
export function readAnimationSpeed(): AnimationSpeed {
  const v = storage()?.getItem(ANIMATION_SPEED_STORAGE_KEY) ?? null;
  return isSpeed(v) ? v : DEFAULT_ANIMATION_SPEED;
}

/** Stores the viewer's choice (only the viewer's own gesture calls this).
 *  Throws on anything that is not a speed, or with no storage to write to. */
export function writeAnimationSpeed(next: AnimationSpeed): void {
  if (!isSpeed(next)) {
    throw new Error(
      `animation speed: '${String(next)}' is not one of ${ANIMATION_SPEEDS.join(', ')}`,
    );
  }
  const s = storage();
  if (!s) throw new Error('animation speed: no localStorage to write to');
  s.setItem(ANIMATION_SPEED_STORAGE_KEY, next);
  notify();
}

/** True when the user asks the system for reduced motion. */
export function prefersReducedMotionMedia(): boolean {
  if (
    typeof window === 'undefined' ||
    typeof window.matchMedia !== 'function'
  ) {
    return false;
  }
  return window.matchMedia(REDUCE_QUERY).matches;
}

/** How a scale treats the reduced-motion preference. */
export interface ScaleOptions {
  /** 'ignore': the speed alone, for a pause that is not motion (a trick
   *  left on the table), or a style that opts out of reduced motion.
   *  Default 'respect': 0 under prefers-reduced-motion. */
  reducedMotion?: 'respect' | 'ignore';
}

/** The multiplier in effect: the speed's, or 0 under reduced motion. */
export function animationScale(o: ScaleOptions = {}): number {
  if (
    (o.reducedMotion ?? 'respect') === 'respect' &&
    prefersReducedMotionMedia()
  ) {
    return 0;
  }
  return ANIMATION_SPEED_SCALE[readAnimationSpeed()];
}

/**
 * The multiplier for a HOLD: a pause that shows the player something (a
 * finished trick left on the table, a combat reveal) rather than moving it.
 * Slow lengthens it and Fast shortens it, like motion. Instant leaves it at
 * its base (Instant drops motion, and a hold is information, not motion).
 * Reduced motion does not touch it at all: a viewer with reduced motion who
 * picks Slow gets Slow's longer holds, because nothing moves in a hold. The
 * View panel's note says so.
 */
export function animationHoldScale(): number {
  const speed = readAnimationSpeed();
  return speed === 'instant' ? 1 : ANIMATION_SPEED_SCALE[speed];
}

/** A HOLD's base duration (see animationHoldScale) at the viewer's speed, in
 *  whole ms: Slow 2.5x, Fast 0.5x, Instant 1x, whether or not reduced motion
 *  is on. Use
 *  it for a pause that shows the player something (narration read at a
 *  readable pace, a beat on a card before it resolves). Throws like scaleMs. */
export function scaleHoldMs(baseMs: number): number {
  return scaleMs(baseMs, animationHoldScale());
}

/**
 * A base duration at the viewer's speed, in whole ms. `scale` defaults to
 * the scale in effect now; a React component passes useAnimationScale()'s so
 * it re-renders on a change. Throws on a negative or non-finite base.
 */
export function scaleMs(
  baseMs: number,
  scale: number = animationScale(),
): number {
  if (!Number.isFinite(baseMs) || baseMs < 0) {
    throw new Error(
      `scaleMs: base must be a finite ms >= 0, got ${String(baseMs)}`,
    );
  }
  if (!Number.isFinite(scale) || scale < 0) {
    throw new Error(
      `scaleMs: scale must be finite and >= 0, got ${String(scale)}`,
    );
  }
  return Math.round(baseMs * scale);
}

let mediaInstalled = false;
let storageInstalled = false;

/** Subscribe to the speed (any tab) and to the reduced-motion preference. */
export function subscribeAnimationSpeed(listener: () => void): () => void {
  listeners.add(listener);
  if (typeof window !== 'undefined') {
    if (!storageInstalled) {
      storageInstalled = true;
      window.addEventListener('storage', e => {
        if (e.key === null || e.key === ANIMATION_SPEED_STORAGE_KEY) notify();
      });
    }
    if (!mediaInstalled && typeof window.matchMedia === 'function') {
      mediaInstalled = true;
      const mq = window.matchMedia(REDUCE_QUERY);
      mq.addEventListener('change', notify);
    }
  }
  return () => {
    listeners.delete(listener);
  };
}

/** Writes the scale in effect to `--anim-scale` on the document's root. */
export function applyAnimationScale(doc: Document = document): void {
  doc.documentElement.style.setProperty(
    ANIMATION_SCALE_VAR,
    String(animationScale()),
  );
}

/** Keeps `--anim-scale` on <html> current from now on (the app's entry calls
 *  it once, before the first render). Returns the unsubscribe. */
export function installAnimationScaleRoot(
  doc: Document = document,
): () => void {
  applyAnimationScale(doc);
  return subscribeAnimationSpeed(() => {
    applyAnimationScale(doc);
  });
}
