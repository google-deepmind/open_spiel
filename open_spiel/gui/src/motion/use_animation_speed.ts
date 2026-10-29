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

// React bindings for the viewer's Animation speed (animationSpeed.ts).
import {useCallback, useSyncExternalStore} from 'react';
import {
  animationHoldScale,
  animationScale,
  prefersReducedMotionMedia,
  readAnimationSpeed,
  subscribeAnimationSpeed,
  writeAnimationSpeed,
  type AnimationSpeed,
  type ScaleOptions,
} from './animation_speed.ts';

/** The viewer's speed and its setter (stored on this browser, every game). */
export function useAnimationSpeed(): [
  AnimationSpeed,
  (next: AnimationSpeed) => void,
] {
  const speed = useSyncExternalStore(
    subscribeAnimationSpeed,
    readAnimationSpeed,
    readAnimationSpeed,
  );
  const set = useCallback((next: AnimationSpeed) => {
    writeAnimationSpeed(next);
  }, []);
  return [speed, set];
}

/** The multiplier in effect (0 under reduced motion unless `reducedMotion:
 *  'ignore'`, 0 at Instant); re-renders when the speed or the system
 *  preference changes. Pass it to scaleMs(). */
export function useAnimationScale(o: ScaleOptions = {}): number {
  const mode = o.reducedMotion ?? 'respect';
  const read = useCallback(() => animationScale({reducedMotion: mode}), [mode]);
  return useSyncExternalStore(subscribeAnimationSpeed, read, read);
}

/** True while the system asks for reduced motion (re-renders on a change). */
export function useReducedMotion(): boolean {
  return useSyncExternalStore(
    subscribeAnimationSpeed,
    prefersReducedMotionMedia,
    () => false,
  );
}

/** animationHoldScale(), re-rendering on a change of speed. */
export function useAnimationHoldScale(): number {
  return useSyncExternalStore(
    subscribeAnimationSpeed,
    animationHoldScale,
    animationHoldScale,
  );
}
