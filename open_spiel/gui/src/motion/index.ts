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

export {
  ANIMATION_SCALE_VAR,
  ANIMATION_SPEED_LABELS,
  ANIMATION_SPEED_SCALE,
  ANIMATION_SPEED_STORAGE_KEY,
  ANIMATION_SPEEDS,
  DEFAULT_ANIMATION_SPEED,
  animationHoldScale,
  animationScale,
  applyAnimationScale,
  installAnimationScaleRoot,
  prefersReducedMotionMedia,
  readAnimationSpeed,
  scaleHoldMs,
  scaleMs,
  subscribeAnimationSpeed,
  writeAnimationSpeed,
  type AnimationSpeed,
  type ScaleOptions,
} from './animation_speed.ts';
export {
  useAnimationHoldScale,
  useAnimationScale,
  useAnimationSpeed,
  useReducedMotion,
} from './use_animation_speed.ts';
