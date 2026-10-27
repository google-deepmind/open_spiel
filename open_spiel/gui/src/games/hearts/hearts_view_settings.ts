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

// Hearts view settings: your hand's shape (flat or curved) and the trick's
// layout, persisted on this browser. The fields sit in the View panel's
// Hearts section (hearts_view_controls.tsx).
import {
  readStoredSetting,
  storedSettingKey,
  usePersistedSetting,
} from '../shared/view_settings.ts';
import {
  DEFAULT_HEARTS_HAND_SHAPE,
  HEARTS_HAND_SHAPES,
  type HeartsHandShape,
} from './hearts_layout.ts';
import {TRICK_LAYOUTS, type TrickLayout} from '../../cards/library/index.ts';

/** The hand shape setting's key. */
export const HAND_SHAPE_KEY = 'hearts.handShape';

/** Where the hand shape persists in localStorage. */
export const storedHandShapeKey = storedSettingKey(HAND_SHAPE_KEY);

/** The labels the View panel shows. */
export const HAND_SHAPE_LABELS: Readonly<Record<HeartsHandShape, string>> = {
  flat: 'Flat',
  curved: 'Curved',
};

/** The hand shape last chosen on this browser, or the default (flat) when none
 *  was, or what is stored is not a shape. */
export function readStoredHandShape(): HeartsHandShape {
  return readStoredSetting(
    HAND_SHAPE_KEY,
    HEARTS_HAND_SHAPES,
    DEFAULT_HEARTS_HAND_SHAPE,
  );
}

/** Hook to get/set your hand's shape (persisted on this browser). */
export function useHeartsHandShape(): [
  HeartsHandShape,
  (next: HeartsHandShape) => void,
] {
  return usePersistedSetting<HeartsHandShape>(
    HAND_SHAPE_KEY,
    HEARTS_HAND_SHAPES,
    DEFAULT_HEARTS_HAND_SHAPE,
  );
}

// ---- The trick: toward players, or overlapping (cards/library trick_area). ----

/** The trick layout the table starts with: each card toward its player. */
export const DEFAULT_HEARTS_TRICK_LAYOUT: TrickLayout = 'toward';

/** Where the trick layout persists. */
export const TRICK_LAYOUT_KEY = 'hearts.trickLayout.standard';

/** The trick layout last chosen on this browser, or the default. */
export function readStoredTrickLayout(): TrickLayout {
  return readStoredSetting(
    TRICK_LAYOUT_KEY,
    TRICK_LAYOUTS,
    DEFAULT_HEARTS_TRICK_LAYOUT,
  );
}

/** Hook to get/set the trick layout (persisted on this browser). */
export function useHeartsTrickLayout(): [
  TrickLayout,
  (next: TrickLayout) => void,
] {
  return usePersistedSetting<TrickLayout>(
    TRICK_LAYOUT_KEY,
    TRICK_LAYOUTS,
    DEFAULT_HEARTS_TRICK_LAYOUT,
  );
}
