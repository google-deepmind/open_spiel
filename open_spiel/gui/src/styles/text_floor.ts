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

// The client's text floor: shared by fitted HTML boards and scaled SVG
// boards.

/** The legibility floor for rendered text, in CSS px. */
export const TEXT_FLOOR_PX = 12;

/**
 * The custom property a fitted board's text reads to stay legible.
 * The transform scales text with the cards, so a 13 px rank at scale 0.65
 * renders at 8.4 px. The inner node carries `--fit-min-font` = the size that
 * renders at TEXT_FLOOR_PX after the scale, and board CSS writes
 * `font-size: max(13px, var(--fit-min-font, 12px))`: the cards shrink, the
 * text never drops below the floor. Outside a fitted board the fallback is
 * the floor itself.
 */
export const FIT_MIN_FONT_VAR = '--fit-min-font';

/**
 * `--fit-min-font` for a scale: TEXT_FLOOR_PX / scale, never below the floor,
 * rounded UP to 1/100 px so the rendered size is never a hair under 12.
 */
export function fitMinFont(scale: number): string {
  const px = TEXT_FLOOR_PX / Math.min(1, scale > 0 ? scale : 1);
  return `${(Math.ceil(px * 100) / 100).toFixed(2)}px`;
}
