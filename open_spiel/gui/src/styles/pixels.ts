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

/**
 * The number of pixels in a CSS length such as `'12px'` or `'0'`; NaN for
 * any other unit or a malformed value, which parseFloat would read silently.
 * The empty string, a property jsdom does not compute, reads as 0.
 */
export function pixels(value: string): number {
  return Number(value.replace(/px$/, ''));
}

/**
 * The number of pixels in a computed length that layout code depends on.
 *
 * @throws Error naming the value when it is not a length in px, such as a
 *     gap that computes to `normal`.
 */
export function requirePixels(value: string): number {
  const result = pixels(value);
  if (!Number.isFinite(result)) {
    throw new Error(`Not a length in px: '${value}'.`);
  }
  return result;
}
