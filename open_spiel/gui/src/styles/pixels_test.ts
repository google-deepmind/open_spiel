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

import {describe, expect, it} from 'vitest';

import {pixels, requirePixels} from './pixels.ts';

describe('pixels', () => {
  it('reads a length in px, and a unitless zero', () => {
    expect(pixels('12px')).toBe(12);
    expect(pixels('0.5px')).toBe(0.5);
    expect(pixels('0')).toBe(0);
  });

  it('reads the empty string, a property jsdom does not compute, as 0', () => {
    expect(pixels('')).toBe(0);
  });

  it('is NaN for another unit or a malformed value', () => {
    expect(pixels('1rem')).toBeNaN();
    expect(pixels('12pt')).toBeNaN();
    expect(pixels('12px;')).toBeNaN();
    expect(pixels('normal')).toBeNaN();
  });
});

describe('requirePixels', () => {
  it('reads a length in px', () => {
    expect(requirePixels('12px')).toBe(12);
  });

  it('throws, naming the value, for anything else', () => {
    expect(() => requirePixels('normal')).toThrow(
      "Not a length in px: 'normal'.",
    );
    expect(() => requirePixels('1rem')).toThrow("Not a length in px: '1rem'.");
  });
});
