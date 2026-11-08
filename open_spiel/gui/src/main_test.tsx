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

import type {Root} from 'react-dom/client';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import {fakeColorScheme} from './testing/color_scheme.ts';

describe('main', () => {
  let mounted: Root | undefined;

  beforeEach(() => {
    vi.resetModules();
    fakeColorScheme(false);
  });

  afterEach(() => {
    mounted?.unmount();
    mounted = undefined;
    document.body.replaceChildren();
    vi.unstubAllGlobals();
  });

  it('mounts the app in #root', async () => {
    document.body.innerHTML = '<div id="root"></div>';
    ({root: mounted} = await import('./main.tsx'));
    // ThemeProvider sets the theme in an effect, which may run after the
    // first render is in the document.
    await vi.waitFor(() => {
      expect(document.querySelector('#root .app__home')?.textContent).toBe(
        'OpenSpiel',
      );
      expect(document.documentElement.dataset['theme']).toBe('light');
    });
  });

  it('refuses a page with no #root', async () => {
    await expect(import('./main.tsx')).rejects.toThrow('no #root element');
  });
});
