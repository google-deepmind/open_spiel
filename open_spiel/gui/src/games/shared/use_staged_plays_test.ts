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

// Plays that arrive together join the table one at a time (two bots' hearts
// plays flew in the same frame at Slow).
import {act, createElement} from 'react';
import {createRoot} from 'react-dom/client';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import '../../testing/render.tsx';
import {
  PLAY_REST_MS,
  type StagedPlaysTiming,
  useStagedPlays,
} from './use_staged_plays.ts';

interface Props {
  readonly k: readonly string[];
  readonly t: StagedPlaysTiming;
}

interface Hooked<P, R> {
  readonly result: {current: R};
  readonly rerender: (props: P) => void;
}

/** A hook's result, rendered in a component, and how to render again. */
function renderHook<P extends object, R>(
  hook: (props: P) => R,
  {initialProps}: {initialProps: P},
): Hooked<P, R> {
  const result: {current: R} = {current: undefined as R};
  function Probe(props: P): null {
    result.current = hook(props);
    return null;
  }
  const root = createRoot(document.createElement('div'));
  const rerender = (props: P): void => {
    act(() => {
      root.render(createElement(Probe, props));
    });
  };
  rerender(initialProps);
  return {result, rerender};
}

const T = {legMs: 240, restMs: PLAY_REST_MS};
const FRAME = T.legMs + T.restMs;

describe('useStagedPlays', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  const mount = (keys: string[], timing = T) =>
    renderHook(({k, t}) => useStagedPlays(k, t).shown, {
      initialProps: {k: keys, t: timing},
    });

  it('draws everything on the first render', () => {
    expect(mount(['W:2c', 'N:5c']).result.current).toBe(2);
  });

  it('two bot plays in one update: the first flies at once, the second after it lands and rests', () => {
    const h = mount([]);
    h.rerender({k: ['W:2c', 'N:5c'], t: T});
    expect(h.result.current).toBe(1);
    act(() => {
      vi.advanceTimersByTime(FRAME - 1);
    });
    expect(h.result.current).toBe(1);
    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(h.result.current).toBe(2);
  });

  it('keeps what is drawn and stages only the new plays of the same trick', () => {
    const h = mount(['W:2c']);
    h.rerender({k: ['W:2c', 'N:5c', 'E:Kc'], t: T});
    expect(h.result.current).toBe(2);
    act(() => {
      vi.advanceTimersByTime(FRAME);
    });
    expect(h.result.current).toBe(3);
  });

  it('a single new play flies alone, unstaged', () => {
    const h = mount(['W:2c']);
    h.rerender({k: ['W:2c', 'N:5c'], t: T});
    expect(h.result.current).toBe(2);
  });

  it('never pulls a drawn play off the table when the list comes back reordered', () => {
    const h = mount(['W:2c', 'N:5c']);
    h.rerender({k: ['N:5c', 'W:2c', 'E:Kc', 'S:Ac'], t: T});
    expect(h.result.current).toBe(4);
  });

  it('a new trick stages from its first play', () => {
    const h = mount(['W:2c', 'N:5c', 'E:Kc', 'S:Ac']);
    h.rerender({k: ['S:3d', 'W:9d', 'N:Td'], t: T});
    expect(h.result.current).toBe(1);
    act(() => {
      vi.advanceTimersByTime(FRAME);
    });
    expect(h.result.current).toBe(2);
    act(() => {
      vi.advanceTimersByTime(FRAME);
    });
    expect(h.result.current).toBe(3);
  });

  it('at Instant (no flight) draws everything at once', () => {
    const h = mount([], {legMs: 0, restMs: 0});
    h.rerender({k: ['W:2c', 'N:5c'], t: {legMs: 0, restMs: 0}});
    expect(h.result.current).toBe(2);
  });

  it('at Slow each play lands and rests 2.5x as long', () => {
    const slow = {legMs: 600, restMs: 500};
    const h = mount([], slow);
    h.rerender({k: ['W:2c', 'N:5c'], t: slow});
    act(() => {
      vi.advanceTimersByTime(1099);
    });
    expect(h.result.current).toBe(1);
    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(h.result.current).toBe(2);
  });
});

describe('useStagedPlays settling', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('settles only when the last staged play has landed, one flight after it left', () => {
    const h = renderHook(({k}) => useStagedPlays(k, T), {
      initialProps: {k: [] as string[]},
    });
    expect(h.result.current.settling).toBe(false);
    h.rerender({k: ['W:2c', 'N:5c', 'E:Kc']});
    expect(h.result.current).toEqual({shown: 1, settling: true});
    act(() => {
      vi.advanceTimersByTime(FRAME);
    });
    act(() => {
      vi.advanceTimersByTime(FRAME);
    });
    expect(h.result.current).toEqual({shown: 3, settling: true});
    act(() => {
      vi.advanceTimersByTime(T.legMs - 1);
    });
    expect(h.result.current.settling).toBe(true);
    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(h.result.current).toEqual({shown: 3, settling: false});
  });

  it('a single new play is not a staging: nothing to settle', () => {
    const h = renderHook(({k}) => useStagedPlays(k, T), {
      initialProps: {k: ['W:2c']},
    });
    h.rerender({k: ['W:2c', 'N:5c']});
    expect(h.result.current).toEqual({shown: 2, settling: false});
  });

  it('refuses a negative time', () => {
    expect(() =>
      renderHook(({k, t}: Props) => useStagedPlays(k, t).shown, {
        initialProps: {k: [], t: {legMs: -1, restMs: 0}},
      }),
    ).toThrow(/finite ms/);
  });
});
