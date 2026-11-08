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

import {type RefObject, useCallback, useLayoutEffect, useRef} from 'react';

/** How near its bottom, in pixels, a list still counts as at the bottom. */
export const BOTTOM_SLACK_PX = 8;

/** What useStickToBottom gives the scrolling element. */
export interface StickToBottom<T extends HTMLElement> {
  readonly ref: RefObject<T | null>;
  readonly onScroll: () => void;
}

/**
 * Keeps a list scrolled to its bottom as `count` grows, while the reader
 * has left it there; once they scroll up, it stays where they put it. Only
 * a change of `count` moves it, so stepping through a replay, which keeps
 * the rows and moves the current one, never scrolls.
 */
export function useStickToBottom<T extends HTMLElement>(
  count: number,
): StickToBottom<T> {
  const ref = useRef<T>(null);
  const pinned = useRef(true);

  useLayoutEffect(() => {
    const element = ref.current;
    if (element !== null && pinned.current) {
      element.scrollTop = element.scrollHeight;
    }
  }, [count]);

  const onScroll = useCallback(() => {
    const element = ref.current;
    if (element === null) return;
    const below =
      element.scrollHeight - element.scrollTop - element.clientHeight;
    pinned.current = below <= BOTTOM_SLACK_PX;
  }, []);

  return {ref, onScroll};
}
