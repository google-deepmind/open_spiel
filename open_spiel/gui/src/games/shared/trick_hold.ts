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

// The trick hold for trick-taking boards (Hearts, Euchre, Bridge): the
// viewer's setting (off / short / normal, persisted on this browser and shared
// by every trick game) and the session-to-feed glue around the shared card
// library's useTrickCollect (cards/library/trick_collect.ts). The config field is
// TrickHoldField.tsx.
import {useMemo} from 'react';
import {
  scaleMs,
  useAnimationHoldScale,
  useAnimationScale,
} from '../../motion/index.ts';
import {
  DEFAULT_TRICK_HOLD,
  TRICK_COLLECT_MS,
  TRICK_HOLDS,
  trickHoldMs,
  useTrickCollect,
  type CompletedTrick,
  type TrickCollectState,
  type TrickFeed,
  type TrickHold,
} from '../../cards/library/index.ts';
import {usePersistedSetting} from './view_settings.ts';

/** What the trick hold reads of a board's session: its id, how many actions
 *  reached the position shown, and whether it is a history view. */
export interface TrickSession {
  readonly sessionId: string;
  readonly state?: {readonly history?: readonly unknown[]} | null;
  readonly historyView?: unknown;
}

/** Whether an earlier position than the live one is shown. */
export interface ReplayState {
  readonly isActive: boolean;
  readonly step: number;
  readonly totalSteps: number;
}

/** The trick hold setting's key. */
export const TRICK_HOLD_KEY = 'cards.trickHold';

/** The viewer's trick hold (persisted on this browser). */
export function useTrickHold(): [TrickHold, (next: TrickHold) => void] {
  return usePersistedSetting<TrickHold>(
    TRICK_HOLD_KEY,
    TRICK_HOLDS,
    DEFAULT_TRICK_HOLD,
  );
}

/** The replay fields that say whether an earlier position is on screen. */
export type ReplayPosition = Pick<
  ReplayState,
  'isActive' | 'step' | 'totalSteps'
>;

/** An earlier position of a replay is on screen. In analysis mode the
 *  Actions panel keeps a replay active over the live game, at its end; only a
 *  step short of the end is a scrub. */
export function replayScrubbing(r: ReplayPosition): boolean {
  return r.isActive && r.step < r.totalSteps;
}

/** The detector's feed from a served payload: its session, history length,
 *  whether it is scrubbed (a history view, or `replayScrub`: a replay short of
 *  its end), and the last finished trick the game parsed from its struct. */
export function trickFeedOf<C, S>(
  session: TrickSession | null | undefined,
  replayScrub: boolean,
  last: CompletedTrick<C, S> | null,
): TrickFeed<C, S> | null {
  if (!session?.state) return null;
  return {
    session: session.sessionId,
    history: session.state.history?.length ?? 0,
    scrubbing: Boolean(session.historyView) || replayScrub,
    last,
  };
}

/** useTrickCollect for a session board, at the viewer's trick hold and
 *  Animation speed: Slow lengthens the hold and the collect, Fast shortens
 *  them; Instant and reduced motion drop the collect's motion but keep the
 *  hold, which is information (animationHoldScale). */
export function useSessionTrickCollect<C, S>(
  session: TrickSession | null | undefined,
  replay: ReplayPosition,
  last: CompletedTrick<C, S> | null,
  /** The table is still drawing plays that arrived together (useStagedPlays):
   *  the hold starts when they have landed. */
  settling = false,
): TrickCollectState<C, S> {
  const [hold] = useTrickHold();
  const scrub = replayScrubbing(replay);
  const feed = useMemo(
    () => trickFeedOf(session, scrub, last),
    [session, scrub, last],
  );
  const holdMs = scaleMs(trickHoldMs(hold), useAnimationHoldScale());
  const collectMs = scaleMs(TRICK_COLLECT_MS, useAnimationScale());
  return useTrickCollect(feed, {holdMs, collectMs, settling});
}
