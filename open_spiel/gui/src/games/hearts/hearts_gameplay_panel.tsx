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

// The hearts Gameplay panel: the status row over the table, the notices
// drawn over it, and the table itself (sections/table_layout.tsx), fed from
// the session (hearts_session.ts) through the board's hooks.

import {type JSX, type ReactNode, useMemo, useState} from 'react';

import {TableCardStyle} from '../../cards/table_cards/index.ts';
import {
  GameplayNotice,
  GameplayNoticeLayer,
  GameplayNoticeLayerProvider,
} from '../shared/gameplay_notice.tsx';
import {useHeartsBoard, type ReadyBoard} from './hearts_board.ts';
import {useHeartsSession} from './hearts_session.ts';
import {cardList} from './hearts_turn.ts';
import {TableLayout} from './sections/index.ts';
import './hearts.css';

/** The board's notices, drawn over the table. */
function HeartsNotices({b}: {readonly b: ReadyBoard}): JSX.Element {
  const {error, pass, turn, heartsState} = b;
  return (
    <>
      {error && <GameplayNotice tone="error">{error}</GameplayNotice>}
      {pass.passError && (
        <GameplayNotice tone="error">{pass.passError}</GameplayNotice>
      )}
      {turn.outOfReach && (
        <GameplayNotice
          tone="turn"
          dismissible={false}
          testId="hearts-out-of-reach"
        >
          {`${turn.onMoveName} to ${heartsState.phase === 'pass' ? 'pass' : 'play'}: their hand is hidden. `}
          <button
            type="button"
            className="gameplay-notice__action"
            data-testid="hearts-turn-to-mover"
            onClick={() => {
              b.setAnalysisViewSeat(turn.currentPlayerSlot);
            }}
          >
            Take their seat
          </button>
          {' or turn on Show all cards (View panel).'}
        </GameplayNotice>
      )}
    </>
  );
}

/**
 * The trick being played ("Trick 2 of 13"), as the table shows it: a
 * finished trick still on show, or queued behind it, is not counted until it
 * is collected, as euchre counts its tricks, so the count never steps ahead
 * of the cards on the table.
 */
export function shownTrickNumber(
  completedTricks: number,
  onShow: number,
  tricksPerHand: number,
): number {
  const done = completedTricks - onShow;
  if (done < 0) {
    throw new Error(
      `hearts: ${String(onShow)} tricks on show but ${String(completedTricks)} completed`,
    );
  }
  return Math.min(done + 1, tricksPerHand);
}

/** The pass prompt, or the trick and whose turn it is, over the table. */
function HeartsStateMeta({b}: {readonly b: ReadyBoard}): JSX.Element {
  const {heartsState, pass, turn, session, trickCollect} = b;
  const {sentPass, passFlight} = pass;
  const direction = heartsState.passDirection;
  const struct = session.state.struct as {completedTricks?: number} | undefined;
  const onShow = (trickCollect.trick ? 1 : 0) + trickCollect.pending.length;
  // A 52-card deck dealt out evenly: 13 tricks with four players.
  const tricksPerHand = Math.floor(52 / heartsState.players.length);
  return (
    <div className="ogs-state-meta">
      {heartsState.phase === 'pass' ? (
        <span>
          {turn.notParticipant
            ? `Players pass three cards ${direction}`
            : sentPass && !passFlight
              ? `Passing ${cardList(sentPass.cards)} ${direction}...`
              : sentPass
                ? `You passed ${cardList(sentPass.cards)} ${direction}; waiting for the other players to pass`
                : `Select three cards to pass ${direction}`}
        </span>
      ) : (
        <>
          <span>
            {heartsState.isTerminal
              ? 'Game Over'
              : `Trick ${String(
                  shownTrickNumber(
                    struct?.completedTricks ?? 0,
                    onShow,
                    tricksPerHand,
                  ),
                )} of ${String(tricksPerHand)}`}
          </span>
          {/* Whose turn it is, once the table has caught up with it. */}
          {!heartsState.isTerminal && onShow === 0 && (
            <span>{heartsState.statusMessage}</span>
          )}
          {turn.waitingOnBot && onShow === 0 && (
            <span>
              {turn.currentAutoPlay ? 'Bot thinking...' : 'Waiting for bot...'}
            </span>
          )}
        </>
      )}
    </div>
  );
}

/** The panel, with the notice layer over everything in it. */
function HeartsShell({children}: {readonly children: ReactNode}): JSX.Element {
  const [noticeLayer, setNoticeLayer] = useState<HTMLElement | null>(null);
  const noticeContext = useMemo(() => ({layer: noticeLayer}), [noticeLayer]);
  return (
    <div className="hearts-gameplay gameplay-shell">
      <GameplayNoticeLayerProvider value={noticeContext}>
        {children}
        <GameplayNoticeLayer placement="top" layerRef={setNoticeLayer} />
      </GameplayNoticeLayerProvider>
    </div>
  );
}

/** The hearts board in the Gameplay panel. */
export function HeartsGameplay(): JSX.Element {
  const b = useHeartsBoard(useHeartsSession());
  if (b.heartsState === null || b.tableProps === null) {
    return (
      <HeartsShell>
        <div className="ogs-status">Loading game state...</div>
      </HeartsShell>
    );
  }
  const ready: ReadyBoard = {...b, heartsState: b.heartsState};
  return (
    <HeartsShell>
      <HeartsNotices b={ready} />
      <HeartsStateMeta b={ready} />
      <div
        className="hearts-gameplay-wrapper"
        style={b.wrapperStyle}
        data-trick-phase={b.trickCollect.phase ?? 'none'}
      >
        <TableCardStyle {...b.cardScope}>
          <TableLayout
            // A new bottom seat redraws the table where it is, rather than
            // flying every card to its new seat.
            key={b.viewerSeat}
            {...b.tableProps}
          />
        </TableCardStyle>
      </div>
    </HeartsShell>
  );
}
