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

import {type JSX, useMemo} from 'react';

import type {Session} from '../session/use_session.ts';
import {type HistoryRow, historyRows, rowText} from './history_rows.ts';
import {NoFrame, Note} from './note.tsx';
import {useStickToBottom} from './use_stick_to_bottom.ts';

import './actions.css';

/** Where a row stands relative to the position shown. */
type Standing = 'current' | 'played' | 'ahead';

function standing({first, last}: HistoryRow, ply: number): Standing {
  if (first <= ply && ply <= last) return 'current';
  return last < ply ? 'played' : 'ahead';
}

/**
 * Every action of the live game, one row each after the initial position,
 * as the viewer may see them: a turn column, "#3", and the action's label.
 * The row of the position shown is marked current. An omniscient viewer can
 * show any position from its row, except while a request is in flight. For
 * any other viewer the rows are read-only, a run of actions withheld from
 * them is a single row, and an empty history says so.
 */
export function History({session}: {session: Session}): JSX.Element {
  const {frame, busy, commands} = session;
  const history = frame?.history;
  const rows = useMemo(() => historyRows(history ?? []), [history]);
  const {ref, onScroll} = useStickToBottom<HTMLOListElement>(
    history?.length ?? 0,
  );
  if (frame === null) return <NoFrame />;
  if (!frame.omniscient && frame.history.length === 0) {
    return <Note>No actions yet.</Note>;
  }
  return (
    <ol ref={ref} className="history" aria-label="History" onScroll={onScroll}>
      {rows.map(row => {
        const rowStanding = standing(row, frame.ply);
        const {turn, label} = rowText(row);
        const columns = (
          <>
            <span className="history__turn">{turn}</span>
            <span className="history__label">{label}</span>
          </>
        );
        return (
          <li
            key={row.first}
            className="history__row"
            data-standing={rowStanding}
            data-collapsed={row.first === row.last ? undefined : 'true'}
            aria-current={rowStanding === 'current' ? 'step' : undefined}
          >
            {frame.omniscient ? (
              <button
                type="button"
                className="history__item"
                aria-disabled={busy ? 'true' : undefined}
                onClick={() => {
                  if (!busy) commands.goTo(row.first);
                }}
              >
                {columns}
              </button>
            ) : (
              <div className="history__item">{columns}</div>
            )}
          </li>
        );
      })}
    </ol>
  );
}
