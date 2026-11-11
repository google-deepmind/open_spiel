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

// The poker board's list in Actions › History: every action in the words of
// the table's own Action History ("player 0 Check", "player 1 Bet 2",
// "player 1 Call 1"), not the engine's ("Pass", "Bet"), so the two lists
// never name one move two ways. Built from the same derived table.

import type {JSX} from 'react';

import type {HistoryRendererProps} from '../../actions/history_renderer.ts';
import {pokerSession} from './poker_session.ts';
import {derivePokerState, historyEntryText} from './poker_state.ts';

/** One row: the position after `ply` plies, and its words. */
interface Row {
  readonly ply: number;
  readonly text: string;
}

/** The rows of the frame's history in the table's words; throws on a record
 *  the table cannot be drawn from. */
function pokerRows(props: HistoryRendererProps): Row[] {
  const {frame} = props;
  const presentation = frame?.presentation ?? null;
  if (frame === undefined || presentation === null) {
    throw new Error('The poker History list needs the frame it lists.');
  }
  const state = derivePokerState(pokerSession(frame, presentation));
  if (state === null) throw new Error('The poker History list has no table.');
  return state.history.map((entry, index) => {
    const {who, what} = historyEntryText(entry, state.seats);
    return {ply: index + 1, text: `${who} ${what}`};
  });
}

/** The poker History list: a row per action, the current one marked. */
export function PokerHistoryRenderer(props: HistoryRendererProps): JSX.Element {
  const {currentStep, isBusy, isAnalysisMode, goToStep} = props;
  let rows: Row[];
  try {
    rows = pokerRows(props);
  } catch (caught) {
    return (
      <p className="poker-history-list__error" role="alert">
        {`This hand's record cannot be listed: ${caught instanceof Error ? caught.message : String(caught)}`}
      </p>
    );
  }
  // An omniscient viewer may show any position, the initial one included.
  const shown = isAnalysisMode ? [{ply: 0, text: 'Initial'}, ...rows] : rows;
  if (shown.length === 0) {
    return <p className="poker-history-list__empty">No actions yet.</p>;
  }
  return (
    <ol className="poker-history-list" aria-label="History">
      {shown.map(row => {
        const standing =
          row.ply === currentStep
            ? 'current'
            : row.ply < currentStep
              ? 'played'
              : 'ahead';
        const columns = (
          <>
            <span className="history__turn">
              {row.ply === 0 ? '' : `#${String(row.ply)}`}
            </span>
            <span className="history__label">{row.text}</span>
          </>
        );
        return (
          <li
            key={row.ply}
            className="history__row"
            data-standing={standing}
            aria-current={standing === 'current' ? 'step' : undefined}
          >
            {isAnalysisMode ? (
              <button
                type="button"
                className="history__item"
                aria-disabled={isBusy ? 'true' : undefined}
                onClick={() => {
                  if (!isBusy) goToStep(row.ply);
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
