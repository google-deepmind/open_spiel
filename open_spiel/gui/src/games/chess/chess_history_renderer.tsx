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

import type {JSX} from 'react';

import type {HistoryRendererProps} from '../../actions/history_renderer.ts';
import {toFigurine} from './chess_table.ts';

/** A move in the list: figurine notation on screen, the SAN letters for a screen reader. */
function MoveText({label}: {label: string}): JSX.Element {
  const shown = toFigurine(label);
  if (shown === label) return <>{label}</>;
  // Each piece glyph in its own span, so it is drawn from a font that
  // draws it solid (chess.css).
  // toFigurine swaps one letter for one glyph, so the two line up.
  const parts = Array.from({length: shown.length}, (_, index) => {
    const glyph = shown.charAt(index);
    return glyph === label.charAt(index) ? (
      glyph
    ) : (
      <span className="chess-history-figurine__piece" key={index}>
        {glyph}
      </span>
    );
  });
  return (
    <span className="chess-history-figurine" aria-label={label} title={label}>
      {parts}
    </span>
  );
}

/** One move of the list: its index in the history, action and label. */
interface MoveEntry {
  index: number;
  action: number;
  label: string;
}

/** A turn of the list: White's move and Black's, if made. */
interface Turn {
  turnNumber: number;
  white: MoveEntry | null;
  black: MoveEntry | null;
}

/** The moves in pairs (turn 1: white + black, turn 2: white + black, etc.) */
function buildTurns({
  history,
  labels,
  fallbackLabels,
  getActionLabel,
}: HistoryRendererProps): Turn[] {
  const turns: Turn[] = [];
  for (let i = 0; i < history.length; i += 2) {
    const turnNumber = Math.floor(i / 2) + 1;
    const whiteAction = history[i] ?? -1;
    const whiteLabel = getActionLabel(i, whiteAction, labels, fallbackLabels);

    let black: Turn['black'] = null;
    if (i + 1 < history.length) {
      const blackAction = history[i + 1] ?? -1;
      const blackLabel = getActionLabel(
        i + 1,
        blackAction,
        labels,
        fallbackLabels,
      );
      black = {index: i + 1, action: blackAction, label: blackLabel};
    }

    turns.push({
      turnNumber,
      white: {index: i, action: whiteAction, label: whiteLabel},
      black,
    });
  }
  return turns;
}

/** A move that shows its position when pressed. */
function MoveButton({
  move,
  props,
}: {
  move: MoveEntry;
  props: HistoryRendererProps;
}): JSX.Element {
  const {currentStep, isBusy, goToStep} = props;
  const step = move.index + 1;
  const isCurrent = currentStep === step;
  const isPlayed = currentStep >= step;
  return (
    <button
      className={`chess-history-move ${isCurrent ? 'chess-history-move--current' : ''} ${isPlayed ? 'chess-history-move--played' : ''}`}
      onClick={() => {
        if (!isBusy) goToStep(step, {raw: true});
      }}
      disabled={isBusy}
      aria-current={isCurrent ? 'step' : undefined}
    >
      <MoveText label={move.label} />
    </button>
  );
}

/** The chess move list: the moves in pairs, a turn to a row. */
export function ChessHistoryRenderer(props: HistoryRendererProps): JSX.Element {
  const {currentStep, isBusy, isAnalysisMode, goToStep} = props;
  const turns = buildTurns(props);

  if (isAnalysisMode) {
    return (
      <div className="chess-history-list">
        <div className="chess-history-row chess-history-row--header">
          <button
            className={`chess-history-initial ${currentStep === 0 ? 'chess-history-initial--current' : ''}`}
            onClick={() => {
              if (!isBusy) goToStep(0, {raw: true});
            }}
            disabled={isBusy}
          >
            Initial
          </button>
        </div>
        {turns.map(turn => (
          <div key={turn.turnNumber} className="chess-history-row">
            <span className="chess-history-turn">{turn.turnNumber}.</span>
            {turn.white && <MoveButton move={turn.white} props={props} />}
            {turn.black ? (
              <MoveButton move={turn.black} props={props} />
            ) : (
              <span className="chess-history-placeholder" />
            )}
          </div>
        ))}
      </div>
    );
  }

  // Read-only mode (not analysis)
  return (
    <div className="chess-history-list chess-history-list--readonly">
      {turns.map(turn => (
        <div key={turn.turnNumber} className="chess-history-row">
          <span className="chess-history-turn">{turn.turnNumber}.</span>
          {turn.white && (
            <span className="chess-history-move chess-history-move--played">
              <MoveText label={turn.white.label} />
            </span>
          )}
          {turn.black ? (
            <span className="chess-history-move chess-history-move--played">
              <MoveText label={turn.black.label} />
            </span>
          ) : (
            <span className="chess-history-placeholder" />
          )}
        </div>
      ))}
    </div>
  );
}
