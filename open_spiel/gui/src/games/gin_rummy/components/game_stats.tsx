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

import {Fragment, type JSX, useRef} from 'react';
import {formatNumber} from '../use_gin_rummy_derived_state.ts';
import {useFitFontScale} from '../../../styles/use_fit_font_scale.ts';
import {GIN_BOT_NOTE} from '../gin_turn_line.ts';

interface GameStatsProps {
  knockCard: number | null;
  playerDeadwood: number | null;
  /**
   * The finished hand in a few words (ginResultFigure): the hand type as the
   * label, the outcome in bold. The full accounting is behind the action
   * bar's Result button.
   */
  result?: {label: string; outcome: string} | null | undefined;
  /**
   * Each player's return, for a board with no hand result to give; not
   * drawn when `result` is.
   */
  returns?: readonly number[] | undefined;
  statusText: string;
  waitingOnBot?: boolean | undefined;
  currentAutoPlay?: boolean | undefined;
  interactionBlockReason?: string | null | undefined;
  canInteract?: boolean | undefined;
}

function Figures({
  knockText,
  deadwoodText,
  result,
  returnsParts,
}: {
  knockText: string;
  deadwoodText: string;
  result: {label: string; outcome: string} | null;
  returnsParts: string[] | null;
}): JSX.Element {
  return (
    <div className="gin-stats__figures">
      <div className="gin-stat">
        Knock card: <strong>{knockText}</strong>
      </div>
      <div className="gin-stat">
        Deadwood: <strong>{deadwoodText}</strong>
      </div>
      {result !== null && (
        <div
          className="gin-stat gin-stat--returns"
          data-testid="gin-stat-result"
        >
          {result.label}:{' '}
          <strong>
            <span className="gin-stat__part">{result.outcome}</span>
          </strong>
        </div>
      )}
      {returnsParts !== null && (
        <div className="gin-stat gin-stat--returns">
          Returns:{' '}
          <strong>
            {returnsParts.map((part, i) => (
              <Fragment key={i}>
                {i > 0 && ' • '}
                <span className="gin-stat__part">{part}</span>
              </Fragment>
            ))}
          </strong>
        </div>
      )}
    </div>
  );
}

/**
 * The stats region: the figures (knock card, deadwood, the hand's result) and the turn
 * line. In the wide layout's column beside the stock they stack; in the
 * narrow layout's short row below the supply they sit side by side
 * (gin_rummy.css, the @container rule). The layout sizes the region to hold
 * this text at 12 px or more (gin_layout_stats.ts ginStatsHeight);
 * should it not, the font shrinks until it fits, never under the text floor
 * (useFitFontScale), so it is never drawn over the stock, the upcard or the
 * action buttons.
 */
export function GameStats({
  knockCard,
  playerDeadwood,
  result = null,
  returns,
  statusText,
  waitingOnBot = false,
  currentAutoPlay = false,
  interactionBlockReason,
  canInteract = true,
}: GameStatsProps): JSX.Element {
  const botNote = waitingOnBot
    ? currentAutoPlay
      ? GIN_BOT_NOTE.thinking
      : GIN_BOT_NOTE.waiting
    : null;
  const blockNote =
    !canInteract &&
    interactionBlockReason !== null &&
    interactionBlockReason !== undefined &&
    interactionBlockReason !== ''
      ? interactionBlockReason
      : null;
  // Each player's return is kept whole; the line wraps between them.
  const returnsParts =
    result === null && returns !== undefined && returns.length > 0
      ? returns.map(
          (value, index) => `P${String(index)}: ${formatNumber(value)}`,
        )
      : null;
  const deadwoodText =
    playerDeadwood !== null ? formatNumber(playerDeadwood) : '—';
  const knockText = knockCard === null ? '—' : String(knockCard);

  const statsRef = useRef<HTMLDivElement>(null);
  useFitFontScale(statsRef);

  return (
    <div className="gin-stats-box">
      <div className="gin-stats" ref={statsRef} aria-live="polite">
        <Figures
          knockText={knockText}
          deadwoodText={deadwoodText}
          result={result}
          returnsParts={returnsParts}
        />
        <div className="gin-stats__turn">
          <span>{statusText}</span>
          {botNote !== null && (
            <span className="gin-stats__note">· {botNote}</span>
          )}
          {blockNote !== null && (
            <span className="gin-stats__note">· {blockNote}</span>
          )}
        </div>
      </div>
    </div>
  );
}
