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

// What the hearts table draws in its regions: the centre (game over, the
// pass button, "No Pass"), the status message, and each seat's name.
import type {JSX, ReactNode} from 'react';

import {CENTRE_REGION, MESSAGE_REGION, SEAT_SLOT} from '../hearts_layout.ts';
import type {Player, PlayerPosition} from '../types.ts';
import {PASS_COUNT, type TableLayoutProps} from './table_types.ts';

/** The game over plaque: each player's points, the winner marked. */
function GameOver({
  players,
  rewards,
}: {
  readonly players: readonly Player[];
  readonly rewards: readonly number[] | undefined;
}): JSX.Element {
  return (
    <div className="hearts-game-over">
      <div className="hearts-game-over__title">Game Over</div>
      <div className="hearts-game-over__scores">
        {players.map((player, idx) => {
          const reward = rewards?.[idx] ?? 0;
          const isWinner = rewards && reward === Math.max(...rewards);
          return (
            <div
              key={player.position}
              className={`hearts-game-over__player ${isWinner ? 'hearts-game-over__player--winner' : ''}`}
            >
              <span className="hearts-game-over__name">{player.name}</span>
              <span className="hearts-game-over__score">
                {player.score ?? 0} pts
              </span>
              {isWinner && (
                <span className="hearts-game-over__badge">Winner!</span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** The pass button, or for a spectator the direction as a fact. */
function PassCentre({p}: {readonly p: TableLayoutProps}): JSX.Element {
  const {passDirection, onPassCards, selectedCards, passPending = false} = p;
  const canPass = selectedCards?.size === PASS_COUNT && !passPending;
  if (!onPassCards) {
    // No pass handler (a spectator): the direction as a fact, not a button.
    return (
      <div className="hearts-no-pass-indicator">Passing {passDirection}</div>
    );
  }
  return (
    <button
      className="hearts-pass-button"
      disabled={!canPass}
      onClick={onPassCards}
    >
      {passPending
        ? `Passing ${passDirection}...`
        : `Pass ${passDirection.charAt(0).toUpperCase() + passDirection.slice(1)}`}
    </button>
  );
}

/** What the centre shows in place of the trick, or null for the trick. */
export function centreOf(p: TableLayoutProps): ReactNode {
  const isPassPhase = p.phase === 'pass';
  const isNoPassRound = p.passDirection === 'none';
  if (p.centerContent) return p.centerContent;
  if (p.isTerminal) return <GameOver players={p.players} rewards={p.rewards} />;
  if (isPassPhase && !isNoPassRound) return <PassCentre p={p} />;
  if (isNoPassRound && p.trickPile.length === 0) {
    return <div className="hearts-no-pass-indicator">No Pass</div>;
  }
  return null;
}

/** The table's regions: the centre, the message and the seats' names. */
export function regionsOf(
  p: TableLayoutProps,
  centre: ReactNode,
): Record<string, ReactNode> {
  const regions: Record<string, ReactNode> = {
    [CENTRE_REGION]: centre !== null && (
      <div className="hearts-region hearts-centre">{centre}</div>
    ),
    [MESSAGE_REGION]: p.message && (
      <div className="hearts-region">
        <div className="hearts-message">{p.message}</div>
      </div>
    ),
  };
  for (const player of p.players) {
    const seat: PlayerPosition = player.position;
    regions[SEAT_SLOT[seat].name] = (
      <div className={`hearts-region hearts-seat--${seat}`}>
        <span
          className={`hearts-player-name ${p.currentPlayer === seat ? 'hearts-player-name--active' : ''}`}
          title={player.name}
        >
          {player.name}
        </span>
      </div>
    );
  }
  return regions;
}
