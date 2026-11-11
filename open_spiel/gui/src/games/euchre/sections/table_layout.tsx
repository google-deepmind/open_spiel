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

import {useRef, type JSX} from 'react';
import {PlayerSection} from './player_section.tsx';
import {
  TrickPile,
  BiddingOverlay,
  TrumpIndicator,
  type CardSize,
  type BiddingAction,
} from '../components/index.ts';
import {
  type CardCode,
  type TrickCard,
  type PlayerPosition,
  type Player,
  type GamePhase,
  type Suit,
} from '../types.ts';
import {
  TEAM_NAME,
  pointsText,
  type EuchreOutcome,
  type Team,
} from '../hand_result.ts';
import {useTableFlights, type Pickup} from '../table_flights.ts';

/** The properties of TableLayout. */
export interface TableLayoutProps {
  /** Array of 4 players */
  players: Player[];
  /** Cards in the current trick */
  trickPile: TrickCard[];
  /** Current game phase */
  phase: GamePhase;
  /** The upcard */
  upcard: CardCode | null;
  /** Current trump suit */
  trumpSuit: Suit | null;
  /** Dealer position */
  dealer: PlayerPosition;
  /** Declarer position (who called trump) */
  declarer: PlayerPosition | null;
  /** Whether declarer is going alone */
  goingAlone: boolean;
  /** Which player's turn it is */
  currentPlayer: PlayerPosition;
  /** Which player led the current trick */
  leadPosition?: PlayerPosition | undefined;
  /** Tricks won by each team */
  tricksWon: {NS: number; EW: number};
  /** Each table position's side (the viewer's seat is at the bottom). */
  teamOf: Record<PlayerPosition, Team>;
  /** Whether the game has ended */
  isTerminal?: boolean | undefined;
  /** At game over: what the hand scored and why (engine returns) */
  outcome?: EuchreOutcome | null | undefined;
  /** The seat playing alone, by compass name ("East") */
  aloneSeat?: string | null | undefined;
  /** The lone player's partner, who sits the hand out */
  sittingOut?: PlayerPosition | null | undefined;
  /** Whether to show opponent cards face-up */
  showOpponentCards?: boolean | undefined;
  /** Whether all hands are interactive (analysis mode) */
  allHandsInteractive?: boolean | undefined;
  /** Set of playable card codes */
  playableCards?: Set<string> | undefined;
  /** Available bidding actions */
  biddingActions?: BiddingAction[] | undefined;
  /** Callback when player clicks a card */
  onCardClick?: ((card: CardCode) => void) | undefined;
  /** Callback when a bidding action is selected */
  onBiddingAction?: ((action: number) => void) | undefined;
  /** Card size */
  cardSize?: CardSize | undefined;
  /** Status message to display */
  message?: string | undefined;
  /** Additional CSS class */
  className?: string | undefined;
  /** A finished trick being collected (shared/cards trickCollect). */
  collect?:
    | {key: string; winner: PlayerPosition; durationMs: number}
    | null
    | undefined;
  /** The up-card the dealer took up, in the dealer's hand until the discard. */
  pickup?: Pickup | null | undefined;
  /** One card's flight, ms (the card library's motionDuration at the
   *  viewer's Animation speed); 0 draws arrivals in place. */
  flightMs?: number | undefined;
  /** The flight's CSS easing (the card style's motion.easing). */
  flightEasing?: string | undefined;
}

const NO_CARDS = new Set<string>();

/** Tricks won this hand, per side (points are scored at game over). */
function TeamScores({
  tricksWon,
}: {
  tricksWon: {NS: number; EW: number};
}): JSX.Element {
  return (
    <div className="euchre-team-scores">
      {(['NS', 'EW'] as Team[]).map(team => (
        <div
          key={team}
          className={`euchre-team-score euchre-team-score--${team.toLowerCase()}`}
          data-testid={`euchre-tricks-${team}`}
          role="img"
          aria-label={`${TEAM_NAME[team]}: ${String(tricksWon[team])} ${tricksWon[team] === 1 ? 'trick' : 'tricks'}`}
          title={`${TEAM_NAME[team]} tricks won this hand`}
        >
          <span className="euchre-team-score__label">
            {team === 'NS' ? 'N-S' : 'E-W'} tricks
          </span>
          <span className="euchre-team-score__value">{tricksWon[team]}</span>
        </div>
      ))}
    </div>
  );
}

/** The hand-over panel: how the hand ended, and each side's points. */
function GameOver({
  outcome,
  tricksWon,
}: {
  outcome: EuchreOutcome | null;
  tricksWon: {NS: number; EW: number};
}): JSX.Element {
  return (
    <div className="euchre-game-over">
      <div className="euchre-game-over__title">Hand Over</div>
      <div
        className="euchre-game-over__cause"
        data-testid="euchre-outcome-cause"
      >
        {outcome?.cause}
      </div>
      <div className="euchre-game-over__scores">
        {(['NS', 'EW'] as Team[]).map(team => {
          const won = outcome?.scorer === team;
          return (
            <div
              key={team}
              className={`euchre-game-over__team ${won ? 'euchre-game-over__team--winner' : ''}`}
              data-testid={`euchre-outcome-${team}`}
            >
              <span className="euchre-game-over__team-name">
                {TEAM_NAME[team]}
              </span>
              <span className="euchre-game-over__team-points">
                {outcome
                  ? pointsText(outcome.points[team])
                  : 'Points not served'}
              </span>
              <span className="euchre-game-over__team-tricks">
                {tricksWon[team]} {tricksWon[team] === 1 ? 'trick' : 'tricks'}
              </span>
              {won && <span className="euchre-game-over__badge">Scores</span>}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** The table's settings that every seat's section reads. */
type SeatContext = Pick<
  TableLayoutProps,
  | 'phase'
  | 'dealer'
  | 'declarer'
  | 'goingAlone'
  | 'currentPlayer'
  | 'tricksWon'
  | 'teamOf'
  | 'sittingOut'
  | 'showOpponentCards'
  | 'allHandsInteractive'
  | 'playableCards'
  | 'onCardClick'
  | 'cardSize'
>;

/** The section of `player`, sitting at `position`. */
function SeatSection({
  table,
  player,
  position,
}: {
  table: SeatContext;
  player: Player;
  position: PlayerPosition;
}): JSX.Element {
  const {currentPlayer, showOpponentCards = false} = table;
  // Helper to determine if a position's hand should be interactive:
  // in analysis mode with revealed cards, active player's hand is interactive
  const interactive =
    table.phase === 'playing' &&
    (position === 'bottom' ||
      (Boolean(table.allHandsInteractive) &&
        showOpponentCards &&
        currentPlayer === position));
  const interaction = interactive
    ? {
        // A hand that is not to play has no playable card (undefined
        // would read as all of them).
        playableCards:
          currentPlayer === position ? table.playableCards : NO_CARDS,
        interactive: true,
        onCardClick: table.onCardClick,
      }
    : {};
  return (
    <PlayerSection
      name={player.name}
      hand={player.hand}
      position={position}
      // A seat's side's count: the side follows the seat's compass, not its
      // place at the table (an East viewer sits at the bottom).
      tricksWon={table.tricksWon[table.teamOf[position]]}
      team={table.teamOf[position]}
      faceUp={
        position === 'bottom'
          ? !player.hidden
          : showOpponentCards && !player.hidden
      }
      isActive={currentPlayer === position}
      cardSize={table.cardSize}
      isDealer={table.dealer === position}
      isDeclarer={table.declarer === position}
      goingAlone={table.goingAlone}
      sittingOut={table.sittingOut === position}
      {...interaction}
    />
  );
}

/** The centre of the table: the hand over, the trick, or nothing. */
function TableCenter(props: TableLayoutProps): JSX.Element {
  const {collect = null, cardSize = 'md'} = props;
  return (
    <div className="euchre-center-section">
      {props.isTerminal ? (
        <GameOver outcome={props.outcome ?? null} tricksWon={props.tricksWon} />
      ) : props.phase === 'playing' ? (
        <TrickPile
          cards={props.trickPile}
          leadPosition={props.leadPosition}
          cardSize={cardSize}
          collect={
            collect && {
              key: collect.key,
              toward: collect.winner,
              durationMs: collect.durationMs,
            }
          }
        />
      ) : (
        <div className="euchre-center-placeholder" />
      )}
    </div>
  );
}

/**
 * Main Euchre table layout with 4 players, center area, and bidding overlay
 */
export function TableLayout(props: TableLayoutProps): JSX.Element {
  const {
    players,
    trickPile,
    phase,
    upcard,
    trumpSuit,
    goingAlone,
    currentPlayer,
    tricksWon,
    isTerminal = false,
    aloneSeat = null,
    allHandsInteractive = false,
    biddingActions = [],
    onBiddingAction,
    message,
    className = '',
    pickup = null,
    flightMs = 0,
    flightEasing = 'ease-out',
  } = props;
  // Cards arriving in the trick fly from their hands; an ordered-up card
  // flies to the dealer.
  const root = useRef<HTMLDivElement | null>(null);
  useTableFlights(
    root,
    phase === 'playing' && !isTerminal ? trickPile : [],
    pickup,
    {durationMs: flightMs, easing: flightEasing},
  );

  // Each seat, by its place at the table
  const seat = (position: PlayerPosition): JSX.Element | null => {
    const player = players.find(p => p.position === position);
    return player ? (
      <SeatSection table={props} player={player} position={position} />
    ) : null;
  };

  const isBiddingPhase =
    phase === 'bidding1' || phase === 'bidding2' || phase === 'alone';
  const isPlayPhase = phase === 'playing';
  const isViewerTurn = currentPlayer === 'bottom';

  return (
    <div ref={root} className={`euchre-table ${className}`}>
      <TeamScores tricksWon={tricksWon} />

      {/* Trump indicator */}
      {trumpSuit && (
        <TrumpIndicator
          trumpSuit={trumpSuit}
          goingAlone={goingAlone}
          aloneSeat={aloneSeat}
          className="euchre-table__trump"
        />
      )}

      {seat('top')}
      {seat('left')}
      <TableCenter {...props} />
      {seat('right')}
      {/* Bottom player (you) */}
      {seat('bottom')}

      {/* Bidding overlay */}
      {isBiddingPhase && onBiddingAction && (
        <BiddingOverlay
          phase={phase}
          upcard={upcard}
          actions={biddingActions}
          isViewerTurn={isViewerTurn}
          allHandsInteractive={allHandsInteractive}
          trumpSuit={trumpSuit}
          onAction={onBiddingAction}
          statusMessage={message}
        />
      )}

      {/* Status message (for play phase) */}
      {message && isPlayPhase && (
        <div className="euchre-message">{message}</div>
      )}
    </div>
  );
}
