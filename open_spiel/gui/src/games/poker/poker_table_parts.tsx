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

// The poker table's parts: a card, a seat's plate, a chip in flight, and the
// words the table writes (chips, blinds, stacks, the hand's result). Split
// from poker_gameplay_panel.tsx.

import {type CSSProperties, type JSX, useEffect, useRef} from 'react';

import {scaleMs} from '../../motion/index.ts';
import {
  type PokerCard,
  type PokerEnding,
  type PokerSeat,
  cardAriaLabel,
} from './poker_state.ts';
import type {ChipFlight, PokerMotion} from './use_poker_motion.ts';

/** `value` within [min, max]. */
export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

/** Chips as the table writes them: "19,305". */
export function formatChips(value: number): string {
  if (!Number.isFinite(value)) {
    return '0';
  }
  return value.toLocaleString();
}

const suitSymbols: Record<PokerCard['suit'], string> = {
  c: '♣',
  d: '♦',
  h: '♥',
  s: '♠',
};

/** The hand bar's blinds, or the ante at a table without blinds. */
export function formatBlindText(blinds: number[], ante = 0): string {
  const active = blinds.filter(blind => blind > 0);
  if (active.length === 0) {
    return ante > 0 ? `Ante ${formatChips(ante)}` : 'Blinds 0/0';
  }
  if (active.length === 1) {
    return `Blind ${formatChips(active[0] ?? 0)}`;
  }
  const ordered = [...active].sort((left, right) => left - right);
  return `Blinds ${formatChips(ordered[0] ?? 0)}/${formatChips(
    ordered[ordered.length - 1] ?? 0,
  )}`;
}

/** The hand bar's stacks; null at a table without stacks. */
export function formatStackText(stacks: number[]): string | null {
  const unique = Array.from(new Set(stacks.filter(stack => stack > 0)));
  if (unique.length === 0) {
    return null;
  }
  if (unique.length === 1) {
    return `Stack ${formatChips(unique[0] ?? 0)}`;
  }
  return 'Variable stacks';
}

function seatCoordinates(totalSeats: number, slot: number): [number, number] {
  // Heads-up seats sit left and right; three or more sit at the rail top
  // to bottom, so a plate at the top or bottom clears the board.
  const radiusX = totalSeats >= 3 ? 39 : 37;
  const radiusY = totalSeats >= 3 ? 49 : 39;
  const startAngle = totalSeats === 2 ? 0 : -90;
  const angle = ((startAngle + (360 / totalSeats) * slot) * Math.PI) / 180;
  return [50 + Math.cos(angle) * radiusX, 50 + Math.sin(angle) * radiusY];
}

/** A signed chip count: "+18,201", "-18,201". */
export function formatSignedChips(value: number): string {
  if (value > 0) {
    return `+${formatChips(value)}`;
  }
  return formatChips(value);
}

/**
 * The hand's result line. A hand won by fold says so: nobody showed down.
 * The viewer's own seat is "You" ("You win +2"), as the reference clients
 * say it; `viewerSeat` is null for a spectator or the oracle.
 */
export function terminalSummary(
  seats: PokerSeat[],
  returns: number[] | null,
  ending: PokerEnding | null,
  viewerSeat: number | null = null,
): string {
  const byFold = ending === 'fold' ? ' by fold' : '';
  if (!returns || returns.length === 0) {
    return ending === 'fold' ? 'Won by fold' : 'Hand complete';
  }
  const maxReturn = Math.max(...returns);
  if (maxReturn <= 0) {
    return 'Hand complete: split pot';
  }
  const winners = seats.filter(seat => seat.returnValue === maxReturn);
  const names = winners
    .map(seat => (seat.slot === viewerSeat ? 'You' : seat.name))
    .join(', ');
  const verb =
    winners.length > 1
      ? 'split'
      : winners[0]?.slot === viewerSeat
        ? 'win'
        : 'wins';
  return `${names || 'Winner'} ${verb} ${formatSignedChips(
    maxReturn,
  )}${byFold}`;
}

/**
 * A card's motion: 'deal' (onto the board) or 'reveal' (turned face up on a
 * plate). Its delay and flight are base ms, scaled here to the viewer's
 * Animation speed (use_poker_motion.ts). `onLanded` runs at its own
 * animationend, so a motion carried into the next table state ends where it
 * ends, not where the next state arrives.
 */
export interface CardMotion {
  kind: 'deal' | 'reveal';
  delayBaseMs: number;
  motion: PokerMotion;
  onLanded: () => void;
}

function cardMotionStyle({
  delayBaseMs,
  motion,
}: CardMotion): CSSProperties & Record<string, string> {
  return {
    '--poker-flight-ms': `${String(scaleMs(motion.flightBaseMs, motion.scale))}ms`,
    '--poker-delay-ms': `${String(scaleMs(delayBaseMs, motion.scale))}ms`,
  };
}

/**
 * A card, or an empty slot for a card the viewer may not see. `plays`: the
 * card is one of those that make a winning hand at showdown. It is raised
 * and outlined, as the reference clients mark the winning five; the other
 * cards keep full contrast (never dim).
 */
export function CardView({
  card,
  compact = false,
  motion,
  plays = false,
}: {
  readonly card?: PokerCard | undefined;
  readonly compact?: boolean;
  readonly motion?: CardMotion | undefined;
  readonly plays?: boolean;
}): JSX.Element {
  if (!card) {
    return (
      <span
        className={`poker-card poker-card--empty${
          compact ? ' poker-card--compact' : ''
        }`}
        aria-hidden
      />
    );
  }
  const isRed = card.suit === 'd' || card.suit === 'h';
  return (
    <span
      className={`poker-card poker-card--${card.suit}${
        isRed ? ' poker-card--red' : ''
      }${compact ? ' poker-card--compact' : ''}${
        plays ? ' poker-card--plays' : ''
      }${motion ? ` poker-card--${motion.kind}` : ''}`}
      aria-label={
        plays
          ? `${cardAriaLabel(card)}, in the winning hand`
          : cardAriaLabel(card)
      }
      onAnimationEnd={motion?.onLanded}
      style={motion ? cardMotionStyle(motion) : undefined}
    >
      <span className="poker-card__corner">
        <span>{card.rank}</span>
        <span>{suitSymbols[card.suit]}</span>
      </span>
      <span className="poker-card__pip">{suitSymbols[card.suit]}</span>
      <span className="poker-card__corner poker-card__corner--bottom">
        <span>{card.rank}</span>
        <span>{suitSymbols[card.suit]}</span>
      </span>
    </span>
  );
}

function holeMotion(
  motion: PokerMotion,
  slot: number,
  index: number,
  revealLanded: (key: string) => void,
): CardMotion | undefined {
  const key = `${String(slot)}:${String(index)}`;
  const delayBaseMs = motion.revealDelays.get(key);
  return delayBaseMs === undefined
    ? undefined
    : {
        kind: 'reveal',
        delayBaseMs,
        motion,
        onLanded: () => {
          revealLanded(key);
        },
      };
}

function seatBadges(seat: PokerSeat): string[] {
  return [
    seat.isDealer ? 'D' : null,
    seat.isSmallBlind ? 'SB' : null,
    seat.isBigBlind ? 'BB' : null,
  ].filter((badge): badge is string => badge !== null);
}

/** What a seat's plate is drawn from. */
export interface SeatViewProps {
  readonly seat: PokerSeat;
  readonly totalSeats: number;
  /** The hole cards each seat holds. */
  readonly holeCards: number;
  readonly motion: PokerMotion;
  readonly revealLanded: (key: string) => void;
  /** Stack and In as shown: before any chip of this seat still in flight. */
  readonly shown: {stack: number; contribution: number};
  /** Whether the table has stacks to show. */
  readonly hasStacks: boolean;
  /** Codes of the cards that make a winning hand (empty before showdown). */
  readonly playing: ReadonlySet<string>;
  /** Whether this is the viewer's own seat. */
  readonly isViewer: boolean;
}

function SeatBody({
  seat,
  shown,
  hasStacks,
  isViewer,
}: Pick<
  SeatViewProps,
  'seat' | 'shown' | 'hasStacks' | 'isViewer'
>): JSX.Element {
  return (
    <div className="poker-seat__body">
      <div className="poker-seat__name-row">
        <span className="poker-seat__name">{seat.name}</span>
        {isViewer && (
          <span className="poker-seat__badge poker-seat__you">You</span>
        )}
        {seatBadges(seat).map(badge => (
          <span className="poker-seat__badge" key={badge}>
            {badge}
          </span>
        ))}
      </div>
      <div className="poker-seat__metrics">
        {hasStacks && <span>Stack {formatChips(shown.stack)}</span>}
        <span>In {formatChips(shown.contribution)}</span>
      </div>
      {seat.hasFolded ? (
        <div className="poker-seat__folded">Folded</div>
      ) : (
        seat.bestHand && <div className="poker-seat__best">{seat.bestHand}</div>
      )}
      {typeof seat.returnValue === 'number' && (
        <div
          className={`poker-seat__return${
            seat.returnValue > 0 ? ' poker-seat__return--positive' : ''
          }`}
        >
          {formatSignedChips(seat.returnValue)}
        </div>
      )}
    </div>
  );
}

/** A seat's plate at the rail: its cards, name, badges and chips. */
export function SeatView(props: SeatViewProps): JSX.Element {
  const {seat, totalSeats, holeCards, motion, revealLanded, playing} = props;
  const name = props.isViewer ? `${seat.name} (you)` : seat.name;
  const [x, y] = seatCoordinates(totalSeats, seat.slot);
  const seatStyle: CSSProperties & Record<string, string> = {
    '--seat-x': `${String(x)}%`,
    '--seat-y': `${String(y)}%`,
  };
  return (
    <div
      className={`poker-seat poker-seat--${String(seat.slot)}${
        seat.isActive ? ' poker-seat--active' : ''
      }${seat.hasFolded ? ' poker-seat--folded' : ''}${seat.isWinner ? ' poker-seat--winner' : ''}${
        props.isViewer ? ' poker-seat--viewer' : ''
      }`}
      style={seatStyle}
      aria-current={seat.isActive ? 'true' : undefined}
      aria-label={seat.isActive ? `${name}, to act` : name}
      role="group"
    >
      <div className="poker-seat__cards">
        {Array.from({length: holeCards}, (_, index) => {
          const card = seat.cards[index];
          return (
            <CardView
              card={card}
              compact={totalSeats >= 5}
              key={card?.code ?? `hidden-${String(index)}`}
              motion={
                card
                  ? holeMotion(motion, seat.slot, index, revealLanded)
                  : undefined
              }
              plays={seat.isWinner && playing.has(card?.code ?? '')}
            />
          );
        })}
      </div>
      <SeatBody {...props} />
    </div>
  );
}

/** A bet or call travelling from its seat to the pot: plain text, the amount. */
export function ChipFlightView({
  chip,
  motion,
  onLanded,
}: {
  readonly chip: ChipFlight;
  readonly motion: PokerMotion;
  readonly onLanded: (key: string) => void;
}): JSX.Element {
  const style: CSSProperties & Record<string, string> = {
    left: `${String(chip.x)}px`,
    top: `${String(chip.y)}px`,
    '--chip-dx': `${String(chip.dx)}px`,
    '--chip-dy': `${String(chip.dy)}px`,
    '--poker-flight-ms': `${String(scaleMs(motion.flightBaseMs, motion.scale))}ms`,
    '--poker-delay-ms': `${String(scaleMs(chip.delayBaseMs, motion.scale))}ms`,
  };
  // A cancelled animation fires animationcancel, not animationend, and the
  // pot and the seat's numbers wait for one of them. A native listener:
  // React has no animationcancel prop.
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const cancel = (): void => {
      onLanded(chip.key);
    };
    el.addEventListener('animationcancel', cancel);
    return () => {
      el.removeEventListener('animationcancel', cancel);
    };
  }, [chip.key, onLanded]);
  return (
    <span
      aria-hidden
      className="poker-chip-flight"
      data-slot={chip.slot}
      onAnimationEnd={() => {
        onLanded(chip.key);
      }}
      ref={ref}
      style={style}
    >
      {formatChips(chip.amount)}
    </span>
  );
}

/**
 * The cards that make the winning hand(s) at a showdown: each winner's best
 * five, board and hole cards alike. Empty before a showdown and when the
 * hand was won by fold (no hand is ranked then).
 */
export function winningCards(
  seats: readonly PokerSeat[],
  ending: PokerEnding | null,
): Set<string> {
  const codes = new Set<string>();
  if (ending !== 'showdown') {
    return codes;
  }
  for (const seat of seats) {
    if (seat.isWinner) {
      seat.bestFive.forEach(card => codes.add(card.code));
    }
  }
  return codes;
}
