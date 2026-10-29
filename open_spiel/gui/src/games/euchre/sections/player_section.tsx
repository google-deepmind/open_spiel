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
import {CardFan, type CardSize} from '../components/index.ts';
import {type CardCode, type PlayerPosition} from '../types.ts';
import {TEAM_NAME, type Team} from '../hand_result.ts';

/** The properties of PlayerSection. */
export interface PlayerSectionProps {
  /** Player name */
  name: string;
  /** Cards in hand */
  hand: CardCode[];
  /** Player position on the table */
  position: PlayerPosition;
  /** Tricks won by this player's team */
  tricksWon?: number | undefined;
  /** The seat's side, by its compass seat (not its place at the table) */
  team: Team;
  /** Whether this player is the dealer */
  isDealer?: boolean | undefined;
  /** Whether this player is the declarer (called trump) */
  isDeclarer?: boolean | undefined;
  /** Whether the declarer is playing alone (names the ★ badge) */
  goingAlone?: boolean | undefined;
  /** This seat sits the hand out: its partner, the declarer, plays alone */
  sittingOut?: boolean | undefined;
  /** Whether to show card faces */
  faceUp?: boolean | undefined;
  /** Whether this player is active (their turn) */
  isActive?: boolean | undefined;
  /** Set of selected card codes */
  selectedCards?: Set<string> | undefined;
  /** Set of playable card codes */
  playableCards?: Set<string> | undefined;
  /** Whether the player can interact with cards */
  interactive?: boolean | undefined;
  /** Callback when a card is clicked */
  onCardClick?: ((card: CardCode) => void) | undefined;
  /** Card size */
  cardSize?: CardSize | undefined;
  /** Maximum spread for the card fan */
  maxSpread?: number | undefined;
}

/** The hand's accessible name: whose hand, how many cards, face down, sitting out. */
function handLabelOf(
  name: string,
  count: number,
  faceUp: boolean,
  sittingOut: boolean,
): string {
  return [
    `Hand of ${name}`,
    `${String(count)} ${count === 1 ? 'card' : 'cards'}`,
    !faceUp && count > 0 ? 'face down' : null,
    sittingOut ? 'sits out this hand' : null,
  ]
    .filter(Boolean)
    .join(', ');
}

/** The dealer's D and the declarer's star beside a seat's name. */
function PlayerBadges({
  isDealer,
  isDeclarer,
  declarerName,
}: {
  isDealer: boolean;
  isDeclarer: boolean;
  declarerName: string;
}): JSX.Element {
  return (
    <div className="euchre-player-badges">
      {isDealer && (
        <span
          className="euchre-badge euchre-badge--dealer"
          title="Dealer"
          role="img"
          aria-label="Dealer"
        >
          D
        </span>
      )}
      {isDeclarer && (
        <span
          className="euchre-badge euchre-badge--declarer"
          title={declarerName}
          role="img"
          aria-label={declarerName}
        >
          ★
        </span>
      )}
    </div>
  );
}

/**
 * Player section component showing name, indicators, and card fan
 */
export function PlayerSection({
  name,
  hand,
  position,
  tricksWon,
  team,
  isDealer = false,
  isDeclarer = false,
  goingAlone = false,
  sittingOut = false,
  faceUp = true,
  isActive = false,
  selectedCards,
  playableCards,
  interactive = false,
  onCardClick,
  maxSpread,
}: PlayerSectionProps): JSX.Element {
  // Determine orientation based on position
  const isVertical = position === 'left' || position === 'right';
  const isBottom = position === 'bottom';
  const orientation = isVertical ? 'vertical' : 'horizontal';

  // Default max spread based on position (smaller for Euchre's 5-card hands)
  const defaultMaxSpread = isVertical ? 220 : isBottom ? 350 : 320;
  const spread = maxSpread ?? defaultMaxSpread;

  // Use medium cards for all players
  const effectiveCardSize: CardSize = 'md';

  const teamName = TEAM_NAME[team];
  const declarerName = goingAlone ? 'Declarer, playing alone' : 'Declarer';
  const handLabel = handLabelOf(name, hand.length, faceUp, sittingOut);

  return (
    <div className={`euchre-player-section euchre-player-section--${position}`}>
      <div className="euchre-player-info">
        <span
          className={`euchre-player-name euchre-player-name--${position} ${isActive ? 'euchre-player-name--active' : ''}`}
        >
          {/* A narrow side column wraps the label; keep "Player 1" together. */}
          {name.replace(/ (\d+)$/, '\u00a0$1')}
        </span>
        <PlayerBadges
          isDealer={isDealer}
          isDeclarer={isDeclarer}
          declarerName={declarerName}
        />
        {sittingOut && (
          <span
            className="euchre-sits-out"
            data-testid={`euchre-sits-out-${position}`}
          >
            Sits out
          </span>
        )}
        {tricksWon !== undefined && tricksWon > 0 && (
          <span
            className="euchre-tricks-won"
            role="img"
            title={`${teamName} tricks won`}
            aria-label={`${teamName}: ${String(tricksWon)} ${tricksWon === 1 ? 'trick' : 'tricks'}`}
          >
            {tricksWon}
          </span>
        )}
      </div>

      <CardFan
        cards={hand}
        orientation={orientation}
        position={position}
        faceUp={faceUp}
        selectedCards={selectedCards}
        playableCards={playableCards}
        cardSize={effectiveCardSize}
        maxSpread={spread}
        fanAngle={0}
        interactive={interactive}
        onCardClick={onCardClick}
        label={handLabel}
      />
    </div>
  );
}
