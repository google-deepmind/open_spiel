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
import {Card} from './card.tsx';
import {
  type CardCode,
  type Suit,
  type GamePhase,
  SUIT_META,
  type ActionType,
  parseCard,
} from '../types.ts';

/** A bidding action, as the adapter describes it. */
export interface BiddingAction {
  action: number;
  label: string;
  actionType?: ActionType;
  calledSuit?: string | null;
}

/** The properties of BiddingOverlay. */
export interface BiddingOverlayProps {
  /** Current bidding phase */
  phase: GamePhase;
  /** The upcard being bid on */
  upcard: CardCode | null;
  /** Available bidding actions */
  actions: BiddingAction[];
  /** Whether it's the viewer's turn */
  isViewerTurn: boolean;
  /** Whether in analysis mode (can act for any player) */
  allHandsInteractive?: boolean | undefined;
  /** Current trump suit (if already called) */
  trumpSuit?: Suit | null | undefined;
  /** Callback when an action is selected */
  onAction: (action: number) => void;
  /** Status message to display */
  statusMessage?: string | undefined;
}

/** What a bidding button is given: the action, whether it may be pressed. */
interface BidButtonsProps {
  phase: GamePhase;
  actions: BiddingAction[];
  canAct: boolean;
  onAction: (action: number) => void;
}

/** A second-round button naming a suit as trump. */
function SuitButton({
  action,
  canAct,
  onAction,
}: {
  action: BiddingAction;
  canAct: boolean;
  onAction: (action: number) => void;
}): JSX.Element | null {
  const suit = action.calledSuit as Suit | null;
  if (!suit) return null;

  const meta = SUIT_META[suit];
  const isRed = suit === 'h' || suit === 'd';

  return (
    <button
      className={`euchre-bidding-btn euchre-bidding-btn--suit ${isRed ? 'euchre-bidding-btn--red' : 'euchre-bidding-btn--black'}`}
      onClick={() => {
        onAction(action.action);
      }}
      disabled={!canAct}
      title={meta.name}
    >
      <span className="euchre-bidding-btn__symbol">{meta.symbol}</span>
    </button>
  );
}

/** A bidding button that plays `action` when pressed. */
function BidButton({
  action,
  canAct,
  onAction,
  primary,
  children,
}: {
  action: BiddingAction;
  canAct: boolean;
  onAction: (action: number) => void;
  primary: boolean;
  children: string;
}): JSX.Element {
  return (
    <button
      className={`euchre-bidding-btn euchre-bidding-btn--${primary ? 'primary' : 'secondary'}`}
      onClick={() => {
        onAction(action.action);
      }}
      disabled={!canAct}
    >
      {children}
    </button>
  );
}

/** The bidding buttons of the phase: Order Up or Pass, a suit or Pass, alone or not. */
function BiddingButtons({
  phase,
  actions,
  canAct,
  onAction,
}: BidButtonsProps): JSX.Element {
  // Filter actions by type
  const passAction = actions.find(a => a.actionType === 'pass');
  const suitActions = actions.filter(a => a.actionType === 'call_suit');
  const aloneAction = actions.find(a => a.actionType === 'alone');
  const partnerAction = actions.find(a => a.actionType === 'partner');
  const orderUp = suitActions.length === 1 ? suitActions[0] : undefined;
  const press = {canAct, onAction};

  return (
    <div className="euchre-bidding-actions">
      {/* Round 1: Order Up / Pass */}
      {phase === 'bidding1' && orderUp && (
        <>
          <BidButton action={orderUp} primary {...press}>
            Order Up
          </BidButton>
          {passAction && (
            <BidButton action={passAction} primary={false} {...press}>
              Pass
            </BidButton>
          )}
        </>
      )}

      {/* Round 2: Call suit / Pass */}
      {phase === 'bidding2' && (
        <>
          <div className="euchre-bidding-suits">
            {suitActions.map(action => (
              <SuitButton key={action.action} action={action} {...press} />
            ))}
          </div>
          {passAction && (
            <BidButton action={passAction} primary={false} {...press}>
              Pass
            </BidButton>
          )}
        </>
      )}

      {/* Alone choice */}
      {phase === 'alone' && (
        <>
          {aloneAction && (
            <BidButton action={aloneAction} primary {...press}>
              Go Alone
            </BidButton>
          )}
          {partnerAction && (
            <BidButton action={partnerAction} primary={false} {...press}>
              Play with Partner
            </BidButton>
          )}
        </>
      )}
    </div>
  );
}

/**
 * Centered overlay modal for the bidding phase
 */
export function BiddingOverlay({
  phase,
  upcard,
  actions,
  isViewerTurn,
  allHandsInteractive = false,
  trumpSuit,
  onAction,
  statusMessage,
}: BiddingOverlayProps): JSX.Element | null {
  // Don't show overlay if not in a bidding phase
  if (phase !== 'bidding1' && phase !== 'bidding2' && phase !== 'alone') {
    return null;
  }

  // At the Go Alone step the up-card is in the dealer's hand only if it was
  // ordered up in the first round: the second round may not name its suit,
  // so trump is the up-card's suit exactly when it was picked up.
  const pickedUp =
    phase === 'alone' &&
    upcard &&
    trumpSuit &&
    parseCard(upcard).suit === trumpSuit
      ? upcard
      : null;

  // Can act if it's viewer's turn OR in analysis mode
  const canAct = isViewerTurn || allHandsInteractive;

  return (
    <div className="euchre-bidding-overlay">
      <div className="euchre-bidding-modal">
        {/* Phase title */}
        <h3 className="euchre-bidding-title">
          {phase === 'bidding1' && 'First Round'}
          {phase === 'bidding2' && 'Second Round'}
          {phase === 'alone' && 'Go Alone?'}
        </h3>

        {/* Status message */}
        {statusMessage && (
          <p className="euchre-bidding-status">{statusMessage}</p>
        )}

        {/* Upcard display (bidding phases; at Go Alone, only if picked up) */}
        {upcard && (phase === 'bidding1' || phase === 'bidding2') && (
          <div className="euchre-bidding-upcard" data-euchre-upcard>
            <span className="euchre-bidding-upcard-label">
              {phase === 'bidding1' ? 'Upcard:' : 'Turned down:'}
            </span>
            <Card card={upcard} faceUp size="sm" />
          </div>
        )}
        {pickedUp && (
          <div className="euchre-bidding-upcard">
            <span className="euchre-bidding-upcard-label">
              Dealer picked up:
            </span>
            <Card card={pickedUp} faceUp size="sm" />
          </div>
        )}

        {/* Bidding buttons */}
        <BiddingButtons
          phase={phase}
          actions={actions}
          canAct={canAct}
          onAction={onAction}
        />

        {/* Current trump indicator (if already set) */}
        {trumpSuit && (
          <div className="euchre-bidding-trump">
            Trump:{' '}
            <span
              className={
                SUIT_META[trumpSuit].color === '#e53e3e' ? 'euchre-red' : ''
              }
            >
              {SUIT_META[trumpSuit].symbol} {SUIT_META[trumpSuit].name}
            </span>
          </div>
        )}

        {/* Waiting indicator for non-active player */}
        {!canAct && (
          <div className="euchre-bidding-waiting">Waiting for opponent...</div>
        )}
      </div>
    </div>
  );
}
