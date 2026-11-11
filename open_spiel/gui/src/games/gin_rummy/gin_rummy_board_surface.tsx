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

// The gin board: the felt, then one shared <Table> over the gin layout,
// with the stats and the action bar in its regions. Its parts:
// gin_board_parts.ts (props), gin_board_staging.ts (the last turn's frames),
// gin_board_deal.ts (layout and items), gin_board_policy.ts (gestures).

import {
  type JSX,
  type ReactNode,
  type RefObject,
  useCallback,
  useLayoutEffect,
  useRef,
  useState,
} from 'react';

import {Table, type ItemRef} from '../../cards/library/index.ts';
import {useTableSurface} from '../../cards/table_cards/index.ts';
import {ActionBar} from './components/action_bar.tsx';
import {GameStats} from './components/game_stats.tsx';
import {HandResult} from './components/hand_result.tsx';
import {TurnLineAnnouncer} from './components/turn_line_announcer.tsx';
import {useDeck} from './deck_context.tsx';
import {
  useBoardLayout,
  useDeal,
  type BoardView,
  type DealFields,
} from './gin_board_deal.ts';
import {
  zoneOf,
  type BoardHandResult,
  type GinRummyBoardSurfaceProps,
} from './gin_board_parts.ts';
import {ginBoardPolicy} from './gin_board_policy.ts';
import {useBoardFrame, type BoardFrame} from './gin_board_staging.ts';
import {GinCardStyleProvider} from './gin_card_style.tsx';
import {
  useCardScale,
  useGridStyle,
  useHandView,
} from './gin_rummy_gameplay_config_extras.tsx';
import {
  GIN_MAX_CARD_WIDTH,
  GIN_REGION,
  GIN_SLOT,
  ginMinCardWidth,
} from './gin_rummy_layout.ts';
import {GIN_TABLE_CARDS_GAME} from './gin_table_cards.ts';
import {useHandOrder, type HandOrder} from './hooks/use_hand_order.ts';

export type {GinRummyBoardSurfaceProps} from './gin_board_parts.ts';
export type {GinBoardCue} from './gin_move_cues.ts';

const NO_CARDS: readonly string[] = [];

/**
 * The gin board on its felt. The felt is the shared table surface: its
 * colour and texture are variables on gin's own root, which gin_rummy.css
 * paints with.
 */
export function GinRummyBoardSurface(
  props: GinRummyBoardSurfaceProps,
): JSX.Element {
  const surface = useTableSurface(GIN_TABLE_CARDS_GAME);
  return (
    <div className="gin-render-wrapper" style={surface}>
      <div className="gin-render-surface">
        <GinCardStyleProvider>
          <GinBoard {...props} />
        </GinCardStyleProvider>
      </div>
    </div>
  );
}

/**
 * Pass, Knock and Sort sit beside the upcard while the whole board is in
 * view. A board that scrolls (a card size past the fit, or a panel under the
 * minimum card width) starts top-left, so the column beside the upcard can
 * lie past its edge: the actions are then pinned at the foot of the
 * scrolling table (<Table foot>), always in view. The foot takes no room
 * from the fit, and the table keeps the actions region, so pinning never
 * re-lays the cards out.
 */
function usePinnedActions(): {
  pinned: boolean;
  boardRef: RefObject<HTMLDivElement | null>;
  onOverflowChange: (scrolls: boolean) => void;
} {
  const [pinned, setPinned] = useState(false);
  const pinnedNow = useRef(false);
  const boardRef = useRef<HTMLDivElement | null>(null);
  // The action button that had focus when the bar moved (it remounts under
  // its new parent): focus follows it there instead of falling to <body>.
  const refocus = useRef<string | null>(null);
  const onOverflowChange = useCallback((scrolls: boolean) => {
    if (scrolls === pinnedNow.current) return;
    pinnedNow.current = scrolls;
    const active = document.activeElement;
    refocus.current =
      active instanceof HTMLButtonElement &&
      active.closest('.gin-actions') !== null &&
      boardRef.current?.contains(active) === true
        ? active.textContent
        : null;
    setPinned(scrolls);
  }, []);
  useLayoutEffect(() => {
    const text = refocus.current;
    refocus.current = null;
    if (text === null) return;
    const buttons = Array.from(
      boardRef.current?.querySelectorAll<HTMLButtonElement>(
        '.gin-actions button',
      ) ?? [],
    );
    buttons.find(b => b.textContent === text)?.focus();
  }, [pinned]);
  return {pinned, boardRef, onOverflowChange};
}

/**
 * The hand's accounting opens from the Result button only, never by itself
 * (discoverable, not pushed); a new hand closes it.
 */
function useResultOpen(
  handResult: BoardHandResult | null,
): [boolean, (open: boolean) => void] {
  const [open, setOpen] = useState(false);
  if (handResult === null && open) setOpen(false);
  return [open && handResult !== null, setOpen];
}

function regionsOf(
  p: GinRummyBoardSurfaceProps,
  frame: BoardFrame,
  v: BoardView,
  actionBar: ReactNode,
  pinned: boolean,
): Record<string, ReactNode> {
  return {
    ...(pinned ? {} : {[GIN_REGION.actions]: actionBar}),
    [GIN_REGION.stats]: (
      <>
        <TurnLineAnnouncer text={frame.shownStatusText} />
        <GameStats
          knockCard={p.knockCard}
          playerDeadwood={p.playerDeadwood}
          result={p.handResult?.figure ?? null}
          returns={p.returns}
          statusText={frame.shownStatusText}
          waitingOnBot={p.waitingOnBot ?? false}
          currentAutoPlay={p.currentAutoPlay ?? false}
          interactionBlockReason={p.interactionBlockReason}
          canInteract={p.canInteract ?? true}
        />
      </>
    ),
    ...(v.gridStyle === 'normal' && p.showOpponentCards
      ? {[GIN_REGION.opponentPanel]: <div className="gin-grid-panel" />}
      : {}),
    ...(v.gridStyle === 'normal' && !v.fanned
      ? {[GIN_REGION.playerPanel]: <div className="gin-grid-panel" />}
      : {}),
  };
}

/**
 * The last turn's mark on a card: on the card's side toward the table's
 * middle, under the piles and the opponent's hand, over a card in the hand
 * at the bottom.
 */
function moveCueOverlay(frame: BoardFrame, ref: ItemRef): JSX.Element | null {
  const {cued} = frame;
  const which =
    ref.id === cued.discard && ref.slot === GIN_SLOT.discard
      ? 'discard'
      : ref.id === cued.stock && ref.slot === GIN_SLOT.stock
        ? 'stock'
        : ref.id === cued.taken && ref.slot !== GIN_SLOT.discard
          ? 'taken'
          : null;
  if (which === null) return null;
  const side =
    which === 'taken' && zoneOf(ref.slot) === 'player' ? 'above' : 'below';
  return (
    <span
      className="gin-move-cue"
      data-gin-cue={which}
      data-side={side}
      aria-hidden="true"
    />
  );
}

function useBoardView(p: GinRummyBoardSurfaceProps): BoardView {
  const [handView] = useHandView();
  const [gridStyle] = useGridStyle();
  const deckConfig = useDeck();
  // A face-down near hand has no faces to lay out in a grid: it is drawn
  // fanned (overlapped unless the viewer chose the arc).
  const handHidden = p.playerHandHidden === true && p.playerHand.length > 0;
  const fanned = handHidden || handView === 'flat' || handView === 'arc';
  return {handView, gridStyle, deckConfig, handHidden, fanned};
}

function dealFields(
  p: GinRummyBoardSurfaceProps,
  v: BoardView,
  frame: BoardFrame,
  hand: HandOrder,
): DealFields {
  return {
    deckConfig: v.deckConfig,
    showOpponentCards: p.showOpponentCards,
    shownOpponentStates: frame.shownOpponentStates,
    shownOpponentCount: frame.shownOpponentCount,
    shownHiddenHandCount: frame.shownHiddenHandCount,
    takenFaceUp: frame.takenFaceUp,
    cueZone: frame.cue?.zone ?? null,
    shownStockCount: frame.shownStockCount,
    shownDiscard: frame.shownDiscard,
    staging: frame.staging,
    canDrawDiscard: p.onDrawDiscard !== undefined,
    handHidden: v.handHidden,
    fanned: v.fanned,
    order: hand.order,
    picked: hand.picked,
    playerGhostCards: p.playerGhostCards ?? NO_CARDS,
    playerCardStates: p.playerCardStates,
  };
}

function GinActionBar({
  p,
  v,
  pinned,
  onResult,
}: {
  readonly p: GinRummyBoardSurfaceProps;
  readonly v: BoardView;
  readonly pinned: boolean;
  readonly onResult: (() => void) | undefined;
}): JSX.Element {
  return (
    <ActionBar
      passDisabled={p.passDisabled ?? true}
      knockDisabled={p.knockDisabled ?? true}
      onPass={p.onPass}
      onKnock={p.onKnock}
      atWall={p.atWall ?? false}
      onSort={v.fanned && !v.handHidden ? p.onSort : undefined}
      onResult={onResult}
      laydown={p.laydown ?? null}
      pinned={pinned}
    />
  );
}

/**
 * A display fault the board recovered from (oneCardOnePlace): shown, not
 * absorbed, so a recurrence is seen and reported.
 */
function DisplayFault({
  faults,
}: {
  readonly faults: readonly string[];
}): JSX.Element | null {
  if (faults.length === 0) return null;
  return (
    <div
      className="gin-board__fault"
      role="alert"
      data-testid="gin-display-fault"
    >
      Display fault, please report: {faults.join('; ')}
    </div>
  );
}

function GinBoard(p: GinRummyBoardSurfaceProps): JSX.Element {
  const [cardScale] = useCardScale();
  const v = useBoardView(p);
  const frame = useBoardFrame(p, v.handHidden);
  const hand = useHandOrder(
    v.handHidden ? NO_CARDS : p.playerHand,
    p.sortTrigger ?? 0,
    p.playerMelds,
    p.playerBestMeldIds,
  );
  const ownCount = v.fanned && !v.handHidden ? hand.order.length : 0;
  const {opts, spec} = useBoardLayout(p, v, frame, hand.order.length);
  const {items, faults} = useDeal(
    dealFields(p, v, frame, hand),
    frame.lastDealRef,
  );
  const {handHidden, fanned} = v;
  const policy = ginBoardPolicy({p, frame, handHidden, fanned, hand, ownCount});
  const {pinned, boardRef, onOverflowChange} = usePinnedActions();
  const handResult = p.handResult ?? null;
  const [resultOpen, setResultOpen] = useResultOpen(handResult);
  const actionBar = (
    <GinActionBar
      p={p}
      v={v}
      pinned={pinned}
      onResult={
        handResult === null
          ? undefined
          : () => {
              setResultOpen(true);
            }
      }
    />
  );
  return (
    <div
      ref={boardRef}
      className={['gin-board', `gin-board--grid-${v.gridStyle}`].join(' ')}
      data-testid="gin-board"
      data-card-scale={cardScale}
      data-gin-staging={frame.phase ?? undefined}
      data-actions-pinned={pinned ? 'true' : undefined}
    >
      <DisplayFault faults={faults} />
      <Table
        spec={spec}
        items={items}
        policy={policy}
        regions={regionsOf(p, frame, v, actionBar, pinned)}
        fit="contain"
        minCardWidth={ginMinCardWidth(opts)}
        maxCardWidth={GIN_MAX_CARD_WIDTH}
        cardScale={cardScale / 100}
        onOverflowChange={onOverflowChange}
        foot={
          pinned ? (
            <div className="gin-board__pinned-actions">{actionBar}</div>
          ) : undefined
        }
        tableId="gin"
        testId="gin-table"
        renderOverlay={ref => moveCueOverlay(frame, ref)}
      />
      {resultOpen && handResult !== null && (
        <HandResult
          result={handResult.result}
          name={handResult.name}
          onClose={() => {
            setResultOpen(false);
          }}
        />
      )}
    </div>
  );
}
