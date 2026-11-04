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

// The felt and its info box: one layout for every classic card table. What
// is not a card is said once, in a box on the felt, never off to the side:
// the felt fills everything.
//
//   <FeltTable info={<TableInfoBox …/>}>{the card table}</FeltTable>
//
// The felt fills the whole Gameplay panel, edge to edge: no white column, no
// prompt line above it, no controls under it. Everything a player reads that
// is not a card (whose turn and the prompt, the hand and the game's target,
// the score, the buttons a decision offers) is said once, in ONE compact box
// on the felt, in its top-left corner, the same place and style at every
// table (as a classic contract table keeps its board box in a corner).
//
// The box takes only its own height (a strip the box's width down the whole
// felt would leave an empty column under it, and squeeze the hand into the
// rest). The cards lay out over the WHOLE felt and centre on its
// midline; only what the box actually overlaps yields room: a <Table> keeps
// every card out of the box's corner (felt_keep_out.ts, the scene's keepOut),
// and a game that lays out rows itself asks for a row's keep-out
// (useFeltKeepOut). The box never covers a card, at its tallest too; when
// it is taller than the felt it scrolls inside. On a phone the box stacks
// over the cards, still on the felt.
import type {CSSProperties, JSX, ReactNode, Ref} from 'react';
import './felt_table.css';

/** A felt table: its info box and its cards. */
export interface FeltTableProps {
  /** The info box (a <TableInfoBox>), drawn in the felt's top-left corner. */
  info: ReactNode;
  /** The card table: it lays out over the whole felt, clear of the box. */
  children: ReactNode;
  /** The felt's test id (default 'felt-table'). */
  testId?: string | undefined;
  /** The cards area's test id (default 'felt-cards'). */
  cardsTestId?: string | undefined;
  className?: string | undefined;
  /** The cards area's own class (a game's layout of its rows). */
  cardsClassName?: string | undefined;
  /** The table surface settings (useTableSurface) and other inline vars. */
  style?: CSSProperties | undefined;
  /** The cards area, for a game that measures its room. */
  cardsRef?: Ref<HTMLDivElement> | undefined;
  /** Extra data attributes on the felt (data-can-act, data-audience, …). */
  data?: Readonly<
    Record<`data-${string}`, string | number | boolean | undefined>
  >;
}

/** The felt filling the Gameplay panel, with the info box in its corner. */
export function FeltTable({
  info,
  children,
  testId = 'felt-table',
  cardsTestId = 'felt-cards',
  className,
  cardsClassName,
  style,
  cardsRef,
  data,
}: FeltTableProps): JSX.Element {
  return (
    <div
      className={className ? `felt-table ${className}` : 'felt-table'}
      data-felt-table=""
      data-testid={testId}
      style={style}
      {...data}
    >
      <div className="felt-table__grid">
        {/* First in reading order; it floats over the felt's corner, which
          the cards keep clear of (felt_keep_out.ts finds it by
          data-felt-info). */}
        <div className="felt-table__info" data-felt-info="">
          {info}
        </div>
        <div
          className={
            cardsClassName
              ? `felt-table__cards ${cardsClassName}`
              : 'felt-table__cards'
          }
          data-felt-cards=""
          data-testid={cardsTestId}
          ref={cardsRef}
        >
          {children}
        </div>
      </div>
    </div>
  );
}

/** The felt's info box: the game's state and its buttons. */
export interface TableInfoBoxProps {
  /** The box's test id (default 'table-info'). */
  testId?: string | undefined;
  /** Its accessible name (default 'Game info'). */
  label?: string | undefined;
  /** Where the game stands: "Hand 1 · game to 1,000", "Trick 3 of 13". */
  progress?: ReactNode | undefined;
  progressTestId?: string | undefined;
  /** Whose turn and what to do: "Your lead". Two lines reserved. */
  prompt: ReactNode;
  promptTestId?: string | undefined;
  /** The prompt is the game's result. */
  done?: boolean | undefined;
  /** Why a card cannot be played, or a hint: one line under the prompt
   *  (absent: no line). Empty, the line takes no height: the cards keep
   *  clear of the box at whatever height it has, so reserving it protected
   *  no card and left a dead band in every box. */
  hint?: ReactNode | undefined;
  hintTestId?: string | undefined;
  /** The hint is a refusal (warning tone). */
  refusal?: boolean | undefined;
  /** The buttons the decision offers (Knock, Draw, a meld list). */
  actions?: ReactNode | undefined;
  actionsTestId?: string | undefined;
  /** The score and the rest, as <InfoSection>s. */
  children?: ReactNode | undefined;
}

function blank(v: ReactNode) {
  return v === '' || v === null || v === undefined || v === false;
}

/** ONE compact box on the felt: progress, prompt, hint, buttons, score. */
export function TableInfoBox({
  testId = 'table-info',
  label = 'Game info',
  progress,
  progressTestId,
  prompt,
  promptTestId,
  done,
  hint,
  hintTestId,
  refusal,
  actions,
  actionsTestId,
  children,
}: TableInfoBoxProps): JSX.Element {
  return (
    <section
      className="table-info"
      aria-label={label}
      data-table-info=""
      data-testid={testId}
    >
      {progress !== undefined && (
        <p className="table-info__progress" data-testid={progressTestId}>
          {blank(progress) ? ' ' : progress}
        </p>
      )}
      <p
        className="table-info__prompt"
        role="status"
        data-testid={promptTestId}
        data-done={done ? 'true' : undefined}
      >
        {blank(prompt) ? ' ' : prompt}
      </p>
      {hint !== undefined && (
        <p
          className="table-info__hint"
          aria-live="polite"
          data-testid={hintTestId}
          data-refusal={refusal ? 'true' : undefined}
          data-empty={blank(hint) ? 'true' : undefined}
        >
          {blank(hint) ? ' ' : hint}
        </p>
      )}
      {!blank(actions) && (
        <div className="table-info__actions" data-testid={actionsTestId}>
          {actions}
        </div>
      )}
      {children}
    </section>
  );
}

/** A titled section of the felt's info box. */
export interface InfoSectionProps {
  title?: ReactNode | undefined;
  testId?: string | undefined;
  label?: string | undefined;
  children: ReactNode;
}

/** A titled part of the info box (the score, a deal's tally). */
export function InfoSection({
  title,
  testId,
  label,
  children,
}: InfoSectionProps): JSX.Element {
  return (
    <div
      className="table-info__section"
      data-testid={testId}
      aria-label={label}
      role={label ? 'group' : undefined}
    >
      {title !== undefined && <h3 className="table-info__title">{title}</h3>}
      {children}
    </div>
  );
}
