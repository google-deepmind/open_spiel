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

import {type JSX, useMemo, useRef} from 'react';

import type {Action, ChanceOutcome, Frame} from '../contract/contract.ts';
import type {Session} from '../session/use_session.ts';
import {WINDOW_ABOVE, duplicateLabelIds, sortById} from './action_list.ts';
import {LegalWindow} from './legal_window.tsx';
import {NoFrame, Note} from './note.tsx';
import {GAME_FINISHED, playBlocker} from './play_blocker.ts';
import {useKeepFocus} from './use_keep_focus.ts';

import './actions.css';

/** A row of the legal list: an action, or a chance outcome and its odds. */
type Choice = Action & {readonly probability?: number};

// What the list offers at `frame`: the chance outcomes the client picks at
// a chance node under `expose`, otherwise the legal actions; null while
// they are withheld.
function choices(frame: Frame | null): readonly Choice[] | null {
  if (frame === null) return null;
  const outcomes: readonly ChanceOutcome[] = frame.chance_outcomes ?? [];
  return outcomes.length > 0 ? outcomes : frame.legal_actions;
}

/** The name of the legal list: "Legal (N)", or "Legal" when it is empty. */
export function legalLabel(frame: Frame | null): string {
  const count = choices(frame)?.length ?? 0;
  return count === 0 ? 'Legal' : `Legal (${String(count)})`;
}

/** The properties of LegalActions. */
export interface LegalActionsProps {
  readonly session: Session;
  /** Whether the list is out of sight; a hidden list does no work. */
  readonly hidden?: boolean;
}

/**
 * The legal actions at the position shown, in id order, each row showing
 * its id; at a chance node the client picks, the outcomes, each with its
 * probability. When the viewer cannot play, a sentence says why, and the
 * rows are text rather than buttons. When a row with the focus goes, as a
 * played action does, the focus stays in the section.
 */
export function LegalActions({
  session,
  hidden = false,
}: LegalActionsProps): JSX.Element {
  const ref = useRef<HTMLDivElement>(null);
  const keepFocus = useKeepFocus(ref);
  const {frame, busy, commands} = session;
  const offered = choices(frame);
  const sorted = useMemo(
    () => (hidden || offered === null ? [] : sortById(offered)),
    [hidden, offered],
  );
  // A hidden list keeps its box, empty, so that its group still offers its
  // tab: a section that draws nothing has none.
  return (
    <div ref={ref} className="legal" tabIndex={-1} onFocus={keepFocus}>
      {hidden ? null : frame === null ? (
        <NoFrame />
      ) : (
        <LegalContent
          frame={frame}
          sorted={sorted}
          busy={busy}
          act={commands.act}
          chooseChance={commands.chooseChance}
        />
      )}
    </div>
  );
}

interface LegalContentProps {
  readonly frame: Frame;
  /** What the list offers, in id order. */
  readonly sorted: readonly Choice[];
  readonly busy: boolean;
  readonly act: (action: number) => void;
  readonly chooseChance: (outcome: number) => void;
}

function LegalContent({
  frame,
  sorted,
  busy,
  act,
  chooseChance,
}: LegalContentProps): JSX.Element {
  if ((frame.chance_outcomes ?? []).length > 0) {
    return (
      <LegalRows
        actions={sorted}
        label={legalLabel(frame)}
        play={chooseChance}
        busy={busy}
      />
    );
  }
  if (frame.to_move.kind === 'terminal') return <Note>{GAME_FINISHED}</Note>;
  const blocker = playBlocker(frame);
  if (frame.legal_actions === null) {
    return <Note>{blocker ?? 'The legal actions are withheld.'}</Note>;
  }
  return (
    <>
      {blocker !== null && <Note>{blocker}</Note>}
      {sorted.length === 0 ? (
        blocker === null && <Note>No legal actions available.</Note>
      ) : (
        <LegalRows
          actions={sorted}
          label={legalLabel(frame)}
          play={blocker === null ? act : null}
          busy={busy}
        />
      )}
    </>
  );
}

interface LegalRowsProps {
  readonly actions: readonly Choice[];
  readonly label: string;
  /** Plays a row's id; null when the viewer cannot play. */
  readonly play: ((id: number) => void) | null;
  readonly busy: boolean;
}

/** The rows: all of them, or above WINDOW_ABOVE only those in view. */
function LegalRows(props: LegalRowsProps): JSX.Element {
  const {actions, label} = props;
  const duplicates = useMemo(() => duplicateLabelIds(actions), [actions]);
  const renderRow = (action: Choice, tabIndex?: 0 | -1): JSX.Element => (
    <LegalRow
      action={action}
      duplicate={duplicates.has(action.id)}
      play={props.play}
      busy={props.busy}
      tabIndex={tabIndex}
    />
  );
  return (
    <>
      {duplicates.size > 0 && (
        <p className="actions-note">
          {String(duplicates.size)} actions share their label with another;
          their ids tell them apart.
        </p>
      )}
      {actions.length > WINDOW_ABOVE ? (
        <LegalWindow
          actions={actions}
          label={label}
          focusableRows={props.play !== null}
          renderRow={renderRow}
        />
      ) : (
        <ul className="legal__list" aria-label={label}>
          {actions.map((action, index) => (
            <li key={action.id} data-row-index={index}>
              {renderRow(action)}
            </li>
          ))}
        </ul>
      )}
    </>
  );
}

interface LegalRowProps extends Pick<LegalRowsProps, 'play' | 'busy'> {
  readonly action: Choice;
  readonly duplicate: boolean;
  /**
   * Whether the button is in the tab order: undefined for a button's
   * default, which is in it. Required, so a list cannot forget to pass it.
   */
  readonly tabIndex: 0 | -1 | undefined;
}

/**
 * One row: a button when the viewer can play it, text otherwise. While a
 * request is in flight the button refuses a press but keeps the focus.
 */
function LegalRow({
  action,
  duplicate,
  play,
  busy,
  tabIndex,
}: LegalRowProps): JSX.Element {
  const content = (
    <>
      <span className="legal__id">{action.id}</span>
      <span className="legal__label">
        {action.label}
        {action.probability !== undefined && (
          <span className="legal__probability">
            p={action.probability.toFixed(2)}
          </span>
        )}
      </span>
    </>
  );
  const duplicated = duplicate ? 'true' : undefined;
  if (play === null) {
    return (
      <span className="legal__row" data-duplicate={duplicated}>
        {content}
      </span>
    );
  }
  return (
    <button
      type="button"
      className="legal__row"
      data-duplicate={duplicated}
      aria-disabled={busy ? 'true' : undefined}
      title={action.label}
      tabIndex={tabIndex}
      onClick={() => {
        if (!busy) play(action.id);
      }}
    >
      {content}
    </button>
  );
}
