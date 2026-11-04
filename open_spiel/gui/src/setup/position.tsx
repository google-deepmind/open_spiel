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
import {useId, useState} from 'react';

import type {Frame} from '../contract/contract.ts';
import type {RequestError} from '../session/errors.ts';
import {parseHistory} from './position_text.ts';
import {ResetControl} from './reset_control.tsx';

import './setup.css';

/** The properties of Position. */
export interface PositionProps {
  frame: Frame;
  busy: boolean;
  /** Why the latest request failed, or null. */
  error: RequestError | null;
  /** Restarts the game and replays `history`; the seats stay. */
  onReset: (history: readonly number[]) => void;
}

/**
 * Replaces the live position: Set state replays typed action ids from the
 * start of the game, and Reset, once confirmed, starts the game again.
 */
export function Position({
  frame,
  busy,
  error,
  onReset,
}: PositionProps): JSX.Element {
  return (
    <section className="setup-card" aria-label="Position">
      <div className="setup-card__header">
        <p className="setup-card__description">
          Set state or reset. Each replaces the current position.
        </p>
      </div>
      {/* A new session starts the controls again, closed. */}
      <div key={frame.session.id} className="setup-position__buttons">
        <SetState busy={busy} error={error} onReset={onReset} />
        <ResetControl
          busy={busy}
          hasHistory={frame.num_plies > 0}
          onReset={() => {
            onReset([]);
          }}
        />
      </div>
    </section>
  );
}

// Where the history typed stands: being typed, sent, applied, or refused
// with a reason, by the parser or the server.
type Outcome =
  | {readonly kind: 'idle' | 'sent' | 'applied'}
  | {readonly kind: 'problem'; readonly problem: string};

const IDLE: Outcome = {kind: 'idle'};

function SetState({
  busy,
  error,
  onReset,
}: Omit<PositionProps, 'frame'>): JSX.Element {
  const id = useId();
  const [open, setOpen] = useState(false);
  const [text, setText] = useState('');
  const [outcome, setOutcome] = useState<Outcome>(IDLE);
  // The answer to the history sent: applied, or the server's reason.
  if (outcome.kind === 'sent' && !busy) {
    setOutcome(
      error === null
        ? {kind: 'applied'}
        : {kind: 'problem', problem: error.message},
    );
    if (error === null) setOpen(false);
  }
  const apply = (): void => {
    const parsed = parseHistory(text);
    setOutcome(
      parsed.ok ? {kind: 'sent'} : {kind: 'problem', problem: parsed.problem},
    );
    if (parsed.ok) onReset(parsed.history);
  };
  const sending = outcome.kind === 'sent';
  const problem = outcome.kind === 'problem' ? outcome.problem : null;
  return (
    <div className="setup-set-state">
      <button
        type="button"
        className="button button--ghost"
        aria-expanded={open}
        aria-controls={`${id}-panel`}
        disabled={busy && !open}
        onClick={() => {
          setOpen(!open);
        }}
      >
        Set state
      </button>
      {open && (
        <div id={`${id}-panel`} className="setup-set-state__panel">
          <label className="setup-set-state__label" htmlFor={id}>
            Action sequence (comma or space separated, brackets ok)
          </label>
          <textarea
            id={id}
            className="setup-input"
            rows={3}
            spellCheck={false}
            placeholder="e.g. [1, 4, 9] or 1 4 9"
            value={text}
            disabled={sending}
            aria-invalid={problem !== null}
            aria-describedby={problem === null ? undefined : `${id}-problem`}
            onChange={event => {
              setText(event.currentTarget.value);
            }}
          />
          <SetStateActions
            sending={sending}
            busy={busy}
            applied={outcome.kind === 'applied'}
            problem={problem}
            problemId={`${id}-problem`}
            onApply={apply}
            onClose={() => {
              setOpen(false);
              setOutcome(IDLE);
            }}
          />
        </div>
      )}
    </div>
  );
}

interface SetStateActionsProps {
  readonly sending: boolean;
  readonly busy: boolean;
  readonly applied: boolean;
  readonly problem: string | null;
  readonly problemId: string;
  readonly onApply: () => void;
  readonly onClose: () => void;
}

// Apply and Close, then what became of the history last applied.
function SetStateActions({
  sending,
  busy,
  applied,
  problem,
  problemId,
  onApply,
  onClose,
}: SetStateActionsProps): JSX.Element {
  return (
    <div className="setup-set-state__actions">
      <button
        type="button"
        className="button"
        disabled={sending || busy}
        onClick={onApply}
      >
        {sending ? 'Applying…' : 'Apply'}
      </button>
      <button
        type="button"
        className="button button--ghost"
        disabled={sending}
        onClick={onClose}
      >
        Close
      </button>
      {applied && (
        <span className="setup-set-state__applied" role="status">
          Applied
        </span>
      )}
      {problem !== null && (
        <span id={problemId} className="setup-set-state__problem" role="alert">
          {problem}
        </span>
      )}
    </div>
  );
}
