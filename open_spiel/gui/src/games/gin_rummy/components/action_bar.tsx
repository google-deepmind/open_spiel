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

interface ActionBarProps {
  passDisabled?: boolean | undefined;
  knockDisabled?: boolean | undefined;
  onPass?: (() => void) | undefined;
  onKnock?: (() => void) | undefined;
  /** The fanned hand's Sort (melds first, then by rank); absent in grid view. */
  onSort?: (() => void) | undefined;
  /**
   * The knock's laydown compose is open: Done replaces Pass and Knock in
   * this region, so the table never rescales when it appears mid-knock.
   * `onDone` is absent while the staged laydown cannot be submitted yet.
   */
  laydown?:
    | {stage: string | null; onDone?: (() => void) | undefined}
    | null
    | undefined;
  /**
   * The hand is over: Result (the hand's full accounting) replaces Pass and
   * Knock, which no longer do anything.
   */
  onResult?: (() => void) | undefined;
  /**
   * Drawn in the bar pinned at the foot of a scrolling board, in a row,
   * rather than in its column beside the upcard.
   */
  pinned?: boolean | undefined;
}

function LaydownDone({
  onDone,
}: {
  onDone?: (() => void) | undefined;
}): JSX.Element {
  return (
    <button
      type="button"
      className={`gin-action${onDone === undefined ? '' : ' gin-action--primary'}`}
      onClick={onDone}
      disabled={onDone === undefined}
      title="Submit the staged laydown"
    >
      Done
    </button>
  );
}

function PassKnock({
  passDisabled,
  knockDisabled,
  onPass,
  onKnock,
}: Pick<ActionBarProps, 'onPass' | 'onKnock'> & {
  passDisabled: boolean;
  knockDisabled: boolean;
}): JSX.Element {
  return (
    <>
      <button
        type="button"
        className={`gin-action${passDisabled ? '' : ' gin-action--primary'}`}
        onClick={passDisabled ? undefined : onPass}
        disabled={passDisabled}
        title="Pass on the upcard"
      >
        Pass
      </button>
      <button
        type="button"
        className={`gin-action${knockDisabled ? '' : ' gin-action--primary'}`}
        onClick={knockDisabled ? undefined : onKnock}
        disabled={knockDisabled}
        title="Knock: end the hand and lay down your melds"
      >
        Knock
      </button>
    </>
  );
}

/**
 * Pass, Knock and Sort (Done while laying down, Result once the hand is
 * over): the action region beside the upcard, or the bar pinned at the foot
 * of a scrolling board.
 */
export function ActionBar({
  passDisabled = true,
  knockDisabled = true,
  onPass,
  onKnock,
  onSort,
  onResult,
  laydown,
  pinned = false,
}: ActionBarProps): JSX.Element {
  return (
    <div
      className={[
        'gin-actions',
        ...(laydown === null || laydown === undefined
          ? []
          : ['gin-laydown-bar']),
        ...(pinned ? ['gin-actions--pinned'] : []),
      ].join(' ')}
      role="group"
      aria-label="Actions"
      data-laydown-stage={laydown?.stage ?? undefined}
    >
      {laydown !== null && laydown !== undefined ? (
        <LaydownDone onDone={laydown.onDone} />
      ) : onResult !== undefined ? (
        <button
          type="button"
          className="gin-action"
          onClick={onResult}
          title="How the hand was scored: melds, lay-offs, deadwood and points"
          data-testid="gin-result-button"
        >
          Result
        </button>
      ) : (
        <PassKnock
          passDisabled={passDisabled}
          knockDisabled={knockDisabled}
          onPass={onPass}
          onKnock={onKnock}
        />
      )}
      {onSort !== undefined && (
        <button
          type="button"
          className="gin-action gin-action--sort"
          onClick={onSort}
          title="Sort cards (melds first, then by rank)"
        >
          Sort
        </button>
      )}
    </div>
  );
}
