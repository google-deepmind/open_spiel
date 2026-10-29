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

import {CheckIcon, CopyIcon, FailedIcon} from './icons.tsx';
import type {CopyStatus} from './use_copy.ts';
import {useCopy} from './use_copy.ts';

import './controls.css';

const NAMES: Record<CopyStatus, string> = {
  idle: 'Copy to clipboard',
  copied: 'Copied',
  failed: 'Copy failed',
};

const TITLES: Record<CopyStatus, string> = {
  idle: 'Copy to clipboard',
  copied: 'Copied!',
  failed: 'Copy failed',
};

/**
 * An icon button that copies `text` to the clipboard. For a moment after a
 * press it shows a check mark, or the failure, in its icon and its name, and
 * says the outcome to screen readers.
 */
export function CopyButton({text}: {readonly text: string}): JSX.Element {
  const {status, message, copy} = useCopy();
  return (
    <>
      <button
        type="button"
        className="button button--icon copy-button"
        data-status={status}
        aria-label={NAMES[status]}
        title={TITLES[status]}
        onClick={() => {
          void copy(text);
        }}
      >
        {status === 'copied' ? (
          <CheckIcon />
        ) : status === 'failed' ? (
          <FailedIcon />
        ) : (
          <CopyIcon />
        )}
      </button>
      <span className="visually-hidden" role="status" aria-live="polite">
        {message}
      </span>
    </>
  );
}

/** The properties of CopyableText. */
export interface CopyableTextProps {
  readonly text: string;
  /** The accessible name of the text. */
  readonly label: string;
  /** Further classes of the text's <pre>. */
  readonly className?: string;
  /**
   * 'compact': a card sized to a few lines (rewards, returns, a seat's
   * message), with a smaller inset. Default: a card that fills its panel.
   */
  readonly variant?: 'fill' | 'compact';
  /** The text's role: 'group' (default), or 'region' for a landmark. */
  readonly role?: 'group' | 'region';
}

/**
 * Preformatted text in a card, with its copy button in the card's top right
 * corner. The text takes the focus, so that a keyboard can scroll it, and
 * keeps clear of the button.
 */
export function CopyableText({
  text,
  label,
  className,
  variant = 'fill',
  role = 'group',
}: CopyableTextProps): JSX.Element {
  const compact = variant === 'compact';
  return (
    <div className={compact ? 'copyable copyable--compact' : 'copyable'}>
      <CopyButton text={text} />
      <pre
        className={`copyable__text${compact ? ' copyable__text--compact' : ''}${className === undefined ? '' : ` ${className}`}`}
        role={role}
        tabIndex={0}
        aria-label={label}
      >
        {text}
      </pre>
    </div>
  );
}

/**
 * Preformatted text in the same card as CopyableText, sized to its few
 * lines, with the same copy button in its top right corner: rewards and
 * returns.
 */
export function CompactText({
  text,
  label,
}: {
  readonly text: string;
  /** The accessible name of the text. */
  readonly label: string;
}): JSX.Element {
  return <CopyableText text={text} label={label} variant="compact" />;
}
