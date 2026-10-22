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

import {useCallback, useEffect, useRef, useState} from 'react';

/** How long the outcome of a copy stays shown, in milliseconds. */
export const COPY_FEEDBACK_MS = 1200;

/** Whether the last copy, if one is still being reported, succeeded. */
export type CopyStatus = 'idle' | 'copied' | 'failed';

/** What useCopy returns. */
export interface CopyState {
  status: CopyStatus;
  /** A sentence for screen readers; empty while idle. */
  message: string;
  copy: (text: string) => Promise<void>;
}

const MESSAGES: Record<CopyStatus, string> = {
  idle: '',
  copied: 'Copied to the clipboard.',
  failed: 'The browser did not allow copying to the clipboard.',
};

/**
 * Copies `text` through a hidden, selected text area: the one way to reach
 * the clipboard outside a secure context, where the Clipboard API is absent.
 * The text area sits beside the focused element, so focus never leaves the
 * container that holds it (a menu that closes on losing focus stays open).
 * The focus and selection it borrows are given back, and the page does not
 * scroll to the text area.
 */
function copyBySelection(text: string): boolean {
  const focused = document.activeElement;
  const selection = document.getSelection();
  const ranges =
    selection === null
      ? []
      : Array.from({length: selection.rangeCount}, (_, index) =>
          selection.getRangeAt(index),
        );
  const area = document.createElement('textarea');
  area.value = text;
  area.setAttribute('readonly', '');
  area.className = 'visually-hidden';
  area.style.position = 'fixed';
  area.style.top = '0';
  const container =
    focused instanceof HTMLElement && focused !== document.body
      ? focused.parentElement
      : null;
  (container ?? document.body).append(area);
  area.focus({preventScroll: true});
  area.select();
  try {
    // eslint-disable-next-line @typescript-eslint/no-deprecated -- the only path outside a secure context.
    return document.execCommand('copy');
  } finally {
    if (focused instanceof HTMLElement) {
      focused.focus({preventScroll: true});
    }
    area.remove();
    if (selection !== null) {
      selection.removeAllRanges();
      for (const range of ranges) {
        selection.addRange(range);
      }
    }
  }
}

async function writeClipboard(text: string): Promise<boolean> {
  try {
    // Absent outside a secure context, whatever the type says.
    // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
    if (navigator.clipboard !== undefined) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // Refused, as when the document lacks focus: try the selection instead.
  }
  return copyBySelection(text);
}

/**
 * Copies text to the clipboard and reports the outcome for
 * COPY_FEEDBACK_MS, through `status` and `message`.
 */
export function useCopy(): CopyState {
  const [status, setStatus] = useState<CopyStatus>('idle');
  const timer = useRef<number | undefined>(undefined);
  const mounted = useRef(false);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      window.clearTimeout(timer.current);
    };
  }, []);

  const copy = useCallback(async (text: string) => {
    const copied = await writeClipboard(text);
    if (!mounted.current) {
      return;
    }
    window.clearTimeout(timer.current);
    setStatus(copied ? 'copied' : 'failed');
    timer.current = window.setTimeout(() => {
      setStatus('idle');
    }, COPY_FEEDBACK_MS);
  }, []);

  return {status, message: MESSAGES[status], copy};
}
