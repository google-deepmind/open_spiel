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
import {act, useEffect} from 'react';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import type {Rendered} from '../testing/render.tsx';
import {render} from '../testing/render.tsx';
import {restoreProperties, stubProperty} from '../testing/stub_property.ts';
import type {CopyState} from './use_copy.ts';
import {COPY_FEEDBACK_MS, useCopy} from './use_copy.ts';

let rendered: Rendered | undefined;
let latest: CopyState | undefined;

/** Reports what useCopy returns after every render. */
function Probe({report}: {report: (seen: CopyState) => void}): JSX.Element {
  const seen = useCopy();
  useEffect(() => {
    report(seen);
  });
  return <></>;
}

function remember(seen: CopyState): void {
  latest = seen;
}

async function copy(text: string): Promise<CopyState> {
  rendered ??= render(<Probe report={remember} />);
  await act(async () => {
    await latest?.copy(text);
  });
  if (latest === undefined) {
    throw new Error('The probe did not render.');
  }
  return latest;
}

/** Stubs the Clipboard API with `writeText`; jsdom has none. */
function stubClipboard(writeText?: (text: string) => Promise<void>): void {
  stubProperty(navigator, 'clipboard', writeText && {writeText});
}

/** Stubs document.execCommand, which jsdom lacks, to answer `result`. */
function stubExecCommand(result: boolean): ReturnType<typeof vi.fn> {
  const execCommand = vi.fn(() => result);
  stubProperty(document, 'execCommand', execCommand);
  return execCommand;
}

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  rendered?.unmount();
  rendered = undefined;
  latest = undefined;
  vi.useRealTimers();
  vi.restoreAllMocks();
  restoreProperties();
});

describe('useCopy', () => {
  it('is idle and silent before a copy', () => {
    rendered = render(<Probe report={remember} />);
    expect(latest).toMatchObject({status: 'idle', message: ''});
  });

  it('writes through the Clipboard API', async () => {
    const writeText = vi.fn(() => Promise.resolve());
    stubClipboard(writeText);
    const execCommand = stubExecCommand(true);
    expect(await copy('[0, 4, 8]')).toMatchObject({
      status: 'copied',
      message: 'Copied to the clipboard.',
    });
    expect(writeText).toHaveBeenCalledExactlyOnceWith('[0, 4, 8]');
    expect(execCommand).not.toHaveBeenCalled();
  });

  it(`clears the outcome after ${String(COPY_FEEDBACK_MS)} ms`, async () => {
    stubClipboard(() => Promise.resolve());
    await copy('text');
    act(() => {
      vi.advanceTimersByTime(COPY_FEEDBACK_MS - 1);
    });
    expect(latest?.status).toBe('copied');
    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(latest).toMatchObject({status: 'idle', message: ''});
  });

  it('stays silent when a copy settles after unmounting', async () => {
    let settle = (): void => undefined;
    stubClipboard(
      () =>
        new Promise<void>(resolve => {
          settle = resolve;
        }),
    );
    const view = render(<Probe report={remember} />);
    const pending = latest?.copy('text');
    view.unmount();
    settle();
    await pending;
    expect(vi.getTimerCount()).toBe(0);
  });

  it('restarts the interval on a second copy', async () => {
    stubClipboard(() => Promise.resolve());
    await copy('first');
    act(() => {
      vi.advanceTimersByTime(COPY_FEEDBACK_MS - 1);
    });
    await copy('second');
    act(() => {
      vi.advanceTimersByTime(COPY_FEEDBACK_MS - 1);
    });
    expect(latest?.status).toBe('copied');
  });
});

describe('useCopy through the selection', () => {
  it('copies the selection when there is no Clipboard API', async () => {
    stubClipboard();
    let selected: string | undefined;
    const execCommand = vi.fn(() => {
      const area = document.querySelector('textarea');
      selected = area?.value.slice(area.selectionStart, area.selectionEnd);
      return true;
    });
    stubProperty(document, 'execCommand', execCommand);
    expect((await copy('[0, 4, 8]')).status).toBe('copied');
    expect(execCommand).toHaveBeenCalledExactlyOnceWith('copy');
    expect(selected).toBe('[0, 4, 8]');
    expect(document.querySelector('textarea')).toBeNull();
  });

  it('gives back the focus and selection it borrows', async () => {
    stubClipboard();
    stubExecCommand(true);
    const button = document.createElement('button');
    const paragraph = document.createElement('p');
    paragraph.textContent = 'chosen text';
    document.body.append(button, paragraph);
    button.focus();
    const chosen = document.createRange();
    chosen.selectNodeContents(paragraph);
    document.getSelection()?.removeAllRanges();
    document.getSelection()?.addRange(chosen);
    const focus = vi.spyOn(HTMLTextAreaElement.prototype, 'focus');

    await copy('text');

    expect(focus).toHaveBeenCalledExactlyOnceWith({preventScroll: true});
    expect(document.activeElement).toBe(button);
    expect(document.getSelection()?.getRangeAt(0).toString()).toBe(
      'chosen text',
    );
    button.remove();
    paragraph.remove();
  });

  it('copies the selection when the Clipboard API refuses', async () => {
    stubClipboard(() =>
      Promise.reject(new DOMException('', 'NotAllowedError')),
    );
    const execCommand = stubExecCommand(true);
    expect((await copy('text')).status).toBe('copied');
    expect(execCommand).toHaveBeenCalledOnce();
  });

  it('reports a copy that every path refused', async () => {
    stubClipboard();
    stubExecCommand(false);
    expect(await copy('text')).toMatchObject({
      status: 'failed',
      message: 'The browser did not allow copying to the clipboard.',
    });
  });
});
