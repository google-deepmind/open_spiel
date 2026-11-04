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

// Dockview keeps a tab that only partly fits its strip in the strip, where a
// stub such as "Sta" reads as a different word, and lists it in the group's
// overflow menu as well. These functions hide such stubs, keep the active tab
// whole, and undo the sideways scroll that bringing a tab into view leaves on
// the dock's own containers.

/** The class of a tab that does not fit whole in its strip; CSS hides it. */
export const CLIPPED_TAB_CLASS = 'dock-tab-clipped';

// Sub-pixel layout rounding is not clipping.
const TOLERANCE = 0.5;

/**
 * Marks each tab partly outside its strip as clipped; the active tab never is.
 * A strip whose active tab was hidden, or is cut while `reveal` is set, first
 * scrolls so that the active tab shows whole and the strip starts on a whole
 * tab.
 */
export function markClippedTabs(root: ParentNode, reveal: boolean): void {
  for (const strip of root.querySelectorAll<HTMLElement>(
    '.dv-tabs-container',
  )) {
    const tabs = [...strip.querySelectorAll<HTMLElement>(':scope > .dv-tab')];
    const box = strip.getBoundingClientRect();
    if (box.width <= 0) {
      continue;
    }
    const active = tabs.findIndex(tab =>
      tab.classList.contains('dv-active-tab'),
    );
    const activeTab = tabs[active];
    if (activeTab !== undefined) {
      const wasHidden = activeTab.classList.contains(CLIPPED_TAB_CLASS);
      if (wasHidden || (reveal && isClipped(activeTab, box))) {
        revealTab(strip, activeTab, tabs.slice(0, active + 1), box);
      }
    }
    for (const tab of tabs) {
      tab.classList.toggle(
        CLIPPED_TAB_CLASS,
        tab !== activeTab && isClipped(tab, box),
      );
    }
  }
}

function isClipped(tab: HTMLElement, box: DOMRect): boolean {
  const rect = tab.getBoundingClientRect();
  return rect.left < box.left - TOLERANCE || rect.right > box.right + TOLERANCE;
}

/**
 * Scrolls `strip` to the active tab if it is cut on the left, and otherwise
 * to the earliest of `upToActive` (the tabs up to and including the active
 * one) from which the active tab fits.
 */
function revealTab(
  strip: HTMLElement,
  activeTab: HTMLElement,
  upToActive: readonly HTMLElement[],
  box: DOMRect,
): void {
  // Tab edges in the strip's scrolled content, which scrolling does not move.
  const left = (tab: HTMLElement): number =>
    tab.getBoundingClientRect().left - box.left + strip.scrollLeft;
  const activeLeft = left(activeTab);
  const activeRight = activeLeft + activeTab.getBoundingClientRect().width;
  const cutOnTheLeft = activeLeft < strip.scrollLeft - TOLERANCE;
  const start = cutOnTheLeft
    ? activeTab
    : upToActive.find(tab => activeRight - left(tab) <= box.width + TOLERANCE);
  if (start !== undefined) {
    strip.scrollLeft = left(start);
  }
}

/**
 * Undoes a sideways scroll of a container in the dock's frame. Dockview
 * brings a tab picked from the overflow menu into view with `scrollIntoView`,
 * which also scrolls every scrollable ancestor and leaves part of a group off
 * screen. Tab strips keep their scroll.
 */
export function resetDockScroll(element: Element): void {
  if (
    element instanceof HTMLElement &&
    element.scrollLeft !== 0 &&
    !element.classList.contains('dv-tabs-container')
  ) {
    element.scrollLeft = 0;
  }
}

/**
 * Keeps the tabs under `root` marked as the dock changes, at most once a
 * frame. Dockview sizes its groups and tabs with inline styles and marks the
 * active tab with a class, so a change to either outside panel content is a
 * resize, a tab or group change, or a new active tab, and reveals the active
 * tab. A scroll re-marks without revealing, so a strip the user scrolled stays
 * where it is. Returns a function that stops watching.
 */
export function watchTabStrips(root: HTMLElement): () => void {
  let frame = 0;
  let reveal = false;
  const schedule = (revealActive: boolean): void => {
    reveal ||= revealActive;
    if (frame !== 0) {
      return;
    }
    frame = requestAnimationFrame(() => {
      frame = 0;
      markClippedTabs(root, reveal);
      reveal = false;
    });
  };
  const changed = new MutationObserver(records => {
    if (records.some(record => isDockFrame(record.target))) {
      schedule(true);
    }
  });
  changed.observe(root, {
    subtree: true,
    childList: true,
    attributeFilter: ['class', 'style'],
  });
  const scrolled = ({target}: Event): void => {
    if (target instanceof Element && isDockFrame(target)) {
      resetDockScroll(target);
      schedule(false);
    }
  };
  root.addEventListener('scroll', scrolled, {capture: true});
  return () => {
    cancelAnimationFrame(frame);
    changed.disconnect();
    root.removeEventListener('scroll', scrolled, {capture: true});
  };
}

/** Whether `node` is part of the dock's frame rather than of panel content. */
function isDockFrame(node: Node): boolean {
  return (
    node instanceof Element && node.closest('.dv-content-container') === null
  );
}
