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

import type {FocusEvent, JSX, KeyboardEvent, ReactNode} from 'react';
import {useEffect, useId, useRef, useState} from 'react';

import type {ThemePreference} from '../theme/theme.ts';
import {THEME_PREFERENCES} from '../theme/theme.ts';
import {useTheme} from '../theme/theme_provider.tsx';
import {GearIcon} from './icons.tsx';
import {Segmented} from './segmented.tsx';

import './controls.css';

const THEME_LABELS: Record<ThemePreference, string> = {
  system: 'System',
  light: 'Light',
  dark: 'Dark',
};

const THEME_OPTIONS = THEME_PREFERENCES.map(value => ({
  value,
  label: THEME_LABELS[value],
}));

/**
 * The Settings button, a gear, and the dialog it opens under it: a
 * "Settings" title, the theme, then any further settings given as
 * `children`, each a label over its options. Opening moves focus into the dialog. It
 * closes on Escape, which returns focus to the button, on a press anywhere
 * outside it, and when focus moves outside it.
 */
export function SettingsMenu({children}: {children?: ReactNode}): JSX.Element {
  const [open, setOpen] = useState(false);
  const menu = useRef<HTMLDivElement>(null);
  const dialog = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const dialogId = useId();

  useEffect(() => {
    if (!open) {
      return;
    }
    dialog.current?.querySelector('button')?.focus();
    const onPointerDown = (event: PointerEvent): void => {
      const inside =
        event.target instanceof Node && menu.current?.contains(event.target);
      if (inside !== true) {
        setOpen(false);
      }
    };
    document.addEventListener('pointerdown', onPointerDown);
    return () => {
      document.removeEventListener('pointerdown', onPointerDown);
    };
  }, [open]);

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    if (open && event.key === 'Escape') {
      event.stopPropagation();
      setOpen(false);
      trigger.current?.focus();
    }
  };

  // Focus leaving for nowhere (the page itself, another window) keeps the
  // dialog open; a press outside closes it then.
  const onBlur = (event: FocusEvent<HTMLDivElement>): void => {
    const next = event.relatedTarget;
    if (next !== null && !event.currentTarget.contains(next)) {
      setOpen(false);
    }
  };

  return (
    <div
      ref={menu}
      className="settings-menu"
      onKeyDown={onKeyDown}
      onBlur={onBlur}
    >
      <button
        ref={trigger}
        type="button"
        className="button button--icon settings-menu__trigger"
        aria-label="Settings"
        title="Settings"
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-controls={open ? dialogId : undefined}
        onClick={() => {
          setOpen(!open);
        }}
      >
        <GearIcon />
      </button>
      {open && (
        <div
          ref={dialog}
          id={dialogId}
          role="dialog"
          aria-label="Settings"
          className="settings-menu__dialog"
        >
          <SettingsContent>{children}</SettingsContent>
        </div>
      )}
    </div>
  );
}

// The dialog's title, the theme, then the further settings.
function SettingsContent({children}: {children?: ReactNode}): JSX.Element {
  const {preference, setPreference} = useTheme();
  return (
    <>
      <h2 className="settings-menu__title" aria-hidden>
        Settings
      </h2>
      <Segmented
        label="Theme"
        value={preference}
        options={THEME_OPTIONS}
        onChange={setPreference}
      />
      {children}
    </>
  );
}
