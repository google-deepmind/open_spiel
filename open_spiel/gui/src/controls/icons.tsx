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

// The client's icons. The client ships no images, so each is drawn in CSS,
// in the text colour (controls.css, .icon). Every icon is hidden from
// assistive technology: the control that holds one carries its name.

import type {JSX} from 'react';

import './controls.css';

function Icon({name}: {readonly name: string}): JSX.Element {
  return <span className={`icon icon--${name}`} aria-hidden />;
}

/** Two sheets, one over the other: copy. */
export function CopyIcon(): JSX.Element {
  return <Icon name="copy" />;
}

/** A check mark: done. */
export function CheckIcon(): JSX.Element {
  return <Icon name="check" />;
}

/** An exclamation mark in a circle: it failed. */
export function FailedIcon(): JSX.Element {
  return <Icon name="failed" />;
}

/** Three dots, one above another: more options. */
export function KebabIcon(): JSX.Element {
  return <Icon name="kebab" />;
}

/** A cross: close. */
export function CloseIcon(): JSX.Element {
  return <Icon name="close" />;
}

/** A toothed wheel: settings. */
export function GearIcon(): JSX.Element {
  return <Icon name="gear" />;
}

/** A window with an arrow leaving it: open in a panel of its own. */
export function PopOutIcon(): JSX.Element {
  return <Icon name="pop-out" />;
}

/** A window with a title bar: put back in its panel. */
export function DockIcon(): JSX.Element {
  return <Icon name="dock" />;
}
