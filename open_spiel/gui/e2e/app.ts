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

// The client as a person uses it, for the end-to-end tests. Every control is
// found by its accessible role and name, the names the panels give it:
//
// - the dock: a tab per panel, titled as in src/game/game_panels.ts;
// - a panel's sections, where a panel has them: a tablist "<Panel> sections"
//   of tabs, each with its tabpanel, and a "Pop out <section>" button;
// - the viewer: View's "Perspective" list (or a list or group named "Viewer"
//   or "View as"), offering "Seat <n>" (or "Player <n>"), "Oracle" and
//   "Spectator"; with none, the address's `viewer`;
// - Gameplay: the status line, which gives the returns at the end, and the
//   board, the game's text, a region named "State" or "<seat>'s
//   observation";
// - Actions: the History, Legal and Text sections; the "Legal (<n>)" list,
//   whose chance outcomes read "p=…", and, at the end, "Game finished. No
//   legal actions.";
// - copying: a "Copy to clipboard" button beside the text it copies, named
//   "Copied" for a moment after;
// - Setup › Players: the "Seats" list, a "Player <n>" item per seat, its
//   "Settings" toggle ("Show settings for Player <n>" when read only),
//   "Controller", "Autoplay", "Step" and "Save settings for Player <n>";
// - Setup › Game options: a control per parameter, named as the parameter
//   for people ("Board Size"), which starts a new game when it changes;
// - Setup › Position: "Set state", its "Action sequence …" field and
//   "Apply", and "Reset", confirmed by the "Reset Game?" dialog's "Reset";
// - the replay control: the "Replay" group in the footer of Actions ›
//   History, "Go to start", "Previous move", "Next move" and "Go to end";
// - the header: the "Saved layouts" group, its "Layouts ▾" menu, the
//   "Panel options" button and the "Settings" button, its dialog and the
//   "Theme" group;
// - saved layouts: in the "Layouts" menu, "Save for this game…", the
//   "New layout name" field and "Save"; each saved layout a button in the
//   "Saved layouts" group, named by the layout.
//
// The dock's tabs are the one exception: the dock library draws them as
// plain elements with no role, so they are found by its class names.

import {type Locator, type Page, expect} from '@playwright/test';

import type {CreateSession, Frame} from '../src/contract/contract.ts';
import {type FrameLog, type ViewerChoice, isViewer} from './frames.ts';

/** A controller as the Players panel offers it. */
export interface ControllerChoice {
  /** The name the server knows it by. */
  readonly name: 'human' | 'random' | 'mcts';
  /** The label the panel shows. */
  readonly label: 'Human' | 'Random' | 'MCTS';
}

/**
 * The controllers every test uses. A person, playing through the Legal list.
 */
export const HUMAN: ControllerChoice = {name: 'human', label: 'Human'};
/** A bot that plays a uniformly random legal action. */
export const RANDOM: ControllerChoice = {name: 'random', label: 'Random'};
/** A bot that plays by Monte Carlo tree search. */
export const MCTS: ControllerChoice = {name: 'mcts', label: 'MCTS'};

const VIEWER_CONTROL = /^(viewer|view as|perspective)$/i;

/** `text`, its regular expression syntax escaped. */
function escape(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/** A regular expression matching `text` exactly. */
export function exactly(text: string): RegExp {
  return new RegExp(`^${escape(text)}$`);
}

function viewerOption(choice: ViewerChoice): RegExp {
  if (choice === 'oracle') return /^oracle\b/i;
  if (choice === 'spectator') return /^spectator\b/i;
  return new RegExp(`^(seat|player) ${String(choice)}\\b`, 'i');
}

/** The client, driven through the page. */
export class App {
  readonly page: Page;
  readonly frames: FrameLog;

  constructor(page: Page, frames: FrameLog) {
    this.page = page;
    this.frames = frames;
  }

  /**
   * Opens `game` at its page and waits for its first frame.
   *
   * @returns The first frame of the game's session.
   */
  async openGame(game: string): Promise<Frame> {
    const before = this.frames.count;
    await this.page.goto(`/game/${encodeURIComponent(game)}`);
    await expect(this.dockTab('Gameplay')).toBeVisible();
    return this.frames.waitFor(
      frame => frame.session.game === game,
      before,
      `the first ${game} frame`,
    );
  }

  /**
   * Starts a session of `body` through the server's API, for a session no
   * panel starts (chance chosen in the browser, a seat count Setup does not
   * offer), and opens its game page as `viewer`.
   *
   * @returns The first frame the page shows.
   */
  async openSession(body: CreateSession, viewer: ViewerChoice): Promise<Frame> {
    const response = await this.page.request.post(
      `/api/sessions?viewer=${String(viewer)}`,
      {data: body},
    );
    expect(response.ok(), await response.text()).toBe(true);
    const {session} = (await response.json()) as Frame;
    const before = this.frames.count;
    const query = new URLSearchParams({
      session: session.id,
      viewer: String(viewer),
    });
    await this.page.goto(`/game/${encodeURIComponent(body.game)}?${query}`);
    await expect(this.dockTab('Gameplay')).toBeVisible();
    return this.frames.waitFor(
      frame =>
        frame.session.id === session.id && isViewer(frame.viewer, viewer),
      before,
      `the first frame of session ${session.id}`,
    );
  }

  /** The dock's tab for the panel titled `title`. */
  dockTab(title: string): Locator {
    return this.page.locator('.dv-tab').filter({hasText: exactly(title)});
  }

  /**
   * Brings the panel titled `title` to the front of its group.
   *
   * @returns The group, showing the panel's content.
   */
  async showPanel(title: string): Promise<Locator> {
    const tab = this.dockTab(title);
    await tab.click();
    return this.page.locator('.dv-groupview').filter({has: tab});
  }

  /**
   * Shows the section `name` of the panel titled `panel`: its tab, where
   * the panel has a tab per section, else the whole panel.
   */
  async showSection(panel: string, name: string): Promise<Locator> {
    const group = await this.showPanel(panel);
    const sections = group.getByRole('tablist', {name: `${panel} sections`});
    if ((await sections.count()) === 0) return group;
    const tab = sections.getByRole('tab', {name: new RegExp(`^${name}\\b`)});
    await tab.click();
    return group.getByRole('tabpanel', {name: new RegExp(`^${name}\\b`)});
  }

  /** The Gameplay panel, in front. */
  gameplay(): Promise<Locator> {
    return this.showPanel('Gameplay');
  }

  /**
   * Views the game as `choice`, and waits for the first frame built for
   * that viewer.
   */
  async setViewer(choice: ViewerChoice): Promise<Frame> {
    // From the frame on screen, which is the answer when `choice` is the
    // viewer already.
    const before = Math.max(0, this.frames.count - 1);
    let control = await this.viewerControl();
    if (control === null && (await this.dockTab('View').count()) > 0) {
      await this.showPanel('View');
      control = await this.viewerControl();
    }
    if (control === null) {
      // No panel on screen chooses the viewer: the address does, as a
      // reload keeps the session and shows it to the viewer named.
      const url = new URL(this.page.url());
      url.searchParams.set('viewer', String(choice));
      await this.page.goto(url.toString());
      return this.frames.waitFor(
        frame => isViewer(frame.viewer, choice),
        before,
        `a frame for viewer ${String(choice)}`,
      );
    }
    const option = viewerOption(choice);
    const kind = await control.evaluate(element => element.tagName);
    if (kind === 'SELECT') {
      const labels = await control.getByRole('option').allTextContents();
      const label = labels.find(text => option.test(text.trim()));
      if (label === undefined) {
        throw new Error(
          `The viewer list offers no ${String(option)}: ${labels.join(', ')}.`,
        );
      }
      await control.selectOption({label});
    } else {
      await control
        .getByRole('button', {name: option})
        .or(control.getByRole('radio', {name: option}))
        .click();
    }
    return this.frames.waitFor(
      frame => isViewer(frame.viewer, choice),
      before,
      `a frame for viewer ${String(choice)}`,
    );
  }

  private async viewerControl(): Promise<Locator | null> {
    const control = this.page
      .getByRole('combobox', {name: VIEWER_CONTROL})
      .or(this.page.getByRole('group', {name: VIEWER_CONTROL}))
      .or(this.page.getByRole('radiogroup', {name: VIEWER_CONTROL}))
      .filter({visible: true})
      .first();
    return (await control.count()) === 0 ? null : control;
  }

  /** Actions › Legal, in front: the chance outcomes and the legal list. */
  legal(): Promise<Locator> {
    return this.showSection('Actions', 'Legal');
  }

  /** The buttons of the legal list in `container`. */
  legalButtons(container: Locator): Locator {
    return container
      .getByRole('list', {name: /^Legal \(\d+\)$/})
      .getByRole('button');
  }

  /** The status line's returns, given once the game is over. */
  returnsSentence(gameplay: Locator): Locator {
    return gameplay.getByText(/Returns: seat 0: /);
  }

  /**
   * Plays the first legal action in Actions › Legal, then the next, until
   * the game is over; the other seats play as their controllers do.
   *
   * @returns The terminal frame.
   * @throws Error when the game is not over after `maxMoves` moves.
   */
  async playToEnd(game: string, maxMoves: number): Promise<Frame> {
    const legal = await this.legal();
    const over = legal.getByText('Game finished. No legal actions.');
    for (let move = 0; move < maxMoves; move++) {
      const first = this.legalButtons(legal).first();
      await expect(over.or(first)).toBeVisible();
      if (await over.isVisible()) {
        return this.terminalFrame(game);
      }
      await Promise.all([
        this.page.waitForResponse(
          response =>
            response.request().method() === 'POST' &&
            new URL(response.url()).pathname.endsWith('/actions'),
        ),
        first.click(),
      ]);
    }
    await expect(over).toBeVisible();
    return this.terminalFrame(game);
  }

  private terminalFrame(game: string): Promise<Frame> {
    return this.frames.waitFor(
      frame => frame.session.game === game && frame.status === 'terminal',
      0,
      `a terminal ${game} frame`,
    );
  }

  /** Setup › Players' row for `seat`. */
  async seat(seat: number): Promise<Locator> {
    const players = await this.showSection('Setup', 'Players');
    return players
      .getByRole('list', {name: 'Seats'})
      .getByRole('listitem', {name: `Player ${String(seat)}`, exact: true});
  }

  /**
   * Hands `seat` to `controller` through Setup › Players, and waits for the
   * frame that shows it.
   */
  async setController(
    seat: number,
    controller: ControllerChoice,
    autoplay: boolean,
  ): Promise<Frame> {
    const name = `Player ${String(seat)}`;
    const row = await this.seat(seat);
    const show = row.getByRole('button', {name: 'Settings', exact: true});
    if (await show.isVisible()) await show.click();
    await row
      .getByRole('combobox', {name: 'Controller'})
      .selectOption({label: controller.label});
    if (controller.name !== HUMAN.name) {
      await row.getByRole('checkbox', {name: 'Autoplay'}).setChecked(autoplay);
    }
    const before = this.frames.count;
    await row.getByRole('button', {name: `Save settings for ${name}`}).click();
    const autoplays = autoplay && controller.name !== HUMAN.name;
    return this.frames.waitFor(
      frame => {
        const view = frame.seats[seat];
        return (
          view?.controller === controller.name && view.autoplay === autoplays
        );
      },
      before,
      `a frame with ${name} played by ${controller.label}`,
    );
  }

  /**
   * Sets the game parameter labelled `label` to `value` in Setup › Game
   * options, which starts a new game with it.
   *
   * @param name The parameter as the session names it, such as `board_size`.
   * @param label The parameter as the panel names it, such as "Board Size".
   * @returns The new session's first frame.
   */
  async startWithParameter(
    name: string,
    label: string,
    value: string,
  ): Promise<Frame> {
    const options = await this.showSection('Setup', 'Game options');
    const before = this.frames.count;
    const list = options.getByRole('combobox', {name: label, exact: true});
    if ((await list.count()) > 0) {
      await list.selectOption({value});
    } else {
      const field = options
        .getByRole('spinbutton', {name: label, exact: true})
        .or(options.getByRole('textbox', {name: label, exact: true}));
      await field.fill(value);
      await field.press('Enter');
    }
    return this.frames.waitFor(
      frame => String(frame.session.parameters[name]) === value,
      before,
      `a new session with ${name} ${value}`,
    );
  }

  /** Resets the game through Setup › Position, confirming the reset. */
  async resetGame(): Promise<void> {
    const position = await this.showSection('Setup', 'Position');
    await position.getByRole('button', {name: 'Reset', exact: true}).click();
    await this.page
      .getByRole('dialog', {name: 'Reset Game?'})
      .getByRole('button', {name: 'Reset', exact: true})
      .click();
  }

  /** The replay control, in the footer of Actions › History, on screen. */
  async replay(): Promise<Locator> {
    await this.showSection('Actions', 'History');
    return this.page
      .getByRole('group', {name: 'Replay', exact: true})
      .filter({visible: true})
      .first();
  }

  /** Chooses `theme` ("System", "Light" or "Dark") in the settings. */
  async chooseTheme(theme: 'System' | 'Light' | 'Dark'): Promise<void> {
    // The header's: a seat's Settings toggle shares the name.
    const settings = this.page.getByRole('banner').getByRole('button', {
      name: 'Settings',
      exact: true,
    });
    if ((await settings.getAttribute('aria-expanded')) !== 'true') {
      await settings.click();
    }
    const dialog = this.page.getByRole('dialog', {name: 'Settings'});
    const option = dialog
      .getByRole('group', {name: 'Theme'})
      .getByRole('button', {name: theme, exact: true});
    await option.click();
    await expect(option).toHaveAttribute('aria-pressed', 'true');
    await this.page.keyboard.press('Escape');
    await expect(dialog).toBeHidden();
  }

  /** Opens the "Panel options" drawer, unless it is open. */
  async openPanelOptions(): Promise<void> {
    const drawer = this.page.getByRole('button', {
      name: 'Panel options',
      exact: true,
    });
    if ((await drawer.getAttribute('aria-expanded')) !== 'true') {
      await drawer.click();
    }
  }

  /** The header's button that restores the saved layout `name`. */
  savedLayout(name: string): Locator {
    return this.page
      .getByRole('group', {name: 'Saved layouts'})
      .getByRole('button', {name: new RegExp(`^${escape(name)}\\b`)});
  }

  /** Saves the dock's arrangement as the layout `name`, for this game. */
  async saveLayoutAs(name: string): Promise<void> {
    await this.page.getByRole('button', {name: /^Layouts\b/}).click();
    const menu = this.page.getByRole('menu', {name: 'Layouts'});
    await menu.getByRole('menuitem', {name: 'Save for this game…'}).click();
    await menu.getByRole('textbox', {name: 'New layout name'}).fill(name);
    await menu.getByRole('button', {name: 'Save', exact: true}).click();
    await expect(this.savedLayout(name)).toHaveAttribute(
      'aria-pressed',
      'true',
    );
  }
}
