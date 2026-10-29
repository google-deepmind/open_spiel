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

import {
  type CSSProperties,
  type JSX,
  type ReactNode,
  type Ref,
  useEffect,
  useId,
  useRef,
  useState,
} from 'react';

import {CloseIcon} from '../controls/icons.tsx';
import {
  MAX_NAME_LENGTH,
  type SavedLayout,
  nameProblem,
} from './layout_store.ts';
import {type SavedLayouts, defaultName} from './use_saved_layouts.ts';

/** What the menu's name field is for, while it is shown. */
type Editing =
  | {readonly mode: 'save'}
  | {readonly mode: 'default-game'}
  | {readonly mode: 'default-all'}
  | {readonly mode: 'rename'; readonly layout: SavedLayout};

/** A command that asks "Really …?" before it acts. */
type Risky = 'delete' | 'reset' | 'default' | 'built-in';

interface MenuProps {
  readonly ref: Ref<HTMLDivElement>;
  readonly style: CSSProperties;
  readonly layouts: SavedLayouts;
  /** Closes the menu. */
  readonly onDone: () => void;
}

/**
 * The "Layouts ▾" menu: the layout in use, then the commands. A command
 * that names a layout opens a name field in place of the commands; one that
 * loses work asks first, in place. A failure stays in the open menu.
 */
export function LayoutsMenu({
  ref,
  style,
  layouts,
  onDone,
}: MenuProps): JSX.Element {
  const [editing, setEditing] = useState<Editing | null>(null);
  const items = useRef<HTMLDivElement>(null);
  useEffect(() => {
    items.current?.querySelector<HTMLElement>('[role="menuitem"]')?.focus();
  }, [editing]);
  const {reading, error} = layouts;
  return (
    <div
      ref={ref}
      className="layouts__menu"
      role="menu"
      aria-label="Layouts"
      style={style}
    >
      {reading.ok ? (
        <div className="layouts__using">{layouts.using}</div>
      ) : (
        <div className="layouts__using layouts__problem" role="alert">
          {`${reading.problem} They are left as they are.`}
        </div>
      )}
      {error !== null && (
        <div className="layouts__error">
          <span>{error}</span>
          <button
            type="button"
            className="button button--icon"
            aria-label="Dismiss"
            onClick={layouts.dismissError}
          >
            <CloseIcon />
          </button>
        </div>
      )}
      <div ref={items} className="layouts__items">
        {editing !== null ? (
          <NameForm
            editing={editing}
            layouts={layouts}
            onCancel={() => {
              setEditing(null);
            }}
            onDone={onDone}
          />
        ) : reading.ok ? (
          <MenuItems layouts={layouts} onEdit={setEditing} onDone={onDone} />
        ) : (
          <Item
            onPress={() => {
              if (layouts.clear()) onDone();
            }}
          >
            Clear saved layouts
          </Item>
        )}
      </div>
    </div>
  );
}

function Item({
  onPress,
  disabled = false,
  danger = false,
  title,
  children,
}: {
  readonly onPress: () => void;
  readonly disabled?: boolean;
  readonly danger?: boolean;
  readonly title?: string;
  readonly children: ReactNode;
}): JSX.Element {
  return (
    <button
      type="button"
      role="menuitem"
      className={`layouts__item${danger ? ' layouts__item--danger' : ''}`}
      disabled={disabled}
      title={title}
      onClick={onPress}
    >
      {children}
    </button>
  );
}

interface ItemsProps {
  readonly layouts: SavedLayouts;
  readonly onEdit: (editing: Editing) => void;
  readonly onDone: () => void;
}

/** The commands, in the order the menu lists them. */
function MenuItems({layouts, onEdit, onDone}: ItemsProps): JSX.Element {
  const [risky, setRisky] = useState<Risky | null>(null);
  const {active, effective} = layouts;
  // Asks first; acts, and closes the menu, on the second press.
  const ask = (which: Risky, act: () => boolean): void => {
    if (risky !== which) {
      setRisky(which);
      return;
    }
    if (act()) onDone();
  };
  const run = (act: () => boolean): void => {
    if (act()) onDone();
  };
  return (
    <>
      <Item
        title="Save what is on screen as a layout of this game"
        onPress={() => {
          onEdit({mode: 'save'});
        }}
      >
        Save for this game…
      </Item>
      {active !== null && (
        <ActiveItems layouts={layouts} risky={risky} ask={ask} run={run} />
      )}
      {active === null && (
        <Item
          title="Save what is on screen for this game and open this game with it"
          onPress={() => {
            onEdit({mode: 'default-game'});
          }}
        >
          Make current default for this game…
        </Item>
      )}
      {active?.game !== null && (
        <Item
          title="Copy what is on screen into a layout for all games and open every game without its own default with it"
          onPress={() => {
            onEdit({mode: 'default-all'});
          }}
        >
          Make default for all games…
        </Item>
      )}
      <StopDefaultItems layouts={layouts} run={run} />
      {active !== null && (
        <EditItems layouts={layouts} risky={risky} ask={ask} onEdit={onEdit} />
      )}
      <div className="layouts__separator" role="separator" />
      {effective !== null && (
        <Item
          danger={risky === 'default'}
          title="Put the default layout back as saved; saved layouts are kept"
          onPress={() => {
            ask('default', layouts.resetToDefault);
          }}
        >
          {risky === 'default'
            ? `Really reset to ‘${effective.layout.name}’?`
            : `Reset to ${defaultName(effective)}…`}
        </Item>
      )}
      <Item
        danger={risky === 'built-in'}
        title="Put this game's standard panels back; saved layouts are kept"
        onPress={() => {
          ask('built-in', layouts.resetToBuiltIn);
        }}
      >
        {risky === 'built-in'
          ? 'Really reset to the built-in layout?'
          : 'Reset to built-in layout…'}
      </Item>
    </>
  );
}

/** The commands that stop a layout being a default. */
function StopDefaultItems({
  layouts,
  run,
}: Pick<ActiveProps, 'layouts' | 'run'>): JSX.Element {
  const {gameDefault, allDefault} = layouts;
  return (
    <>
      {gameDefault !== null && (
        <Item
          onPress={() => {
            run(() => layouts.stopDefault('game'));
          }}
        >
          {`Stop using ‘${gameDefault.name}’ as default for this game`}
        </Item>
      )}
      {allDefault !== null && (
        <Item
          onPress={() => {
            run(() => layouts.stopDefault('all'));
          }}
        >
          {`Stop using ‘${allDefault.name}’ as default for all games`}
        </Item>
      )}
    </>
  );
}

interface ActiveProps {
  readonly layouts: SavedLayouts;
  readonly risky: Risky | null;
  readonly ask: (which: Risky, act: () => boolean) => void;
  readonly run: (act: () => boolean) => void;
}

/** The commands that save the active layout and make it a default. */
function ActiveItems({layouts, run}: ActiveProps): JSX.Element {
  const {active, modified, gameDefault, allDefault} = layouts;
  if (active === null) throw new Error('No layout is in use.');
  const name = activeName(layouts, active);
  return (
    <>
      <Item
        title="Make what is on screen the saved layout"
        onPress={() => {
          run(layouts.saveActive);
        }}
      >
        {`Save ${name}${modified ? ' (modified)' : ''}`}
      </Item>
      <Item
        disabled={gameDefault?.id === active.id}
        title="Open this game with this layout"
        onPress={() => {
          run(() => layouts.makeGameDefault());
        }}
      >
        {`Make ${name} default for this game`}
      </Item>
      {active.game === null && (
        <Item
          disabled={allDefault?.id === active.id}
          title="Open every game without its own default with this layout"
          onPress={() => {
            run(() => layouts.makeAllDefault());
          }}
        >
          {`Make ${name} default for all games`}
        </Item>
      )}
    </>
  );
}

/** The commands that reset, rename, move and delete the active layout. */
function EditItems({
  layouts,
  risky,
  ask,
  onEdit,
}: Omit<ActiveProps, 'run'> & {
  readonly onEdit: (editing: Editing) => void;
}): JSX.Element {
  const {active, modified} = layouts;
  if (active === null) throw new Error('No layout is in use.');
  const name = activeName(layouts, active);
  const siblings =
    active.game === null ? layouts.allLayouts : layouts.gameLayouts;
  const index = siblings.findIndex(({id}) => id === active.id);
  return (
    <>
      <Item
        disabled={!modified}
        danger={risky === 'reset'}
        title="Throw away your changes and go back to the saved layout"
        onPress={() => {
          ask('reset', layouts.resetToSaved);
        }}
      >
        {risky === 'reset'
          ? `Really reset ${name}? Your changes are lost`
          : `Reset ${name} to saved…`}
      </Item>
      <Item
        onPress={() => {
          onEdit({mode: 'rename', layout: active});
        }}
      >
        {`Rename ${name}…`}
      </Item>
      <Item
        disabled={index <= 0}
        onPress={() => {
          layouts.move(active, index - 1);
        }}
      >
        {`Move ${name} left`}
      </Item>
      <Item
        disabled={index >= siblings.length - 1}
        onPress={() => {
          layouts.move(active, index + 1);
        }}
      >
        {`Move ${name} right`}
      </Item>
      <Item
        danger={risky === 'delete'}
        onPress={() => {
          ask('delete', () => layouts.remove(active));
        }}
      >
        {risky === 'delete' ? `Really delete ${name}?` : `Delete ${name}…`}
      </Item>
    </>
  );
}

// The active layout's name as the menu says it: with its level when a
// layout of the other level shares the name.
function activeName(layouts: SavedLayouts, active: SavedLayout): string {
  const other = active.game === null ? layouts.gameLayouts : layouts.allLayouts;
  const shared = other.some(
    ({name}) =>
      name.localeCompare(active.name, undefined, {sensitivity: 'base'}) === 0,
  );
  if (!shared) return `‘${active.name}’`;
  return `‘${active.name}’ (${active.game === null ? 'all games' : 'this game'})`;
}

const FIELD_NAMES: Record<Editing['mode'], string> = {
  save: 'New layout name',
  'default-game': 'New layout name',
  'default-all': 'Name for all games',
  rename: 'Layout name',
};

const SUBMIT_NAMES: Record<Editing['mode'], string> = {
  save: 'Save',
  'default-game': 'Save and make default',
  'default-all': 'Save and make default',
  rename: 'Rename',
};

/** The name field a command that names a layout opens. */
function NameForm({
  editing,
  layouts,
  onCancel,
  onDone,
}: {
  readonly editing: Editing;
  readonly layouts: SavedLayouts;
  readonly onCancel: () => void;
  readonly onDone: () => void;
}): JSX.Element {
  const problemId = useId();
  const [name, setName] = useState(
    editing.mode === 'rename' ? editing.layout.name : '',
  );
  const forAll =
    editing.mode === 'default-all' ||
    (editing.mode === 'rename' && editing.layout.game === null);
  const problem = nameProblem(
    name,
    forAll ? layouts.allLayouts : layouts.gameLayouts,
    editing.mode === 'rename' ? editing.layout.id : undefined,
  );
  const submit = (): boolean => {
    switch (editing.mode) {
      case 'save':
        return layouts.saveAs(name);
      case 'default-game':
        return layouts.makeGameDefault(name);
      case 'default-all':
        return layouts.makeAllDefault(name);
      case 'rename':
        return layouts.rename(editing.layout, name);
      default: {
        const unreachable: never = editing;
        throw new Error(`Unhandled layout command: ${String(unreachable)}`);
      }
    }
  };
  return (
    <form
      className="layouts__form"
      onSubmit={event => {
        event.preventDefault();
        if (problem === null && submit()) onDone();
      }}
    >
      <input
        autoFocus
        className="control"
        aria-label={FIELD_NAMES[editing.mode]}
        aria-invalid={name !== '' && problem !== null}
        aria-describedby={problem === null ? undefined : problemId}
        value={name}
        maxLength={MAX_NAME_LENGTH}
        onChange={event => {
          setName(event.currentTarget.value);
        }}
        onKeyDown={event => {
          if (event.key === 'Escape') {
            event.stopPropagation();
            onCancel();
          }
        }}
      />
      {name !== '' && problem !== null && (
        <span id={problemId} className="layouts__problem">
          {problem}
        </span>
      )}
      <button type="submit" className="button" disabled={problem !== null}>
        {SUBMIT_NAMES[editing.mode]}
      </button>
    </form>
  );
}
